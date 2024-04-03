import os
import torch
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
from PIL import Image

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


class DDPM():
    def __init__(self,
                 unet_dict = { 'sample_size':128,  
                               'in_channels':1, 
                               'out_channels':1, 
                               'layers_per_block':2, 
                               'block_out_channels':(128, 128, 256, 256, 512, 512),
                               'down_block_types':(
                                    "DownBlock2D", 
                                    "DownBlock2D",
                                    "DownBlock2D",
                                    "DownBlock2D",
                                    "AttnDownBlock2D", 
                                    "DownBlock2D",
                                ),
                               'up_block_types':(
                                    "UpBlock2D", 
                                    "AttnUpBlock2D", 
                                    "UpBlock2D",
                                    "UpBlock2D",
                                    "UpBlock2D",
                                    "UpBlock2D",
                                ),
                             }  
                ):


        self.unet = UNet2DModel(**unet_dict)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        
    def load_weights(self, unet_path):
        self.unet.load_state_dict(torch.load(unet_path, map_location='cuda'))
    
    def train( self, 
               loader,
               output_dir,
               num_epochs = 10,
               gradient_accumulation_steps = 1,
               learning_rate = 1e-4,
               lr_warmup_steps = 100,
               save_image_epochs = 10,
               save_model_epochs = 10,
               mixed_precision = "fp16", 
               seed = 0,
             ):

        # Initialize accelerator
        accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(output_dir, "logs"),
        )
        if accelerator.is_main_process:
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
            accelerator.init_trackers("train_example")

        # opt, lr scheduler
        optimizer = torch.optim.AdamW(self.unet.parameters(), lr=learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=(len(loader) * num_epochs),
        )

        # to accelerator
        self.unet, optimizer, loader, lr_scheduler = accelerator.prepare(
            self.unet, optimizer, loader, lr_scheduler
        )
        
        # Train loop
        global_step = 0
        for epoch in range(num_epochs):
            progress_bar = tqdm(total=len(loader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(loader):
                imgs = batch
                noise = torch.randn(imgs.shape).to(imgs.device)
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (imgs.shape[0],),
                    device=imgs.device
                ).long()
                noisy_images = self.noise_scheduler.add_noise(imgs, noise, timesteps)
                
                with accelerator.accumulate(self.unet):
                    noise_pred = self.unet(noisy_images, timesteps, return_dict=False)[0]
                    loss = F.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            
                # End batch
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

            # End Epoch
            if accelerator.is_main_process:
                        
                if (epoch + 1) % save_image_epochs == 0 or epoch == num_epochs - 1:
                    images = self.infer(n=4, seed=seed)
                    grid = make_grid(images, rows=1, cols=4)
                    
                    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
                    grid.save(f"{output_dir}/samples/{epoch:04d}.png")
                    
                if (epoch + 1) % save_model_epochs == 0 or epoch == num_epochs - 1:
                    pipeline = DDPMPipeline(unet=accelerator.unwrap_model(self.unet), scheduler=self.noise_scheduler)
                    pipeline.save_pretrained(output_dir)
                

    def infer(self, n=4, seed=1, output_type='pil', num_inference_steps=30,  device='cuda'):
        pipeline = DDPMPipeline(unet=self.unet.to(device), scheduler=self.noise_scheduler).to(device)
        images = pipeline( batch_size=n,
                           generator=torch.manual_seed(seed),
                           output_type=output_type,
                           num_inference_steps=num_inference_steps,
                         ).images
        
        return images
