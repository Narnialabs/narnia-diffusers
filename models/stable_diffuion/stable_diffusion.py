import os
import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers import StableDiffusionPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
from PIL import Image

def image_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


class SD():
    def __init__(self,
                 pretrained_model = 'runwayml/stable-diffusion-v1-5',
                 unet_dict = None,
                ):
        
        if unet_dict is not None:
            self.unet = UNet2DConditionModel(**unet_dict)
        else:
            self.unet = UNet2DConditionModel.from_pretrained( pretrained_model, 
                                                              subfolder="unet")
        
        self.pretrained_model = pretrained_model

        self.tokenizer = CLIPTokenizer.from_pretrained( pretrained_model,
                                                        subfolder="tokenizer")    
        self.text_encoder = CLIPTextModel.from_pretrained( pretrained_model,
                                                           subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained( pretrained_model, 
                                                  subfolder="vae")
        self.noise_scheduler = DDPMScheduler.from_pretrained( pretrained_model, 
                                                              subfolder="scheduler")


        
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
               infer_prompts=[],
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

        
        # device
        self.text_encoder.to('cuda', dtype=torch.float16)
        self.vae.to('cuda', dtype=torch.float16)

        # Freeze vae and text_encoder
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.train()

        # Train loop
        global_step = 0
        for epoch in range(num_epochs):
            progress_bar = tqdm(total=len(loader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(loader):
                imgs = batch['pixel_values'].to('cuda')
                tokens = batch['tokens'].to('cuda')
                
                latents = self.vae.encode(imgs.to(torch.float16)).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                noise = torch.randn_like(latents)
                
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (imgs.shape[0],),
                    device=latents.device,
                ).long()

                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                
                with accelerator.accumulate(self.unet):
                    encoder_hidden_states = self.text_encoder(tokens)[0]
                    noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction='mean')
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
                    
                    if len(infer_prompts)!=0:
                        images = self.infer(prompts=infer_prompts, n=4, seed=seed)
                        grid = image_grid(images, rows=1, cols=len(infer_prompts))
                        os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
                        grid.save(f"{output_dir}/samples/{epoch:04d}.png")
                    
                if (epoch + 1) % save_model_epochs == 0 or epoch == num_epochs - 1:
                    pipeline = StableDiffusionPipeline.from_pretrained( 
                        self.pretrained_model,
                        text_encoder=self.text_encoder, 
                        tokenizer=self.tokenizer, 
                        vae=self.vae, 
                        unet=accelerator.unwrap_model(self.unet), 
                        scheduler=self.noise_scheduler
                        )
                    pipeline.save_pretrained(output_dir)
                

    def infer(self, prompts=[], n=1, seed=1, output_type='pil', num_inference_steps=20,  device='cuda'):
        pipeline = StableDiffusionPipeline.from_pretrained( 
           self.pretrained_model,
           unet=self.unet, 
           ).to(device)
        images = pipeline( prompts,
                           output_type=output_type,
                           num_images_per_prompt=n,
                           num_inference_steps=num_inference_steps,
                           generator=torch.manual_seed(seed),
                         ).images

        return images
    
