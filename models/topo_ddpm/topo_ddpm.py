import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
from utils import make_grid
from topo_ddpm_pipeline import TopoDDPMPipeline


class TopoEmbedder(nn.Module):
    def __init__(self, shape = (77,768), dtype=torch.float16):
        super(TopoEmbedder, self).__init__()
        self.shape = shape
        self.dtype = dtype
        
    def forward(self, vfs):
        vfs_list = []
        for value in vfs:
            vfs_tensor = torch.full(self.shape, value, dtype=torch.float16)
            vfs_list.append(vfs_tensor)
        vfs_tensor = torch.stack(vfs_list)
        return vfs_tensor
        
def transfer_log(x, reverse=False):
    if not reverse:
        transformed = torch.log(x + 1) 
    else:
        transformed = torch.exp(x) - 1  
    return transformed


class TopoDDPM():
    def __init__(self,
                unet_dict =  { 'sample_size':128,
                               'in_channels':1,
                               'out_channels':1,
                               'time_embedding_dim':1280,
                               'encoder_hid_dim': 768,
                               'cross_attention_dim' : 77
                               },
                ):

        self.unet_dict = unet_dict
        emb_shape = (unet_dict['cross_attention_dim'], unet_dict['encoder_hid_dim'])
        self.topo_embedder = TopoEmbedder(shape=emb_shape)
        self.unet = UNet2DConditionModel(**unet_dict)
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
        
        # save model info
        train_info = {
           'loader' : loader,
           'output_dir' : output_dir,
           'num_epochs' : num_epochs,
           'gradient_accumulation_steps' = gradient_accumulation_steps,
           'learning_rate' = learning_rate,
           'lr_warmup_steps' = lr_warmup_steps,
           'save_image_epochs' = save_image_epochs,
           'save_model_epochs' = save_model_epochs,
           'mixed_precision' = mixed_precision, 
           'seed' = seed,
        }
        with open(f'{output_dir}/train_info.yaml', 'w') as file:
            yaml.dump({'train_info': train_info}, file, default_flow_style=False)
        with open(f'{output_dir}/unet_dict.yaml', 'w') as file:
            yaml.dump({'unet_dict': self.unet_dict}, file, default_flow_style=False)

        
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
                imgs, vfs = batch
                noise = torch.randn(imgs.shape).to(imgs.device)
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (imgs.shape[0],),
                    device=imgs.device
                ).long()
                noisy_images = self.noise_scheduler.add_noise(imgs, noise, timesteps)
                
                with accelerator.accumulate(self.unet):
                    encoder_hidden_states = self.topo_embedder(vfs)
                    noise_pred = self.unet(noisy_images, timesteps, encoder_hidden_states.cuda()).sample

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
                    images = self.infer(seed=seed)
                    grid = make_grid(images, rows=1, cols=4)
                    
                    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
                    grid.save(f"{output_dir}/samples/{epoch:04d}.png")
                    
                if (epoch + 1) % save_model_epochs == 0 or epoch == num_epochs - 1:
                    torch.save(self.unet.state_dict(), f'{output_dir}/unet_weights_{epoch:04d}.pth')


    def infer(self, 
              topo_cond=0.5,
              n=4, 
              seed=1, 
              output_type='pil', 
              num_inference_steps=30,  
              device='cuda'):
        
        pipe = TopoDDPMPipeline( unet=self.unet, 
                                 scheduler=self.noise_scheduler,
                                 embedder=self.topo_embedder).to(device)
        images = pipe( batch_size=n,
                       seed=seed,
                       output_type=output_type,
                       num_inference_steps=num_inference_steps,
                       topo_cond=[topo_cond]
                     ).images
        
        return images
