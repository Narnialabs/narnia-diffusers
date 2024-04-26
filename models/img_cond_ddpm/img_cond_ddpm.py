import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, ControlNetModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
from utils import grid_imgs
from img_cond_ddpm_pipeline import ImgCondDDPMPipeline
import yaml
from diffusers import DDPMScheduler,DDIMScheduler,\
                      PNDMScheduler,EulerDiscreteScheduler,\
                      EulerAncestralDiscreteScheduler,\
                      DPMSolverMultistepScheduler,\
                      HeunDiscreteScheduler,KDPM2DiscreteScheduler,\
                      DPMSolverSinglestepScheduler,UniPCMultistepScheduler,\
                      DEISMultistepScheduler



class VFEmbedder(nn.Module):
    def __init__(self, shape=(77,768), dtype=torch.float16):
        super(VFEmbedder, self).__init__()
        self.shape = shape
        self.dtype = dtype
            
    def forward(self, vfs):
        vfs_list = []
        for value in vfs:
            vfs_tensor = torch.full(self.shape, value, dtype=self.dtype)
            vfs_list.append(vfs_tensor)
        vfs_tensor = torch.stack(vfs_list)
        
        return vfs_tensor


def get_scheduler(scheduler):
    scheduler_dict = {'ddpm' : DDPMScheduler,
                    'ddim' : DDIMScheduler,
                    'pndm' : PNDMScheduler,
                    'euler_d': EulerDiscreteScheduler,
                    'euler_ad': EulerAncestralDiscreteScheduler,
                    'dpm_solver': DPMSolverMultistepScheduler,
                    'heun': HeunDiscreteScheduler,
                    'kdpm2':KDPM2DiscreteScheduler,
                    'dpm_solver_single': DPMSolverSinglestepScheduler,
                    'unipc': UniPCMultistepScheduler,
                    'deis' : DEISMultistepScheduler }
    return scheduler_dict[scheduler]



class ImgCondDDPM():
    def __init__(self,
                unet_dict =  { 'sample_size':128,
                               'in_channels':1,
                               'out_channels':1,
                               'encoder_hid_dim': 768,
                               'cross_attention_dim' : 1,
                               'layers_per_block': 1,
                             }
                 scheduler = 'ddpm',
                 conditioning_channels = 1,
                ):
    

        # base modules
        self.unet_dict = unet_dict
        self.unet = UNet2DConditionModel(**unet_dict)

        emb_shape = (int(unet_dict['cross_attention_dim']), int(unet_dict['encoder_hid_dim']))
        self.vf_embedder = VFEmbedder(shape=emb_shape)
        self.noise_scheduler = get_scheduler(scheduler)(num_train_timesteps=1000)

        self.conditioning_channels = conditioning_channels
        self.controlnet = ControlNetModel.from_unet(self.unet, 
                          conditioning_channels=self.conditioning_channels)

    def load_cond_weight(self, path):
        self.controlnet.load_state_dict(torch.load(path, map_location='cuda'))
        
    def load_unet_weight(self, path):
        self.unet.load_state_dict(torch.load(path, map_location='cuda'))

    def prepare_infer(self, cond=False, dtype='fp16'):
        
        # Initialize accelerator
        accelerator = Accelerator(
            mixed_precision=dtype,
        )

        if cond:
            self.unet, self.controlnet = accelerator.prepare(
                self.unet, self.controlnet
            )
        else:
            self.unet = accelerator.prepare(
                self.unet
            )


    def infer(self, 
              vf_cond=0.2,
              n=4, 
              seed=1, 
              output_type='pil', 
              num_inference_steps=30,  
              device='cuda'):
        
        pipe = TopoDDPMPipeline( unet=self.unet.to(device), 
                                 scheduler=self.noise_scheduler,
                                 embedder=self.topo_embedder.to(device)).to(device)
        images = pipe( batch_size=n,
                       seed=seed,
                       output_type=output_type,
                       num_inference_steps=num_inference_steps,
                       topo_cond=[topo_cond]
                     ).images
        
        return images


    def save_train_info():
              
        # save model info
        train_info = {
           'output_dir' : output_dir,
           'num_epochs' : num_epochs,
           'gradient_accumulation_steps' : gradient_accumulation_steps,
           'learning_rate' : learning_rate,
           'lr_warmup_steps' : lr_warmup_steps,
           'save_image_epochs' : save_image_epochs,
           'save_model_epochs' : save_model_epochs,
           'mixed_precision' : mixed_precision, 
           'seed' : seed,
        }
        
        with open(f'{output_dir}/train_info.yaml', 'w') as file:
            yaml.dump({'train_info': train_info}, file, default_flow_style=False)
        with open(f'{output_dir}/unet_dict.yaml', 'w') as file:
            yaml.dump({'unet_dict': self.unet_dict}, file, default_flow_style=False)
        

    def train(  self, 
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
               eval_cond =  0.5
             ):
        
        train_info = {
           'output_dir' : output_dir,
           'num_epochs' : num_epochs,
           'gradient_accumulation_steps' : gradient_accumulation_steps,
           'learning_rate' : learning_rate,
           'lr_warmup_steps' : lr_warmup_steps,
           'save_image_epochs' : save_image_epochs,
           'save_model_epochs' : save_model_epochs,
           'mixed_precision' : mixed_precision, 
           'seed' : seed,
           'eval_cond' : eval_cond
        }

        self.controlnet = ControlNetModel.from_unet(self.unet, 
                          conditioning_channels=self.conditioning_channels)
        self.unet.requires_grad_(False)
        self.controlnet.train()


        # opt, lr scheduler
        optimizer = torch.optim.AdamW(self.controlnet.parameters(), lr=learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=(len(loader) * num_epochs),
        )

        # accelerator
        accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            project_dir=os.path.join(output_dir, "logs"),
        )

        self.controlnet, optimizer, loader, lr_scheduler = accelerator.prepare(
            self.controlnet, optimizer, loader, lr_scheduler
        )
        if accelerator.is_main_process:
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
            accelerator.init_trackers("train_example")

        # save
        self.save_train_info(train_info)
        
        # Train loop
        global_step = 0
        for epoch in range(num_epochs):
            progress_bar = tqdm(total=len(loader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(loader):
                imgs, vfs, img_conds = batch
                noise = torch.randn(imgs.shape).to(imgs.device)
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (imgs.shape[0],),
                    device=imgs.device
                ).long()
                noisy_images = self.noise_scheduler.add_noise(imgs, noise, timesteps)
                
                with accelerator.accumulate(self.controlnet):
                    encoder_hidden_states = self.topo_embedder(vfs)
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        noisy_images,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=img_conds.to(dtype=mixed_precision),
                        return_dict=False,
                    )
    
                    # Predict the noise residual
                    model_pred = self.unet(
                        noisy_images,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=[
                            sample.to(dtype=mixed_precision) for sample in down_block_res_samples
                        ],
                        mid_block_additional_residual=mid_block_res_sample.to(dtype=mixed_precision),
                        return_dict=False,
                    )[0]

                    
                    loss = F.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(self.controlnet.parameters(), 1.0)
                    
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
                    images = self.infer(seed=seed, topo_cond=eval_cond)                    
                    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
                    grid_imgs(images, save=f"{output_dir}/samples/{epoch:04d}.png", size=3)
                    
                if (epoch + 1) % save_model_epochs == 0 or epoch == num_epochs - 1:
                    torch.save(self.controlnet.state_dict(), f'{output_dir}/controlnet_weights_{epoch:04d}.pth')