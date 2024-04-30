import os, yaml, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import pickle

from diffusers import UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from accelerate import Accelerator
from tqdm.auto import tqdm

from utils import grid_imgs
from vec_cond_ddpm_pipeline import VecCondDDPMPipeline
from data_loader import Loader

class VecCondDDPM():
    def __init__(self, 
                 loader : Loader):

        self.loader = loader
        
        unet_dict =  { 'sample_size':128,
                       'in_channels':1,
                       'out_channels':1,
                       'encoder_hid_dim': self.loader.emb_dims,
                       'cross_attention_dim' : 1+len(self.loader.vec_cols),
                       'layers_per_block': 1,
                       'block_out_channels':  (128, 256, 512, 512),
                     }
        self.unet_dict = unet_dict
        self.unet_dict['time_embedding_dim'] = unet_dict['block_out_channels'][-1]
        self.unet = UNet2DConditionModel(**unet_dict)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    
    def load_weights(self, unet_path):
        self.unet.load_state_dict(torch.load(unet_path, map_location='cuda'))

    def prepare(self, mixed_precision='fp16'):
        # Initialize accelerator
        accelerator = Accelerator(
            mixed_precision=mixed_precision,
        )
        # to accelerator
        self.unet = accelerator.prepare(
            self.unet
        )
        
    def train( self, 
               output_dir,
               num_epochs = 10,
               gradient_accumulation_steps = 1,
               learning_rate = 1e-4,
               lr_warmup_steps = 100,
               save_image_steps = 100,
               save_model_steps = 100,
               mixed_precision = "fp16", 
               seed = 0,
               eval_conds = []
             ):
        
        # fix random seed
        torch.manual_seed(seed)
        random.seed(seed)
        dataset = self.loader.get_ds()
        
        # Initialize accelerator
        accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            project_dir=os.path.join(output_dir, "logs"),
        )

        # opt, lr scheduler
        optimizer = torch.optim.AdamW(self.unet.parameters(), lr=learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=(len(dataset) * num_epochs),
        )

        # to accelerator
        self.unet, optimizer, dataset, lr_scheduler = accelerator.prepare(
            self.unet, optimizer, dataset, lr_scheduler
        )
        
        if accelerator.is_main_process:
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
            accelerator.init_trackers("train_example")
        
        # save model info
        train_info = {
           'output_dir' : output_dir,
           'num_epochs' : num_epochs,
           'gradient_accumulation_steps' : gradient_accumulation_steps,
           'learning_rate' : learning_rate,
           'lr_warmup_steps' : lr_warmup_steps,
           'mixed_precision' : mixed_precision, 
           'seed' : seed,
        }
        
        with open(f'{output_dir}/train_info.yaml', 'w') as file:
            yaml.dump({'train_info': train_info}, file, default_flow_style=False)
        with open(f'{output_dir}/unet_dict.yaml', 'w') as file:
            yaml.dump({'unet_dict': self.unet_dict}, file, default_flow_style=False)
        with open(f'{output_dir}/loader.pkl', 'wb') as f:
            pickle.dump(self.loader, f)
    
        # Train loop
        global_step = 0
        for epoch in range(num_epochs):
            progress_bar = tqdm(total=len(dataset), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(dataset):
                imgs, vec_conds = batch
        
                noise = torch.randn(imgs.shape).to(imgs.device)
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (imgs.shape[0],),
                    device=imgs.device
                ).long()
                noisy_images = self.noise_scheduler.add_noise(imgs, noise, timesteps)
                
                with accelerator.accumulate(self.unet):
                    noise_pred = self.unet(noisy_images, 
                                           timesteps, 
                                           vec_conds.cuda()).sample

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
                    
                    if (global_step) % save_image_steps == 0:
                        images = self.infer(eval_conds, seed=seed)                    
                        os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
                        grid_imgs(images, save=f"{output_dir}/samples/{global_step:09d}_{epoch}.png", size=3)
                        
                    if (global_step) % save_model_steps == 0:
                        torch.save(self.unet.state_dict(), f'{output_dir}/unet_weights_{global_step:09d}_{epoch}.pth')


    def infer(self, 
              vec_conds=[],
              n=4, 
              seed=1, 
              output_type='pil', 
              num_inference_steps=30,  
              device='cuda'):
        
        vec_conds = self.loader.transfer_vec(vec_conds).unsqueeze(0)
        pipe = VecCondDDPMPipeline( unet=self.unet.to(device), 
                                    scheduler=self.noise_scheduler).to(device)
        images = pipe( batch_size=n,
                       seed=seed,
                       output_type=output_type,
                       num_inference_steps=num_inference_steps,
                       vec_conds=vec_conds
                     ).images
        
        return images
