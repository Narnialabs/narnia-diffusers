import os, yaml, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from diffusers import DDPMScheduler,DDIMScheduler,\
                      PNDMScheduler,EulerDiscreteScheduler,\
                      EulerAncestralDiscreteScheduler,\
                      DPMSolverMultistepScheduler,\
                      HeunDiscreteScheduler,KDPM2DiscreteScheduler,\
                      DPMSolverSinglestepScheduler,UniPCMultistepScheduler,\
                      DEISMultistepScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from vec_cond_ddpm_pipeline import VecCondDDPMPipeline
from accelerate import Accelerator
from tqdm.auto import tqdm
from utils import grid_imgs
import cv2

class VecEmbedder(nn.Module):
    def __init__(self, emb_dims=256, dtype=torch.float16):
        super(VecEmbedder, self).__init__()
        self.emb_dims = emb_dims
        self.dtype = dtype
        
    def forward(self, x):
        xs = []
        for batch in x:
            batch_x = []
            for param in batch:
                batch_x.append(
                    torch.full((1, self.emb_dims), param, dtype = self.dtype))
                )
            
            batch_x = torch.cat(batch_x)
            xs.append(batch_x)
        x = torch.stack(xs)
        return x

class ImgEmbedder(nn.Module):
    def __init__(self, resize=16, dtype=torch.float16):
        super(ImgEmbedder, self).__init__()
        self.resize = resize
        self.dtype = dtype
        
    def forward(self, x):
        x = F.interpolate(
            x, 
            size=(self.resize, self.resize), 
            mode='bilinear', 
            align_corners=False
        )
        x = torch.flatten(x, start_dim=1)  # 배치 차원을 유지하고 1차원으로 평탄화      
        return x

def get_scheduler(scheduler):
    scheduler_dict = {
        'ddpm' : DDPMScheduler,
        'ddim' : DDIMScheduler,
        'pndm' : PNDMScheduler,
        'euler_d': EulerDiscreteScheduler,
        'euler_ad': EulerAncestralDiscreteScheduler,
        'dpm_solver': DPMSolverMultistepScheduler,
        'heun': HeunDiscreteScheduler,
        'kdpm2':KDPM2DiscreteScheduler,
        'dpm_solver_single': DPMSolverSinglestepScheduler,
        'unipc': UniPCMultistepScheduler,
        'deis' : DEISMultistepScheduler
    }
    return scheduler_dict[scheduler]



class VecCondDDPM():
    def __init__(self,
                 sample_size = 128,
                 in_channels = 1,
                 out_channels = 1,
                 encoder_hid_dim = 128,
                 cross_attention_dim = 128,
                 layers_per_block = 1,
                 block_out_channels = [64, 128, 256, 512],
                 scheduler = 'ddpm',
                 dtype = torch.float16,
                 device = 'cuda'
                ):
    
        # init
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_hid_dim = encoder_hid_dim
        self.cross_attention_dim = cross_attention_dim
        self.layers_per_block = layers_per_block
        self.block_out_channels = block_out_channels
        
        self.scheduler = scheduler
        self.dtype = dtype
        self.device = device
        if self.dtype == torch.float16:
            self.mp = 'fp16'
        else: 
            self.mp = 'no'
        
        # base modules
        self.unet = UNet2DConditionModel(
            sample_size = sample_size,
            in_channels = in_channels,
            out_channels = out_channels,
            encoder_hid_dim = encoder_hid_dim,
            cross_attention_dim = cross_attention_dim,
            layers_per_block = layers_per_block,
            block_out_channels = block_out_channels,

        )
        self.vec_embedder = VecEmbedder(emb_dims=encoder_hid_dim, device=self.device)
        self.noise_scheduler = get_scheduler(scheduler)(num_train_timesteps=1000)

        # Initialize model
        accelerator = Accelerator(mixed_precision=self.mp)
        self.unet, self.vec_embedder = accelerator.prepare(self.unet, self.vec_embedder)
    
        
    def load_unet_weight(self, path):
        self.unet.load_state_dict(torch.load(path, map_location='cuda'))

    def infer(self, 
              vec_conds=[],
              n=4, 
              seed=1, 
              output_type='pil', 
              num_inference_steps=30,
              ):
        
        pipe = VecCondDDPMPipeline( 
            unet=self.unet.to(self.device), 
            scheduler=self.noise_scheduler,
            embedder=self.vec_embedder.to(self.device)
        ).to(self.device)
        
        images = pipe( batch_size=n,
                       seed=seed,
                       output_type=output_type,
                       num_inference_steps=num_inference_steps,
                       vec_conds=vec_conds
                     ).images
        return images
        
        
    def train( self, 
               dataset,
               output_dir,
               num_epochs = 10,
               gradient_accumulation_steps = 1,
               learning_rate = 1e-4,
               lr_warmup_steps = 100,
               save_image_steps = 100,
               save_model_per_epoch = True,
               seed = 0,
               eval_conds = [],
             ):
        
        # fix random seed
        torch.manual_seed(seed)
        random.seed(seed)
      
        # Initialize accelerator
        accelerator = Accelerator(
            mixed_precision=self.mp,
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
        
        # save info parameters
        model_info = {
            'sample_size' : self.sample_size,
            'in_channels' : self.in_channels,
            'out_channels' : self.out_channels,
            'encoder_hid_dim' : self.encoder_hid_dim,
            'cross_attention_dim' : self.cross_attention_dim,
            'layers_per_block' : self.layers_per_block,
            'block_out_channels' : self.block_out_channels,
            'scheduler' : self.scheduler,
            
        }
        train_info = {
           'output_dir' : output_dir,
           'num_epochs' : num_epochs,
           'gradient_accumulation_steps' : gradient_accumulation_steps,
           'learning_rate' : learning_rate,
           'lr_warmup_steps' : lr_warmup_steps,
           'seed' : seed,
        }
        with open(f'{output_dir}/model_info.yaml', 'w') as file:
            yaml.dump(model_info, file, default_flow_style=False)
        with open(f'{output_dir}/train_info.yaml', 'w') as file:
            yaml.dump(train_info, file, default_flow_style=False)
    
        # Train loop
        global_step = 0
        for epoch in range(num_epochs):
            progress_bar = tqdm(total=len(dataset), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(dataset):
                imgs, vec_conds = batch
        
                noise = torch.randn(imgs.shape).to(self.device)
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (imgs.shape[0],),
                    device=self.device
                ).long()
                noisy_images = self.noise_scheduler.add_noise(imgs, noise, timesteps)
                
                with accelerator.accumulate(self.unet):
                    encoder_hidden_states = self.vec_embedder(vec_conds)
                    noise_pred = self.unet(noisy_images, 
                                           timesteps, 
                                           encoder_hidden_states.cuda()).sample

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
                        images = self.infer(
                            eval_conds, 
                            seed=seed
                        )
                        
                        grid_imgs(
                            images, 
                            save=f"{output_dir}/{global_step:09d}_{epoch}.png", 
                            size=3
                        )
                        
            if save_model_per_epoch:
                os.makedirs(os.path.join(output_dir, "weights"), exist_ok=True)
                torch.save(
                    self.unet.state_dict(),
                    f'{output_dir}/weights/unet_weights_{epoch:05d}.pth'
                )

        # final model
        torch.save(
            self.unet.state_dict(),
            f'{output_dir}/weights/unet_weights.pth'
        )
