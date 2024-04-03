from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDPMScheduler,DDIMScheduler,\
                      PNDMScheduler,EulerDiscreteScheduler,\
                      EulerAncestralDiscreteScheduler,DPMSolverMultistepScheduler,\
                      HeunDiscreteScheduler,KDPM2DiscreteScheduler,\
                      DPMSolverSinglestepScheduler,UniPCMultistepScheduler,\
                      DEISMultistepScheduler
from PIL import Image
import numpy as np
import torch, cv2, os


class ControlNetAslan():
    def __init__(self,
                 pipe, 
                 device='cuda'):
      
        self._pipe = pipe
        self._device = device
        self._schedulers={'ddpm' : DDPMScheduler,
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
                          
    def run(self, 
                  sketch_path,
                  prompt,
                  save_dir,
                  lora=None,
                  scheduler = 'unipc',
                  new_w = 768,
                  new_h = 512,
                  negative_prompt = '',
                  guide_scale = 10.,
                  cond_scale = 1.6,
                  num_infer_steps = 50,
                  num_images = 4,
                  seed = -1,
            ):

        self.set_pipe(lora)
        self.generate(
                  sketch_path,
                  prompt,
                  scheduler,
                  new_w,
                  new_h,
                  negative_prompt,
                  guide_scale,
                  cond_scale,
                  num_infer_steps,
                  num_images,
                  seed,
                  save_dir,
            )

    def set_pipe(self, lora=None):
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(self._pipe).to(self._device)
        if lora is not None:
            self.pipe.unet.load_attn_procs(lora)

    def generate( self,
                  sketch_path,
                  prompt,
                  scheduler = 'unipc',
                  new_w = 768,
                  new_h = 512,
                  negative_prompt = '',
                  guide_scale = 10.,
                  cond_scale = 1.6,
                  num_infer_steps = 50,
                  num_images = 4,
                  seed = -1,
                  save_dir=None
              ):
      
        # fixed parameters
        fixed_prompt = f'Photorealistic camera shot of {prompt}, with highly detailed body, especially premium detailed wheel design. Ultra-detailed 8K resolution. Crisp quality.'
        fixed_negative_prompt = " worst quality, low quality, watermark, logo."
        
        self.pipe.scheduler = self._schedulers[scheduler].from_config(self.pipe.scheduler.config)

        # condition image process
        sketch = Image.open(sketch_path).convert('RGB')
        w, h = sketch.size
        w_ratio = w / new_w
        h_ratio = h / new_h
        
        if w_ratio >= h_ratio:
            tmp_size = int(new_w*h/w)
            sketch = sketch.resize((int(new_w), tmp_size))
            pad_size = new_h-tmp_size
            arr = np.array(sketch)
            if pad_size >0:
                pad = np.full((pad_size, new_w, 3), 255.)
                new_sketch = np.concatenate([pad, arr])
        else: 
            tmp_size = int(new_h*w/h)
            sketch = sketch.resize((tmp_size, int(new_h)))
            pad_size = new_w-tmp_size
            arr = np.array(sketch)
        
            if pad_size >0:
                pad = np.full((new_h, pad_size, 3), 255.)
                new_sketch = np.concatenate([arr, pad], axis=1)
                
        new_sketch = 255.-new_sketch
        new_sketch = new_sketch.astype(np.uint8)
        new_sketch = Image.fromarray(new_sketch)

        # generate with pipe
        if seed == -1:
            seed = np.random.randint(0,100000)
        outputs = self.pipe(  fixed_prompt,
                              new_sketch,
                              num_inference_steps=num_infer_steps,
                              guidance_scale=guide_scale,
                              num_images_per_prompt=num_images,
                              controlnet_conditioning_scale=cond_scale,
                              generator=torch.Generator(device=self._device).manual_seed(seed),
                              negative_prompt = negative_prompt+fixed_negative_prompt).images
        
        if save_dir is None:   
          return new_sketch, outputs
        else:
          new_sketch.save(f'{save_dir}/sketch_img.png')
          for i, img in enumerate(outputs):
            img_path = os.path.join(save_dir, f'gen_img_{i}.png')
            img.save(img_path)
