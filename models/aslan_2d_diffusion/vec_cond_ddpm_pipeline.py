from typing import List, Optional, Tuple, Union
import torch
from diffusers import DiffusionPipeline, ImagePipelineOutput


class VecCondDDPMPipeline(DiffusionPipeline):

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, embedder):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, embedder=embedder)
        
    @torch.no_grad()
    def __call__(
        self,
        vec_conds : list,
        batch_size: int = 1,
        seed : Optional[int] = 0,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        device : str = 'cuda',
    ) -> Union[ImagePipelineOutput, Tuple]:

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)
            
        generator=torch.manual_seed(seed)
        image = torch.randn(image_shape, generator=generator).to(device)
        vec_conds = torch.tensor([vec_conds]*batch_size, dtype=torch.float16)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
       
        for t in self.progress_bar(self.scheduler.timesteps):
            encoder_hidden_states = self.embedder(vec_conds)
            model_output = self.unet(
                image, 
                t, 
                encoder_hidden_states.to(device)
            ).sample
            image = self.scheduler.step(
                model_output, 
                t, 
                image, 
                generator=generator
            ).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)