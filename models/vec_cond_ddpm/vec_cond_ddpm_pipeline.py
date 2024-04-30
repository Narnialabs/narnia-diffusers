# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Optional, Tuple, Union
import torch
from diffusers import DiffusionPipeline, ImagePipelineOutput


class VecCondDDPMPipeline(DiffusionPipeline):

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        
    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        seed : Optional[int] = 0,
        num_inference_steps: int = 25,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        vec_conds : torch.Tensor = torch.tensor([]),
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
        
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        vec_conds = vec_conds.repeat(4, 1, 1)
        for t in self.progress_bar(self.scheduler.timesteps):
            model_output = self.unet(image, t, vec_conds.to(device)).sample
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)