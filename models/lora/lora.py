import torch, os, math
import diffusers, transformers
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel, DiffusionPipeline
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from PIL import Image
from accelerate import Accelerator
from tqdm import tqdm
import torch.nn.functional as F


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


class LoRA():

    def __init__(self,
                 base_model_path = 'runwayml/stable-diffusion-v1-5',
                 device='cuda',
                 seed = 1):

        self.base_model_path = base_model_path
        self.device = device
        self.seed = seed
        print(f'- base_model_path: {base_model_path}')
        print(f'- device: {device}')
        print(f'- seed: {seed}')

    def set_pipe(self, unet=None, weight_dtype=torch.float16):

        if unet is None:
            self.pipe = DiffusionPipeline.from_pretrained(
                self.base_model_path,
                safety_checker = None,
                torch_dtype=weight_dtype
            )

        else:
            self.pipe = DiffusionPipeline.from_pretrained(
                self.base_model_path,
                unet=unet,
                torch_dtype=weight_dtype,
                safety_checker = None
            )
        self.pipe = self.pipe.to(self.device)

    def set_models(self, pipe):
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.noise_scheduler = pipe.scheduler
        self.vae = pipe.vae
        self.unet = pipe.unet

    def _get_lora_layers(self, unet, rank = 4):
        lora_attn_procs = {}

        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=rank,
            )
        return lora_attn_procs
  
    def infer(self, prompts=[], w=512, h=512, n_infer_steps = 20, n=1):

        images = self.pipe(prompt=prompts,
                        width=w,
                        height=h,
                        num_inference_steps=n_infer_steps,
                        num_images_per_prompt=n,
                        generator=torch.manual_seed(self.seed)
                        ).images
        
        return images

    def _train_loop(self, batch, weight_dtype):
        latents = self.vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states = self.text_encoder(batch["tokens"])[0]
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        return loss



    def train(  self,
                train_dataloader,
                output_dir = None,
                # dataset
                img_w=512,
                img_h=512,
                # train
                mixed_precision = "fp16", #["no", "bf16"]
                num_train_epochs = 1,
                gradient_accumulation_steps=1, # Number of updates steps to accumulate before performing a backward/update pass.
                learning_rate = 1e-4,
                adam_beta1 = 0.9,
                adam_beta2 = 0.999,
                adam_weight_decay = 1e-2,
                adam_epsilon = 1e-08,
                max_train_steps = None,
                checkpointing_steps = None,
                lr_scheduler = "constant", # ["linear", "cosine", "cosine_with_restarts", "polynomial","constant", "constant_with_warmup"]'
                lr_warmup_steps = 500,
                # infer
                infer_prompts=[],
                n_infer_steps=20,
                n_infer_imgs=1,
                ):

        bsz = train_dataloader.batch_size
        # set data type
        if mixed_precision == 'fp16': weight_dtype = torch.float16
        elif mixed_precision == "bf16": weight_dtype = torch.bfloat16
        else: weight_dtype = torch.float32

        # set accelerator
        accelerator = Accelerator(
          gradient_accumulation_steps=gradient_accumulation_steps,
          mixed_precision=mixed_precision,
          log_with=None,
          project_dir=os.path.join(output_dir, "logs"),
        )

        # Handle the repository creation
        if accelerator.is_main_process and output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        # freeze parameters of models to save more memory
        self.set_models(self.pipe)
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Move unet, vae and text_encoder to device and cast to weight_dtype
        self.unet.to(accelerator.device, dtype=weight_dtype)
        self.vae.to(accelerator.device, dtype=weight_dtype)
        self.text_encoder.to(accelerator.device, dtype=weight_dtype)

        # set lora layers
        lora_attn_procs = self._get_lora_layers(self.unet)
        self.unet.set_attn_processor(lora_attn_procs)
        lora_layers = AttnProcsLayers(self.unet.attn_processors)

        # set optimizer
        optimizer = torch.optim.AdamW(
            lora_layers.parameters(),
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        if max_train_steps is None:
            max_train_steps = num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True


        # lr schedular
        lr_scheduler = get_scheduler(
            lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps * accelerator.num_processes,
            num_training_steps=max_train_steps * accelerator.num_processes,
        )

        # accelerator
        lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            lora_layers, optimizer, train_dataloader, lr_scheduler
        )


        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        if overrode_max_train_steps:
            max_train_steps = num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers("text2image-fine-tune")


        # Train!
        total_batch_size = bsz * accelerator.num_processes * gradient_accumulation_steps
        if checkpointing_steps is None:
          checkpointing_steps = int(max_train_steps / num_train_epochs)
        txt = f'''
        ***** Running training *****
          Project directory = {output_dir}
          Num examples = {len(train_dataloader)}
          Num Epochs = {num_train_epochs}
          Instantaneous batch size per device = {bsz}
          Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}
          Gradient Accumulation steps = {gradient_accumulation_steps}
          Total optimization steps = {max_train_steps}
          Checkpointing steps = {checkpointing_steps}
        '''
        print(txt)


        # Only show the progress bar once on each machine.
        global_step = 0
        progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")


        for epoch in range(num_train_epochs):
            self.unet.train()
            train_loss = 0.0
            for step, batch in enumerate(train_dataloader):

                with accelerator.accumulate(self.unet):

                    loss = self._train_loop(batch, weight_dtype)

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(bsz)).mean()
                    train_loss += avg_loss.item() / gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = lora_layers.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()


                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                    if global_step % checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            checkpoints = os.listdir(output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            save_path = os.path.join(output_dir, f"checkpoint-{global_step:06d}")
                            accelerator.save_state(save_path)

                            print(f"Saved state to {save_path}")

                            if infer_prompts is not None:
                                print(f"Running validation with prompt: {infer_prompts}")

                                # create pipeline
                                self.set_pipe(unet=accelerator.unwrap_model(self.unet), weight_dtype = weight_dtype)
                                self.pipe.set_progress_bar_config(disable=True)

                                # run inference
                                images = self.infer(prompts=infer_prompts, w=img_w, h=img_h, n_infer_steps=n_infer_steps, n=n_infer_imgs)
                                grid = image_grid(images, len(infer_prompts), n_infer_imgs)
                                grid.save(f"{output_dir}/{global_step:06d}.png")

                                del self.pipe
                                torch.cuda.empty_cache()

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)


                if global_step >= max_train_steps:
                    break

            # -- End Batch Loop---


        # -- End Epoch Loop ---

        # Save the lora layers
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            self.unet = self.unet.to(torch.float32)
            self.unet.save_attn_procs(output_dir)
        accelerator.end_training()
