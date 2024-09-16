import torch
import pyrallis

from diffusers import UNet2DConditionModel, LCMScheduler
from dataclasses import dataclass
from torch.utils.data import DataLoader

from pipelines.sdxl_pipeline import NoiseTrainingPipeline
from data.dataset import ImagePromptDataset

@dataclass
class Config:
    dataset_dir: str = "/home/dcor/orlichter/TML_project/data/single_image_dataset"
    default_prompt: str = "fuji pagoda"
    target_prompt: str = "fuji pagoda"
    device: str = "cpu"
    batch_size: int = 1
    epochs: int = 100
    max_steps: int = 1000
    use_lora: bool = True
    

@pyrallis.wrap()
def main(cfg: Config):
    torch_dtype = torch.float32 if cfg.device == "cpu" else torch.float16
    if cfg.use_lora:
        pipe = NoiseTrainingPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch_dtype
        ).to(cfg.device)
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
        guidance_scale = 0.0

    else:
        unet = UNet2DConditionModel.from_pretrained(
            "latent-consistency/lcm-sdxl",
            torch_dtype=torch_dtype,
            variant="fp16",
        )
        
        pipe = NoiseTrainingPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch_dtype
        ).to(cfg.device)
        guidance_scale = 8.0

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    
    pertubation = torch.randn(1, 4, 128, 128, device=cfg.device, dtype=torch_dtype, requires_grad=True)
    
    generator = torch.manual_seed(0)
    
    dataset = ImagePromptDataset(cfg.dataset_dir, cfg.default_prompt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    step_counter = -1
    for _ in range(cfg.epochs):
        for batch in dataloader:
            step_counter += 1
            if step_counter >= cfg.max_steps:
                break

            source_image, prompt = batch            
            encoded_source_image = pipe.vae.encode(source_image).latent_dist.sample(generator) * pipe.vae.config.scaling_factor
            encoded_source_image += pertubation
            
            noise = torch.randn_like(encoded_source_image)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (1,), device=cfg.device)
            noisy_model_input = pipe.scheduler.add_noise(encoded_source_image, noise, timesteps).to(dtype=torch_dtype)

            output_latents = pipe(
                prompt=prompt[0],
                num_inference_steps=1,
                generator=generator,
                guidance_scale=guidance_scale,
                latents=noisy_model_input,
                timesteps=timesteps,
                output_type="latents",
            ).images[0]
                

if __name__ == '__main__':
    main()