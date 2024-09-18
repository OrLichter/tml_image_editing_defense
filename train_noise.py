import torch
import pyrallis
import wandb
import numpy as np
import cv2

from diffusers import UNet2DConditionModel, LCMScheduler, AutoencoderKL, AutoencoderTiny
from dataclasses import dataclass
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import trange

from pipelines.sdxl_pipeline import NoiseTrainingPipeline
from data.dataset import ImagePromptDataset
from losses.losses import LpRegularization, LpDistance, CosineSimilarity


@dataclass
class Config:
    dataset_dir: str = "/home/dcor/orlichter/TML_project/data/single_image_dataset"
    default_prompt: str = "fuji pagoda"
    target_prompt: str = "fuji pagoda"
    device: str = "cuda:0"
    batch_size: int = 1
    epochs: int = 1000
    max_steps: int = 1000
    use_lora: bool = True
    validate_every_k_steps: int = 5
    l1_regularization_coeff: float = 0
    l2_image_coeff: float = 1.0
    latent_images_cosine_similarity_coeff: float = 1.0
    experiment_name: str = "image_loss"
    seed: int = 0


@pyrallis.wrap()
def main(cfg: Config):
    wandb.login()
    wandb.init(
        project="TML Project",
        config=cfg,
    )
    wandb.run.name = cfg.experiment_name + " | " + wandb.run.name

    torch.manual_seed(cfg.seed)
    torch_dtype = torch.float32 if cfg.device == "cpu" else torch.float16
    if cfg.use_lora:
        pipe = NoiseTrainingPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch_dtype
        )
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
        )
        guidance_scale = 8.0

    preview_vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl").to(cfg.device, torch_dtype)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype)
    
    pipe.vae = vae
    pipe.to(cfg.device)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    
    pertubation = torch.randn(1, 4, 128, 128, device=cfg.device, dtype=torch_dtype, requires_grad=True)
    optimizer = torch.optim.Adam([pertubation], lr=1e-3)
    
    generator = torch.manual_seed(0)
    
    dataset = ImagePromptDataset(cfg.dataset_dir, cfg.default_prompt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    # Define Losses
    l2_distance = LpDistance(2)  # L1 give inf for some reason
    cosine_similarity = CosineSimilarity()
    
    step_counter = -1
    for _ in trange(cfg.epochs, desc="Epochs"):
        for batch in dataloader:
            step_counter += 1
            if step_counter >= cfg.max_steps:
                break
            
            # Set up the batch
            source_image, prompt = batch
            source_image = source_image.to(cfg.device, dtype=torch_dtype)

            encoded_source_image = pipe.vae.encode(source_image).latent_dist.sample(generator) * pipe.vae.config.scaling_factor       

            encoded_source_image += pertubation

            noise = torch.randn_like(encoded_source_image)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (1,))
            noisy_model_input = pipe.scheduler.add_noise(encoded_source_image, noise, timesteps).to(dtype=torch_dtype)

            # Generate the outputs
            output_latents = pipe(
                prompt=prompt[0],
                num_inference_steps=1,
                generator=generator,
                guidance_scale=guidance_scale,
                latents=noisy_model_input,
                timesteps=timesteps,
                output_type="latent",
            ).images[0]
            
            output_image = preview_vae.decode(encoded_source_image / preview_vae.config.scaling_factor, return_dict=False)[0]
            
            # Apply losses
            l2_distance_loss = l2_distance(output_image, source_image)
            cos_sim_loss = cosine_similarity(output_latents, encoded_source_image)
            
            loss = 0.            
            loss += l2_distance_loss * cfg.l2_image_coeff
            loss += cos_sim_loss * cfg.latent_images_cosine_similarity_coeff
            
            # Optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Logging
            wandb.log({"loss": loss.item(), "image_l2_distance": l2_distance_loss.item(), "latent_cosine_similarity": cos_sim_loss.item()})

            # Validation
            if step_counter % cfg.validate_every_k_steps == 0:
                with torch.no_grad():
                    validation_image = pipe(
                        prompt=prompt[0],
                        num_inference_steps=1,
                        generator=generator,
                        guidance_scale=guidance_scale,
                        latents=noisy_model_input,
                        timesteps=timesteps,
                    ).images[0]
                    validation_image = np.array(validation_image)
                    source_image = rearrange(source_image[0], "c h w -> h w c").cpu().numpy()
                    source_image = ((source_image / 2 + 0.5).clip(0, 1) * 255).astype(np.uint8)
                    output_image = rearrange(output_image[0], "c h w -> h w c").detach().cpu().numpy()
                    output_image = ((output_image / 2 + 0.5).clip(0, 1) * 255).astype(np.uint8)
                    validation_image = np.hstack([source_image, output_image, validation_image])
                    validation_image = cv2.putText(validation_image, f"timestep: {timesteps[0].item()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    wandb.log({"output_image": wandb.Image(validation_image)})    
                

if __name__ == '__main__':
    main()