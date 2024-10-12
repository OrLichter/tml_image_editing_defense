import torch
import pyrallis
import wandb
import numpy as np
import cv2
import os

from PIL import Image
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
	dataset_dir: str = "/home/dcor/orlichter/TML_project/data/images"
	target_image: str = "/home/dcor/orlichter/TML_project/data/shrug.jpg"
	default_prompt: str = "fuji pagoda"
	target_prompt: str = "fuji pagoda on fire"
	device: str = "cuda:0"
	batch_size: int = 1
	epochs: int = 2000
	max_steps: int = 2000
	use_lora: bool = True
	validate_every_k_steps: int = 5
	l2_image_coeff: float = 0.0
	l_inf_image_coeff: float = 1e5
	l2_latent_coeff: float = 1e3
	lr: float = 1e-2
	experiment_name: str = "PGD | pertubations on image | target latents (1e3) & l_inf regularization (1e5) (float32) | lr 1e-2"
	seed: int = 0
	apply_image_pertubation: bool = True

	# Parameters taken from `super_l2` in  https://github.com/MadryLab/photoguard/blob/main/notebooks/demo_complex_attack_inpainting.ipynb
	grad_reps: int = 1
	eps: float = 16
	step_size: float = 1


@pyrallis.wrap()
def main(cfg: Config):
	wandb.login()
	wandb.init(
		project="TML Project",
		config=cfg,
	)
	wandb.run.name = cfg.experiment_name + " | " + wandb.run.name
	wandb.save(os.path.basename(__file__))

	torch.manual_seed(cfg.seed)
	torch_dtype = torch.float32  # if cfg.device == "cpu" else torch.float16
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
	pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, torch_dtype=torch_dtype)
	pipe.unet.requires_grad_(False)
	pipe.vae.requires_grad_(False)
	pipe.text_encoder.requires_grad_(False)
	pipe.text_encoder_2.requires_grad_(False)
	preview_vae.requires_grad_(False)

	if cfg.apply_image_pertubation:
		perturbation = torch.zeros(1, 3, 1024, 1024, device=cfg.device, dtype=torch_dtype, requires_grad=True)
	else:
		perturbation = torch.zeros(1, 4, 128, 128, device=cfg.device, dtype=torch_dtype, requires_grad=True)

	optimizer = torch.optim.Adam([perturbation], lr=cfg.lr)

	generator = torch.manual_seed(0)

	dataset = ImagePromptDataset(cfg.dataset_dir, cfg.default_prompt)
	dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

	# Get target image
	target_image = Image.open(cfg.target_image)
	target_image = dataset.image_transforms(target_image).unsqueeze(0).to(cfg.device, dtype=torch_dtype)
	target_latents = vae.encode(target_image).latent_dist.sample(generator) * vae.config.scaling_factor

	# Define Losses
	l2_distance = LpDistance(2)
	l_inf_distance = LpDistance(float("inf"))
	l2_latent_distance = LpDistance(2)

	step_counter = -1
	pbar = trange(cfg.epochs, desc="Epochs")
	for _ in pbar:
		for batch in dataloader:
			step_counter += 1
			if step_counter >= cfg.max_steps:
				break

			# Set up the batch
			source_image, _ = batch
			source_image = source_image.to(cfg.device, dtype=torch_dtype)

			target_image = source_image.clone()  # In the loss this makes the output_image the source image and vice versa

			if cfg.apply_image_pertubation:
				source_image += perturbation

			encoded_source_image = pipe.vae.encode(source_image).latent_dist.sample(
				generator) * pipe.vae.config.scaling_factor

			if not cfg.apply_image_pertubation:
				encoded_source_image += perturbation

			all_grads = []
			losses = []

			for _ in range(cfg.grad_reps):

				# Compute loss
				noise = torch.randn_like(encoded_source_image)
				timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (1,))
				noisy_model_input = pipe.scheduler.add_noise(encoded_source_image, noise, timesteps)

				output_latents = pipe(
					prompt=cfg.target_prompt,
					num_inference_steps=1,
					generator=generator,
					guidance_scale=guidance_scale,
					latents=noisy_model_input,
					timesteps=timesteps,
					output_type="latent",
				).images[0]

				if cfg.apply_image_pertubation:
					l2_distance_loss = l2_distance(source_image, target_image)
					l_inf_distance_loss = l_inf_distance(source_image, target_image)
					l2_latent_distance_loss = l2_latent_distance(output_latents, target_latents)

				else:
					source_image = \
						preview_vae.decode(encoded_source_image / preview_vae.config.scaling_factor, return_dict=False)[
							0]

					# Apply losses
					l2_distance_loss = l2_distance(source_image, target_image)
					l_inf_distance_loss = l_inf_distance(source_image, target_image)
					l2_latent_distance_loss = l2_latent_distance(encoded_source_image, target_latents)

				loss = 0.
				loss += l2_distance_loss * cfg.l2_image_coeff
				loss += l_inf_distance_loss * cfg.l_inf_image_coeff
				loss += l2_latent_distance_loss * cfg.l2_latent_coeff

				# Compute gradients
				loss.backward()
				all_grads.append(perturbation.grad.clone())
				losses.append(loss.item())

				perturbation.grad.zero_()

			# Average gradients
			grad = torch.stack(all_grads).mean(0)

			pbar.set_description(f'AVG Loss: {np.mean(losses):.3f}')

			wandb.log({"avg_loss": np.mean(losses)})

			# Normalize gradient
			grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, *([1] * (len(grad.shape) - 1)))
			grad_normalized = grad / (grad_norm + 1e-10)

			# Update perturbation
			perturbation.data = perturbation.data - grad_normalized * cfg.step_size

			# Project perturbation
			perturbation.data = torch.clamp(perturbation.data, -cfg.eps, cfg.eps)

			# Ensure the perturbed image is within valid range
			if cfg.apply_image_pertubation:
				perturbed_image = torch.clamp(source_image + perturbation, -1, 1)
				perturbation.data = perturbed_image - source_image

			# Logging
			wandb.log({
				"loss": loss.item(),
				"l2_latent_distance_loss": l2_latent_distance_loss.item(),
				"l2_distance_loss": l2_distance_loss.item(),
				"l_inf_distance_loss": l_inf_distance_loss.item(),
			})
			pbar.set_postfix({"loss": loss.item()})

			# Validation
			if step_counter % cfg.validate_every_k_steps == 0:
				with torch.no_grad():
					validation_image = pipe(
						prompt=cfg.target_prompt,
						num_inference_steps=1,
						generator=generator,
						guidance_scale=guidance_scale,
						latents=noisy_model_input,
						timesteps=timesteps,
					).images[0]
					validation_image = np.array(validation_image)
					source_image = rearrange(source_image[0], "c h w -> h w c").cpu().numpy()
					source_image = ((source_image / 2 + 0.5).clip(0, 1) * 255).astype(np.uint8)
					target_image = rearrange(target_image[0], "c h w -> h w c").detach().cpu().numpy()
					target_image = ((target_image / 2 + 0.5).clip(0, 1) * 255).astype(np.uint8)
					validation_image = np.hstack([source_image, target_image, validation_image])
					validation_image = cv2.putText(validation_image, f"timestep: {timesteps[0].item()}", (10, 30),
												   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

					wandb.log({"output_image": wandb.Image(validation_image)})


if __name__ == '__main__':
	main()
