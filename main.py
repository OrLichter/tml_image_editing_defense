import dataclasses
import inspect
import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
import wandb
from PIL import Image
from diffusers import AutoPipelineForImage2Image, AutoencoderKL, LCMScheduler
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm
import torchvision.transforms as T

from configs import TrainConfig, InferenceConfig, INFERENCE_PROMPTS
from data.dataset import ImagePromptDataset
from losses import losses


class Trainer:

	def __init__(self, cfg: TrainConfig, use_sdxl: bool = True, use_lcm: bool = False):
		self.cfg = cfg
		self.use_sdxl = use_sdxl
		self.use_lcm = use_lcm
		self.pipeline = self.load_models(use_sdxl=use_sdxl, use_lcm=use_lcm, dtype=torch.float16)
		self.device = torch.device("cuda")

	def run(self) -> Image.Image:
		""" Main training loop """
		wandb.init(
			project="TML Project",
			config=dataclasses.asdict(self.cfg),
			name=self.cfg.experiment_name,
		)
		wandb.save(os.path.basename(__file__))

		source_image, target_image = self._process_images()

		X_adv = source_image.clone()
		target_latent = self.pipeline.vae.encode(target_image).latent_dist.sample()

		iterator = tqdm(range(self.cfg.n_optimization_steps))

		for iteration in iterator:
			all_grads = []
			losses = []

			output_image = None
			prompt = self.cfg.prompts[np.random.randint(0, len(self.cfg.prompts))]
			for i in range(self.cfg.grad_reps):
				# Randomly sample one of the prompts in the set
				c_grad, loss, output_image = self.compute_grad(
					cur_image=X_adv,
					prompt=prompt,
					source_image=source_image,
					target_image=target_image,
					target_latent=target_latent,
				)
				all_grads.append(c_grad)
				losses.append(loss)

			# Aggregate gradients
			grad = torch.stack(all_grads).mean(0)

			# Display average loss
			iterator.set_description_str(f'AVG Loss: {np.mean(losses):.3f}')
			logs = {"avg_loss": np.mean(losses)}

			# Apply the perturbation step (either L2 or Linf)
			X_adv = self.perturbation_step(X_adv, grad, source_image)

			if iteration % self.cfg.image_visualization_interval == 0:
				images = Image.fromarray(np.concatenate([
					T.ToPILImage()((X_adv[0] / 2 + 0.5).clamp(0, 1)),
					T.ToPILImage()((output_image[0] / 2 + 0.5).clamp(0, 1)),
				], axis=1))
				logs.update({
					"train_images": wandb.Image(images, caption=prompt),
				})

			wandb.log(logs)

		torch.cuda.empty_cache()

		X_adv = (X_adv / 2 + 0.5).clamp(0, 1)
		adversarial_image = T.ToPILImage()(X_adv[0]).convert("RGB")
		wandb.log({"final_adversarial_image": wandb.Image(adversarial_image)})
		return adversarial_image

	def compute_grad(self,
					 cur_image: torch.Tensor,
					 prompt: str,
					 source_image: torch.Tensor,
					 target_image: torch.Tensor,
					 target_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		torch.set_grad_enabled(True)
		cur_image = cur_image.clone()
		cur_image.requires_grad = True

		output_latent = self.attack_forward(image=cur_image, prompt=prompt)
		output_image = self.pipeline.vae.decode(output_latent).sample

		# Compute general loss between output image and target image
		if self.cfg.apply_loss_on_images:
			rec_loss = (output_image - target_image).norm(p=2)
		elif self.cfg.apply_loss_on_latents:
			rec_loss = (output_latent - target_latent).norm(p=2)
		else:
			raise ValueError("Please specify whether to apply loss on images or latents")

		# Optionally add loss to minimize the strength of the perturbations applied to the source image
		if self.cfg.perturbation_loss_lambda > 0:
			pert_loss = losses.perturbation_loss(output_image, source_image)
			loss = rec_loss + self.cfg.perturbation_loss_lambda * pert_loss
		else:
			loss = rec_loss

		grad = torch.autograd.grad(loss, [cur_image])[0]
		return grad, loss.item(), output_image

	def attack_forward(self,
					   prompt: Union[str, List[str]],
					   image: Union[torch.Tensor, Image.Image]) -> torch.Tensor:

		# Encode the prompt
		embeds = self._encode_prompt(prompt)
		prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = embeds
		prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
		prompt_embeds = prompt_embeds.detach()

		# Encode the current image into its latents
		image_latents = self.pipeline.vae.encode(image).latent_dist.sample() * 0.18215

		# Get the timesteps to be used here
		self.pipeline.scheduler.set_timesteps(self.cfg.n_denoising_steps_per_iteration)
		timesteps_tensor = self.pipeline.scheduler.timesteps.to(self.device)

		# Limit the timesteps since we know that for editing, we only really want a subset of the timesteps
		if self.cfg.limit_timesteps:
			timesteps_tensor = torch.tensor([t for t in timesteps_tensor if 100 < t < 800], device=self.device)

		# Get additional inputs if using SDXL
		timestep_cond, added_cond_kwargs = None, None
		if self.use_sdxl:
			added_cond_kwargs = self.get_sdxl_additional_inputs(
				prompt_embeds=prompt_embeds,
				pooled_prompt_embeds=pooled_prompt_embeds.detach(),
				negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.detach()
			)

		# Add noise to the input latent
		noise = randn_tensor(image_latents.shape, device=self.device, dtype=torch.float16)
		latents = self.pipeline.scheduler.add_noise(image_latents, noise, timesteps_tensor[:1])

		extra_step_kwargs = {}
		if 'eta' in inspect.signature(self.pipeline.scheduler.step).parameters:
			extra_step_kwargs = {'eta': self.cfg.eta}

		# Forward pass through the UNet
		extra_kwargs = {
			"timestep_cond": timestep_cond,
			"cross_attention_kwargs": {},
			"added_cond_kwargs": added_cond_kwargs,
		} if self.use_sdxl else {}

		for i, t in enumerate(timesteps_tensor):
			latent_model_input = torch.cat([latents] * 2)
			latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)

			noise_pred = self.pipeline.unet(
				latent_model_input,
				t,
				encoder_hidden_states=prompt_embeds,
				**extra_kwargs
			).sample

			noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
			noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)
			latents = self.pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs,
												   return_dict=True).prev_sample

		latents = 1 / 0.18215 * latents
		return latents

	def perturbation_step(self, X_adv: torch.Tensor, grad: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
		"""Apply the perturbation step for either L2 or Linf norm."""
		if self.cfg.norm_type == 'l2':
			# Normalize gradient for L2 norm
			l = len(X.shape) - 1
			grad_norm = torch.norm(grad.detach().reshape(grad.shape[0], -1), dim=1).view(-1, *([1] * l))
			grad_normalized = grad.detach() / (grad_norm + 1e-10)
			X_adv = X_adv - grad_normalized * self.cfg.step_size

			# Apply L2 constraint
			d_x = X_adv - X.detach()
			d_x_norm = torch.renorm(d_x, p=2, dim=0, maxnorm=self.cfg.eps)
			X_adv.data = torch.clamp(X + d_x_norm, self.cfg.min_value, self.cfg.max_value)

		elif self.cfg.norm_type == 'linf':
			# Apply Linf norm step
			X_adv = X_adv - grad.detach().sign() * self.cfg.step_size
			X_adv = torch.minimum(torch.maximum(X_adv, X - self.cfg.eps), X + self.cfg.eps)
			X_adv.data = torch.clamp(X_adv, self.cfg.min_value, self.cfg.max_value)

		return X_adv

	@staticmethod
	def load_models(use_sdxl: bool = True,
					use_lcm: bool = False,
					dtype: torch.dtype = torch.float16) -> AutoPipelineForImage2Image:
		if use_sdxl:
			pipeline = AutoPipelineForImage2Image.from_pretrained(
				"stabilityai/stable-diffusion-xl-base-1.0",
				torch_dtype=dtype,
			)
			pipeline = pipeline.to("cuda")
			vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype).to('cuda')
			pipeline.vae = vae
			if use_lcm:
				pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
				pipeline.load_lora_weights("latent-consistency/lcm-lora-sdxl")
				pipeline.fuse_lora()
		else:
			pipeline = AutoPipelineForImage2Image.from_pretrained(
				"runwayml/stable-diffusion-v1-5",
				torch_dtype=dtype,
			)
			pipeline = pipeline.to("cuda")
			if use_lcm:
				pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
				pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
				pipeline.fuse_lora()

		return pipeline

	def _process_images(self) -> Tuple[torch.Tensor, torch.Tensor]:
		image_transforms = ImagePromptDataset.get_image_transforms()
		source_image = image_transforms(self.cfg.source_image).unsqueeze(0).to('cuda', dtype=torch.float16)
		target_image = image_transforms(self.cfg.target_image).unsqueeze(0).to('cuda', dtype=torch.float16)
		return source_image, target_image

	def _encode_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		if self.use_sdxl:
			(
				prompt_embeds,
				negative_prompt_embeds,
				pooled_prompt_embeds,
				negative_pooled_prompt_embeds,
			) = self.pipeline.encode_prompt(
				prompt=prompt,
				device=self.pipeline.device,
				num_images_per_prompt=1,
				do_classifier_free_guidance=True,
				negative_prompt="",
			)
		else:
			(
				prompt_embeds,
				negative_prompt_embeds,
			) = self.pipeline.encode_prompt(
				prompt=prompt,
				device=self.pipeline.device,
				num_images_per_prompt=1,
				do_classifier_free_guidance=True,
				negative_prompt="",
			)
			negative_pooled_prompt_embeds, pooled_prompt_embeds = None, None
		return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

	def get_sdxl_additional_inputs(self,
								   prompt_embeds: torch.Tensor,
								   pooled_prompt_embeds: torch.Tensor,
								   negative_pooled_prompt_embeds: torch.Tensor) -> dict:
		add_text_embeds = pooled_prompt_embeds
		text_encoder_projection_dim = self.pipeline.text_encoder_2.config.projection_dim
		add_time_ids = self._get_add_time_ids(
			self.pipeline.unet,
			original_size=(512, 512),
			crops_coords_top_left=(0, 0),
			target_size=(512, 512),
			dtype=prompt_embeds.dtype,
			text_encoder_projection_dim=text_encoder_projection_dim,
		)
		add_neg_time_ids = self._get_add_time_ids(
			self.pipeline.unet,
			original_size=(512, 512),
			crops_coords_top_left=(0, 0),
			target_size=(512, 512),
			dtype=prompt_embeds.dtype,
			text_encoder_projection_dim=text_encoder_projection_dim,
		)
		add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
		add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)
		added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids.to('cuda')}
		return added_cond_kwargs

	@staticmethod
	def _get_add_time_ids(unet,
						  original_size: Tuple[int, int] = (512, 512),
						  crops_coords_top_left: Tuple[int, int] = (0, 0),
						  target_size: Tuple[int, int] = (512, 512),
						  dtype: torch.dtype = torch.float16,
						  text_encoder_projection_dim: int = 1280) -> torch.Tensor:
		add_time_ids = list(original_size + crops_coords_top_left + target_size)
		passed_add_embed_dim = (
				unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
		)
		expected_add_embed_dim = unet.add_embedding.linear_1.in_features
		if expected_add_embed_dim != passed_add_embed_dim:
			raise ValueError(
				f"Model expects an added time embedding vector of length {expected_add_embed_dim}, "
				f"but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. "
				f"Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
			)
		add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
		return add_time_ids


class Inference:

	@staticmethod
	def run_inference(cfg: InferenceConfig,
					  adversarial_image: Image.Image,
					  inference_prompts: List[str],
					  use_sdxl: bool = True,
					  use_lcm: bool = False) -> List[Image.Image]:
		""" Main inference loop """
		wandb.init(
			project="TML Project",
			config=dataclasses.asdict(cfg),
			name=cfg.experiment_name,
		)

		pipeline = Trainer.load_models(use_sdxl=use_sdxl, use_lcm=use_lcm, dtype=torch.float32)
		source_image = Image.open(cfg.source_image_path).convert("RGB")
		target_image = Image.open(cfg.target_image_path).convert("RGB")
		torch.manual_seed(cfg.seed)
		output_images = []
		for prompt in inference_prompts:
			output_clean = pipeline.__call__(
				prompt=prompt,
				image=source_image,
				num_inference_steps=cfg.n_steps,
				guidance_scale=cfg.guidance_scale,
				strength=cfg.strength
			).images[0]
			output_adversarial = pipeline.__call__(
				prompt=prompt,
				image=adversarial_image,
				num_inference_steps=cfg.n_steps,
				guidance_scale=cfg.guidance_scale,
				strength=cfg.strength
			).images[0]

			# Join all the images together side by side
			images = [
				source_image.resize((512, 512)),
				target_image.resize((512, 512)),
				adversarial_image.resize((512, 512)),
				output_clean.resize((512, 512)),
				output_adversarial.resize((512, 512))
			]
			joined_image = Image.fromarray(np.concatenate(images, axis=1))
			save_name = "-".join(prompt[:30].split())
			joined_image.save(cfg.output_path / f"{save_name}.png")
			wandb.log({f"val_images": wandb.Image(joined_image, caption=prompt)})
			output_images.append(joined_image)
		return output_images


if __name__ == '__main__':
	use_sdxl = False
	use_lcm = False

	# Source image path
	source_image_path = Path("data/images/japan.jpg")
	target_image_path = Path("data/images/japan.jpg")
	output_path = Path("/data/yuval/")

	# # Part 1: Training
	train_cfg = TrainConfig(
		source_image_path=source_image_path,
		target_image_path=target_image_path,
		output_path=output_path,
		n_optimization_steps=200,
	)
	trainer = Trainer(
		cfg=train_cfg,
		use_sdxl=use_sdxl,
		use_lcm=use_lcm
	)
	adversarial_image = trainer.run()
	adversarial_image.save(output_path / "adversarial_image.png")

	adversarial_image = Image.open(output_path / "adversarial_image.png").convert("RGB")

	# Part 2: Inference
	inference_cfg = InferenceConfig(
		source_image_path=source_image_path,
		target_image_path=target_image_path,
		output_path=output_path,
		n_steps=4 if use_lcm else 50,
		guidance_scale=5.0,
		strength=0.6,
	)
	Inference.run_inference(
		cfg=inference_cfg,
		adversarial_image=adversarial_image,
		inference_prompts=INFERENCE_PROMPTS,
		use_sdxl=use_sdxl,
		use_lcm=use_lcm
	)
