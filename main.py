import dataclasses
import inspect
import os
from pathlib import Path
from typing import List, Tuple, Union, Optional
from transformers import pipeline

import numpy as np
import torch
import wandb
from PIL import Image
from diffusers import AutoPipelineForImage2Image, AutoencoderKL, LCMScheduler
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm
import torchvision.transforms as T

from configs import TrainConfig, InferenceConfig, INFERENCE_PROMPTS, NEGATIVE_PROMPT
from data.dataset import ImagePromptDataset
from losses import losses
from pipelines.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
from utils.vis_utils import create_table_plot


class Trainer:
	
	def __init__(self, cfg: TrainConfig, use_sdxl: bool = True, use_lcm: bool = False):
		self.cfg = cfg
		self.use_sdxl = use_sdxl
		self.use_lcm = use_lcm
		self.device = cfg.device
		# self.dtype = torch.float32 if cfg.device == "cpu" else torch.float16
		self.dtype = torch.float32
		self.pipeline = self.load_models(
			use_sdxl=use_sdxl,
			use_lcm=use_lcm,
			device=self.device,
			dtype=self.dtype
		)
		self.noises = None
		if self.cfg.use_fixed_noise:
			noise_shape = (1, 4, 64, 64)
			self.noises = [
				randn_tensor(noise_shape, device=self.device, dtype=self.dtype) for _ in range(self.cfg.n_noise)
			]
	
	def run(self) -> Tuple[Image.Image, torch.Tensor]:
		""" Main training loop """
		# Verify that the W&B API key is defined as env variable
		# if "WANDB_API_KEY" in os.environ:
		# 	print('Using W&B API key from env')
		# else:
		# 	raise Exception("Please provide a valid W&B API key in env")
		wandb.init(
			project="TML Project",
			config=dataclasses.asdict(self.cfg),
			name=self.cfg.experiment_name,
		)
		wandb.save(os.path.basename(__file__))
		
		# Make dataset with the source images, their captions, and target images
		dataset = ImagePromptDataset(
			source_images=self.cfg.source_images,
			target_images=self.cfg.target_images,
			pipeline=self.pipeline,
			device=self.device,
			dtype=self.dtype,
			cfg=self.cfg
		)
		dataloader = torch.utils.data.DataLoader(
			dataset,
			batch_size=1,
			shuffle=False
		)
		
		iterator = tqdm(range(self.cfg.n_optimization_steps))
		
		# Initialize the perturbation
		perturbation = torch.zeros((1, 3, 512, 512), requires_grad=True, device=self.device, dtype=self.dtype)
		
		for iteration in iterator:
			all_grads = []
			losses = []
			loss_dict = {}
			
			# Get next batch
			source_image, target_image, target_latent, source_image_caption = next(iter(dataloader))
			source_image = source_image.to(self.device, dtype=self.dtype)
			target_image = target_image.to(self.device, dtype=self.dtype)
			target_latent = target_latent.to(self.device, dtype=self.dtype)
			source_image_caption = source_image_caption[0]
			
			# Initialize the adversarial image
			X_adv = source_image.clone() + perturbation
			
			output_image = None
			prompt = self.cfg.prompts[np.random.randint(0, len(self.cfg.prompts))]
			prompt = f"{source_image_caption} {prompt}" if source_image_caption != "" else prompt
			prompt = f"{prompt}, detailed"
			for i in range(self.cfg.grad_reps):
				# Randomly sample one of the prompts in the set
				c_grad, loss, output_image, loss_dict = self.compute_grad(
					cur_image=X_adv,
					prompt=prompt,
					source_image=source_image,
					target_image=target_image,
					target_latent=target_latent,
					noise=self.noises,
				)
				all_grads.append(c_grad)
				losses.append(loss)
			
			# Aggregate gradients
			grad = torch.stack(all_grads).mean(0)
			
			# Display average loss
			iterator.set_description_str(f'AVG Loss: {np.mean(losses):.3f}')
			logs = {"avg_loss": np.mean(losses)}
			logs.update(loss_dict)
			
			# Apply the perturbation step (either L2 or Linf)
			X_adv, perturbation = self.perturbation_step(
				X_adv=X_adv,
				grad=grad,
				X=source_image,
				X_mask=None
			)
			
			if iteration % self.cfg.image_visualization_interval == 0 or iteration == self.cfg.n_optimization_steps - 1:
				vis_adversarial_image = T.ToPILImage()(
					(X_adv[0] / 2 + 0.5).clamp(0, 1)
				)
				vis_diff = T.ToPILImage()(
					((source_image[0] - X_adv[0]) / 2 + 0.5).clamp(0, 1)
				)
				vis_output = T.ToPILImage()(
					(output_image[0] / 2 + 0.5).clamp(0, 1)
				)
				images = create_table_plot(
					images=[vis_adversarial_image, vis_diff, vis_output],
					captions=[f'Current Adversarial Image', 'Difference Image', f'Edited Image ({prompt})']
				)
				logs.update({
					"train_images": wandb.Image(images, caption=prompt),
				})
			
			wandb.log(logs)
		
		torch.cuda.empty_cache()
		
		# Log the final adversarial image (if working with multiple, just use the last one)
		X_adv = (X_adv / 2 + 0.5).clamp(0, 1)
		adversarial_image = T.ToPILImage()(X_adv[0]).convert("RGB")
		wandb.log({"final_adversarial_image": wandb.Image(adversarial_image)})
		
		# Let's also log the perturbation
		perturbation = perturbation.detach().cpu()
		perturbation_image = T.ToPILImage()((perturbation / 2 + 0.5).clamp(0, 1)[0]).convert("RGB")
		wandb.log({"perturbation": wandb.Image(perturbation_image)})
		return adversarial_image, perturbation
	
	def compute_grad(self,
					 cur_image: torch.Tensor,
					 prompt: str,
					 source_image: torch.Tensor,
					 target_image: torch.Tensor,
					 target_latent: torch.Tensor,
					 noise: Optional[List[torch.Tensor]] = None) -> Tuple[
		torch.Tensor, torch.Tensor, torch.Tensor, dict]:
		torch.set_grad_enabled(True)
		
		cur_image = cur_image.clone()
		# cur_image.requires_grad = True
		
		output_latent = self.attack_forward(image=cur_image, prompt=prompt, noise=noise)
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
			loss = self.cfg.rec_loss_lambda * rec_loss + self.cfg.perturbation_loss_lambda * pert_loss
		else:
			loss = self.cfg.rec_loss_lambda * rec_loss
			pert_loss = torch.tensor(0.0)
		
		loss_dict = {'rec_loss': rec_loss.item(), 'pert_loss': pert_loss.item()}
		
		grad = torch.autograd.grad(loss, [cur_image])[0]
		return grad, loss.item(), output_image, loss_dict
	
	def attack_forward(self,
					   prompt: Union[str, List[str]],
					   image: Union[torch.Tensor, Image.Image],
					   noise: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
		
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
			timesteps_tensor = torch.tensor([t for t in timesteps_tensor if t < 700], device=self.device)
		
		# Get additional inputs if using SDXL
		timestep_cond, added_cond_kwargs = None, None
		if self.use_sdxl:
			added_cond_kwargs = self.get_sdxl_additional_inputs(
				prompt_embeds=prompt_embeds,
				pooled_prompt_embeds=pooled_prompt_embeds.detach(),
				negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.detach()
			)
		
		# Add noise to the input latent
		if noise is None:
			noise = [randn_tensor(image_latents.shape, device=self.device, dtype=self.dtype)]
		
		# Randomly sample a noise tensor from the list of noise tensors
		selected_noise = noise[np.random.randint(0, len(noise))]
		latents = self.pipeline.scheduler.add_noise(image_latents, selected_noise, timesteps_tensor[:1])
		
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
	
	def perturbation_step(self,
						  X_adv: torch.Tensor,
						  grad: torch.Tensor,
						  X: torch.Tensor,
						  X_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Apply the perturbation step for either L2 or Linf norm."""
		if self.cfg.norm_type == 'l2':
			# Normalize gradient for L2 norm
			l = len(X.shape) - 1
			grad_norm = torch.norm(grad.detach().reshape(grad.shape[0], -1), dim=1).view(-1, *([1] * l))
			grad_normalized = grad.detach() / (grad_norm + 1e-10)
			
			if X_mask is not None:
				grad_normalized = grad_normalized * X_mask.repeat(1, 3, 1, 1)
			
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
		
		# Get the perturbation
		perturbation = X_adv - X
		
		return X_adv, perturbation
	
	@staticmethod
	def load_models(use_sdxl: bool = True,
					use_lcm: bool = False,
					device: str = "cuda",
					dtype: torch.dtype = torch.float16) -> AutoPipelineForImage2Image:
		if use_sdxl:
			pipeline = AutoPipelineForImage2Image.from_pretrained(
				"stabilityai/stable-diffusion-xl-base-1.0",
				torch_dtype=dtype,
				safety_checker=None,
			)
			pipeline = pipeline.to(device)
			vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype).to(device)
			pipeline.vae = vae
			if use_lcm:
				pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
				pipeline.load_lora_weights("latent-consistency/lcm-lora-sdxl")
				pipeline.fuse_lora()
		else:
			pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
				"runwayml/stable-diffusion-v1-5",
				torch_dtype=dtype,
				safety_checker=None,
			)
			vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=dtype).to(device)
			pipeline.vae = vae
			pipeline = pipeline.to(device)
			if use_lcm:
				pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
				pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
				pipeline.fuse_lora()
		return pipeline
	
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
				# negative_prompt=NEGATIVE_PROMPT,
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
				# negative_prompt=NEGATIVE_PROMPT,
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
		added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids.to(self.device)}
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
	def transfer_perturbation(original_perturbation, original_image, new_image, max_perturbation_value: int = 20):
		# Calculate scaling factor based on standard deviation
		std_ratio = np.std(new_image) / np.std(original_image)
		# Cap the scaling factor to prevent amplification
		scale_factor = min(1, std_ratio)
		# Scale the perturbation
		scaled_perturbation = original_perturbation * scale_factor
		# Normalize the perturbation to have a maximum absolute value
		scaled_perturbation = np.clip(scaled_perturbation, -max_perturbation_value, max_perturbation_value)
		# Apply the scaled perturbation to the new image
		# perturbed_image = new_image + scaled_perturbation
		perturbed_image = new_image - scaled_perturbation
		# Clip pixel values to valid range
		perturbed_image = np.clip(perturbed_image, 0, 255)
		return perturbed_image.astype(np.uint8)
	
	@staticmethod
	def run_inference(cfg: InferenceConfig,
					  perturbation: torch.Tensor,
					  inference_prompts: List[str],
					  use_sdxl: bool = True,
					  use_lcm: bool = False,
					  noises: Optional[List[torch.Tensor]] = None) -> List[Image.Image]:
		""" Main inference loop """
		wandb.init(
			project="TML Project",
			config=dataclasses.asdict(cfg),
			name=cfg.experiment_name,
		)
		torch.manual_seed(cfg.seed)
		
		pipeline = Trainer.load_models(use_sdxl=use_sdxl, use_lcm=use_lcm, dtype=torch.float32)
		resize_transforms = T.Compose([
			T.Resize(512, interpolation=T.InterpolationMode.BILINEAR),
			T.CenterCrop(512),
		])
		transforms = T.Compose([
			T.Resize(512, interpolation=T.InterpolationMode.BILINEAR),
			T.CenterCrop(512),
			T.ToTensor(),
			T.Normalize([0.5], [0.5]),
		])
		source_images = [resize_transforms(Image.open(path).convert("RGB")) for path in cfg.source_image_paths]
		target_images = [resize_transforms(Image.open(path).convert("RGB")) for path in cfg.target_image_paths]
		source_tensors = [transforms(image) for image in source_images]

		# Define the adversarial image by adding the perturbation to the source image
		adversarial_images = [
			(((image + perturbation[0]).numpy() + 1) * 127.5).astype(np.uint8).transpose(1, 2, 0)
			for image in source_tensors
		]
		adversarial_images = [Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)) for image in adversarial_images]

		source_image_captions = [''] * len(source_images)
		if cfg.default_source_image_caption != "" or cfg.add_image_caption_to_prompts:
			if cfg.default_source_image_caption != "":
				source_image_captions = [cfg.default_source_image_caption] * len(source_images)
			else:
				source_image_captions = [
					ImagePromptDataset._get_image_caption(image, dtype=torch.float32) for image in source_images
				]
		
		output_images = []
		
		# Concat inference prompts and training prompts with a tuple that has a prefix of type
		all_prompts = [(prompt, "Validation") for prompt in inference_prompts]

		for source_image, target_image, adversarial_image, prefix in zip(source_images, target_images, adversarial_images, source_image_captions):
		
			# for prompt, prompt_type in all_prompts:
			for prompt, prompt_type in []:
				
				# If noises are not given, get n_noise random noise tensors for each prompt
				noises_for_prompt = noises
				if noises is None:
					noises_for_prompt = [
						randn_tensor((1, 4, 64, 64), device='cuda', dtype=torch.float32) for _ in range(cfg.n_noise)
					]
				
				for noise_idx, noise in enumerate(noises_for_prompt):
					prompt = f"{prefix} {prompt}" if prefix != "" else prompt
					prompt = f"{prompt}, detailed"
					with torch.no_grad():
						output_clean = pipeline.__call__(
							prompt=prompt,
							image=source_image,
							num_inference_steps=cfg.n_steps,
							guidance_scale=cfg.guidance_scale,
							strength=cfg.strength,
						).images[0]
						output_adversarial = pipeline.__call__(
							prompt=prompt,
							image=adversarial_image,
							num_inference_steps=cfg.n_steps,
							guidance_scale=cfg.guidance_scale,
							strength=cfg.strength,
							noise=noise,
						).images[0]
	
					# Join all the images together side by side
					images = [
						source_image.resize((512, 512)),
						target_image.resize((512, 512)),
						adversarial_image.resize((512, 512)),
						output_clean.resize((512, 512)),
						output_adversarial.resize((512, 512))
					]
					labels = [
						'Source Image',
						'Target Image',
						'Adversarial Image',
						f'Edit on Original ({prompt})',
						f'Edit on Adversarial ({prompt})'
					]
					joined_image = create_table_plot(images=images, captions=labels)
					save_name = "-".join(prompt[:30].split()) if len(prompt) > 0 else 'empty_prompt'
					joined_image.save(cfg.output_path / f"{save_name}_noise_{noise_idx}.png")
					wandb.log({f"Train Images - {prompt_type} Prompts": wandb.Image(joined_image, caption=prompt)})
					output_images.append(joined_image)
		
		if cfg.validation_images_path is not None:
			
			# Read paths to validation images
			with open(cfg.validation_images_path, "r") as f:
				validation_images_paths = f.readlines()
				validation_images_paths = [Path(img.strip()) for img in validation_images_paths]
			
			for val_image_path in validation_images_paths:
				
				val_image = resize_transforms(Image.open(val_image_path).convert("RGB"))
				# For the original image, take the average image of those seen during training
				avg_image = torch.stack(source_tensors).mean(0)
				avg_image_np = ((avg_image.numpy().transpose(1, 2, 0) + 1) * 127.5).astype('uint8')
				# Extract perturbation from one of the source images
				perturbation_np = np.array(adversarial_images[0]) - np.array(source_images[0])
				val_image_np = np.array(val_image)
				val_image_adversarial = Inference.transfer_perturbation(
					original_perturbation=perturbation_np,
					original_image=avg_image_np,
					new_image=val_image_np,
					max_perturbation_value=20
				)
				val_image_adversarial = Image.fromarray(val_image_adversarial).convert("RGB")
				
				source_image_caption = ""  # Don't use prefix for this test
				
				for prompt, prompt_type in all_prompts:
					
					# If noises are not given, get n_noise random noise tensors for each prompt
					noises_for_prompt = noises
					if noises is None:
						noises_for_prompt = [
							randn_tensor((1, 4, 64, 64), device='cuda', dtype=torch.float32) for _ in range(cfg.n_noise)
						]
					
					for noise_idx, noise in enumerate(noises_for_prompt):
						prompt = f"{source_image_caption} {prompt}" if source_image_caption != "" else prompt
						prompt = f"{prompt}, detailed"
						with torch.no_grad():
							val_output_clean = pipeline.__call__(
								prompt=prompt,
								image=val_image,
								num_inference_steps=cfg.n_steps,
								guidance_scale=cfg.guidance_scale,
								strength=cfg.strength,
							).images[0]
							val_output_adversarial = pipeline.__call__(
								prompt=prompt,
								image=val_image_adversarial,
								num_inference_steps=cfg.n_steps,
								guidance_scale=cfg.guidance_scale,
								strength=cfg.strength,
								noise=noise,
							).images[0]

						# Join all the images together side by side
						images = [
							val_image.resize((512, 512)),
							val_image_adversarial.resize((512, 512)),
							val_output_clean.resize((512, 512)),
							val_output_adversarial.resize((512, 512))
						]
						labels = [
							'Val Original Image',
							'Val Adversarial Image',
							f'Edit on Original ({prompt})',
							f'Edit on Adversarial ({prompt})'
						]
						joined_image = create_table_plot(images=images, captions=labels)
						save_name = "-".join(prompt[:30].split()) if len(prompt) > 0 else 'empty_prompt'
						joined_image.save(cfg.output_path / f"{save_name}_noise_{noise_idx}.png")
						wandb.log({f"Val Images - {prompt_type} Prompt": wandb.Image(joined_image, caption=prompt)})
		
		return output_images
	
	
def main():
	use_sdxl = False
	use_lcm_training = True
	use_lcm_inference = True
	
	# Source image path
	source_image_paths = [
		Path("./images/pexels-burcin-altinyay-1182404935-28191722.jpg"),
		Path("./images/pexels-lorna-pauli-1320744316-28992825.jpg"),
	]
	target_image_paths = [
		Path("./images/pexels-burcin-altinyay-1182404935-28191722.jpg"),
		Path("./images/pexels-lorna-pauli-1320744316-28992825.jpg"),
	]
	output_path = Path("/data/yuval/")
	
	# # Part 1: Training
	train_cfg = TrainConfig(
		source_image_paths=source_image_paths,
		target_image_paths=target_image_paths,
		output_path=output_path,
		n_optimization_steps=100,
		guidance_scale=4.0,
		n_noise=1,
		use_fixed_noise=True,
	)
	trainer = Trainer(
		cfg=train_cfg,
		use_sdxl=use_sdxl,
		use_lcm=use_lcm_training
	)
	# adversarial_image, perturbation = trainer.run()
	# adversarial_image.save(output_path / "adversarial_image.png")
	# torch.save(trainer.noises, output_path / "noise.pt")
	# torch.save(perturbation, output_path / "perturbation.pt")
	
	trainer.noises = torch.load(output_path / "noise.pt")
	perturbation = torch.load(output_path / "perturbation.pt")
	
	# Part 2: Inference
	inference_cfg = InferenceConfig(
		experiment_name='use_train_noises',
		source_image_paths=source_image_paths,
		target_image_paths=target_image_paths,
		output_path=output_path,
		n_steps=4 if use_lcm_inference else 50,
		guidance_scale=4.0,
		strength=0.60,
		use_fixed_noise=True,
		n_noise=train_cfg.n_noise,
		validation_images_path=Path("validation_images.txt"),
		default_source_image_caption="",
	)
	
	inference_noises = None
	if inference_cfg.use_fixed_noise:
		inference_noises = trainer.noises
	
	Inference.run_inference(
		cfg=inference_cfg,
		perturbation=perturbation,
		inference_prompts=INFERENCE_PROMPTS,
		use_sdxl=False,
		use_lcm=use_lcm_inference,
		noises=inference_noises,
	)


if __name__ == '__main__':
	main()