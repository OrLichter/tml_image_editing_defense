from pathlib import Path
from typing import Union, List

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from tqdm import tqdm

from configs import INFERENCE_PROMPTS
from photoguard.photoguard_utils import prepare_mask_and_masked_image, recover_image, prepare_image

to_pil = T.ToPILImage()


# A differentiable version of the forward function of the inpainting stable diffusion model!
def attack_forward(
		self,
		prompt: Union[str, List[str]],
		masked_image: Union[torch.FloatTensor, Image.Image],
		mask: Union[torch.FloatTensor, Image.Image],
		height: int = 512,
		width: int = 512,
		num_inference_steps: int = 50,
		guidance_scale: float = 7.5,
		eta: float = 0.0,
):
	text_inputs = self.tokenizer(
		prompt,
		padding="max_length",
		max_length=self.tokenizer.model_max_length,
		return_tensors="pt",
	)
	text_input_ids = text_inputs.input_ids
	text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
	
	uncond_tokens = [""]
	max_length = text_input_ids.shape[-1]
	uncond_input = self.tokenizer(
		uncond_tokens,
		padding="max_length",
		max_length=max_length,
		truncation=True,
		return_tensors="pt",
	)
	uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
	seq_len = uncond_embeddings.shape[1]
	text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
	
	text_embeddings = text_embeddings.detach()
	
	num_channels_latents = self.vae.config.latent_channels
	
	latents_shape = (1, num_channels_latents, height // 8, width // 8)
	latents = torch.randn(latents_shape, device=self.device, dtype=text_embeddings.dtype)
	
	mask = torch.nn.functional.interpolate(mask, size=(height // 8, width // 8))
	mask = torch.cat([mask] * 2)
	
	masked_image_latents = self.vae.encode(masked_image).latent_dist.sample()
	masked_image_latents = 0.18215 * masked_image_latents
	masked_image_latents = torch.cat([masked_image_latents] * 2)
	
	latents = latents * self.scheduler.init_noise_sigma
	
	self.scheduler.set_timesteps(num_inference_steps)
	timesteps_tensor = self.scheduler.timesteps.to(self.device)
	
	for i, t in enumerate(timesteps_tensor):
		latent_model_input = torch.cat([latents] * 2)
		latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
		noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
		noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
		noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
		latents = self.scheduler.step(noise_pred, t, latents, eta=eta).prev_sample
	
	latents = 1 / 0.18215 * latents
	image = self.vae.decode(latents).sample
	return image


def compute_grad(cur_mask, cur_masked_image, prompt, target_image, **kwargs):
	torch.set_grad_enabled(True)
	cur_mask = cur_mask.clone()
	cur_masked_image = cur_masked_image.clone()
	cur_mask.requires_grad = False
	cur_masked_image.requires_grad_()
	image_nat = attack_forward(pipe_inpaint, mask=cur_mask,
	                           masked_image=cur_masked_image,
	                           prompt=prompt,
	                           **kwargs)
	
	loss = (image_nat - target_image).norm(p=2)
	grad = torch.autograd.grad(loss, [cur_masked_image])[0] * (1 - cur_mask)
	
	return grad, loss.item(), image_nat.data.cpu()


def super_l2(cur_mask, X, prompt, step_size, iters, eps, clamp_min, clamp_max, grad_reps=5, target_image=0, **kwargs):
	X_adv = X.clone()
	iterator = tqdm(range(iters))
	for i in iterator:
		
		all_grads = []
		losses = []
		for i in range(grad_reps):
			c_grad, loss, last_image = compute_grad(cur_mask, X_adv, prompt, target_image=target_image, **kwargs)
			all_grads.append(c_grad)
			losses.append(loss)
		grad = torch.stack(all_grads).mean(0)
		
		iterator.set_description_str(f'AVG Loss: {np.mean(losses):.3f}')
		
		l = len(X.shape) - 1
		grad_norm = torch.norm(grad.detach().reshape(grad.shape[0], -1), dim=1).view(-1, *([1] * l))
		grad_normalized = grad.detach() / (grad_norm + 1e-10)
		
		# actual_step_size = step_size - (step_size - step_size / 100) / iters * i
		actual_step_size = step_size
		X_adv = X_adv - grad_normalized * actual_step_size
		
		d_x = X_adv - X.detach()
		d_x_norm = torch.renorm(d_x, p=2, dim=0, maxnorm=eps)
		X_adv.data = torch.clamp(X + d_x_norm, clamp_min, clamp_max)
	
	torch.cuda.empty_cache()
	
	return X_adv, last_image


def super_linf(cur_mask, X, prompt, step_size, iters, eps, clamp_min, clamp_max, grad_reps=5, target_image=0, **kwargs):
	X_adv = X.clone()
	iterator = tqdm(range(iters))
	for i in iterator:
		
		all_grads = []
		losses = []
		for i in range(grad_reps):
			c_grad, loss, last_image = compute_grad(cur_mask, X_adv, prompt, target_image=target_image, **kwargs)
			all_grads.append(c_grad)
			losses.append(loss)
		grad = torch.stack(all_grads).mean(0)
		
		iterator.set_description_str(f'AVG Loss: {np.mean(losses):.3f}')
		
		# actual_step_size = step_size - (step_size - step_size / 100) / iters * i
		actual_step_size = step_size
		X_adv = X_adv - grad.detach().sign() * actual_step_size
		
		X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
		X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
	
	torch.cuda.empty_cache()
	
	return X_adv, last_image


pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
	"stable-diffusion-v1-5/stable-diffusion-inpainting",
	torch_dtype=torch.float32,
)
pipe_inpaint = pipe_inpaint.to("cuda")

output_root = Path('/data/yuval/tml_experiments/photoguard_results')
output_root.mkdir(exist_ok=True, parents=True)

image_paths = [p for p in Path('./images').glob("*") if p.suffix in ['.jpg', '.png', '.jpeg']]

for idx, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
	
	print(f"Running on path {idx} out of {len(image_paths)}")
	
	experiment_root = output_root / image_path.stem
	experiment_root.mkdir(exist_ok=True, parents=True)
	
	init_image = Image.open(image_path).convert('RGB').resize((512, 512))
	target_image = Image.open(image_path).convert("RGB").resize((512, 512))
	
	# Create an all white mask
	mask_image = Image.new("L", init_image.size, 0).convert('RGB')
	
	prompt = ""
	torch.manual_seed(torch.randint(0, 2 ** 32, (1,)).item())
	
	strength = 0.7
	guidance_scale = 7.5
	num_inference_steps = 4
	
	cur_mask, cur_masked_image = prepare_mask_and_masked_image(init_image, mask_image)
	
	cur_mask = cur_mask.cuda()
	cur_masked_image = cur_masked_image.cuda()
	target_image_tensor = prepare_image(target_image)
	target_image_tensor = 0 * target_image_tensor.cuda()
	
	# result, last_image = super_l2(cur_mask, cur_masked_image,
	#                               prompt=prompt,
	#                               target_image=target_image_tensor,
	#                               eps=16,
	#                               step_size=1,
	#                               iters=200,
	#                               clamp_min=-1,
	#                               clamp_max=1,
	#                               eta=1,
	#                               num_inference_steps=num_inference_steps,
	#                               guidance_scale=guidance_scale,
	#                               grad_reps=10)
	#
	# adv_X = (result / 2 + 0.5).clamp(0, 1)
	# adv_image = to_pil(adv_X[0]).convert("RGB")
	# adv_image = recover_image(adv_image, init_image, mask_image, background=True)
	# adv_image.save(experiment_root / f"adversarial_image.png")
	
	adv_image = Image.open(experiment_root / f"adversarial_image.png").convert("RGB")
	
	# Run inference on all prompts
	for prompt in INFERENCE_PROMPTS:
		seed = torch.randint(0, 2 ** 32, (1,)).item()
		
		torch.manual_seed(seed)
		strength = 0.6
		guidance_scale = 7.5
		num_inference_steps = 100
		
		# Make all white mask image
		mask_image = Image.new("L", init_image.size, 255).convert('RGB')
		
		image_nat = pipe_inpaint(prompt=prompt,
		                         image=init_image,
		                         mask_image=mask_image,
		                         eta=1,
		                         num_inference_steps=num_inference_steps,
		                         guidance_scale=guidance_scale,
		                         strength=strength).images[0]
		torch.manual_seed(seed)
		image_adv = pipe_inpaint(prompt=prompt,
		                         image=adv_image,
		                         mask_image=mask_image,
		                         eta=1,
		                         num_inference_steps=num_inference_steps,
		                         guidance_scale=guidance_scale,
		                         strength=strength).images[0]
		
		# Save the results
		prompt_root = experiment_root / prompt
		prompt_root.mkdir(exist_ok=True, parents=True)
		
		image_nat.save(prompt_root / f"natural_v2.png")
		image_adv.save(prompt_root / f"adversarial_v2.png")
