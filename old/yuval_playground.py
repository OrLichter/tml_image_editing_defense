import inspect

from PIL import Image
import numpy as np

import time
import torch
from tqdm import tqdm
from diffusers import AutoPipelineForInpainting, LCMScheduler, AutoencoderKL
import torchvision.transforms as T
from typing import Union, List
import torch.nn.functional as F


from data.dataset import ImagePromptDataset

to_pil = T.ToPILImage()

USE_SDXL = True
USE_LCM = True

if USE_SDXL:
    pipe_inpaint = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
    )
    pipe_inpaint = pipe_inpaint.to("cuda")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to('cuda')
    pipe_inpaint.vae = vae
    if USE_LCM:
        pipe_inpaint.scheduler = LCMScheduler.from_config(pipe_inpaint.scheduler.config)
        pipe_inpaint.load_lora_weights("latent-consistency/lcm-lora-sdxl")
        pipe_inpaint.fuse_lora()
else:
    pipe_inpaint = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
    )
    pipe_inpaint = pipe_inpaint.to("cuda")
    if USE_LCM:
        pipe_inpaint.scheduler = LCMScheduler.from_config(pipe_inpaint.scheduler.config)
        pipe_inpaint.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        pipe_inpaint.fuse_lora()


def attack_forward(
        self,
        prompt: Union[str, List[str]],
        image: Union[torch.Tensor, Image.Image],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 40,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        limit_timesteps: bool = False,
):

    if USE_SDXL:
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt="",
        )
    else:
        (
            prompt_embeds,
            negative_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt="",
        )
        negative_pooled_prompt_embeds, pooled_prompt_embeds = None, None

    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    prompt_embeds = prompt_embeds.detach()

    num_channels_latents = self.vae.config.latent_channels

    latents_shape = (1, num_channels_latents, height // 8, width // 8)
    latents = torch.randn(latents_shape, device=self.device, dtype=prompt_embeds.dtype)

    masked_image_latents = self.vae.encode(image).latent_dist.sample()
    masked_image_latents = 0.18215 * masked_image_latents
    masked_image_latents = torch.cat([masked_image_latents] * 2)

    mask = torch.ones_like(latents)[:, :1]
    mask = torch.cat([mask] * 2)

    latents = latents * self.scheduler.init_noise_sigma

    self.scheduler.set_timesteps(num_inference_steps)
    timesteps_tensor = self.scheduler.timesteps.to(self.device)

    # Limit the timesteps since we know that for editing, we only really want a subset of the timesteps
    if limit_timesteps:
        timesteps_tensor = torch.tensor([t for t in timesteps_tensor if 100 < t < 800], device=self.device)

    timestep_cond, added_cond_kwargs = None, None
    if USE_SDXL:
        add_text_embeds = pooled_prompt_embeds
        text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
        add_time_ids = _get_add_time_ids(
            self.unet,
            original_size=(512, 512),
            crops_coords_top_left=(0, 0),
            target_size=(512, 512),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_neg_time_ids = _get_add_time_ids(
            self.unet,
            original_size=(512, 512),
            crops_coords_top_left=(0, 0),
            target_size=(512, 512),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        timestep_cond = None
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids.to('cuda')}

    prompt_embeds = prompt_embeds.detach()

    for i, t in enumerate(timesteps_tensor):

        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

        if USE_SDXL:
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs={},
                added_cond_kwargs=added_cond_kwargs,
            ).sample
        else:
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        extra_step_kwargs = {'eta': eta} if 'eta' in inspect.signature(self.scheduler.step).parameters else {}
        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True).prev_sample

    latents = 1 / 0.18215 * latents
    return latents


def perturbation_loss(adv_image, source_image):
    """ Compute L2 loss between the adversarial image and the source image. This keeps perturbations small. """
    return F.mse_loss(adv_image, source_image)


def compute_grad(cur_image: torch.Tensor,
                 prompt: str,
                 source_image: torch.Tensor,
                 target_image: torch.Tensor,
                 apply_loss_on_images: bool = True,
                 apply_loss_on_latents: bool = False,
                 limit_timesteps: bool = False,
                 perturbation_loss_lambda: float = 0.0,
                 **kwargs):
    torch.set_grad_enabled(True)
    cur_image = cur_image.clone()
    cur_image.requires_grad_()
    target_latent = kwargs.pop("target_latents", None)
    start_time = time.time()
    output_latent = attack_forward(pipe_inpaint,
                                   image=cur_image,
                                   prompt=prompt,
                                   limit_timesteps=limit_timesteps,
                                   **kwargs)
    print(f"Time taken for forward pass: {time.time() - start_time}")
    output_image = None
    if apply_loss_on_images:
        output_image = pipe_inpaint.vae.decode(output_latent).sample
        rec_loss = (output_image - target_image).norm(p=2)
    elif apply_loss_on_latents:
        rec_loss = (output_latent - target_latent).norm(p=2)
    else:
        raise ValueError("Please specify whether to apply loss on images or latents")

    if perturbation_loss_lambda > 0:
        if output_image is None:
            output_image = pipe_inpaint.vae.decode(output_latent).sample
        pert_loss = perturbation_loss(output_image, source_image)
        loss = rec_loss + perturbation_loss_lambda * pert_loss
    else:
        loss = rec_loss

    grad = torch.autograd.grad(loss, [cur_image])[0]
    # Image.fromarray(((image[0].cpu().detach().permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8))

    return grad, loss.item()


def super_l2(X: torch.Tensor,
             prompts: List[str],
             step_size: int,
             iters: int,
             eps: int,
             clamp_min: int,
             clamp_max: int,
             grad_reps: int = 5,
             target_image: Union[int, torch.Tensor] = 0,
             apply_loss_on_images: bool = True,
             apply_loss_on_latents: bool = False,
             limit_timesteps: bool = False,
             perturbation_loss_lambda: float = 0.0,
             **kwargs):
    X_adv = X.clone()
    target_latents = pipe_inpaint.vae.encode(target_image).latent_dist.sample()
    iterator = tqdm(range(iters))
    for _ in iterator:
        all_grads = []
        losses = []
        for i in range(grad_reps):
            # Randomly sample one of the prompts in the set
            prompt = prompts[np.random.randint(0, len(prompts))]
            c_grad, loss = compute_grad(X_adv,
                                        prompt,
                                        source_image=X,
                                        target_image=target_image,
                                        apply_loss_on_images=apply_loss_on_images,
                                        apply_loss_on_latents=apply_loss_on_latents,
                                        target_latents=target_latents,
                                        limit_timesteps=limit_timesteps,
                                        perturbation_loss_lambda=perturbation_loss_lambda,
                                        **kwargs)
            all_grads.append(c_grad)
            losses.append(loss)
        grad = torch.stack(all_grads).mean(0)

        iterator.set_description_str(f'AVG Loss: {np.mean(losses):.3f}')

        l = len(X.shape) - 1
        grad_norm = torch.norm(grad.detach().reshape(grad.shape[0], -1), dim=1).view(-1, *([1] * l))
        grad_normalized = grad.detach() / (grad_norm + 1e-10)

        actual_step_size = step_size
        X_adv = X_adv - grad_normalized * actual_step_size

        d_x = X_adv - X.detach()
        d_x_norm = torch.renorm(d_x, p=2, dim=0, maxnorm=eps)
        X_adv.data = torch.clamp(X + d_x_norm, clamp_min, clamp_max)

    torch.cuda.empty_cache()
    return X_adv


def super_linf(X: torch.Tensor,
               prompts: List[str],
               step_size: int,
               iters: int,
               eps: int,
               clamp_min: int,
               clamp_max: int,
               grad_reps: int = 5,
               target_image: Union[int, torch.Tensor] = 0,
               apply_loss_on_images: bool = True,
               apply_loss_on_latents: bool = False,
               limit_timesteps: bool = False,
               perturbation_loss_lambda: float = 0.0,
               **kwargs):
    X_adv = X.clone()
    target_latents = pipe_inpaint.vae.encode(target_image).latent_dist.sample()
    iterator = tqdm(range(iters))
    for _ in iterator:

        all_grads = []
        losses = []
        for i in range(grad_reps):

            # Randomly sample one of the prompts in the set
            prompt = prompts[np.random.randint(0, len(prompts))]
            start_time = time.time()
            c_grad, loss = compute_grad(X_adv,
                                        prompt,
                                        source_image=X,
                                        target_image=target_image,
                                        apply_loss_on_images=apply_loss_on_images,
                                        apply_loss_on_latents=apply_loss_on_latents,
                                        target_latents=target_latents,
                                        limit_timesteps=limit_timesteps,
                                        perturbation_loss_lambda=perturbation_loss_lambda,
                                        **kwargs)
            print(f"Time taken for grad computation: {time.time() - start_time}")
            all_grads.append(c_grad)
            losses.append(loss)

        grad = torch.stack(all_grads).mean(0)

        iterator.set_description_str(f'AVG Loss: {np.mean(losses):.3f}')

        actual_step_size = step_size
        X_adv = X_adv - grad.detach().sign() * actual_step_size

        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)

    torch.cuda.empty_cache()
    return X_adv


def _get_add_time_ids(unet, original_size=(512, 512), crops_coords_top_left=(0, 0),
                      target_size=(512, 512), dtype=torch.float16, text_encoder_projection_dim=1280):
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


SEED = 786349
torch.manual_seed(SEED)

num_inference_steps = 4
iters = 200

apply_loss_on_images = False
apply_loss_on_latents = True
limit_timesteps = True
perturbation_loss_lambda = 1.0

prompts = [
    ""
]

image_transforms = ImagePromptDataset.get_image_transforms()

source_image_pil = Image.open('../data/images/japan.jpg').convert("RGB")
source_image = image_transforms(source_image_pil).unsqueeze(0).to('cuda', dtype=torch.float16)

target_image_path = "../data/images/stick-figure-sticker.jpg"
target_image_pil = Image.open(target_image_path).convert("RGB")
target_image_tensor = image_transforms(target_image_pil).unsqueeze(0).to('cuda', dtype=torch.float16)

result = super_linf(
    source_image,
    prompts=prompts,
    target_image=target_image_tensor,
    eps=16,
    step_size=1,
    iters=iters,
    clamp_min=-1,
    clamp_max=1,
    eta=1.,
    num_inference_steps=num_inference_steps,
    guidance_scale=7.5,
    grad_reps=10,
    apply_loss_on_images=apply_loss_on_images,
    apply_loss_on_latents=apply_loss_on_latents,
    limit_timesteps=limit_timesteps,
    perturbation_loss_lambda=perturbation_loss_lambda,
)
adv_X = (result / 2 + 0.5).clamp(0, 1)
adv_image = to_pil(adv_X[0]).convert("RGB")

""" Inference time """

prompt = "a fuji pagoda on fire"
SEED = 9209
strength = 0.7
guidance_scale = 7.5 if not USE_LCM else 4.0
num_inference_steps = 100 if not USE_LCM else 4

# Make the mask all ones using PIL
mask_image = Image.new("RGB", (512, 512), (255, 255, 255))

torch.manual_seed(SEED)
image_nat = pipe_inpaint(prompt=prompt,
                         image=source_image_pil,
                         mask_image=mask_image,
                         eta=1,
                         num_inference_steps=num_inference_steps,
                         guidance_scale=guidance_scale,
                         strength=strength).images[0]
torch.manual_seed(SEED)
image_adv = pipe_inpaint(prompt=prompt,
                         image=adv_image,
                         mask_image=mask_image,
                         eta=1,
                         num_inference_steps=num_inference_steps,
                         guidance_scale=guidance_scale,
                         strength=strength).images[0]

# Join all the images together side by side
images = [
    source_image_pil.resize((512, 512)),
    target_image_pil.resize((512, 512)),
    adv_image.resize((512, 512)),
    image_nat.resize((512, 512)),
    image_adv.resize((512, 512))
]
joined_image = Image.fromarray(np.concatenate(images, axis=1))
joined_image.save("/data/yuval/joined_image.png")
