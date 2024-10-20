from typing import List

from PIL import Image
from diffusers import AutoPipelineForImage2Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from transformers import Blip2ForConditionalGeneration, AutoProcessor

from configs import TrainConfig


class ImagePromptDataset(Dataset):
	
	def __init__(self,
	             source_images: List[Image.Image],
	             target_images: List[Image.Image],
	             pipeline: AutoPipelineForImage2Image,
	             device: str,
	             dtype: torch.dtype,
	             cfg: TrainConfig):
		super(ImagePromptDataset, self).__init__()
		self.source_images = source_images
		self.target_images = target_images
		self.device = device
		self.dtype = dtype
		self.cfg = cfg
		self.source_image_captions = self.get_source_captions()
		self.transform = self.get_image_transforms()
		self.target_latents = self.compute_latents(target_images, pipeline=pipeline)
	
	def __getitem__(self, idx: int):
		source_tensor = self.transform(self.source_images[idx]).clone()
		target_tensor = self.transform(self.target_images[idx]).clone()
		target_latent = self.target_latents[idx]
		source_image_caption = self.source_image_captions[idx]
		return source_tensor, target_tensor, target_latent, source_image_caption
	
	def compute_latents(self, images: List[Image.Image], pipeline):
		images = [self.transform(image).unsqueeze(0).to(self.device, self.dtype) for image in images]
		return [pipeline.vae.encode(image).latent_dist.sample() for image in images]
	
	def get_source_captions(self):
		source_image_captions = [''] * len(self.cfg.source_images)
		if self.cfg.default_source_image_caption != "" or self.cfg.add_image_caption_to_prompts:
			if self.cfg.default_source_image_caption != "":
				source_image_captions = [self.cfg.default_source_image_caption] * len(self.cfg.source_images)
			else:
				source_image_captions = [
					self._get_image_caption(image, device=self.device, dtype=self.dtype)
					for image in self.cfg.source_images
				]
		return source_image_captions
	
	@staticmethod
	def _get_image_caption(image: Image.Image, device='cuda:0', dtype=torch.float16) -> str:
		processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
		model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", torch_dtype=dtype)
		question = "what is shown in the image?"
		inputs = processor(image, question, return_tensors="pt").to(device, dtype)
		generated_ids = model.generate(**inputs, max_new_tokens=20)
		generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
		return generated_text
	
	@staticmethod
	def get_image_transforms():
		return transforms.Compose(
			[
				transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
				transforms.CenterCrop(512),
				transforms.ToTensor(),
				transforms.Normalize([0.5], [0.5]),
			]
		)
	
	def __len__(self):
		return len(self.source_images)
