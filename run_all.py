import os
import random
from pathlib import Path
import torch
from PIL import Image

from configs import TrainConfig, PROMPTS_LIST, InferenceConfig, INFERENCE_PROMPTS
from main import Trainer, Inference


output_root = Path('/data/yuval/tml_experiments')
output_root.mkdir(exist_ok=True, parents=True)

image_paths = [p for p in Path('./images').glob("*") if p.suffix in ['.jpg', '.png', '.jpeg']]

# Split the image_paths into two sets
image_paths = image_paths[:len(image_paths) // 2]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# image_paths = image_paths[len(image_paths) // 2:]
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

for image_path in image_paths:
	image_output_path = output_root / image_path.stem
	for n_prompts in [1, 10, 25, None]:
		
		# Sample the prompts to use - we will use the same prompts for all the noise experiments for a given image
		if n_prompts is None:
			sampled_prompts = PROMPTS_LIST
		elif n_prompts == 1:
			sampled_prompts = [""]
		else:
			sampled_prompts = [""] + random.sample(PROMPTS_LIST, n_prompts - 1)
		
		for n_noises in [1, 3, 5, None]:
		
			experiment_output_root = image_output_path / f"n_noises_{n_noises}" / f"n_prompts_{n_prompts}"
			experiment_output_root.mkdir(exist_ok=True, parents=True)
			
			# Randomly sample a seed
			seed = random.randint(0, 2 ** 32 - 1)
			
			train_cfg = TrainConfig(
				experiment_name=f'{image_path.stem}_n_noises_{n_noises}_n_prompts_{n_prompts}',
				source_image_path=image_path,
				target_image_path=image_path,
				default_source_image_caption="",
				output_path=image_output_path,
				n_optimization_steps=250,
				n_noise=n_noises,
				use_fixed_noise=True if n_noises is not None else False,
				prompts=sampled_prompts,
				seed=seed,
				guidance_scale=3.0,
			)
			trainer = Trainer(
				cfg=train_cfg,
				use_sdxl=False,
				use_lcm=True
			)
			adversarial_image = trainer.run()
			adversarial_image.save(experiment_output_root / "adversarial_image.png")
			torch.save(trainer.noises, experiment_output_root / "noise.pt")
			
			adversarial_image = Image.open(experiment_output_root / "adversarial_image.png").convert("RGB")
			trainer.noises = torch.load(experiment_output_root / "noise.pt")
			
			# Part 2: Inference
			inference_cfg = InferenceConfig(
				experiment_name=f'{image_path.stem}_n_noises_{n_noises}_n_prompts_{n_prompts}',
				source_image_path=image_path,
				target_image_path=image_path,
				output_path=experiment_output_root,
				n_steps=4,
				guidance_scale=4.0,
				strength=0.60,
				use_fixed_noise=True if n_noises is not None else False,
				n_noise=len(trainer.noises) if n_noises is not None else 1,
			)
			
			inference_noises = None
			if n_noises is not None:
				inference_noises = trainer.noises
				
			Inference.run_inference(
				cfg=inference_cfg,
				adversarial_image=adversarial_image,
				inference_prompts=INFERENCE_PROMPTS,
				use_sdxl=False,
				use_lcm=True,
				noises=inference_noises,
				training_prompts=train_cfg.prompts,
			)
