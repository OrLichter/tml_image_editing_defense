import random
from dataclasses import dataclass
from pathlib import Path

import pyrallis
import torch

from configs import TrainConfig, PROMPTS_LIST, InferenceConfig, INFERENCE_PROMPTS
from main import Trainer, Inference


@dataclass
class Config:
	n_optimization_steps: int = 200


@pyrallis.wrap()
def main(cfg: Config):
	output_root = Path('./universal_perturbations')
	output_root.mkdir(exist_ok=True, parents=True)
	
	image_paths = [p for p in Path('./images').glob("*") if p.suffix in ['.jpg', '.png', '.jpeg']]
	
	for experiment_idx in range(20):
		
		sampled_image_paths = random.sample(image_paths, 10)  # We'll use at most 10 images
		validation_images_paths = random.sample([p for p in image_paths if p not in sampled_image_paths], 20)
		
		for n_images in [10, 5, 1]:
			
			image_paths_subset = sampled_image_paths[:n_images]
	
			sampled_prompts = PROMPTS_LIST
			n_noises = 1
			
			experiment_output_root = output_root / f'experiment_{experiment_idx}___n_images_{n_images}'
			experiment_output_root.mkdir(exist_ok=True, parents=True)
			
			# Randomly sample a seed
			seed = random.randint(0, 2 ** 32 - 1)
			
			train_cfg = TrainConfig(
				experiment_name=f'experiment_idx_{experiment_idx}__n_images_{n_images}',
				source_image_paths=image_paths_subset,
				target_image_paths=image_paths_subset,
				output_path=experiment_output_root,
				n_optimization_steps=cfg.n_optimization_steps,
				n_noise=n_noises,
				use_fixed_noise=True,
				prompts=sampled_prompts,
				seed=seed,
				guidance_scale=4.0,
				add_image_caption_to_prompts=False,
				default_source_image_caption="",
			)
			trainer = Trainer(
				cfg=train_cfg,
				use_sdxl=False,
				use_lcm=True
			)
			adversarial_image, perturbation = trainer.run()
			adversarial_image.save(experiment_output_root / "adversarial_image.png")
			torch.save(trainer.noises, experiment_output_root / "noise.pt")
			torch.save(perturbation, experiment_output_root / "perturbation.pt")
			
			trainer.noises = torch.load(experiment_output_root / "noise.pt")
			perturbation = torch.load(experiment_output_root / "perturbation.pt")
			
			# Part 2: Inference
			inference_cfg = InferenceConfig(
				experiment_name=f'experiment_idx_{experiment_idx}__n_images_{n_images}',
				source_image_paths=image_paths_subset,
				target_image_paths=image_paths_subset,
				output_path=experiment_output_root,
				n_steps=4,
				guidance_scale=4.0,
				strength=0.60,
				use_fixed_noise=True,
				n_noise=len(trainer.noises),
				add_image_caption_to_prompts=False,
				default_source_image_caption="",
				validation_images_paths=validation_images_paths,
			)
			
			inference_noises = None
			if n_noises is not None:
				inference_noises = trainer.noises
			
			Inference.run_inference(
				cfg=inference_cfg,
				perturbation=perturbation,
				inference_prompts=INFERENCE_PROMPTS,
				use_sdxl=False,
				use_lcm=True,
				noises=inference_noises,
			)


if __name__ == '__main__':
	main()

