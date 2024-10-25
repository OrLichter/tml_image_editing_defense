import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from evaluation.clip_eval import ImageDirEvaluator

data_root = Path('./images/')

root = Path('./final_experiments/')

evaluator = ImageDirEvaluator(device='cuda', clip_model='ViT-B/32')


for n_noise in [1, 3, 5, 'None']:

	# for n_prompts in [1, 10, 25, 'None']:
	for n_prompts in ['None']:
		
		natural_image_similarities, natural_text_similarities = {}, {}
		image_similarities, text_similarities = {}, {}

		for image_root in tqdm(root.iterdir()):
			
			if not image_root.is_dir():
				continue
			
			image_similarities[image_root.stem] = {}
			text_similarities[image_root.stem] = {}
			natural_image_similarities[image_root.stem] = {}
			natural_text_similarities[image_root.stem] = {}
			
			# Need to find the image since the extension may be differnet
			image_path = [p for p in data_root.glob(f'{image_root.stem}.*')][0]
			input_image = Image.open(image_path).convert('RGB')
			
			output_root = image_root / f'n_noises_{n_noise}' / f'n_prompts_{n_prompts}'
			prompt_files = [p for p in output_root.iterdir() if p.suffix == '.png' and p.stem != 'adversarial_image']
			if len(prompt_files) == 0:
				continue
			
			for prompt_file in prompt_files:
				
				prompt = " ".join(prompt_file.stem.split(",")[0].split("-"))
				
				joined_image = np.array(Image.open(prompt_file).convert('RGB'))
				
				# Need to crop the relevant regions
				natural_image = Image.fromarray(joined_image[0:512, 1536:2048])
				adversarial_image = Image.fromarray(joined_image[0:512, 2048:])
				
				# Compute the cosine similarity between the adversarial and natural images
				sim_samples_to_img, sim_samples_to_text = evaluator.evaluate(gen_samples=[adversarial_image],
				                                                             src_images=[input_image],
				                                                             target_text=[prompt])
				image_similarities[image_root.stem][prompt] = sim_samples_to_img.item()
				text_similarities[image_root.stem][prompt] = sim_samples_to_text.item()
				
				sim_samples_to_img, sim_samples_to_text = evaluator.evaluate(gen_samples=[natural_image],
				                                                             src_images=[input_image],
				                                                             target_text=[prompt])
				natural_image_similarities[image_root.stem][prompt] = sim_samples_to_img.item()
				natural_text_similarities[image_root.stem][prompt] = sim_samples_to_text.item()
				
		# Compute the averages across all prompts
		natural_image_similarities_avg = {k: sum(v.values()) / len(v) for k, v in natural_image_similarities.items()}
		natural_text_similarities_avg = {k: sum(v.values()) / len(v) for k, v in natural_text_similarities.items()}
		adversarial_image_similarities_avg = {k: sum(v.values()) / len(v) for k, v in image_similarities.items()}
		adversarial_text_similarities_avg = {k: sum(v.values()) / len(v) for k, v in text_similarities.items()}
		
		# Compute the average across all images
		natural_image_similarities_avg = sum(natural_image_similarities_avg.values()) / len(natural_image_similarities_avg)
		natural_text_similarities_avg = sum(natural_text_similarities_avg.values()) / len(natural_text_similarities_avg)
		adversarial_image_similarities_avg = sum(adversarial_image_similarities_avg.values()) / len(adversarial_image_similarities_avg)
		adversarial_text_similarities_avg = sum(adversarial_text_similarities_avg.values()) / len(adversarial_text_similarities_avg)
		
		print(f"Natural Image Similarities: {natural_image_similarities_avg}")
		print(f"Natural Text Similarities: {natural_text_similarities_avg}")
		print(f"Adversarial Image Similarities: {adversarial_image_similarities_avg}")
		print(f"Adversarial Text Similarities: {adversarial_text_similarities_avg}")
		
		# Save results to a file
		with open(root / f'n_noises_{n_noise}___n_prompts_{n_prompts}_image_similarity.json', 'w') as f:
			json.dump(image_similarities, f, indent=4, sort_keys=False)
		with open(root / f'n_noises_{n_noise}___n_prompts_{n_prompts}_text_similarity.json', 'w') as f:
			json.dump(text_similarities, f, indent=4, sort_keys=False)
		with open(root / f'n_noises_{n_noise}___n_prompts_{n_prompts}_natural_image_similarity.json', 'w') as f:
			json.dump(natural_image_similarities, f, indent=4, sort_keys=False)
		with open(root / f'n_noises_{n_noise}___n_prompts_{n_prompts}_natural_text_similarity.json', 'w') as f:
			json.dump(natural_text_similarities, f, indent=4, sort_keys=False)
		
		# Save the average results to text file
		with open(root / f'n_noises_{n_noise}___n_prompts_{n_prompts}_avg_similarity.txt', 'w') as f:
			f.write(f"Adversarial Image Similarities: {adversarial_image_similarities_avg}\n")
			f.write(f"Adversarial Text Similarities: {adversarial_text_similarities_avg}\n")
			f.write(f"Natural Image Similarities: {natural_image_similarities_avg}\n")
			f.write(f"Natural Text Similarities: {natural_text_similarities_avg}")
