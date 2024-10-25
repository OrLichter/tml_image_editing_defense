from pathlib import Path

from PIL import Image
from tqdm import tqdm

from evaluation.clip_eval import ImageDirEvaluator

data_root = Path('/data/yuval/code/tau/tml_final_project/images/')

root = Path('/data/yuval/tml_experiments/photoguard_results/')
# root = Path('/data/yuval/tml_experiments/photoguard_results_img2img/')

evaluator = ImageDirEvaluator(device='cuda', clip_model='ViT-B/32')

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
	
	for prompt_root in image_root.iterdir():
		
		if not prompt_root.is_dir():
			continue
			
		if 'detailed' in prompt_root.name:
			continue
		
		print(prompt_root)
		
		prompt = prompt_root.stem
		natural_image = Image.open(prompt_root / 'natural_v2.png').convert('RGB')
		adversarial_image = Image.open(prompt_root / 'adversarial_v2.png').convert('RGB')
		# Compute the cosine similarity between the adversarial and natural images
		sim_samples_to_img, sim_samples_to_text = evaluator.evaluate(gen_samples=[adversarial_image],
		                                                             src_images=[input_image],
		                                                             target_text=[prompt])
		image_similarities[image_root.stem][prompt] = sim_samples_to_img
		text_similarities[image_root.stem][prompt] = sim_samples_to_text
		
		# Compute the cosine similarity between the adversarial and natural images
		sim_samples_to_img, sim_samples_to_text = evaluator.evaluate(gen_samples=[natural_image],
		                                                             src_images=[input_image],
		                                                             target_text=[prompt])
		natural_image_similarities[image_root.stem][prompt] = sim_samples_to_img
		natural_text_similarities[image_root.stem][prompt] = sim_samples_to_text
		
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