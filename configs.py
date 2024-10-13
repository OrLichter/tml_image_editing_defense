from dataclasses import field, dataclass
from pathlib import Path
from typing import List

from PIL import Image

PROMPTS_LIST = [
	"",
	"in space",
	"on fire",
	# "as a liquid",
	# "melting",
	# "as a lava",
	# "crumbling",
	# "as a lego",
	# "rusted and old",
	# "covered in gold",
	# "as a origami",
	# "made of of candy",
	# "as a hologram",
	# "as a neon sign",
	# "as a plush toy",
	# "exploded",
	# "on mars",
	# "on the moon",
	# "in a cartoon style",
	# "in a pixel art style",
	# "cubism painting",
	# "abstract painting",
	# "pencil drawing",
	# "oil painting",
	# "watercolor painting",
	# "ink drawing",
	# "pastel drawing",
	# "as if submerged underwater",
	# "in a dystopian world",
	# "in a utopian world",
	# "as a mosaic",
	# "in a desert",
	# "in a forest",
	# "in a city",
	# "in the style of picasso",
	# "in the style of van gogh",
	# "in the style of monet",
	# "in a apocalypse",
	# "in a apocalypse",
	# "in a cyberpunk world",
	# "in a steampunk world",
	# "in a fantasy world",
	# "as a crystal",
	# "made of ice",
	# "as a balloon",
	# "made of glass",
	# "as a snow globe",
	# "as a robot",
	# "in low poly style",
	# "as a wireframe",
	# "as a steampunk machine",
	# "wrapped in vines",
	# "as a wooden sculpture",
	# "as a papercraft",
	# "as if made of shadows",
	# "glowing in the dark",
	# "as a glitch effect",
	# "as a 3D printed object",
	# "as a pixelated glitch",
	# "surrounded by lightning",
	# "as a graffiti mural",
	# "as a comic book panel",
	# "as a stained glass window",
	# "as a shadow puppet",
	# "as a chalk drawing",
	# "as a street art stencil",
	# "as if made of stars",
	# "as a holographic projection",
	# "in a futuristic city",
	# "in an underwater cave",
	# "in a sci-fi world",
	# "in a medieval setting",
	# "as if made of electricity",
	# "as a giant sculpture",
	# "shattered into pieces",
	# "transformed into light",
	# "as a floating cloud",
	# "as a time-traveling object",
]
INFERENCE_PROMPTS = [
	"",
	"in space",
	"on fire",
	# "covered in gold",
	# "frozen in ice",
	# "in space",
	# "as a black and white pencil sketch",
	# "in the style of picasso",
]


@dataclass
class TrainConfig:
	# Source image path
	source_image_path: Path = Path("data/images/japan.jpg")
	# Target image path
	target_image_path: Path = Path("data/images/stick-figure-sticker.jpg")
	# Target image prompt
	default_source_image_caption: str = "fuji pagoda"
	# Output path
	output_path: Path = Path("./output")
	# Experiment name
	experiment_name: str = 'experiment_l2'
	# Number of steps for optimization
	n_optimization_steps: int = 200
	# Number of denoising steps per iteration during optimization
	n_denoising_steps_per_iteration: int = 4
	# Whether to apply loss on images (after decoding with VAE)
	apply_loss_on_images: bool = True
	# Whether to apply loss between latents (instead of decoding with VAE at each iteration)
	apply_loss_on_latents: bool = False
	# Whether to limit the timesteps considered during optimization
	limit_timesteps: bool = True
	# Loss lambda for L2 loss between the output image and target image
	rec_loss_lambda: float = 1.0
	# Loss lambda for minimizing the strength of the perturbations applied to the source image
	perturbation_loss_lambda: float = 1.0
	# Seed to use for training
	seed: int = 42
	# Default prompt to use
	prompts: List[str] = field(default_factory=lambda: PROMPTS_LIST)
	# Device to use for training
	device: str = "cuda:0"
	
	""" Various parameters for optimization"""
	# Norm type
	norm_type: str = "l2"  # or "l2"
	# Epsilon
	eps: float = 0.1  # 16 for l2
	# Step size
	step_size: float = 0.006  # 1 for l2
	# Min value for clamp
	min_value: int = -1
	# Max value for clamp
	max_value: int = 1
	# Guidance scale for training
	guidance_scale: float = 3.0
	# Number of repetitions per iteration
	grad_reps: int = 5  # 10 for l2
	# Eta value for scheduler
	eta: float = 1.0
	# Whether to add a prefix to each prompt describe the object in the image
	add_image_caption_to_prompts: bool = False
	# Whether to allow perturbations only on salient regions of the image
	use_segmentation_mask: bool = True
	
	""" For visualization purposes """
	image_visualization_interval: int = 25
	
	def __post_init__(self):
		self.output_path.mkdir(exist_ok=True, parents=True)
		self.source_image = Image.open(self.source_image_path).convert("RGB")
		self.target_image = Image.open(self.target_image_path).convert("RGB")
		if self.norm_type == "l2":
			self.eps = 32
			self.step_size = 10
			self.grad_reps = 10
		else:
			self.eps = 0.1
			self.step_size = 0.006
			self.grad_reps = 5


@dataclass
class InferenceConfig:
	# Source image path
	source_image_path: Path = Path("data/images/japan.jpg")
	# Target image path
	target_image_path: Path = Path("data/images/stick-figure-sticker.jpg")
	# Target image prompt
	default_source_image_caption: str = "fuji pagoda"
	# Output path
	output_path: Path = Path("./output")
	# Experiment name
	experiment_name: str = 'experiment_inference'
	# Number of denoising steps
	n_steps: int = 100
	# Strength for SDEdit
	strength: float = 0.6
	# Guidance scale
	guidance_scale: float = 7.5
	# Seed to use for inference
	seed: int = 42
	add_image_caption_to_prompts: bool = False
	
	def __post_init__(self):
		self.output_path.mkdir(exist_ok=True, parents=True)
		self.source_image = Image.open(self.source_image_path).convert("RGB")
