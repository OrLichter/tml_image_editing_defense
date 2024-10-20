from dataclasses import field, dataclass
from pathlib import Path
from typing import List, Optional

from PIL import Image

PROMPTS_LIST = [
	"",
	"melting",
	"shattered",
	"moldy",
	"plush",
	"futuristic",
	"glowing",
	"wet",
	"marble",
	"origami",
	"hologram",
	"made of glass",
	"covered in moss",
	
	"painting",
	"sketch",
	"mosaic",
	"oil painting",
	"pencil drawing",
	"charcoal drawing",
	"pastel drawing",
	"ink drawing",
	"3d rendering",
	"comic drawing",
	"animation",
	"anime",
	"pixel art",
	"concept art",
	"minimalist art",
	"in the style of picasso",
	"in the style of van gogh",
	"in the style of monet",
	"wooden sculpture",
	"street art stencil",
	"chalk drawing",
	
	"underwater",
	"on mars",
	"in utopian world",
	"in a desert",
	"in a city",
	"in an apocalypse",
	"in a fantasy world",
	"in a lightning storm",
	"in a medieval setting",
	"in a futuristic city",
	"in a forest",
	"in a jungle",
	"in a mountain",
	"on an alien planet",
	"during a sunset",
	"in an enchanted forest",
]
INFERENCE_PROMPTS = [
	"on fire",
	"colorful",
	"frozen",
	"muddy",
	"gold",
	"lego",
	"made of candy",
	"watercolor painting",
	"cartoon",
	"pixel art",
	"grafiti",
	"abstract art",
	"cubism",
	"in space",
	"underwater",
	"in a snowstorm",
	"on a beach",
	"expressionist style",
	"disney style",
	"in a sci-fi world",
]
NEGATIVE_PROMPT = '(worst quality, low quality, blurry:1.3), (bad teeth, deformed teeth, deformed lips), (bad anatomy, bad proportions:1.1), (deformed iris, deformed pupils), (deformed eyes, bad eyes), (deformed face, ugly face, bad face), (deformed hands, bad hands, fused fingers), morbid, mutilated, mutation, disfigured'


@dataclass
class TrainConfig:
	# Source image path
	source_image_paths: List[Path] = field(default_factory=lambda: [Path("data/images/japan.jpg")])
	# Target image path
	target_image_paths: List[Path] = field(default_factory=lambda: [Path("data/images/japan.jpg")])
	# Target image prompt
	default_source_image_caption: str = ""
	# Output path
	output_path: Path = Path("./output")
	# Experiment name
	experiment_name: str = 'experiment_name'
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
	eta: float = 0.9
	# Whether to add a prefix to each prompt describe the object in the image
	add_image_caption_to_prompts: bool = False
	# Whether to allow perturbations only on salient regions of the image
	use_segmentation_mask: bool = False
	# Whether to use a fixed noise for training / inference
	use_fixed_noise: bool = True
	# How many noise inputs to allow during training
	n_noise: int = 1
	
	""" For visualization purposes """
	image_visualization_interval: int = 25
	
	def __post_init__(self):
		self.output_path.mkdir(exist_ok=True, parents=True)
		self.source_images = [Image.open(path).convert("RGB") for path in self.source_image_paths]
		self.target_images = [Image.open(path).convert("RGB") for path in self.target_image_paths]
		if self.norm_type == "l2":
			self.eps = 32
			self.step_size = 7.5
			self.grad_reps = 10
		else:
			self.eps = 0.1
			self.step_size = 0.006
			self.grad_reps = 5


@dataclass
class InferenceConfig:
	# Source image path
	source_image_paths: List[Path] = field(default_factory=lambda: [Path("data/images/japan.jpg")])
	# Target image path
	target_image_paths: List[Path] = field(default_factory=lambda: [Path("data/images/japan.jpg")])
	# Target image prompt
	default_source_image_caption: str = ""
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
	# Whether to add a prefix to each prompt describe the object in the image
	add_image_caption_to_prompts: bool = False
	# Whether to use the same noises as training
	use_fixed_noise: bool = True
	# How many noises to use for inference if not running with fixed noise
	n_noise: int = 1
	# Validation images (not seen during training)
	validation_images_path: Optional[Path] = Path("validation_images.txt")
	
	def __post_init__(self):
		self.output_path.mkdir(exist_ok=True, parents=True)
		self.source_images = [Image.open(path).convert("RGB") for path in self.source_image_paths]
