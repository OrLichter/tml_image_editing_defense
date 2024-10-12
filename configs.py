from dataclasses import field, dataclass
from pathlib import Path
from typing import List

from PIL import Image


@dataclass
class TrainConfig:
	# Source image path
	source_image_path: Path = Path("data/images/japan.jpg")
	# Target image path
	target_image_path: Path = Path("data/images/stick-figure-sticker.jpg")
	# Output path
	output_path: Path = Path("./output")
	# Number of steps for optimization
	n_optimization_steps: int = 200
	# Number of denoising steps per iteration during optimization
	n_denoising_steps_per_iteration: int = 4
	# Whether to apply loss on images (after decoding with VAE)
	apply_loss_on_images: bool = True
	# Whether to apply loss between latents (instead of decoding with VAE at each iteration)
	apply_loss_on_latents: bool = False
	# Whether to limit the timesteps considered during optimization
	limit_timesteps: bool = False
	# Loss lambda for minimizing the strength of the perturbations applied to the source image
	perturbation_loss_lambda: float = 0.0
	# Seed to use for training
	seed: int = 42
	# Default prompt to use
	prompts: List[str] = field(default_factory=lambda: [""])

	""" Various parameters for optimization"""
	# Norm type
	norm_type: str = "l2"  # or "linf"
	# Epsilon
	eps: int = 16
	# Step size
	step_size: int = 1
	# Min value for clamp
	min_value: int = -1
	# Max value for clamp
	max_value: int = 1
	# Guidance scale for training
	guidance_scale: float = 7.5
	# Number of repetitions per iteration
	grad_reps: int = 10

	def __post_init__(self):
		self.output_path.mkdir(exist_ok=True, parents=True)
		self.source_image = Image.open(self.source_image_path).convert("RGB")
		self.target_image = Image.open(self.target_image_path).convert("RGB")


@dataclass
class InferenceConfig:
	# Source image path
	source_image_path: Path = Path("data/images/japan.jpg")
	# Target image path
	target_image_path: Path = Path("data/images/stick-figure-sticker.jpg")
	# Output path
	output_path: Path = Path("./output")
	# Number of denoising steps
	n_steps: int = 100
	# Strength for SDEdit
	strength: float = 0.6
	# Guidance scale
	guidance_scale: float = 7.5
	# Seed to use for inference
	seed: int = 42

	def __post_init__(self):
		self.output_path.mkdir(exist_ok=True, parents=True)
		self.source_image = Image.open(self.source_image_path).convert("RGB")
