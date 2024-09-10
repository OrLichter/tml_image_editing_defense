import torch
import pyrallis

from diffusers import UNet2DConditionModel, LCMScheduler
from dataclasses import dataclass
from torch.utils.data import DataLoader

from pipelines.sdxl_img2img_pipeline import NoiseTrainingPipeline
from data.dataset import ImagePromptDataset

@dataclass
class Config:
    dataset_dir: str = "/home/dcor/orlichter/TML_project/data/single_image_dataset"
    default_prompt: str = "fuji pagoda"
    target_prompt: str = "fuji pagoda"
    device: str = "cpu"
    batch_size: int = 1
    epochs: int = 100
    max_steps: int = 1000
    

@pyrallis.wrap()
def main(cfg: Config):
    unet = UNet2DConditionModel.from_pretrained(
        "latent-consistency/lcm-sdxl",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe = NoiseTrainingPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16
    ).to(cfg.device)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    generator = torch.manual_seed(0)
    
    dataset = ImagePromptDataset(cfg.dataset_dir, cfg.default_prompt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    step_counter = -1
    for _ in range(cfg.epochs):
        for batch in dataloader:
            source_image, prompt = batch
            step_counter += 1
            if step_counter >= cfg.max_steps:
                break
            image = pipe(
                image=source_image[0], prompt=prompt[0], num_inference_steps=1, generator=generator, guidance_scale=8.0
            ).images[0]
                

if __name__ == '__main__':
    main()