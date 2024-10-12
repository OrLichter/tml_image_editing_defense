from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path


class ImagePromptDataset(Dataset):
    def __init__(self, image_dir: str, default_prompt: str):
        self.images = []
        self.default_prompt = default_prompt
        self.image_transforms = self.get_image_transforms()
        
        for image_path in Path(image_dir).rglob('*.jpg'):
            self.images.append(Image.open(image_path))

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
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.image_transforms(image)
        return image, self.default_prompt