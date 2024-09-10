from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

class ImagePromptDataset(Dataset):
    def __init__(self, image_dir: str, defualt_prompt: str, center_crop: bool = False):
        self.images = []
        self.default_prompt = defualt_prompt
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(1024),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
        for image_path in Path(image_dir).rglob('*.jpg'):
            self.images.append(Image.open(image_path))
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.image_transforms(image)
        return image, self.default_prompt