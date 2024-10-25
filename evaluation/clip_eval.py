import clip
import torch
from torchvision import transforms


class CLIPEvaluator(object):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess

        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] +
                                             clip_preprocess.transforms[:2] +
                                             clip_preprocess.transforms[4:])

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    @torch.no_grad()
    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = [self.clip_preprocess(image).to(self.device) for image in images]
        return self.model.encode_image(torch.stack(images).cuda())

    def get_text_features(self, text: str, norm: bool = True) -> torch.Tensor:
        tokens = clip.tokenize(text).to(self.device)
        text_features = self.encode_text(tokens).detach()
        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)
        return image_features

    def img_to_img_similarity(self, src_images, generated_images):
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)
        return (src_img_features @ gen_img_features.T).mean()

    def txt_to_img_similarity(self, text, generated_images):
        text_features = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)
        return (text_features @ gen_img_features.T).mean()


class ImageDirEvaluator(CLIPEvaluator):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        super().__init__(device, clip_model)

    def evaluate(self, gen_samples, src_images, target_text, placeholder="*"):
        sim_samples_to_img = self.img_to_img_similarity(src_images, gen_samples)
        sim_samples_to_text = self.txt_to_img_similarity(target_text, gen_samples)
        return sim_samples_to_img, sim_samples_to_text
