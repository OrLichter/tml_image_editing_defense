import torch
import torch.nn.functional as F
from typing import List, Union


def perturbation_loss(adv_image, source_image):
    """ Compute L2 loss between the adversarial image and the source image. This keeps perturbations small. """
    return F.mse_loss(adv_image, source_image)
