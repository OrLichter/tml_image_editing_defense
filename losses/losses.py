import torch

from typing import List, Union


class LpRegularization:
    """
    Apply Lp regularization to a list of parameters.
    """
    def __init__(self, p: int):
        self.p = p

    def __call__(self, regularization_parameters: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if isinstance(regularization_parameters, torch.Tensor):
            regularization_parameters = [regularization_parameters]
        return sum([torch.norm(p, self.p) for p in regularization_parameters])


class LpDistance:
    """
    Compute Lp distance between two tensors.
    """
    def __init__(self, p: int):
        self.p = p

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.norm(x - y, self.p)


class CosineSimilarity:
    """
    Compute cosine similarity between two tensors.
    Add 1 to the result to make it non-negative.
    """
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (torch.nn.functional.cosine_similarity(x, y) + 1).mean()