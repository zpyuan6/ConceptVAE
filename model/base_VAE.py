import torch.nn as nn
from torch import Tensor
from typing import List, Any
from abc import abstractmethod

class AbstractVAE(nn.Module):

    def __init__(self) -> None:
        super(AbstractVAE).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

    @abstractmethod
    def get_hyperparamters(self) -> Tensor:
        pass

    