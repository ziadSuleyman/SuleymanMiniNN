import numpy as np
from .module import Module
from ..core.tensor import Tensor

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x

        mask = (np.random.rand(*x.data.shape) > self.p).astype(np.float32)

        scale = 1.0 / (1.0 - self.p)
        
        final_mask = mask * scale

        mask_tensor = Tensor(final_mask, requires_grad=False)

        return x * mask_tensor