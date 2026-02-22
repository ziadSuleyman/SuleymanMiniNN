
import numpy as np
from .module import Module
from .parameter import Parameter
from ..core.tensor import Tensor


class BatchNorm2d(Module):
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.gamma = Parameter(np.ones((1, num_features, 1, 1), dtype=np.float32))
        self.beta = Parameter(np.zeros((1, num_features, 1, 1), dtype=np.float32))
        self.running_mean = np.zeros((1, num_features, 1, 1), dtype=np.float32)
        self.running_var = np.ones((1, num_features, 1, 1), dtype=np.float32)

    def forward(self, x):
        if self.training:
            mean = np.mean(x.data, axis=(0, 2, 3), keepdims=True)
            var = np.var(x.data, axis=(0, 2, 3), keepdims=True)
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            inv_std = 1.0 / np.sqrt(var + self.eps)
            x_norm = (x - Tensor(mean)) * Tensor(inv_std)
        else:
            inv_std = 1.0 / np.sqrt(self.running_var + self.eps)
            x_norm = (x - Tensor(self.running_mean)) * Tensor(inv_std)

        return x_norm * self.gamma + self.beta
