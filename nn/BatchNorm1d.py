import numpy as np
from .module import Module
from ..core.tensor import Tensor
from .parameter import Parameter

class BatchNorm1d(Module):
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        self.gamma = Parameter(np.ones((1, num_features), dtype=np.float32))
        self.beta = Parameter(np.zeros((1, num_features), dtype=np.float32))

        self.running_mean = np.zeros((1, num_features), dtype=np.float32)
        self.running_var = np.ones((1, num_features), dtype=np.float32)

    def forward(self, x: Tensor):
        if self.training:
            mean = np.mean(x.data, axis=0, keepdims=True)
            var = np.var(x.data, axis=0, keepdims=True)
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            mean_t = Tensor(mean, requires_grad=False)
            var_t = Tensor(var, requires_grad=False)
            
            inv_std = 1.0 / np.sqrt(var + self.eps)
            inv_std_t = Tensor(inv_std, requires_grad=False)
            
            x_centered = x - mean_t
            x_norm = x_centered * inv_std_t
            
        else:
            mean_t = Tensor(self.running_mean, requires_grad=False)
            inv_std = 1.0 / np.sqrt(self.running_var + self.eps)
            inv_std_t = Tensor(inv_std, requires_grad=False)
            
            x_centered = x - mean_t
            x_norm = x_centered * inv_std_t

        return x_norm * self.gamma + self.beta