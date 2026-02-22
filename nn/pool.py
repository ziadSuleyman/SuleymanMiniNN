# SuleymanMiniNN/nn/pool.py
from .module import Module
from ..ops.pool import max_pool1d

class MaxPool1d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x):
        return max_pool1d(x, self.kernel_size, self.stride)