# SuleymanMiniNN/nn/flatten.py

from .module import Module

class Flatten(Module):
    """
    Flattens the input tensor to (Batch_Size, -1).
    Useful before Linear layers.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # نستخدم دالة reshape الموجودة في التينسور (والتي تعتمد على ops/array.py)
        return x.reshape(x.shape[0], -1)