# Linear Layer (Linear)
# ├── in_features          
# ├── out_features       
# ├── weight              
# ├── bias                 
# ├── forward(input)      
# │   ├── input           
# │   └── output          
# └── backward(grad_output)

import numpy as np
from .module import Module
from ..core.tensor import Tensor
from ..nn.parameter import Parameter
from ..ops.matrix import matmul
from ..ops.arithmetic import add

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        std = np.sqrt(2.0 / in_features)
        self.weight = Parameter(
            np.random.randn(out_features, in_features).astype(np.float32) * std
        )

        if bias:
            self.bias = Parameter(np.zeros((out_features,), dtype=np.float32))
        else:
            self.bias = None

    def forward(self, x: Tensor):
        out = matmul(x, self.weight.T) 
        if self.bias is not None:
            out = add(out, self.bias)
        return out

    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def __call__(self, x: Tensor):
        return self.forward(x)