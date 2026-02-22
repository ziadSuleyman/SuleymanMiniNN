# activation
# │
# ├── ReLU
# │   ├── forward(x)
# │   └── backward(grad_out)
# │
# ├── Sigmoid
# │   ├── forward(x)
# │   └── backward(grad_out)
# │
# └── Tanh
#     ├── forward(x)
#     └── backward(grad_out)
import numpy as np
from ..core.function import Function
class ReLU(Function):
    def forward(self, x):
        self.ctx.save_for_backward(x)
        return np.maximum(0, x)

    def backward(self, grad_output):
        x, = self.ctx.get_saved()
        grad = grad_output * (x > 0).astype(np.float32)
        return grad

class Sigmoid(Function):
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.ctx.save_for_backward(out)
        return out

    def backward(self, grad_output):
        out, = self.ctx.get_saved()
        grad = grad_output * out * (1 - out)
        return grad

class Tanh(Function):
    def forward(self, x):
        out = np.tanh(x)
        self.ctx.save_for_backward(out)
        return out

    def backward(self, grad_output):
        out, = self.ctx.get_saved()
        grad = grad_output * (1 - out ** 2)
        return grad
def relu(tensor):
    return ReLU.apply(tensor)

def sigmoid(tensor):
    return Sigmoid.apply(tensor)

def tanh(tensor):
    return Tanh.apply(tensor)
