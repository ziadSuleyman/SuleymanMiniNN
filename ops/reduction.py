# reduction
# │
# ├── Sum(Function)
# │   ├── forward(x, axis=None, keepdims=False)
# │   │   └── حفظ: shape الأصلية + axis
# │   └── backward(grad_output)
# │       └── broadcast gradient إلى شكل x
# │
# ├── Mean(Function)
# │   ├── forward(x, axis=None, keepdims=False)
# │   │   ├── حساب mean
# │   │   └── حفظ: shape + axis + count
# │   └── backward(grad_output)
# │       └── grad = grad_sum / count
# │
# ├── sum(tensor, axis=None, keepdims=False)
# │
# └── mean(tensor, axis=None, keepdims=False)
import numpy as np
from ..core.function import Function
from ..core.tensor import Tensor


class Sum(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.ctx.save_for_backward(x.shape)
        return np.sum(x, axis=self.axis, keepdims=self.keepdims)

    def backward(self, grad_output):
        (original_shape,) = self.ctx.get_saved()

        grad = grad_output

        if not self.keepdims and self.axis is not None:
            grad = np.expand_dims(grad, axis=self.axis)

        grad = np.broadcast_to(grad, original_shape)
        return [grad]


class Mean(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.ctx.save_for_backward(x.shape)
        if self.axis is None:
            self.count = x.size
        else:
            self.count = x.shape[self.axis]
        return np.mean(x, axis=self.axis, keepdims=self.keepdims)

    def backward(self, grad_output):
        (original_shape,) = self.ctx.get_saved()

        grad = grad_output / self.count

        if not self.keepdims and self.axis is not None:
            grad = np.expand_dims(grad, axis=self.axis)

        grad = np.broadcast_to(grad, original_shape)
        return [grad]



def sum(tensor, axis=None, keepdims=False):
    return Sum(axis, keepdims).apply(tensor)


def mean(tensor, axis=None, keepdims=False):
    return Mean(axis, keepdims).apply(tensor)
