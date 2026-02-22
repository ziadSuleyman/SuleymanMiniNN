# MSE Loss
# │
# ├── forward(pred: Tensor, target: Tensor)
# │   ├── diff = pred - target
# │   ├── sq = diff ** 2
# │   └── loss = mean(sq)
# │
# └── backward(grad_output)
#     ├── grad_pred = 2 * (pred - target) / N
#     └── grad_target = zeros_like(target)

import numpy as np
from ..core.function import Function
from ..core.tensor import Tensor
class MSELoss(Function):
    def forward(self, pred: Tensor, target: Tensor):
        N = pred.data.size 
        return ((pred - target) ** 2).sum() / N       
    __call__ = forward

    def backward(self, grad_output):
        pred, target = self.ctx.get_saved()
        grad_pred = 2 * (pred - target) / self.N
        grad_pred = grad_pred * grad_output
        grad_target = np.zeros_like(target)
        return [grad_pred, grad_target]

def mse(pred_tensor: Tensor, target_tensor: Tensor):
    return MSELoss().apply(pred_tensor, target_tensor)