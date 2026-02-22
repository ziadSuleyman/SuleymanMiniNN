# loss/cross_entropy.py
# │
# ├── BinaryCrossEntropy(Function)
# │   ├── forward(pred, target)
# │   │   ├── pred: Tensor (probabilities بعد sigmoid)
# │   │   ├── target: Tensor (0 أو 1)
# │   │   └── حفظ pred, target في ctx
# │   └── backward(grad_output)
# │       └── grad = (pred - target) / N
# │
# ├── CrossEntropy(Function)
# │   ├── forward(logits, target)
# │   │   ├── logits: Tensor (softmax)
# │   │   ├── target: Tensor (one-hot أو indices)
# │   │   └── حفظ softmax و target في ctx
# │   └── backward(grad_output)
# │       └── grad = (softmax - target) / N
# │
# ├── binary_cross_entropy(pred_tensor, target_tensor)
# │
# └── cross_entropy(logits_tensor, target_tensor)

import numpy as np
from ..core.function import Function
from ..core.tensor import Tensor
from ..ops.arithmetic import add, mul
from ..ops.activation import sigmoid


import numpy as np
from ..core.function import Function
from ..core.tensor import Tensor

# ==============================================================================
# 1. Binary Cross Entropy Converted to take Logits for Stability
# ==============================================================================
class BinaryCrossEntropy(Function):
    """
    Computes BCE Loss assuming inputs are Logits (before Sigmoid).
    This combines Sigmoid + BCE for numerical stability.
    """
    def forward(self, logits, target):
        self.probs = 1 / (1 + np.exp(-logits))
        
        self.ctx.save_for_backward(self.probs, target)
        
        epsilon = 1e-12
        probs_clipped = np.clip(self.probs, epsilon, 1. - epsilon)
        
        # Formula: - [y * log(p) + (1-y) * log(1-p)]
        loss = - (target * np.log(probs_clipped) + (1 - target) * np.log(1 - probs_clipped))
        return np.mean(loss)

    def backward(self, grad_output):
        pred, target = self.ctx.get_saved()
        N = pred.size  
        grad = (pred - target) / N 
        
        return [grad * grad_output, np.zeros_like(target)]

def binary_cross_entropy(logits_tensor: Tensor, target_tensor: Tensor):
    """
    input: logits_tensor (Raw scores before sigmoid)
    target: target_tensor (0 or 1)
    """
    return BinaryCrossEntropy().apply(logits_tensor, target_tensor)

# ==============================================================================
# 2. Categorical Cross Entropy
# ==============================================================================
class CrossEntropy(Function):
    """
    Computes Cross Entropy Loss assuming inputs are Logits (before Softmax).
    """
    def forward(self, logits, target):
        self.N = logits.shape[0]
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)
        
        self.ctx.save_for_backward(self.probs, target)
        
        if target.ndim == 1:
            correct_logprobs = -np.log(self.probs[np.arange(self.N), target] + 1e-12)
        else:
            correct_logprobs = -np.sum(target * np.log(self.probs + 1e-12), axis=1)
            
        loss = np.mean(correct_logprobs)
        return loss

    def backward(self, grad_output):
        probs, target = self.ctx.get_saved()
        N = self.N

        if target.ndim == 1:
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(N), target] = 1
            target = one_hot
        grad = (probs - target) / N
        
        return [grad * grad_output, np.zeros_like(target)]

def cross_entropy(logits_tensor: Tensor, target_tensor: Tensor):
    return CrossEntropy().apply(logits_tensor, target_tensor)

# ==============================================================================
# 3. Class Wrappers
# ==============================================================================
class CrossEntropyLoss:
    def __call__(self, logits, target):
        return cross_entropy(logits, target)

class BCELoss:
    def __call__(self, logits, target):
        return binary_cross_entropy(logits, target)