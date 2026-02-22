from .optimizer import Optimizer

import numpy as np


import numpy as np
from .optimizer import Optimizer

# SuleymanMiniNN/optim/sgd.py
from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            
            p.data -= self.lr * p.grad