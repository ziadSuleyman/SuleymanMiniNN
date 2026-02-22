import numpy as np
from .optimizer import Optimizer

class Momentum(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9):
        """
        Initializes the Momentum optimizer.

        Args:
            params (list): A list of Tensor objects representing the model's parameters.
            lr (float): The learning rate.
            momentum (float): The momentum factor, typically between 0.9 and 0.99.
        """
        super().__init__(params, lr)
        self.momentum_factor = momentum
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        """
        Performs a single optimization step (parameter update).
        """
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad

        
            self.v[i] = self.momentum_factor * self.v[i] + g

        
            p.data -= self.lr * self.v[i]