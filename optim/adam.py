import numpy as np
from .optimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            
            if p not in self.m:
                self.m[p] = np.zeros_like(p.data)
                self.v[p] = np.zeros_like(p.data)

            g = p.grad
            
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * g
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (g ** 2)
            
            m_hat = self.m[p] / (1 - self.beta1 ** self.t)
            v_hat = self.v[p] / (1 - self.beta2 ** self.t)
            
            update = -self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            p.data += update