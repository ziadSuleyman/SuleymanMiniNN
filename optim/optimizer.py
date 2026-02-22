# optim/
# │
# ├── optimizer.py              
# │   ├── params                
# │   ├── lr                     
# │   ├── step()                 
# │   └── zero_grad()            
# │
# ├── sgd.py                    
# │   ├── inherits Optimizer
# │   ├── momentum (اختياري)
# │   ├── velocity              
# │   └── step()               
# │
# └── adam.py                  
#     ├── inherits Optimizer
#     ├── beta1, beta2, epsilon
#     ├── m, v                  
#     ├── t                      
#     └── step()               

from ..nn.parameter import Parameter

class Optimizer:
    def __init__(self, params, lr=0.001):
        self.params = list(params)
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.zero_grad()
