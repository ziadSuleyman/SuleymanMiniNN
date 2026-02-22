# Module
# ├── __init__()            
# ├── forward(*args)          
# ├── __call__(*args)         
# ├── parameters()            
# ├── zero_grad()             
# ├── add_module(name, module) 
# └── _modules               
from ..core.tensor import Tensor
from .parameter import Parameter

class Module:
    def __init__(self):
        self.training = True  
        self._modules = dict()  

    def forward(self, *args):
        raise NotImplementedError

    def __call__(self, *args):
        return self.forward(*args)

    def add_module(self, name, module):
        self._modules[name] = module

    def parameters(self):
        params = []
        for name, value in self.__dict__.items():
            if hasattr(value, 'requires_grad') and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def train(self):
        self.training = True
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                attr.train()
            if isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Module):
                        item.train()
    
    def eval(self):
        self.training = False 
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                attr.eval()
            if isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Module):
                        item.eval()
