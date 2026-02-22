# Sequential (nn/sequential.py)
# ├── __init__(self, *modules)
# │   └── modules: قائمة Modules (مثل Linear, Activations, ...)
# ├── forward(self, x)
# │   └── يمرر الـ Tensor خلال كل module بالتسلسل
# ├── backward(self, grad_output)  # عادة يتم عبر autograd تلقائياً
# └── __call__(self, x)
#     └── واجهة تجعل Sequential قابل للاستدعاء مثل دالة: model(x)
from .module import Module
from ..nn.parameter import Parameter 

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)

    def forward(self, x):
        out = x
        for module in self.modules:
            out = module(out)
        return out

    def __call__(self, x):
        return self.forward(x)
    def parameters(self):
        params = []
        for module in self.modules:
            if hasattr(module, 'parameters') and callable(module.parameters):
                params.extend(module.parameters())
            else:
                for attr_name, attr_value in module.__dict__.items():
                    if isinstance(attr_value, Parameter):
                        params.append(attr_value)
        
        return params