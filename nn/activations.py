# Activations
# ├── ReLU(Function)
# │   ├── forward(x)
# │   └── backward(grad_output)
# ├── Sigmoid(Function)
# │   ├── forward(x)
# │   └── backward(grad_output)
# └── Tanh(Function)
#     ├── forward(x)
#     └── backward(grad_output)

# user
# ├── relu(tensor)
# ├── sigmoid(tensor)
# └── tanh(tensor)
# nn/activations.py
# SuleymanMiniNN/nn/activations.py

from ..ops.activation import relu, sigmoid, tanh

class ReLU:
    def __init__(self):
        pass

    def forward(self, x):
        return relu(x)

    def __call__(self, x):
        return self.forward(x)

class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        return sigmoid(x)

    def __call__(self, x):
        return self.forward(x)

class Tanh:
    def __init__(self):
        pass

    def forward(self, x):
        return tanh(x)

    def __call__(self, x):
        return self.forward(x)
    

