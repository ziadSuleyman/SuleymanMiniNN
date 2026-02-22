# arithmetic.py
# │
# ├── Add        # x + y
# │   ├── forward(x, y)
# │   └── backward(grad_out)
# │
# ├── Sub        # x - y
# │   ├── forward(x, y)
# │   └── backward(grad_out)
# │
# ├── Mul        # x * y
# │   ├── forward(x, y)
# │   └── backward(grad_out)
# │
# └── Div        # x / y
#     ├── forward(x, y)
#     └── backward(grad_out)
# SuleymanMiniNN/ops/arithmetic.py

import numpy as np
from SuleymanMiniNN.core.function import Function
import numpy as np
from SuleymanMiniNN.core.function import Function

def unbroadcast(grad, original_shape):
    while grad.ndim > len(original_shape):
        grad = grad.sum(axis=0)
    
    for i, dim in enumerate(original_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
            
    return grad
class Add(Function):
    def forward(self, x, y):
        self.ctx.save_for_backward(x.shape, y.shape)
        return x + y

    def backward(self, grad_output):
        x_shape, y_shape = self.ctx.get_saved()
        return unbroadcast(grad_output, x_shape), unbroadcast(grad_output, y_shape)

class Sub(Function):
    def forward(self, x, y):
        self.ctx.save_for_backward(x.shape, y.shape)
        return x - y

    def backward(self, grad_output):
        x_shape, y_shape = self.ctx.get_saved()
        return unbroadcast(grad_output, x_shape), unbroadcast(-grad_output, y_shape)

class Mul(Function):
    def forward(self, x, y):
        self.ctx.save_for_backward(x, y)
        return x * y

    def backward(self, grad_output):
        x, y = self.ctx.get_saved()
        return unbroadcast(grad_output * y, x.shape), unbroadcast(grad_output * x, y.shape)

class Div(Function):
    def forward(self, x, y):
        self.ctx.save_for_backward(x, y)
        return x / y

    def backward(self, grad_output):
        x, y = self.ctx.get_saved()
        # قاعدة القسمة
        grad_x = grad_output / y
        grad_y = -grad_output * x / (y ** 2)
        return unbroadcast(grad_x, x.shape), unbroadcast(grad_y, y.shape)

class Pow(Function):
    def forward(self, x, y):
        self.ctx.save_for_backward(x, y)
        return x ** y

    def backward(self, grad_output):
        x, y = self.ctx.get_saved()
        
        grad_x = grad_output * y * (x ** (y - 1))
        
        safe_x = np.where(x > 0, x, 1.0)
        grad_y = grad_output * (x ** y) * np.log(safe_x)
        
        return unbroadcast(grad_x, x.shape), unbroadcast(grad_y, y.shape)
    

# =============================================================================
# Comparison Operations (Non-differentiable)
# =============================================================================

class Eq(Function):
    def forward(self, x, y):
        self.ctx.save_for_backward(x.shape, y.shape)
        return x == y

    def backward(self, grad_output):
        x_shape, y_shape = self.ctx.get_saved()
        # المقارنة لا تنقل الغرادينت (المشتق صفر)
        return np.zeros(x_shape, dtype=np.float32), np.zeros(y_shape, dtype=np.float32)

class Ne(Function):
    def forward(self, x, y):
        self.ctx.save_for_backward(x.shape, y.shape)
        return x != y

    def backward(self, grad_output):
        x_shape, y_shape = self.ctx.get_saved()
        return np.zeros(x_shape, dtype=np.float32), np.zeros(y_shape, dtype=np.float32)

class Gt(Function):
    def forward(self, x, y):
        self.ctx.save_for_backward(x.shape, y.shape)
        return x > y

    def backward(self, grad_output):
        x_shape, y_shape = self.ctx.get_saved()
        return np.zeros(x_shape, dtype=np.float32), np.zeros(y_shape, dtype=np.float32)

class Lt(Function):
    def forward(self, x, y):
        self.ctx.save_for_backward(x.shape, y.shape)
        return x < y

    def backward(self, grad_output):
        x_shape, y_shape = self.ctx.get_saved()
        return np.zeros(x_shape, dtype=np.float32), np.zeros(y_shape, dtype=np.float32)

class Ge(Function):
    def forward(self, x, y):
        self.ctx.save_for_backward(x.shape, y.shape)
        return x >= y

    def backward(self, grad_output):
        x_shape, y_shape = self.ctx.get_saved()
        return np.zeros(x_shape, dtype=np.float32), np.zeros(y_shape, dtype=np.float32)

class Le(Function):
    def forward(self, x, y):
        self.ctx.save_for_backward(x.shape, y.shape)
        return x <= y

    def backward(self, grad_output):
        x_shape, y_shape = self.ctx.get_saved()
        return np.zeros(x_shape, dtype=np.float32), np.zeros(y_shape, dtype=np.float32)

# =============================================================================
# Helper Functions
# =============================================================================

def eq(x, y): return Eq.apply(x, y)
def ne(x, y): return Ne.apply(x, y)
def gt(x, y): return Gt.apply(x, y)
def lt(x, y): return Lt.apply(x, y)
def ge(x, y): return Ge.apply(x, y)
def le(x, y): return Le.apply(x, y)
# =============================================================================
def add(x, y): return Add.apply(x, y)
def sub(x, y): return Sub.apply(x, y)
def mul(x, y): return Mul.apply(x, y)
def div(x, y): return Div.apply(x, y)
def power(x, y): return Pow.apply(x, y)