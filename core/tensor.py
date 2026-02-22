# Tensor
# ├── data            
# ├── dtype            
# ├── shape            
# ├── requires_grad    
# ├── grad             
# ├── grad_fn          
# ├── is_leaf          
# └── _backward()      


import numpy as np
from .autograd import AutogradEngine

class Tensor:
    def __init__(self, data, requires_grad=False):
        """
        Tensor Wrapper around NumPy array.
        """
        if isinstance(data, Tensor):
            data = data.data
        
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad

        self.grad = None           
        self.grad_fn = None       
        self.is_leaf = True        

    # =================================================================
    # 1. Properties & Meta Data
    # =================================================================
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        """Transposes the tensor."""
        from ..ops.matrix import transpose
        return transpose(self)

    # =================================================================
    # 2. Gradient & Autograd Methods
    # =================================================================
    def backward(self, grad=None):
        """
        Initiates backpropagation from this tensor.
        """
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar tensor")
            grad = np.ones_like(self.data)
          
        engine = AutogradEngine()  
        engine.backward(self, grad)

    def zero_grad(self):
        """Resets the gradient to None."""
        self.grad = None

    def detach(self):
        """
        Returns a new Tensor, detached from the current graph.
        The result will never require gradient.
        Useful for Optimizers and Data Loading.
        """
        return Tensor(self.data, requires_grad=False)

    # =================================================================
    # 3. Python Magic Methods (len, print, item)
    # =================================================================
    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        from ..ops.array import getitem
        return getitem(self, index)

    # =================================================================
    # 4. Arithmetic Operations
    # =================================================================
    def _ensure_tensor(self, t):
        """Helper to convert scalars to Tensors automatically."""
        if isinstance(t, Tensor):
            return t
        return Tensor(t, requires_grad=False)

    def __add__(self, other):
        other = self._ensure_tensor(other)
        from ..ops.arithmetic import add
        return add(self, other)

    def __radd__(self, other): 
        other = self._ensure_tensor(other)
        from ..ops.arithmetic import add
        return add(other, self)

    def __sub__(self, other):
        other = self._ensure_tensor(other)
        from ..ops.arithmetic import sub
        return sub(self, other)

    def __rsub__(self, other): 
        other = self._ensure_tensor(other)
        from ..ops.arithmetic import sub
        return sub(other, self)

    def __mul__(self, other):
        other = self._ensure_tensor(other)
        from ..ops.arithmetic import mul
        return mul(self, other)

    def __rmul__(self, other): 
        other = self._ensure_tensor(other)
        from ..ops.arithmetic import mul
        return mul(other, self)

    def __truediv__(self, other):
        other = self._ensure_tensor(other)
        from ..ops.arithmetic import div
        return div(self, other)

    def __rtruediv__(self, other): 
        other = self._ensure_tensor(other)
        from ..ops.arithmetic import div
        return div(other, self)

    def __pow__(self, other):
        other = self._ensure_tensor(other)
        from ..ops.arithmetic import power
        return power(self, other)

    def __matmul__(self, other):
        other = self._ensure_tensor(other)
        from ..ops.matrix import matmul
        return matmul(self, other)

    # =================================================================
    # 5. Neural Network Operations
    # =================================================================
    def relu(self):
        from ..ops.activation import relu
        return relu(self)

    def sigmoid(self):
        from ..ops.activation import sigmoid
        return sigmoid(self)

    def tanh(self):
        from ..ops.activation import tanh
        return tanh(self)    

    def sum(self, axis=None, keepdims=False):
        from ..ops.reduction import sum as sum_op 
        return sum_op(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        from ..ops.reduction import mean as mean_op
        return mean_op(self, axis=axis, keepdims=keepdims)
    














# ===================================================================== (v_2.0 )
    def reshape(self, *new_shape):
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            new_shape = new_shape[0]
            
        from ..ops.array import reshape
        return reshape(self, new_shape)

    def permute(self, *dims):
        from ..ops.array import permute
        return permute(self, dims)
    def flatten(self):
        from ..ops.array import flatten
        return flatten(self)
    

# أضف هذا داخل class Tensor في core/tensor.py

    # =================================================================
    # 6. Comparison Operators
    # =================================================================
    def __eq__(self, other):
        other = self._ensure_tensor(other)
        from ..ops.arithmetic import eq
        return eq(self, other)

    def __ne__(self, other):
        other = self._ensure_tensor(other)
        from ..ops.arithmetic import ne
        return ne(self, other)

    def __lt__(self, other):
        other = self._ensure_tensor(other)
        from ..ops.arithmetic import lt
        return lt(self, other)

    def __gt__(self, other):
        other = self._ensure_tensor(other)
        from ..ops.arithmetic import gt
        return gt(self, other)

    def __le__(self, other):
        other = self._ensure_tensor(other)
        from ..ops.arithmetic import le
        return le(self, other)

    def __ge__(self, other):
        other = self._ensure_tensor(other)
        from ..ops.arithmetic import ge
        return ge(self, other)