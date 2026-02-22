# Function Node
# ├── inputs         
# ├── output          
# ├── ctx             
# ├── forward(*inputs)
# ├── backward(grad)   
# ├── save_for_backward(*args) 
# └── apply(*inputs)  
from .context import Context
from .tensor import Tensor


class Function:
    @classmethod
    def apply(cls, *inputs, **kwargs):
        ctx = Context()
        fn = cls(**kwargs) 

        # fn = cls()
        fn.ctx = ctx
        fn.parents = inputs

        input_data = [t.data for t in inputs]

        result_data = fn.forward(*input_data)

        requires_grad = any(t.requires_grad for t in inputs)
        out = Tensor(result_data, requires_grad=requires_grad)

        if requires_grad:
            out.grad_fn = fn

        return out
    def forward(self, *args):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError