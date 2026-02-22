# Autograd Engine
# └── backward(tensor, grad=None)       
#     ├── (Internal) build_topo()       
#     └── (Loop) Process Gradients      
import numpy as np
class AutogradEngine:
    def __init__(self):
        pass

    def backward(self, tensor, grad=None):
        if not tensor.requires_grad:
            return

        if grad is None:
            if tensor.data.size == 1:
                grad = np.ones_like(tensor.data)
            else:
                grad = np.ones_like(tensor.data)

        if tensor.grad is None:
            tensor.grad = grad
        else:
            tensor.grad += grad

        
        topo = []
        visited = set()

        def build_topo(t):
            if id(t) not in visited:
                visited.add(id(t))
                if t.grad_fn is not None:
                    for parent in t.grad_fn.parents:
                        build_topo(parent)
                topo.append(t)

        build_topo(tensor)

        for t in reversed(topo):
            if t.grad_fn is not None:
                grads = t.grad_fn.backward(t.grad)
                
                if not isinstance(grads, (list, tuple)):
                    grads = [grads]
                
                for parent, parent_grad in zip(t.grad_fn.parents, grads):
                    if parent.requires_grad:
                        if parent.grad is None:
                            parent.grad = parent_grad
                        else:
                            parent.grad += parent_grad
