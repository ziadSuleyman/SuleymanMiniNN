# Context
# ├── saved_values     
# ├── save_for_backward(*args)  
# └── get_saved()     
_grad_enabled = True  

def is_grad_enabled():
    return _grad_enabled

def set_grad_enabled(mode: bool):
    global _grad_enabled
    _grad_enabled = mode

class no_grad:
    def __init__(self):
        self.prev = True
        
    def __enter__(self):
        global _grad_enabled
        self.prev = _grad_enabled  
        _grad_enabled = False      
        
    def __exit__(self, exc_type, exc_value, traceback):
        global _grad_enabled
        _grad_enabled = self.prev  

import numpy as np

_debug_mode = False

def set_debug_mode(mode: bool):
    global _debug_mode
    _debug_mode = mode
    status = "ON" if mode else "OFF"
    print(f"\n[Context] Debug Mode is now {status}")

def _fmt_arg(arg):
    if hasattr(arg, 'shape'):
        return f"Array/Tensor(shape={arg.shape}, dtype={arg.dtype})"
    if isinstance(arg, (int, float)):
        return f"Scalar({arg})"
    return str(type(arg))

_grad_enabled = True

def is_grad_enabled():
    return _grad_enabled

def set_grad_enabled(mode: bool):
    global _grad_enabled
    _grad_enabled = mode

class no_grad:
    def __init__(self):
        self.prev = True
    def __enter__(self):
        global _grad_enabled
        self.prev = _grad_enabled
        _grad_enabled = False
    def __exit__(self, exc_type, exc_value, traceback):
        global _grad_enabled
        _grad_enabled = self.prev

class Context:
    def __init__(self):
        self.saved_values = ()
        self.id = hex(id(self))[-6:] 

    def save_for_backward(self, *args):
        self.saved_values = args
        
        if _debug_mode:
            print(f"\033[92m[Forward -> Ctx:{self.id}] Saving {len(args)} items for backward:\033[0m")
            for i, arg in enumerate(args):
                print(f"    Item {i}: {_fmt_arg(arg)}")

    def get_saved(self):
        if _debug_mode:
            print(f"\033[94m[Backward <- Ctx:{self.id}] Retrieving {len(self.saved_values)} items...\033[0m")
            
        return self.saved_values