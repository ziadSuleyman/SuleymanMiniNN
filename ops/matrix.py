# matrix.py
# │
# ├── MatMul           # z = x @ y
# │   ├── forward(x, y)
# │   └── backward(grad_out)
# │
# └── Transpose        # z = x.T
#     ├── forward(x)
#     └── backward(grad_out)
import numpy as np
from SuleymanMiniNN.core.function import Function

# def unbroadcast(grad, original_shape):
#     while grad.ndim > len(original_shape):
#         grad = grad.sum(axis=0)
#     for i, dim in enumerate(original_shape):
#         if dim == 1:
#             grad = grad.sum(axis=i, keepdims=True)
#     return grad

# class MatMul(Function):
#     def forward(self, x, y):
#         self.ctx.save_for_backward(x, y)
#         return x @ y

#     def backward(self, grad_output):
#         x, y = self.ctx.get_saved()
        
#         grad_x = grad_output @ y.T
#         grad_y = x.T @ grad_output

#         return unbroadcast(grad_x, x.shape), unbroadcast(grad_y, y.shape)

# class Transpose(Function):
#     def forward(self, x):
#         self.ctx.save_for_backward(None)
#         return x.T

#     def backward(self, grad_output):
#         return [grad_output.T]

# def matmul(x, y): return MatMul.apply(x, y)
# def transpose(x): return Transpose.apply(x)


def unbroadcast(grad, original_shape):
    """
    تقليص الغرادينت ليتطابق مع الأبعاد الأصلية (للتعامل مع Broadcasting).
    """
    # 1. جمع الأبعاد الزائدة في البداية (مثل Batch dim عند حساب مشتق الوزن)
    while grad.ndim > len(original_shape):
        grad = grad.sum(axis=0)
    
    # 2. التعامل مع الأبعاد التي قيمتها 1 في الأصل (Keepdims)
    for i, dim in enumerate(original_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
            
    return grad

class MatMul(Function):
    def forward(self, x, y):
        self.ctx.save_for_backward(x, y)
        return x @ y

    def backward(self, grad_output):
        x, y = self.ctx.get_saved()
        
        # ---------------------------------------------------------
        # تصحيح الخطأ: استخدام swapaxes بدلاً من T
        # T في Numpy تقلب كل الأبعاد، بينما في ضرب المصفوفات
        # نحتاج فقط لقلب آخر بعدين (Rows <-> Cols)
        # ---------------------------------------------------------
        
        # 1. حساب مشتق x:  dL/dx = dL/dz @ y.T
        if y.ndim >= 2:
            y_T = y.swapaxes(-1, -2)
        else:
            y_T = y 
            
        grad_x = grad_output @ y_T

        # 2. حساب مشتق y:  dL/dy = x.T @ dL/dz
        if x.ndim >= 2:
            x_T = x.swapaxes(-1, -2)
        else:
            x_T = x
            
        grad_y = x_T @ grad_output

        return unbroadcast(grad_x, x.shape), unbroadcast(grad_y, y.shape)

class Transpose(Function):
    def forward(self, x):
        self.ctx.save_for_backward(x.shape)
        return x.T

    def backward(self, grad_output):
        # مشتق الـ Transpose هو Transpose آخر
        return grad_output.T

def matmul(x, y): return MatMul.apply(x, y)
def transpose(x): return Transpose.apply(x)