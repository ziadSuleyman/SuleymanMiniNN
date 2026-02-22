# import numpy as np
# from ..core.tensor import Tensor
# from ..core.function import Function 

# class GetItem(Function):
#     def forward(self, ctx, tensor, index):
#         ctx.save_for_backward(tensor.shape, index)
#         return tensor.data[index]

#     def backward(self, ctx, grad):
#         original_shape, index = ctx.saved_tensors
#         full_grad = np.zeros(original_shape, dtype=np.float32)
        
#         full_grad[index] = grad
        
#         return full_grad, None

# def getitem(tensor, index):
#     return Tensor(GetItem.apply(tensor, index), requires_grad=tensor.requires_grad)


# class Reshape(Function):
#     def __init__(self, shape):
#         self.target_shape = shape
#     def forward(self, x):
#         self.ctx.save_for_backward(x.shape)
#         return x.reshape(self.target_shape)
#     def backward(self, grad_output):
#         original_shape, = self.ctx.get_saved()
#         return grad_output.reshape(original_shape)

# def reshape(tensor, shape):
#     return Reshape.apply(tensor, shape=shape)

# class Cat(Function):
#     def __init__(self, axis=0):
#         self.axis = axis

#     def forward(self, *inputs):
#         self.ctx.save_for_backward([x.shape for x in inputs], self.axis)
#         arrays = [x.data for x in inputs]
#         return np.concatenate(arrays, axis=self.axis)

#     def backward(self, grad_output):
#         shapes, axis = self.ctx.get_saved()
#         grads = []
#         current_idx = 0
        
#         for shape in shapes:
#             slc = [slice(None)] * grad_output.ndim
#             dim_len = shape[axis]
#             slc[axis] = slice(current_idx, current_idx + dim_len)
            
#             grads.append(grad_output[tuple(slc)])
#             current_idx += dim_len
            
#         return grads

# def cat(tensors, axis=0):
#     return Cat(axis=axis).apply(*tensors)


import numpy as np
from ..core.function import Function

# =============================================================================
# 1. GetItem (Slicing)
# =============================================================================
class GetItem(Function):
    def __init__(self, index):
        self.index = index

    def forward(self, x):
        # x here is numpy array
        self.ctx.save_for_backward(x.shape)
        return x[self.index]

    def backward(self, grad_output):
        original_shape, = self.ctx.get_saved()
        
        # إنشاء مصفوفة أصفار بنفس حجم المدخل الأصلي
        grad = np.zeros(original_shape, dtype=grad_output.dtype)
        
        # وضع الغرادينت في المكان الصحيح
        # ملاحظة: هذا يعمل مع الـ Slicing العادي.
        # للـ Advanced Indexing المعقد، قد تحتاج np.add.at
        grad[self.index] = grad_output
        
        return grad

def getitem(tensor, index):
    # apply تعيد Tensor، لا حاجة لتغليفها مرة أخرى
    return GetItem.apply(tensor, index=index)


# =============================================================================
# 2. Reshape
# =============================================================================
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.ctx.save_for_backward(x.shape)
        return x.reshape(self.shape)

    def backward(self, grad_output):
        original_shape, = self.ctx.get_saved()
        return grad_output.reshape(original_shape)

def reshape(tensor, shape):
    return Reshape.apply(tensor, shape=shape)


# =============================================================================
# 3. Permute (Transpose axes)
# =============================================================================
class Permute(Function):
    def __init__(self, dims):
        self.dims = dims

    def forward(self, x):
        self.ctx.save_for_backward(self.dims)
        return x.transpose(self.dims)

    def backward(self, grad_output):
        dims, = self.ctx.get_saved()
        # لإيجاد الترتيب العكسي: np.argsort يعيد ترتيب الفهارس للأصل
        inverse_dims = np.argsort(dims)
        return grad_output.transpose(inverse_dims)

def permute(tensor, dims):
    return Permute.apply(tensor, dims=dims)


# =============================================================================
# 4. Flatten (Convenience wrapper around Reshape)
# =============================================================================
class Flatten(Function):
    def forward(self, x):
        self.ctx.save_for_backward(x.shape)
        # تحويل لـ (Batch_Size, -1)
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output):
        original_shape, = self.ctx.get_saved()
        return grad_output.reshape(original_shape)

def flatten(tensor):
    return Flatten.apply(tensor)


# =============================================================================
# 5. Concatenate (Cat)
# =============================================================================
class Cat(Function):
    def __init__(self, axis=0):
        self.axis = axis

    def forward(self, *inputs):
        # inputs: قائمة من مصفوفات numpy
        shapes = [x.shape for x in inputs]
        self.ctx.save_for_backward(shapes, self.axis)
        return np.concatenate(inputs, axis=self.axis)

    def backward(self, grad_output):
        shapes, axis = self.ctx.get_saved()
        grads = []
        current_idx = 0
        
        # تقسيم الغرادينت (Slicing) لإعادته لكل تنسور مساهم
        for shape in shapes:
            # إنشاء Slice لكل الأبعاد
            slc = [slice(None)] * grad_output.ndim
            
            # تحديد مجال القص في المحور المحدد
            dim_len = shape[axis]
            slc[axis] = slice(current_idx, current_idx + dim_len)
            
            # قص الغرادينت
            grads.append(grad_output[tuple(slc)])
            
            current_idx += dim_len
            
        return tuple(grads) # يجب إرجاع tuple لأن هناك عدة مدخلات

def cat(tensors, axis=0):
    # apply تستقبل *args، لذا يجب فك القائمة
    return Cat.apply(*tensors, axis=axis)