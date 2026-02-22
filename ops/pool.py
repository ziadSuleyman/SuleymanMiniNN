import numpy as np
from ..core.function import Function

class MaxPool1d(Function):
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x):
        # x shape: (N, L, C) -> (Batch, Length, Channels)
        N, L, C = x.shape
        k = self.kernel_size
        s = self.stride
        
        # حساب الطول الجديد
        L_out = (L - k) // s + 1
        
        # مصفوفة للمخرجات
        out = np.zeros((N, L_out, C), dtype=x.dtype)
        
        # لحفظ المواقع من أجل backward، سنحفظ المدخلات والمعاملات
        # الطريقة الأبسط هي إعادة حساب القناع (Mask) في backward لتوفير الذاكرة
        self.ctx.save_for_backward(x, (k, s, L_out))
        
        for i in range(L_out):
            start = i * s
            end = start + k
            # أخذ النافذة على طول الزمن
            window = x[:, start:end, :]
            # حساب الماكس على المحور 1 (محور الزمن داخل النافذة)
            out[:, i, :] = np.max(window, axis=1)
            
        return out

    def backward(self, grad_output):
        x, params = self.ctx.get_saved()
        k, s, L_out = params
        
        grad_x = np.zeros_like(x)
        
        for i in range(L_out):
            start = i * s
            end = start + k
            
            # استرجاع النافذة الأصلية
            window = x[:, start:end, :] # Shape: (N, k, C)
            
            # قيمة الماكس (يجب الحفاظ على الأبعاد للمقارنة)
            max_val = np.max(window, axis=1, keepdims=True) # Shape: (N, 1, C)
            
            # إنشاء القناع: 1 مكان الماكس، 0 في غيره
            mask = (window == max_val)
            
            # الغرادينت القادم لهذه الخطوة (N, C) -> نوسعه ليصبح (N, 1, C)
            grad_step = grad_output[:, i, :][:, None, :]
            
            # توزيع الغرادينت فقط على الأماكن التي كانت Max
            grad_x[:, start:end, :] += mask * grad_step
            
        return grad_x

def max_pool1d(x, kernel_size, stride=None):
    return MaxPool1d(kernel_size, stride).apply(x)