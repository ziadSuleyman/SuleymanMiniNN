import numpy as np
from ..core.function import Function
from ..nn.module import Module
class MaxPool2dOp(Function):
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        N, C, H, W = x.shape
        KH, KW = self.kernel_size, self.kernel_size
        s = self.stride
        
        H_out = (H - KH) // s + 1
        W_out = (W - KW) // s + 1
        
        H_used = (H_out - 1) * s + KH
        W_used = (W_out - 1) * s + KW
        
        x_cropped = x[:, :, :H_used, :W_used] # <--- إضافة مهمة
        
        if KH == s and KW == s:
             x_reshaped = x_cropped.reshape(N, C, H_out, s, W_out, s)
             x_reshaped = x_reshaped.transpose(0, 1, 2, 4, 3, 5)
             out = x_reshaped.max(axis=(4, 5))
             self.ctx.save_for_backward(x.shape, x_reshaped, out, (KH, s, True)) # True = simple mode
        else:
             x_reshaped = x_cropped.reshape(N, C, H_out, s, W_out, s)
             x_reshaped = x_reshaped.transpose(0, 1, 2, 4, 3, 5)
             out = x_reshaped.max(axis=(4, 5))
             self.ctx.save_for_backward(x.shape, x_reshaped, out, (KH, s, True))

        return out

    def backward(self, grad_output):
        x_shape, x_reshaped, out, params = self.ctx.get_saved()
        KH, s, simple_mode = params
        
        grad_output_expanded = grad_output[:, :, :, :, None, None]
        out_expanded = out[:, :, :, :, None, None]
        
        mask = (x_reshaped == out_expanded)
        dx_reshaped = mask * grad_output_expanded
        
        dx_reshaped = dx_reshaped.transpose(0, 1, 2, 4, 3, 5)
        
        dx = np.zeros(x_shape, dtype=np.float32)
        
        H_used = dx_reshaped.shape[2] * s
        W_used = dx_reshaped.shape[4] * s
        
        dx_content = dx_reshaped.reshape(x_shape[0], x_shape[1], H_used, W_used)
        dx[:, :, :H_used, :W_used] = dx_content
        
        return dx
    


class AvgPool2dOp(Function):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
    
    def forward(self, x):
        N, C, H, W = x.shape
        KH, KW = self.kernel_size, self.kernel_size
        S = self.stride
        
        H_out = (H - KH) // S + 1
        W_out = (W - KW) // S + 1
        
        # تنفيذ مبسط (Naive)
        out = np.zeros((N, C, H_out, W_out), dtype=np.float32)
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * S
                w_start = j * S
                h_end = h_start + KH
                w_end = w_start + KW
                out[:, :, i, j] = np.mean(x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))
        
        self.ctx.save_for_backward(x.shape, (H_out, W_out, KH, KW, S))
        return out

    def backward(self, grad_output):
        x_shape, params = self.ctx.get_saved()
        H_out, W_out, KH, KW, S = params
        dx = np.zeros(x_shape, dtype=np.float32)
        
        # توزيع الغرادينت (Distribute Gradient)
        # بما أن المتوسط هو sum/n، فالمشتق هو grad/n
        n = KH * KW
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * S
                w_start = j * S
                grad_val = grad_output[:, :, i, j][:, :, None, None] / n
                dx[:, :, h_start:h_start+KH, w_start:w_start+KW] += grad_val
        return dx

# في nn/layers.py (التغليف)
class AvgPool2d(Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self, x):
        return AvgPool2dOp(self.kernel_size, self.stride).apply(x)