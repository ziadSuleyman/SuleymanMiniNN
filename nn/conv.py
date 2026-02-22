import numpy as np
from ..core.tensor import Tensor
from ..nn.module import Module
from ..nn.parameter import Parameter
from ..ops.array import cat

class Conv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        # تهيئة الأوزان
        # الشكل هنا: (kernel_size, in_channels, out_channels)
        # هذا الشكل يسهل عملية الضرب داخل الـ Loop (x @ w)
        k = 1 / np.sqrt(in_channels * kernel_size)
        weight_data = np.random.uniform(-k, k, (kernel_size, in_channels, out_channels))
        self.weight = Parameter(weight_data)
        
        if bias:
            bias_data = np.zeros(out_channels)
            self.bias = Parameter(bias_data)
        else:
            self.bias = None

    def forward(self, x):
        # x shape: (Batch, Seq_Len, In_Channels)
        N, L, C = x.shape
        
        # 1. تطبيق الـ Padding
        # بما أننا نستخدم Tensor Ops، يجب أن ننشئ تينسور أصفار ونقوم بدمجه
        if self.padding > 0:
            # ننشئ مصفوفة أصفار (Batch, Padding, In_Channels)
            zeros = Tensor(np.zeros((N, self.padding, C), dtype=np.float32), requires_grad=False)
            # دمج: [zeros, x, zeros] على المحور 1 (محور الزمن)
            x_padded = cat([zeros, x, zeros], axis=1)
        else:
            x_padded = x

        # حساب طول التسلسل الجديد بعد الحشو
        # L_in + 2*padding
        L_padded = x_padded.shape[1]
        
        # طول المخرج المتوقع: L_out = L_padded - kernel_size + 1
        L_out = L_padded - self.kernel_size + 1
        
        # 2. عملية الـ Convolution (Sliding Window via Loop)
        # سنقوم بتجميع النتائج في هذا المتغير
        # ملاحظة: نبدأ بـ 0 (كـ int) وسيقوم أول جمع بتحويله لتينسور تلقائياً
        output = 0 
        
        for k in range(self.kernel_size):
            # أ) تحديد النافذة (Sliding Window)
            # نأخذ شريحة من x_padded تبدأ من k وتنتهي عند k + L_out
            # Shape: (Batch, L_out, In_Channels)
            x_slice = x_padded[:, k : k + L_out, :]
            
            # ب) جلب الوزن الخاص بهذا الموقع من الكيرنل
            # Shape: (In_Channels, Out_Channels)
            # نستخدم self.weight[k] وهو عملية slicing مدعومة في Tensor
            w_k = self.weight[k]
            
            # ج) العملية الحسابية (Tensor Ops)
            # (Batch, L_out, In) @ (In, Out) -> (Batch, L_out, Out)
            term = x_slice @ w_k
            
            # د) التجميع
            output = output + term

        # 3. إضافة الـ Bias
        if self.bias is not None:
            # Broadcasting سيقوم بتوزيع الـ bias على كل الخطوات الزمنية
            output = output + self.bias
            
        return output