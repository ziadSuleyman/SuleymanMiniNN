# Parameter(Tensor)
# ├── data            # وراثة من Tensor
# ├── requires_grad   # دائمًا True
# ├── grad            # وراثة من Tensor
# └── is_leaf         # True، لأنه يُنشأ مباشرة من المستخدم/Layer

from ..core.tensor import Tensor

class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)
        self.is_leaf = True
