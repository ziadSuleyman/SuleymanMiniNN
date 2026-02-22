from ..nn.activations import *
from ..nn.module import *
from ..nn.sequential import *
from ..nn.parameter import *
from ..nn.linear import *
from ..nn.conv import *
from ..nn.BatchNorm1d import *
from ..nn.Dropout import *
from .flatten import Flatten  # <--- إضافة الاستيراد من الملف الجديد
# أضف هذا السطر في SuleymanMiniNN/nn/__init__.py
from .pool import MaxPool1d
