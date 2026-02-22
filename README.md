توثيق مكتبتي الشاملتوثيق شامل لكتبتي SuleymanMiniNN

اعتمدنا على مكتبة pytorch للفهم و مقارنتها بمكتبتنا 

هذا التوثيق مصمم ليشرح "فلسفة المكتبة"، "كيف تعمل تحت الغطاء"، وكيفية استخدامها

مكتبة تعلم عميق (Deep Learning) ديناميكية مبنية من الصفر باستخدام Python و NumPy.

1. نظرة عامة (Overview)

SuleymanMiniNN ليست مجرد غلاف  حول مكتبات أخرى، بل هي محرك كامل للاشتقاق التلقائي (Autograd Engine) تم بناؤه من الأساس. تهدف المكتبة إلى محاكاة طريقة عمل PyTorch، مما يجعلها أداة تعليمية قوية لفهم ما يحدث خلف كواليس الذكاء الاصطناعي.

الميزات الرئيسية:

غراف حسابي ديناميكي (Dynamic Computational Graph): يتم بناء الغراف أثناء تنفيذ الكود (Define-by-Run).

اشتقاق تلقائي عكسي (Reverse-Mode Autodiff): حساب التدرجات بدقة عالية.

واجهة برمجية مألوفة: تصميم الكلاسات (Tensor, Module, Optimizer) يطابق معايير الصناعة.

2. الهيكلية الداخلية: كيف تعمل؟ (Core Architecture)

تعتمد المكتبة على ثلاثة أعمدة رئيسية:

أ. التنسور (The Tensor) 

هو الوحدة الأساسية للبيانات.

البيانات: يخزن البيانات كمصفوفة numpy.ndarray.

الذاكرة (History): يحتفظ بمرجع للعملية التي أنشأته (grad_fn)، مما يسمح بتتبع تسلسل العمليات.

التدرج (Gradient): يخزن قيمة المشتق (.grad) بعد عملية الـ Backward.

ب. محرك (The Autograd Engine) 

هذا هو "عقل" المكتبة. عند استدعاء .backward():

بناء الطوبولوجيا (Topological Sort): يقوم المحرك بترتيب العقد (Nodes) في الغراف لضمان حساب مشتقات المخرجات قبل المدخلات.

قاعدة السلسلة : يمرر التدرجات من المخرج (Loss) عائداً إلى الأوزان، مستخدماً مشتقات الدوال المحفوظة.

ج. السياق 

أثناء المرور الأمامي (Forward)، تحتاج بعض الدوال لحفظ قيم معينة لاستخدامها في الاشتقاق (مثل input في دالة الضرب، أو indices في MaxPooling). كلاس Context هو الذاكرة المؤقتة التي تحتفظ بهذه القيم حتى يتم استدعاؤها في الـ Backward.

3. المكونات التفصيلية (Components)
1. العمليات الحسابية (Function Nodes)

كل عملية رياضية (جمع، ضرب، Relu، Sigmoid) هي كلاس يرث من Function.

Forward: ينفذ العملية بـ NumPy ويحفظ المتغيرات في ctx.

Backward: يستلم التدرج القادم (grad_output) ويحسب التدرج المحلي ويضربه فيه.

2. طبقات الشبكة (Neural Modules)

تتبع نمط Module، حيث تحتوي على:

Parameters: أوزان قابلة للتدريب (Layer.weight).

Training Mode: دعم لطبقات تتصرف بشكل مختلف أثناء التدريب (مثل Dropout و BatchNorm).

3. المحسّنات (Optimizers)

SGD: يدعم الـ Momentum لتسريع التقارب.

يقوم بتحديث الأوزان بناءً على التدرجات المحسوبة: 

4. دورة حياة التدريب 

Forward Pass:

تدخل البيانات x في الموديل.

يتم إنشاء Function Nodes وربطها ببعضها لبناء الغراف.

يتم حساب الخسارة (Loss).

Backward Pass:

تستدعي loss.backward().

AutogradEngine، يرتب الغراف، ويحسب التدرجات لكل Tensor يحمل requires_grad=True.

Optimization Step:

تستدعي optimizer.step().

يأخذ المحسن التدرجات المخزنة في .grad ويحدث البيانات .لأوزان.

تستدعي optimizer.zero_grad() لتنظيف الغراف للخطوة التالية.


تم التطوير بواسطة: زياد (Suleyman) 


هذا مثال كامل لمكتبتنا


============================================================================================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.gridspec as gridspec
from SuleymanMiniNN.core.tensor import Tensor
from SuleymanMiniNN.nn import Sequential, Linear, ReLU
from SuleymanMiniNN.optim.adam import Adam
from SuleymanMiniNN.loss.cross_entropy import CrossEntropyLoss
from SuleymanMiniNN.utils.data_prepare import prepare_mnist_data, DataLoader
from SuleymanMiniNN.training.trainer import Trainer
from SuleymanMiniNN.nn.BatchNorm1d import BatchNorm1d
from SuleymanMiniNN.nn.Dropout import Dropout
from SuleymanMiniNN.training.tuner import GridSearchTuner

# ============================================================
# 1. init data
# ============================================================
print("Loading Data...")
(x_train, y_train), (x_val, y_val), (x_test, y_test) = prepare_mnist_data()

BATCH_SIZE = 256

train_data = DataLoader(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
val_data   = DataLoader(x_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
test_data  = DataLoader(x_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 2. build model
# ============================================================
def build_model(params):
    hidden_1  = params['hidden_size']
    hidden_2  = params['hidden_size'] // 2 
    drop_rate = params['dropout']
    lr        = params['lr']
    model     = Sequential(
                Linear(784, hidden_1),
                BatchNorm1d(hidden_1),
                ReLU(),
                Dropout(drop_rate),
                #=======2=========#
                Linear(hidden_1, hidden_2),
                BatchNorm1d(hidden_2),
                ReLU(),
                Dropout(drop_rate),
                Linear(hidden_2, 10)
    )
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    
    return model, optimizer, criterion

# ============================================================
# 3. train parameters
# ============================================================
best_fixed_params = {
    'lr'          : [0.01,0.005],           
    'hidden_size' : [256,128],    
    'dropout'     : [0.3,0.4]         
}

print("\nInitializing Training...")
tuner = GridSearchTuner(build_model, best_fixed_params)

best_params = tuner.search(
    train_data, 
    val_data, 
    batch_size=64,       
    epochs_per_trial=1 
)
print("\nStarting Final Extended Training...")
final_model, final_opt, final_crit = build_model(best_params)
final_trainer = Trainer(final_model, final_opt, final_crit)


history = final_trainer.fit(
    train_data, 
    validation_data=val_data, 
    epochs=20
)

loss, acc = final_trainer.evaluate(loader=test_data)
print(f"\nFinal Test Accuracy: {acc*100:.2f}%")

def print_detailed_tree(node, indent="", is_last=True, seen=None):
    if seen is None: seen = set()
    marker = "└── " if is_last else "├── "
    connector = "    " if is_last else "│   "    
    name = "Leaf"
    color = "\033[97m" 
    if not node.is_leaf:
        name = type(node.grad_fn).__name__ if node.grad_fn else "Op"
        color = "\033[94m"
    else:
        color = "\033[92m"

    grad_info = "\033[90mNone\033[0m"
    if node.grad is not None:
        g_mean = np.mean(np.abs(node.grad))
        g_max = np.max(np.abs(node.grad))
        grad_info = f"\033[91mGrad(μ={g_mean:.1e})\033[0m"

    print(f"{indent}{marker}{color}[{name}]\033[0m Shape={node.shape} | {grad_info}")
    
    if id(node) in seen:
        print(f"{indent}{connector}\033[90m... (Shared Node) ...\033[0m")
        return
    seen.add(id(node))


    if node.grad_fn and hasattr(node.grad_fn, 'parents'):
        parents = node.grad_fn.parents
        for i, parent in enumerate(parents):
            if hasattr(parent, 'shape'):
                print_detailed_tree(parent, indent + connector, (i == len(parents) - 1), seen)


print("\n🔍 --- STEP 1: FORWARD PASS ONLY ---")
x_sample, y_sample = next(iter(test_data))
x_sample = Tensor(x_sample.data[:1])
y_sample = Tensor(y_sample.data[:1])


final_model.zero_grad()

out = final_model(x_sample)
loss = final_crit(out, y_sample)

print_detailed_tree(loss)


print("\n--- STEP 2: EXECUTING BACKWARD PASS ---")
loss.backward()
print("... Backward pass finished.")


print("\n--- STEP 3: INSPECTING GRADIENTS ---")
print_detailed_tree(loss)






def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def visualize_dashboard(model, history, x_test, y_test):
    sns.set_theme(style="whitegrid")
    
    fig = plt.figure(figsize=(16, 20), constrained_layout=True)
    fig.suptitle('SuleymanMiniNN Training Dashboard', fontsize=24, fontweight='bold', y=1.02)

    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1.2, 0.5, 0.5])
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_acc = fig.add_subplot(gs[0, 1])

    s_train_loss = smooth_curve(history['train_loss'])
    s_val_loss = smooth_curve(history['val_loss'])
    s_train_acc = smooth_curve(history['train_acc'])
    s_val_acc = smooth_curve(history['val_acc'])

    ax_loss.plot(history['train_loss'], alpha=0.3, color='#3498db')
    ax_loss.plot(history['val_loss'], alpha=0.3, color='#e74c3c')
    ax_loss.plot(s_train_loss, label='Train (Smooth)', linewidth=2, color='#3498db')
    ax_loss.plot(s_val_loss, label='Val (Smooth)', linewidth=2, linestyle='--', color='#e74c3c')
    ax_loss.set_title('Loss History', fontsize=14, fontweight='bold')
    ax_loss.legend()
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')

    ax_acc.plot(history['train_acc'], alpha=0.3, color='#2ecc71')
    ax_acc.plot(history['val_acc'], alpha=0.3, color='#f39c12')
    ax_acc.plot(s_train_acc, label='Train (Smooth)', linewidth=2, color='#2ecc71')
    ax_acc.plot(s_val_acc, label='Val (Smooth)', linewidth=2, linestyle='--', color='#f39c12')
    ax_acc.set_title('Accuracy History', fontsize=14, fontweight='bold')
    ax_acc.legend()
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Accuracy')
    model.eval()
    x_tensor = Tensor(x_test[:2000])
    logits = model(x_tensor)
    if isinstance(logits, Tensor): logits = logits.data
    predictions = np.argmax(logits, axis=1)
    
    if y_test.ndim > 1:
        true_labels = np.argmax(y_test[:2000], axis=1)
    else:
        true_labels = y_test[:2000]

    ax_cm = fig.add_subplot(gs[1, :]) 
    cm = confusion_matrix(true_labels, predictions)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black', ax=ax_cm)
    ax_cm.set_title('Confusion Matrix (Top 2000 Samples)', fontsize=14, fontweight='bold')
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('True')
    gs_images = gs[2:, :].subgridspec(3, 5)
    
    indices = np.random.choice(len(true_labels), 15, replace=False)
    
    for i, idx in enumerate(indices):
        row = i // 5
        col = i % 5
        ax_img = fig.add_subplot(gs_images[row, col])
        
        img = x_test[idx].reshape(28, 28)
        pred = predictions[idx]
        true = true_labels[idx]
        color = 'green' if pred == true else 'red'
        
        ax_img.imshow(img, cmap='gray')
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        
        ax_img.set_xlabel(f"T:{true} | P:{pred}", color=color, fontweight='bold', fontsize=10)
        
        if i == 2:
            ax_img.set_title("Random Prediction Samples", fontsize=14, fontweight='bold', pad=10)

    plt.show()

print("\nGenerating Unified Dashboard...")
visualize_dashboard(final_model, final_trainer.history, x_test, y_test)

===============================================================================================================================================
Loading Data...
Loading MNIST Data...
Data Ready:
Train: (56000, 784)
Val:   (7000, 784)
Test:  (7000, 784)

Initializing Training...

Initializing Grid Search (8 combinations)...

--- Trial 1/8: {'lr': 0.01, 'hidden_size': 256, 'dropout': 0.3} ---
Starting training for 1 epochs...
Epoch 1/1 (3.30s) | Train Loss: 0.3423 | Train Acc: 0.8950 | Val Loss: 0.2388 | Val Acc: 0.9305
--> Validation Accuracy: 0.9305
★ New Best Found!

--- Trial 2/8: {'lr': 0.01, 'hidden_size': 256, 'dropout': 0.4} ---
Starting training for 1 epochs...
Epoch 1/1 (2.73s) | Train Loss: 0.3897 | Train Acc: 0.8796 | Val Loss: 0.2726 | Val Acc: 0.9163
--> Validation Accuracy: 0.9163

--- Trial 3/8: {'lr': 0.01, 'hidden_size': 128, 'dropout': 0.3} ---
Starting training for 1 epochs...
Epoch 1/1 (1.67s) | Train Loss: 0.4571 | Train Acc: 0.8599 | Val Loss: 0.2942 | Val Acc: 0.9141
--> Validation Accuracy: 0.9141

--- Trial 4/8: {'lr': 0.01, 'hidden_size': 128, 'dropout': 0.4} ---
Starting training for 1 epochs...
Epoch 1/1 (1.68s) | Train Loss: 0.5331 | Train Acc: 0.8365 | Val Loss: 0.3580 | Val Acc: 0.8865
--> Validation Accuracy: 0.8865

--- Trial 5/8: {'lr': 0.005, 'hidden_size': 256, 'dropout': 0.3} ---
Starting training for 1 epochs...
Epoch 1/1 (2.74s) | Train Loss: 0.3665 | Train Acc: 0.8879 | Val Loss: 0.2345 | Val Acc: 0.9267
--> Validation Accuracy: 0.9267

--- Trial 6/8: {'lr': 0.005, 'hidden_size': 256, 'dropout': 0.4} ---
Starting training for 1 epochs...
Epoch 1/1 (2.48s) | Train Loss: 0.4249 | Train Acc: 0.8687 | Val Loss: 0.2835 | Val Acc: 0.9110
--> Validation Accuracy: 0.9110

--- Trial 7/8: {'lr': 0.005, 'hidden_size': 128, 'dropout': 0.3} ---
Starting training for 1 epochs...
Epoch 1/1 (1.54s) | Train Loss: 0.6587 | Train Acc: 0.8073 | Val Loss: 0.3986 | Val Acc: 0.8797
--> Validation Accuracy: 0.8797

--- Trial 8/8: {'lr': 0.005, 'hidden_size': 128, 'dropout': 0.4} ---
Starting training for 1 epochs...
Epoch 1/1 (1.57s) | Train Loss: 0.6097 | Train Acc: 0.8169 | Val Loss: 0.4223 | Val Acc: 0.8720
--> Validation Accuracy: 0.8720
=============================================
Best Params Found: {'lr': 0.01, 'hidden_size': 256, 'dropout': 0.3}
Best Validation Accuracy during search: 0.9305
=============================================

Starting Final Extended Training...
Starting training for 20 epochs...
Epoch 1/20 (2.89s) | Train Loss: 0.3336 | Train Acc: 0.8982 | Val Loss: 0.2333 | Val Acc: 0.9295
Epoch 2/20 (2.66s) | Train Loss: 0.1887 | Train Acc: 0.9415 | Val Loss: 0.1899 | Val Acc: 0.9442
Epoch 3/20 (2.61s) | Train Loss: 0.1491 | Train Acc: 0.9544 | Val Loss: 0.1766 | Val Acc: 0.9480
Epoch 4/20 (2.75s) | Train Loss: 0.1263 | Train Acc: 0.9606 | Val Loss: 0.1580 | Val Acc: 0.9552
Epoch 5/20 (3.95s) | Train Loss: 0.1077 | Train Acc: 0.9668 | Val Loss: 0.1485 | Val Acc: 0.9585
Epoch 6/20 (2.86s) | Train Loss: 0.0978 | Train Acc: 0.9696 | Val Loss: 0.1354 | Val Acc: 0.9602
Epoch 7/20 (3.11s) | Train Loss: 0.0902 | Train Acc: 0.9719 | Val Loss: 0.1236 | Val Acc: 0.9644
Epoch 8/20 (2.65s) | Train Loss: 0.0789 | Train Acc: 0.9748 | Val Loss: 0.1296 | Val Acc: 0.9656
Epoch 9/20 (3.54s) | Train Loss: 0.0776 | Train Acc: 0.9758 | Val Loss: 0.1309 | Val Acc: 0.9638
Epoch 10/20 (4.65s) | Train Loss: 0.0683 | Train Acc: 0.9778 | Val Loss: 0.1294 | Val Acc: 0.9645
Epoch 11/20 (5.27s) | Train Loss: 0.0636 | Train Acc: 0.9792 | Val Loss: 0.1215 | Val Acc: 0.9673
Epoch 12/20 (3.38s) | Train Loss: 0.0622 | Train Acc: 0.9795 | Val Loss: 0.1353 | Val Acc: 0.9653
Epoch 13/20 (2.95s) | Train Loss: 0.0592 | Train Acc: 0.9802 | Val Loss: 0.1192 | Val Acc: 0.9697
Epoch 14/20 (3.03s) | Train Loss: 0.0541 | Train Acc: 0.9826 | Val Loss: 0.1287 | Val Acc: 0.9679
Epoch 15/20 (3.12s) | Train Loss: 0.0537 | Train Acc: 0.9825 | Val Loss: 0.1296 | Val Acc: 0.9677
Epoch 16/20 (2.54s) | Train Loss: 0.0535 | Train Acc: 0.9830 | Val Loss: 0.1277 | Val Acc: 0.9674
Epoch 17/20 (2.86s) | Train Loss: 0.0489 | Train Acc: 0.9843 | Val Loss: 0.1257 | Val Acc: 0.9689
Epoch 18/20 (3.31s) | Train Loss: 0.0477 | Train Acc: 0.9845 | Val Loss: 0.1313 | Val Acc: 0.9708
Epoch 19/20 (2.87s) | Train Loss: 0.0484 | Train Acc: 0.9843 | Val Loss: 0.1219 | Val Acc: 0.9698
Epoch 20/20 (2.61s) | Train Loss: 0.0441 | Train Acc: 0.9856 | Val Loss: 0.1363 | Val Acc: 0.9674

Final Test Accuracy: 96.82%

🔍 --- STEP 1: FORWARD PASS ONLY ---
└── [Leaf] Shape=() | None
    ├── [Leaf] Shape=(1, 10) | None
    │   ├── [Leaf] Shape=(1, 10) | None
    │   │   ├── [Leaf] Shape=(1, 128) | None
    │   │   │   ├── [Leaf] Shape=(1, 128) | None
    │   │   │   │   └── [Leaf] Shape=(1, 128) | None
    │   │   │   │       ├── [Leaf] Shape=(1, 128) | None
    │   │   │   │       │   ├── [Leaf] Shape=(1, 128) | None
    │   │   │   │       │   │   ├── [Leaf] Shape=(1, 128) | None
    │   │   │   │       │   │   │   ├── [Leaf] Shape=(1, 128) | None
    │   │   │   │       │   │   │   │   ├── [Leaf] Shape=(1, 128) | None
    │   │   │   │       │   │   │   │   │   ├── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   ├── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   │   └── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   │       ├── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   │       │   ├── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   ├── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   ├── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   ├── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   │   ├── [Leaf] Shape=(1, 784) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   │   └── [Leaf] Shape=(784, 256) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   │       └── [Leaf] Shape=(256, 784) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   └── [Leaf] Shape=(256,) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   └── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   └── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   │       │   └── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   │       └── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   └── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   └── [Leaf] Shape=(256, 128) | None
    │   │   │   │       │   │   │   │   │       └── [Leaf] Shape=(128, 256) | None
    │   │   │   │       │   │   │   │   └── [Leaf] Shape=(128,) | None
    │   │   │   │       │   │   │   └── [Leaf] Shape=(1, 128) | None
    │   │   │   │       │   │   └── [Leaf] Shape=(1, 128) | None
    │   │   │   │       │   └── [Leaf] Shape=(1, 128) | None
    │   │   │   │       └── [Leaf] Shape=(1, 128) | None
    │   │   │   └── [Leaf] Shape=(1, 128) | None
    │   │   └── [Leaf] Shape=(128, 10) | None
    │   │       └── [Leaf] Shape=(10, 128) | None
    │   └── [Leaf] Shape=(10,) | None
    └── [Leaf] Shape=(1, 10) | None

--- STEP 2: EXECUTING BACKWARD PASS ---
... Backward pass finished.

--- STEP 3: INSPECTING GRADIENTS ---
└── [Leaf] Shape=() | Grad(μ=1.0e+00)
    ├── [Leaf] Shape=(1, 10) | Grad(μ=1.8e-01)
    │   ├── [Leaf] Shape=(1, 10) | Grad(μ=1.8e-01)
    │   │   ├── [Leaf] Shape=(1, 128) | Grad(μ=4.2e-01)
    │   │   │   ├── [Leaf] Shape=(1, 128) | Grad(μ=4.0e-01)
    │   │   │   │   └── [Leaf] Shape=(1, 128) | Grad(μ=1.8e-02)
    │   │   │   │       ├── [Leaf] Shape=(1, 128) | Grad(μ=1.8e-02)
    │   │   │   │       │   ├── [Leaf] Shape=(1, 128) | Grad(μ=1.8e-02)
    │   │   │   │       │   │   ├── [Leaf] Shape=(1, 128) | Grad(μ=5.6e+00)
    │   │   │   │       │   │   │   ├── [Leaf] Shape=(1, 128) | Grad(μ=5.6e+00)
    │   │   │   │       │   │   │   │   ├── [Leaf] Shape=(1, 128) | Grad(μ=5.6e+00)
    │   │   │   │       │   │   │   │   │   ├── [Leaf] Shape=(1, 256) | Grad(μ=2.0e+02)
    │   │   │   │       │   │   │   │   │   │   ├── [Leaf] Shape=(1, 256) | Grad(μ=2.0e+02)
    │   │   │   │       │   │   │   │   │   │   │   └── [Leaf] Shape=(1, 256) | Grad(μ=8.0e-02)
    │   │   │   │       │   │   │   │   │   │   │       ├── [Leaf] Shape=(1, 256) | Grad(μ=8.0e-02)
    │   │   │   │       │   │   │   │   │   │   │       │   ├── [Leaf] Shape=(1, 256) | Grad(μ=6.3e-02)
    │   │   │   │       │   │   │   │   │   │   │       │   │   ├── [Leaf] Shape=(1, 256) | Grad(μ=2.0e+01)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   ├── [Leaf] Shape=(1, 256) | Grad(μ=2.0e+01)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   ├── [Leaf] Shape=(1, 256) | Grad(μ=2.0e+01)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   │   ├── [Leaf] Shape=(1, 784) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   │   └── [Leaf] Shape=(784, 256) | Grad(μ=4.6e+00)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   │       └── [Leaf] Shape=(256, 784) | Grad(μ=4.6e+00)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   └── [Leaf] Shape=(256,) | Grad(μ=2.0e+01)
    │   │   │   │       │   │   ├── [Leaf] Shape=(1, 128) | Grad(μ=5.6e+00)
    │   │   │   │       │   │   │   ├── [Leaf] Shape=(1, 128) | Grad(μ=5.6e+00)
    │   │   │   │       │   │   │   │   ├── [Leaf] Shape=(1, 128) | Grad(μ=5.6e+00)
    │   │   │   │       │   │   │   │   │   ├── [Leaf] Shape=(1, 256) | Grad(μ=2.0e+02)
    │   │   │   │       │   │   │   │   │   │   ├── [Leaf] Shape=(1, 256) | Grad(μ=2.0e+02)
    │   │   │   │       │   │   │   │   │   │   │   └── [Leaf] Shape=(1, 256) | Grad(μ=8.0e-02)
    │   │   │   │       │   │   │   │   │   │   │       ├── [Leaf] Shape=(1, 256) | Grad(μ=8.0e-02)
    │   │   │   │       │   │   │   │   │   │   │       │   ├── [Leaf] Shape=(1, 256) | Grad(μ=6.3e-02)
    │   │   │   │       │   │   │   │   │   │   │       │   │   ├── [Leaf] Shape=(1, 256) | Grad(μ=2.0e+01)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   ├── [Leaf] Shape=(1, 256) | Grad(μ=2.0e+01)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   ├── [Leaf] Shape=(1, 256) | Grad(μ=2.0e+01)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   │   ├── [Leaf] Shape=(1, 784) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   │   └── [Leaf] Shape=(784, 256) | Grad(μ=4.6e+00)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   │       └── [Leaf] Shape=(256, 784) | Grad(μ=4.6e+00)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   └── [Leaf] Shape=(256,) | Grad(μ=2.0e+01)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   └── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   └── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   ├── [Leaf] Shape=(1, 128) | Grad(μ=5.6e+00)
    │   │   │   │       │   │   │   │   ├── [Leaf] Shape=(1, 128) | Grad(μ=5.6e+00)
    │   │   │   │       │   │   │   │   │   ├── [Leaf] Shape=(1, 256) | Grad(μ=2.0e+02)
    │   │   │   │       │   │   │   │   │   │   ├── [Leaf] Shape=(1, 256) | Grad(μ=2.0e+02)
    │   │   │   │       │   │   │   │   │   │   │   └── [Leaf] Shape=(1, 256) | Grad(μ=8.0e-02)
    │   │   │   │       │   │   │   │   │   │   │       ├── [Leaf] Shape=(1, 256) | Grad(μ=8.0e-02)
    │   │   │   │       │   │   │   │   │   │   │       │   ├── [Leaf] Shape=(1, 256) | Grad(μ=6.3e-02)
    │   │   │   │       │   │   │   │   │   │   │       │   │   ├── [Leaf] Shape=(1, 256) | Grad(μ=2.0e+01)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   ├── [Leaf] Shape=(1, 256) | Grad(μ=2.0e+01)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   ├── [Leaf] Shape=(1, 256) | Grad(μ=2.0e+01)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   │   ├── [Leaf] Shape=(1, 784) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   │   └── [Leaf] Shape=(784, 256) | Grad(μ=4.6e+00)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   │       └── [Leaf] Shape=(256, 784) | Grad(μ=4.6e+00)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   └── [Leaf] Shape=(256,) | Grad(μ=2.0e+01)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   └── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   └── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   │   └── [Leaf] Shape=(1, 256) | Grad(μ=8.0e-02)
    │   │   │   │       │   │   │   │   │   │   │       ├── [Leaf] Shape=(1, 256) | Grad(μ=8.0e-02)
    │   │   │   │       │   │   │   │   │   │   │       │   ├── [Leaf] Shape=(1, 256) | Grad(μ=6.3e-02)
    │   │   │   │       │   │   │   │   │   │   │       │   │   ├── [Leaf] Shape=(1, 256) | Grad(μ=2.0e+01)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   ├── [Leaf] Shape=(1, 256) | Grad(μ=2.0e+01)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   ├── [Leaf] Shape=(1, 256) | Grad(μ=2.0e+01)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   │   ├── [Leaf] Shape=(1, 784) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   │   └── [Leaf] Shape=(784, 256) | Grad(μ=4.6e+00)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   │       └── [Leaf] Shape=(256, 784) | Grad(μ=4.6e+00)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   └── [Leaf] Shape=(256,) | Grad(μ=2.0e+01)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   └── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   └── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   ├── [Leaf] Shape=(1, 256) | Grad(μ=2.0e+01)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   ├── [Leaf] Shape=(1, 256) | Grad(μ=2.0e+01)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   ├── [Leaf] Shape=(1, 256) | Grad(μ=2.0e+01)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   │   ├── [Leaf] Shape=(1, 784) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   │   └── [Leaf] Shape=(784, 256) | Grad(μ=4.6e+00)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   │       └── [Leaf] Shape=(256, 784) | Grad(μ=4.6e+00)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   └── [Leaf] Shape=(256,) | Grad(μ=2.0e+01)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   └── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   └── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   │   ├── [Leaf] Shape=(1, 784) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   │   └── [Leaf] Shape=(784, 256) | Grad(μ=4.6e+00)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   │       └── [Leaf] Shape=(256, 784) | Grad(μ=4.6e+00)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   └── [Leaf] Shape=(256,) | Grad(μ=2.0e+01)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   └── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   └── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   │       └── [Leaf] Shape=(256, 784) | Grad(μ=4.6e+00)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   │   └── [Leaf] Shape=(256,) | Grad(μ=2.0e+01)
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   └── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   └── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   │   └── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   │       │   │   └── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   │   │       │   └── [Leaf] Shape=(1, 256) | Grad(μ=0.0e+00)
    │   │   │   │       │   │   │   │   │   │   │       └── [Leaf] Shape=(1, 256) | Grad(μ=8.0e-02)
    │   │   │   │       │   │   │   │   │   │   └── [Leaf] Shape=(1, 256) | None
    │   │   │   │       │   │   │   │   │   └── [Leaf] Shape=(256, 128) | Grad(μ=6.2e-04)
    │   │   │   │       │   │   │   │   │       └── [Leaf] Shape=(128, 256) | Grad(μ=6.2e-04)
    │   │   │   │       │   │   │   │   └── [Leaf] Shape=(128,) | Grad(μ=5.6e+00)
    │   │   │   │       │   │   │   └── [Leaf] Shape=(1, 128) | None
    │   │   │   │       │   │   └── [Leaf] Shape=(1, 128) | None
    │   │   │   │       │   └── [Leaf] Shape=(1, 128) | Grad(μ=0.0e+00)
    │   │   │   │       └── [Leaf] Shape=(1, 128) | Grad(μ=1.8e-02)
    │   │   │   └── [Leaf] Shape=(1, 128) | None
    │   │   └── [Leaf] Shape=(128, 10) | Grad(μ=9.9e-05)
    │   │       └── [Leaf] Shape=(10, 128) | Grad(μ=9.9e-05)
    │   └── [Leaf] Shape=(10,) | Grad(μ=1.8e-01)
    └── [Leaf] Shape=(1, 10) | None

Generating Unified Dashboard...




==============================================================================================
pytorch

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.gridspec as gridspec
import itertools 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# ============================================================
# 1. init data
# ============================================================
print("Loading Data...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)), 
    transforms.Lambda(lambda x: torch.flatten(x)) 
])

full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

val_size = 10000
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

BATCH_SIZE = 256

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

x_test_np = test_dataset.data.numpy().reshape(-1, 784) / 255.0
y_test_np = test_dataset.targets.numpy()

# ============================================================
# 2. build model
# ============================================================
def build_model(params):
    hidden_1  = params['hidden_size']
    hidden_2  = params['hidden_size'] // 2 
    drop_rate = params['dropout']
    lr        = params['lr']
    
    model = nn.Sequential(
                nn.Linear(784, hidden_1),
                nn.BatchNorm1d(hidden_1),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                #=======2=========#
                nn.Linear(hidden_1, hidden_2),
                nn.BatchNorm1d(hidden_2),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Linear(hidden_2, 10)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    return model, optimizer, criterion

# ============================================================
# Helper Class: Trainer
# ============================================================
class PyTorchTrainer:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
    def fit(self, train_loader, validation_data, epochs, verbose=True):
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total
            
            # Validation
            val_loss, val_acc = self.evaluate(validation_data)
            
            self.history['train_loss'].append(epoch_loss)
            self.history['train_acc'].append(epoch_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
        return self.history

    def evaluate(self, loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return running_loss / len(loader), correct / total

# ============================================================
# 3. Grid Search Tuning
# ============================================================

param_grid = {
    'lr'          : [0.01, 0.001],    
    'hidden_size' : [128, 256 ],         
    'dropout'     : [0.2, 0.5]          
}

keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"\nInitializing Grid Search ({len(combinations)} combinations)...")

best_acc = 0.0
best_params = None

for i, params in enumerate(combinations):
    print(f"\n--- Trial {i+1}/{len(combinations)}: {params} ---")
    
    model, opt, crit = build_model(params)
    trainer = PyTorchTrainer(model, opt, crit)
    
    hist = trainer.fit(train_loader, val_loader, epochs=1, verbose=False)
    
    final_val_acc = hist['val_acc'][-1]
    print(f"--> Validation Accuracy: {final_val_acc:.4f}")
    
    if final_val_acc > best_acc:
        best_acc = final_val_acc
        best_params = params
        print("★ New Best Found!")

print("\n=============================================")
print(f"Best Params Found: {best_params}")
print(f"Best Validation Accuracy during search: {best_acc:.4f}")
print("=============================================\n")

# ============================================================
# 4. Final Training with Best Params
# ============================================================
print("Starting Final Extended Training with BEST parameters...")
final_model, final_opt, final_crit = build_model(best_params)
final_trainer = PyTorchTrainer(final_model, final_opt, final_crit)

history = final_trainer.fit(
    train_loader, 
    validation_data=val_loader, 
    epochs=20,
    verbose=True
)

loss, acc = final_trainer.evaluate(loader=test_loader)
print(f"\nFinal Test Accuracy: {acc*100:.2f}%")


# ============================================================
# 5. Visualizations (Graph)
# ============================================================

def print_detailed_tree_pytorch(grad_fn, indent="", is_last=True, seen=None):
    if seen is None: seen = set()
    if grad_fn is None: return

    marker = "└── " if is_last else "├── "
    connector = "    " if is_last else "│   "
    
    name = type(grad_fn).__name__
    color = "\033[94m" 
    if 'AccumulateGrad' in name:
        name = "Leaf/Weights"
        color = "\033[92m" 
    
    print(f"{indent}{marker}{color}[{name}]\033[0m")
    
    if id(grad_fn) in seen:
        print(f"{indent}{connector}\033[90m... (Shared Node) ...\033[0m")
        return
    seen.add(id(grad_fn))

    if hasattr(grad_fn, 'next_functions'):
        parents = [fn[0] for fn in grad_fn.next_functions if fn[0] is not None]
        for i, parent in enumerate(parents):
            print_detailed_tree_pytorch(parent, indent + connector, (i == len(parents) - 1), seen)


print("\n --- STEP 1: FORWARD PASS ONLY ---")
x_sample, y_sample = next(iter(test_loader))
x_sample = x_sample[:1].to(device)
y_sample = y_sample[:1].to(device)

final_model.zero_grad()
out = final_model(x_sample)
loss = final_crit(out, y_sample)

print_detailed_tree_pytorch(loss.grad_fn)

print("\n--- STEP 2: EXECUTING BACKWARD PASS ---")
loss.backward()
print("... Backward pass finished.")

print("\n--- STEP 3: INSPECTING GRADIENTS ---")
for name, param in final_model.named_parameters():
    if param.grad is not None and 'weight' in name:
        g_mean = torch.mean(torch.abs(param.grad)).item()
        print(f"Param: {name} | \033[91mGrad Mean Abs: {g_mean:.1e}\033[0m")
        break 

# --- Dashboard Visualization ---
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def visualize_dashboard(model, history, x_test, y_test):
    sns.set_theme(style="whitegrid")
    
    fig = plt.figure(figsize=(16, 20), constrained_layout=True)
    fig.suptitle('PyTorch + Grid Search Dashboard', fontsize=24, fontweight='bold', y=1.02)

    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1.2, 0.5, 0.5])
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_acc = fig.add_subplot(gs[0, 1])

    s_train_loss = smooth_curve(history['train_loss'])
    s_val_loss = smooth_curve(history['val_loss'])
    s_train_acc = smooth_curve(history['train_acc'])
    s_val_acc = smooth_curve(history['val_acc'])

    ax_loss.plot(history['train_loss'], alpha=0.3, color='#3498db')
    ax_loss.plot(history['val_loss'], alpha=0.3, color='#e74c3c')
    ax_loss.plot(s_train_loss, label='Train (Smooth)', linewidth=2, color='#3498db')
    ax_loss.plot(s_val_loss, label='Val (Smooth)', linewidth=2, linestyle='--', color='#e74c3c')
    ax_loss.set_title('Loss History', fontsize=14, fontweight='bold')
    ax_loss.legend()
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')

    ax_acc.plot(history['train_acc'], alpha=0.3, color='#2ecc71')
    ax_acc.plot(history['val_acc'], alpha=0.3, color='#f39c12')
    ax_acc.plot(s_train_acc, label='Train (Smooth)', linewidth=2, color='#2ecc71')
    ax_acc.plot(s_val_acc, label='Val (Smooth)', linewidth=2, linestyle='--', color='#f39c12')
    ax_acc.set_title('Accuracy History', fontsize=14, fontweight='bold')
    ax_acc.legend()
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Accuracy')

    model.eval()
    limit = 2000
    x_tensor_vis = torch.tensor(x_test[:limit], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = model(x_tensor_vis)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
    
    true_labels = y_test[:limit]

    ax_cm = fig.add_subplot(gs[1, :]) 
    cm = confusion_matrix(true_labels, predictions)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black', ax=ax_cm)
    ax_cm.set_title(f'Confusion Matrix (Best Params: {best_params["hidden_size"]} hidden)', fontsize=14, fontweight='bold')
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('True')
    
    gs_images = gs[2:, :].subgridspec(3, 5)
    indices = np.random.choice(len(true_labels), 15, replace=False)
    
    for i, idx in enumerate(indices):
        row = i // 5
        col = i % 5
        ax_img = fig.add_subplot(gs_images[row, col])
        
        img = x_test[idx].reshape(28, 28)
        pred = predictions[idx]
        true = true_labels[idx]
        color = 'green' if pred == true else 'red'
        
        ax_img.imshow(img, cmap='gray')
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_xlabel(f"T:{true} | P:{pred}", color=color, fontweight='bold', fontsize=10)
        
        if i == 2:
            ax_img.set_title("Random Prediction Samples", fontsize=14, fontweight='bold', pad=10)

    plt.show()

print("\nGenerating Unified Dashboard...")
visualize_dashboard(final_model, final_trainer.history, x_test_np, y_test_np)


=========================================================================================================================================

Using device: cpu
Loading Data...

Initializing Grid Search (8 combinations)...

--- Trial 1/8: {'lr': 0.01, 'hidden_size': 128, 'dropout': 0.2} ---
--> Validation Accuracy: 0.9601
★ New Best Found!

--- Trial 2/8: {'lr': 0.01, 'hidden_size': 128, 'dropout': 0.5} ---
--> Validation Accuracy: 0.9453

--- Trial 3/8: {'lr': 0.01, 'hidden_size': 256, 'dropout': 0.2} ---
--> Validation Accuracy: 0.9599

--- Trial 4/8: {'lr': 0.01, 'hidden_size': 256, 'dropout': 0.5} ---
--> Validation Accuracy: 0.9568

--- Trial 5/8: {'lr': 0.001, 'hidden_size': 128, 'dropout': 0.2} ---
--> Validation Accuracy: 0.9455

--- Trial 6/8: {'lr': 0.001, 'hidden_size': 128, 'dropout': 0.5} ---
--> Validation Accuracy: 0.9204

--- Trial 7/8: {'lr': 0.001, 'hidden_size': 256, 'dropout': 0.2} ---
--> Validation Accuracy: 0.9586

--- Trial 8/8: {'lr': 0.001, 'hidden_size': 256, 'dropout': 0.5} ---
--> Validation Accuracy: 0.9436

=============================================
Best Params Found: {'lr': 0.01, 'hidden_size': 128, 'dropout': 0.2}
Best Validation Accuracy during search: 0.9601
=============================================

Starting Final Extended Training with BEST parameters...
Epoch 1/20 | Loss: 0.3048 | Acc: 0.9094 | Val Loss: 0.1484 | Val Acc: 0.9578
Epoch 2/20 | Loss: 0.1523 | Acc: 0.9541 | Val Loss: 0.1118 | Val Acc: 0.9691
Epoch 3/20 | Loss: 0.1201 | Acc: 0.9631 | Val Loss: 0.0975 | Val Acc: 0.9730
Epoch 4/20 | Loss: 0.1021 | Acc: 0.9677 | Val Loss: 0.0907 | Val Acc: 0.9754
Epoch 5/20 | Loss: 0.0897 | Acc: 0.9722 | Val Loss: 0.1012 | Val Acc: 0.9714
Epoch 6/20 | Loss: 0.0812 | Acc: 0.9753 | Val Loss: 0.0888 | Val Acc: 0.9751
Epoch 7/20 | Loss: 0.0754 | Acc: 0.9755 | Val Loss: 0.0824 | Val Acc: 0.9768
Epoch 8/20 | Loss: 0.0676 | Acc: 0.9776 | Val Loss: 0.0836 | Val Acc: 0.9772
Epoch 9/20 | Loss: 0.0662 | Acc: 0.9791 | Val Loss: 0.0929 | Val Acc: 0.9755
Epoch 10/20 | Loss: 0.0611 | Acc: 0.9794 | Val Loss: 0.0907 | Val Acc: 0.9760
Epoch 11/20 | Loss: 0.0580 | Acc: 0.9811 | Val Loss: 0.0924 | Val Acc: 0.9767
Epoch 12/20 | Loss: 0.0537 | Acc: 0.9829 | Val Loss: 0.0892 | Val Acc: 0.9774
Epoch 13/20 | Loss: 0.0553 | Acc: 0.9817 | Val Loss: 0.0916 | Val Acc: 0.9762
Epoch 14/20 | Loss: 0.0507 | Acc: 0.9830 | Val Loss: 0.0914 | Val Acc: 0.9785
Epoch 15/20 | Loss: 0.0495 | Acc: 0.9838 | Val Loss: 0.0871 | Val Acc: 0.9788
Epoch 16/20 | Loss: 0.0462 | Acc: 0.9842 | Val Loss: 0.0942 | Val Acc: 0.9785
Epoch 17/20 | Loss: 0.0479 | Acc: 0.9844 | Val Loss: 0.0914 | Val Acc: 0.9788
Epoch 18/20 | Loss: 0.0441 | Acc: 0.9861 | Val Loss: 0.0840 | Val Acc: 0.9806
Epoch 19/20 | Loss: 0.0428 | Acc: 0.9863 | Val Loss: 0.0894 | Val Acc: 0.9773
Epoch 20/20 | Loss: 0.0455 | Acc: 0.9850 | Val Loss: 0.1011 | Val Acc: 0.9783

Final Test Accuracy: 97.93%

 --- STEP 1: FORWARD PASS ONLY ---
└── [NllLossBackward0]
    └── [LogSoftmaxBackward0]
        └── [AddmmBackward0]
            ├── [Leaf/Weights]
            ├── [ReluBackward0]
            │   └── [NativeBatchNormBackward0]
            │       ├── [AddmmBackward0]
            │       │   ├── [Leaf/Weights]
            │       │   ├── [ReluBackward0]
            │       │   │   └── [NativeBatchNormBackward0]
            │       │   │       ├── [AddmmBackward0]
            │       │   │       │   ├── [Leaf/Weights]
            │       │   │       │   └── [TBackward0]
            │       │   │       │       └── [Leaf/Weights]
            │       │   │       ├── [Leaf/Weights]
            │       │   │       └── [Leaf/Weights]
            │       │   └── [TBackward0]
            │       │       └── [Leaf/Weights]
            │       │           ... (Shared Node) ...
            │       ├── [Leaf/Weights]
            │       └── [Leaf/Weights]
            └── [TBackward0]
                └── [Leaf/Weights]
                    ... (Shared Node) ...

--- STEP 2: EXECUTING BACKWARD PASS ---
... Backward pass finished.

--- STEP 3: INSPECTING GRADIENTS ---
Param: 0.weight | Grad Mean Abs: 1.2e-08

Generating Unified Dashboard...