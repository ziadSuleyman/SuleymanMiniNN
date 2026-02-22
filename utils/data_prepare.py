# ======================================================
# Data Preparation Utilities
# ======================================================
from keras.datasets import mnist
import numpy as np

class DataLoader:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.shape[0]
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))
        self.indices = np.arange(self.n_samples)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for i in range(self.n_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.n_samples)
            batch_indices = self.indices[start:end]
            
            yield self.X[batch_indices], self.y[batch_indices]

    def __len__(self):
        return self.n_batches

# ======================================================
# 1) Load MNIST
# ======================================================
def load_mnist():
    """
    تحميل بيانات MNIST باستخدام Keras
    وإرجاعها موحدة (train + test)
    """
    print("Loading MNIST Data...")
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = mnist.load_data()

    X_all = np.concatenate((x_train_raw, x_test_raw), axis=0)
    Y_all = np.concatenate((y_train_raw, y_test_raw), axis=0)

    return X_all, Y_all


# ======================================================
# 2) Normalize + Flatten
# ======================================================
def normalize_and_flatten(X, flatten=True):
    """
    flatten=True  -> (N, 784)      للموديلات الخطية MLP
    flatten=False -> (N, 1, 28, 28) لموديلات CNN
    """
    X = X.astype(np.float32) / 255.0

    if flatten:
        X = X.reshape(X.shape[0], -1)
    else:
        X = X.reshape(X.shape[0], 1, 28, 28)
        
    return X

# ======================================================
# 3) One-Hot Encoding
# ======================================================
def one_hot_encode(y, num_classes=10):
    """
    تحويل labels إلى One-Hot Encoding
    """
    y = y.astype(int)
    one_hot = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    one_hot[np.arange(y.shape[0]), y] = 1.0
    return one_hot


# ======================================================
# 4) Train / Val / Test Split
# ======================================================
def split_dataset(
    X,
    Y,
    test_ratio=0.1,
    val_ratio=0.1,
    shuffle=True,
    seed=42
):
    """
    تقسيم البيانات إلى:
    Train / Val / Test
    """
    assert len(X) == len(Y), "X and Y must have same length"

    N = len(X)
    indices = np.arange(N)

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    X = X[indices]
    Y = Y[indices]

    test_size = int(N * test_ratio)
    val_size  = int(N * val_ratio)

    X_test = X[:test_size]
    Y_test = Y[:test_size]

    X_val = X[test_size:test_size + val_size]
    Y_val = Y[test_size:test_size + val_size]

    X_train = X[test_size + val_size:]
    Y_train = Y[test_size + val_size:]

    return (
        (X_train, Y_train),
        (X_val, Y_val),
        (X_test, Y_test)
    )


# ======================================================
# 5) Full Pipeline
# ======================================================


def prepare_mnist_data(
        test_ratio=0.1,
        val_ratio=0.1,
        flatten=True):
    
    X_all, Y_all = load_mnist()
    
    X_all = normalize_and_flatten(X_all, flatten=flatten)
    Y_all = one_hot_encode(Y_all, num_classes=10)
    
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_dataset(
        X_all,
        Y_all,
        test_ratio=test_ratio,
        val_ratio=val_ratio
    )

    print("Data Ready:")
    print("Train:", x_train.shape)
    print("Val:  ", x_val.shape)
    print("Test: ", x_test.shape)

    return (
        (x_train, y_train),
        (x_val, y_val),
        (x_test, y_test)
    )
