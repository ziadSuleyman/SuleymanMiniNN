import numpy as np
import time
from ..core.context import no_grad 
from ..utils.data_prepare import DataLoader
from ..core.tensor import Tensor


class Trainer:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def _calculate_accuracy(self, output, target):
        pred_data = output.data if hasattr(output, 'data') else output
        target_data = target.data if hasattr(target, 'data') else target
        
        if pred_data.ndim > 1 and pred_data.shape[1] > 1:
            # Multi-class classification (Logits -> Argmax)
            pred_cls = np.argmax(pred_data, axis=1)
            target_cls = np.argmax(target_data, axis=1) # Assuming One-Hot Target
        else:
            # Binary classification (Logits -> Threshold at 0)
            pred_cls = (pred_data > 0).astype(int).reshape(-1)
            target_cls = target_data.astype(int).reshape(-1)
            
        return np.mean(pred_cls == target_cls)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = len(loader)

        for x, y in loader:
            self.optimizer.zero_grad()
            
            output = self.model(x)
            
            loss = self.criterion(output, y)
            
            loss.backward()
            
            self.optimizer.step()

            total_loss += loss.data
            total_acc += self._calculate_accuracy(output, y)

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        return avg_loss, avg_acc

    def evaluate(self, x=None, y=None, loader=None, batch_size=32):
        if loader is None:
            if x is not None and y is not None:
                loader = DataLoader(x, y, batch_size=batch_size, shuffle=False)
            elif x is not None and hasattr(x, '__iter__'):
                loader = x
            else:
                raise ValueError("Must provide either 'loader' or 'x' and 'y'.")
        
        if hasattr(self.model, 'eval'):
            self.model.eval()
            
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        with no_grad():
            for batch_x, batch_y in loader:
                if not isinstance(batch_x, Tensor): batch_x = Tensor(batch_x)
                if not isinstance(batch_y, Tensor): batch_y = Tensor(batch_y)

                output = self.model(batch_x)
                
                loss = self.criterion(output, batch_y)
                acc = self._calculate_accuracy(output, batch_y) 
                total_loss += loss.data if hasattr(loss, 'data') else loss
                total_acc += acc
                num_batches += 1

        if hasattr(self.model, 'train'):
            self.model.train()

        if num_batches == 0: return 0.0, 0.0
        return total_loss / num_batches, total_acc / num_batches
    def train_step(self, x, y):
        if not isinstance(x, Tensor): x = Tensor(x)
        if not isinstance(y, Tensor): y = Tensor(y)
        output = self.model(x)
        loss = self.criterion(output, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        acc = self._calculate_accuracy(output, y)
        return loss.data, acc

    def val_step(self, x, y):
        if not isinstance(x, Tensor): x = Tensor(x)
        if not isinstance(y, Tensor): y = Tensor(y)

        with no_grad():
            output = self.model(x)
            loss = self.criterion(output, y)
            acc = self._calculate_accuracy(output, y)
        return loss.data, acc

    def fit(self, x, y=None, validation_data=None, epochs=10, batch_size=32, verbose=True):
        if y is None:
            if not hasattr(x, '__iter__'):
                raise ValueError("If y is None, x must be an iterator (DataLoader).")
            train_loader = x
        else:
            train_loader = DataLoader(x, y, batch_size=batch_size, shuffle=True)

        val_loader = None
        if validation_data is not None:
            if isinstance(validation_data, (list, tuple)) and len(validation_data) == 2:
                val_x, val_y = validation_data
                val_loader = DataLoader(val_x, val_y, batch_size=batch_size, shuffle=False)
            elif hasattr(validation_data, '__iter__'):
                val_loader = validation_data

        print(f"Starting training for {epochs} epochs...")
        
        self.model.train()

        for epoch in range(epochs):
            start_time = time.time()
            losses = []
            accuracies = []
            for batch_x, batch_y in train_loader:
                l, a = self.train_step(batch_x, batch_y)
                losses.append(l)
                accuracies.append(a)
            
            avg_train_loss = np.mean(losses)
            avg_train_acc = np.mean(accuracies)
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(avg_train_acc)

            val_msg = ""
            if val_loader:
                self.model.eval()
                val_losses = []
                val_accuracies = []
                for batch_x, batch_y in val_loader:
                    vl, va = self.val_step(batch_x, batch_y)
                    val_losses.append(vl)
                    val_accuracies.append(va)
                
                self.model.train() 
                
                avg_val_loss = np.mean(val_losses)
                avg_val_acc = np.mean(val_accuracies)
                
                self.history['val_loss'].append(avg_val_loss)
                self.history['val_acc'].append(avg_val_acc)
                val_msg = f" | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}"

            if verbose:
                duration = time.time() - start_time
                print(f"Epoch {epoch+1}/{epochs} ({duration:.2f}s) "
                      f"| Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}"
                      f"{val_msg}")
                      
        return self.history