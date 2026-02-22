import itertools
from .trainer import Trainer

class GridSearchTuner:
    def __init__(self, model_builder_func, param_grid):
        self.model_builder = model_builder_func
        self.param_grid = param_grid
        self.best_params = None
        self.best_score = -float('inf')
        self.results = []
    def search(self, train_data, val_data, batch_size=32, epochs_per_trial=1):
        
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        all_combinations = list(itertools.product(*values))
        total_combinations = len(all_combinations)
        
        print(f"\nInitializing Grid Search ({total_combinations} combinations)...")
        
        for i, combination in enumerate(all_combinations):
            current_params = dict(zip(keys, combination))
            
            print(f"\n--- Trial {i+1}/{total_combinations}: {current_params} ---")

            try:
                model, optimizer, criterion = self.model_builder(current_params)
                
                trainer = Trainer(model, optimizer, criterion)
                history = trainer.fit(
                    train_data, 
                    validation_data=val_data,
                    epochs=epochs_per_trial,
                    batch_size=batch_size,  
                    verbose=True 
                )
                
                final_val_acc = max(history['val_acc'])
                
                self.results.append({
                    'params': current_params,
                    'score': final_val_acc
                })

                print(f"--> Validation Accuracy: {final_val_acc:.4f}")

            
                if final_val_acc > self.best_score:
                    self.best_score = final_val_acc
                    self.best_params = current_params
                    print(f"New Best Found!")
            
            except Exception as e:
                print(f"Trial failed with error: {e}")
                import traceback
                traceback.print_exc()

        print("\n=============================================")
        print(f"Best Params Found: {self.best_params}")
        print(f"Best Validation Accuracy during search: {self.best_score:.4f}")
        print("=============================================")
        
        return self.best_params