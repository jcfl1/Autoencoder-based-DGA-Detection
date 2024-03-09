import numpy as np
import torch

# Early Stopping -----------------------------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=7, delta=0, objective='minimize', verbose=True, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.objective = objective
        self.path = path

        if self.objective == 'minimize':
            self.compare_funct = lambda new_score,best_score: new_score < best_score - self.delta
            self.best_score = np.Inf
        elif self.objective == 'maximize':
            self.compare_funct = lambda new_score,best_score: new_score > best_score + self.delta
            self.best_score = -np.Inf
        else:
            raise ValueError(f"Unexpected value atributted to 'objective'.")

    def __call__(self, new_val_score, model):
        if self.compare_funct(new_val_score, self.best_score):     # If comparison between new_val_score and best_score is aligned with our objetive, then lets save current checkpoint and update best_score
            self.save_checkpoint(new_val_score, model)
            self.counter = 0
        else:                                                                   # If comparison between new_val_score and best_score is NOT aligned with our objetive, lets increment patient counter
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}. Current validation score: {new_val_score:.5f}')
            if self.counter > self.patience:
                self.early_stop = True

    def save_checkpoint(self, new_val_score, model):
        if self.verbose:
            print(f'Validation score improved ({self.best_score:.5f} --> {new_val_score:.5f}).  Saving model ...')
        torch.save(model, self.path)
        self.best_score = new_val_score

