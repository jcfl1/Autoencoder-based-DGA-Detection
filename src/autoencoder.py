import pandas as pd
import numpy as np
from math import ceil
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Early Stopping -----------------------------------------------------------------------------------------------
class EarlyStopping:
  def __init__(self, patience=7, delta=0, objective='minimize', verbose=True, path='checkpoint.pt'):
    self.patience = patience
    self.delta = delta
    self.verbose = verbose
    self.counter = 0
    self.early_stop = False
    self.objective = objective
    self.best_score = np.Inf
    self.path = path

    if self.objective == 'minimize':
        self.compare_funct = lambda new_score,best_score: new_score < best_score
    elif self.objective == 'maximize':
        self.compare_funct = lambda new_score,best_score: new_score > best_score
    else:
        raise ValueError(f"Unexpected value atributted to 'objective'.")

  def __call__(self, new_val_score, model):
    if self.compare_funct(new_val_score, self.best_score - self.delta):     # If comparison between new_val_score and best_score is aligned with our objetive, then lets save current checkpoint and update best_score
        self.save_checkpoint(new_val_score, model)
        self.counter = 0
    else:                                                                   # If comparison between new_val_score and best_score is NOT aligned with our objetive, lets increment patient counter
        self.counter += 1
        print(f'EarlyStopping counter: {self.counter} out of {self.patience}. Current validation score: {new_val_score:.5f}')
        if self.counter > self.patience:
            self.early_stop = True

  def save_checkpoint(self, new_val_score, model):
    if self.verbose:
        print(f'Validation score decreased ({self.best_score:.5f} --> {new_val_score:.5f}).  Saving model ...')
    torch.save(model, self.path)
    self.best_score = new_val_score


# Autoencoder -----------------------------------------------------------------------------------------------
class Autoencoder(nn.Module):
  def __init__(self, in_features, hidden_layers_dims, dropout_rate=0.2):
    super().__init__()

    self.in_features = in_features
    self.dropout_rate = dropout_rate
    self.early_stopping = None

    # Encoder
    self.encoder = nn.ModuleList()
    self.encoder.append(nn.Linear(self.input_dim, hidden_layers_dims[0]))
    self.encoder.append(self.activation)
    for i in range(0, len(hidden_layers_dims)//2 ):
        self.encoder.append(nn.Linear(hidden_layers_dims[i], hidden_layers_dims[i+1]))
        self.encoder.append(self.activation)

    # Decoder
    self.decoder = nn.ModuleList()
    for i in range(len(hidden_layers_dims)//2, len(hidden_layers_dims)-1):
        self.decoer.append(nn.Linear(hidden_layers_dims[i], hidden_layers_dims[i+1]))
        self.decoer.append(self.activation)
    self.decoer.append(nn.Linear(hidden_layers_dims[-1], self.output_dim))        
    self.decoer.append(nn.Sigmoid())

  def forward(self, X):
    encoded = self.encoder(X)
    decoded = self.decoder(encoded)
    return decoded

  def compile(self, learning_rate, weight_decay):
    self.criterion = nn.MSEscore()
    self.optimizer = optim.Adam(self.parameters(), lr = learning_rate, weight_decay=weight_decay)

  def fit(self, X_train, num_epochs, batch_size, X_val = None, y_val=None, patience = None, delta = None, es_objective = None):
    if X_val is not None and y_val is not None and len(X_val) == len(y_val)and patience is not None and delta is not None:
      print(f'Using early stopping with patience={patience} and delta={delta}')
      self.early_stopping = EarlyStopping(patience, delta, objective=es_objective)

    val_avg_losses = []
    train_avg_losses = []

    for epoch in range(num_epochs):
      # Calibrando os pesos do modelo
      train_losses = []
      self.train()
      for batch in tqdm(range(0, len(X_train), batch_size)):
        batch_X = X_train[batch:(batch+batch_size)]
        batch_reconstruction = self.forward(batch_X)

        train_loss = self.criterion(batch_reconstruction, batch_X)
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()
        train_losses.append(train_loss.item())
      train_avg_loss = np.mean(train_losses)
      train_avg_losses.append(train_avg_loss)
      print(f'Epoch#{epoch+1}: Train Average Loss = {train_avg_loss:.5f}')

      # Mecanismo de early stopping
      if self.early_stopping is not None:
        val_losses = []
        self.eval()
        with torch.no_grad():
          for batch in range(0, len(X_val), batch_size):
            batch_X = X_val[batch:(batch+batch_size)]
            batch_y = y_val[batch:(batch+batch_size)]
            batch_reconstruction = self.forward(batch_X)
            new_val_anomaly_score = self.criterion(batch_reconstruction, batch_X)
            val_losses.append(new_val_anomaly_score.item())
        val_avg_loss = np.mean(val_losses)
        val_avg_losses.append(val_avg_loss)
        
        if es_criterion = 'aucroc':
            val_aucroc = roc_auc_score(y_val, val_losses)
            es_score = val_aucroc
        elif es_criteriron = 'loss':
            es_score = val_avg_loss
        else:
            raise ValueError(f"Unexpected value atributted to 'es_criteriron'.")
        
        self.early_stopping(val_avg_loss, self)

        if self.early_stopping.early_stop:
          print(f'Stopped by early stopping at epoch {epoch+1}')
          break

    if self.early_stopping is not None:
      self = torch.load('checkpoint.pt')
    self.eval()
    return train_avg_losses, val_avg_losses