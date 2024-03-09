import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from early_stopping import EarlyStopping

# Autoencoder -----------------------------------------------------------------------------------------------
class Autoencoder(nn.Module):
  def __init__(self, in_features, hidden_layers_dims, dropout_rate=0.2):
    super().__init__()

    self.in_features = in_features
    self.dropout_rate = dropout_rate
    self.activation = nn.ReLU()

    # Encoder
    self.encoder = nn.ModuleList()
    self.encoder.append(nn.Linear(self.in_features, hidden_layers_dims[0]))
    self.encoder.append(self.activation)
    for i in range(0, len(hidden_layers_dims)//2 ):
        self.encoder.append(nn.Linear(hidden_layers_dims[i], hidden_layers_dims[i+1]))
        self.encoder.append(self.activation)

    # Decoder
    self.decoder = nn.ModuleList()
    for i in range(len(hidden_layers_dims)//2, len(hidden_layers_dims)-1):
        self.decoder.append(nn.Linear(hidden_layers_dims[i], hidden_layers_dims[i+1]))
        self.decoder.append(self.activation)
    self.decoder.append(nn.Linear(hidden_layers_dims[-1], self.in_features))        

  def forward(self, X):
    encoded = X
    for layer in self.encoder:
      encoded = layer(encoded)
    
    decoded = encoded
    for layer in self.decoder:
      decoded = layer(decoded)
    
    return decoded

  def compile(self, learning_rate, weight_decay):
    self.criterion = nn.MSELoss()
    self.optimizer = optim.Adam(self.parameters(), lr = learning_rate, weight_decay=weight_decay)

  def fit(self, X_train, num_epochs, batch_size, X_val = None, y_val=None, patience = None, delta = None, es_criterion=None):
    if X_val is not None and y_val is not None and len(X_val) == len(y_val)and patience is not None and delta is not None and es_criterion is not None:
      print(f'Using early stopping with patience={patience}, delta={delta} and es_criterion={es_criterion}')
      if es_criterion == 'loss':
         es_objective = 'minimize'
      elif es_criterion == 'aucroc':
         es_objective = 'maximize'
      else:
        raise ValueError(f"Unexpected value atributted to 'es_criterion'.")
      early_stopping = EarlyStopping(patience, delta, objective=es_objective)

    val_avg_losses = []
    train_avg_losses = []

    for epoch in range(num_epochs):
      # Updating models weights
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

      # Early stopping mechanism
      if early_stopping is not None:
        val_losses = []
        self.eval()
        with torch.no_grad():
          for batch in range(0, len(X_val), batch_size):
            batch_X = X_val[batch:(batch+batch_size)]
            batch_reconstruction = self.forward(batch_X)
            batch_val_anomaly_score = torch.mean(torch.pow(batch_reconstruction - batch_X, 2), dim=1).tolist()
            val_losses.extend(batch_val_anomaly_score)
        val_avg_loss = np.mean(val_losses)
        val_avg_losses.append(val_avg_loss)

        if es_criterion == 'aucroc':
            val_aucroc = roc_auc_score(y_val, val_losses)
            es_score = val_aucroc
        elif es_criterion == 'loss':
            es_score = val_avg_loss
        
        early_stopping(es_score, self)

        if early_stopping.early_stop:
          print(f'Stopped by early stopping at epoch {epoch+1}')
          break
    
    model = self
    if early_stopping is not None:
      model = torch.load('checkpoint.pt')
    model.eval()
    return model, train_avg_losses, val_avg_losses