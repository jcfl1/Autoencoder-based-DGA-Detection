import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from early_stopping import EarlyStopping

class LSTMModel(nn.Module):
    def __init__(self, max_features, embedding_dim, hidden_dim, dropout_rate, device='cpu'):
        super(LSTMModel, self).__init__()

        self.max_features = max_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.device = device

        self.embedding_layer = nn.Embedding(self.max_features, self.embedding_dim)
        self.lstm_layer = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.linear_layer = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        embedded = self.embedding_layer(X)
        _, (h_n, c_n) = self.lstm_layer(embedded)
        out = h_n[:,:,:].squeeze(0)
        out = out.squeeze()
        out = self.dropout_layer(out)
        out = self.linear_layer(out)
        out = out.squeeze()
        out = self.sigmoid(out)
        return out

    def compile(self, learning_rate, weight_decay):
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate, weight_decay=weight_decay)

    def fit(self, X_train, y_train, num_epochs, batch_size, X_val = None, y_val=None, patience = None, delta = None):
        early_stopping = None
        if X_val is not None and y_val is not None and len(X_val) == len(y_val)and patience is not None and delta is not None:
            print(f'Using early stopping with patience={patience}, delta={delta}')
            early_stopping = EarlyStopping(patience, delta, objective='minimize')


        val_avg_losses = []
        train_avg_losses = []
        for epoch in range(num_epochs):
            # Updating models weights
            train_losses = []
            self.train()
            for batch in tqdm(range(0, len(X_train), batch_size)):
                batch_X = X_train[batch:(batch+batch_size)]
                batch_y = y_train[batch:(batch+batch_size)]
                batch_y_pred = self.forward(batch_X)
                batch_train_loss = self.criterion(batch_y_pred, batch_y)
                train_losses.append(batch_train_loss.item())
                self.optimizer.zero_grad()
                batch_train_loss.backward()
                self.optimizer.step()
            train_avg_loss = np.mean(train_losses)
            train_avg_losses.append(train_avg_loss)
            print(f'Epoch#{epoch+1}: Train Average Loss = {train_avg_loss:.5f}')

            if early_stopping is not None:
                val_losses = []
                self.eval()
                with torch.no_grad():
                    for batch in tqdm(range(0, len(X_val), batch_size)):
                        batch_X = X_val[batch:(batch+batch_size)]
                        batch_y = y_val[batch:(batch+batch_size)]
                        batch_y_pred = self.forward(batch_X)
                        batch_val_loss = self.criterion(batch_y_pred, batch_y)
                        val_losses.append(batch_val_loss.item())
                val_avg_loss = np.mean(val_losses)
                val_avg_losses.append(val_avg_loss)
                early_stopping(val_avg_loss, self)

                if early_stopping.early_stop:
                    print(f'Stopped by early stopping at epoch {epoch+1}')
                    break

        model = self
        if early_stopping is not None:
            model = torch.load('checkpoint.pt')
        model.eval()
        return model, train_avg_losses, val_avg_losses

    def batch_inference(self, X, batch_size):
        preds = []
        with torch.no_grad():
            for batch in tqdm(range(0, len(X), batch_size)):
                batch_X = X[batch:(batch+batch_size)]
                batch_y_pred = self.forward(batch_X)
                preds.extend(batch_y_pred.tolist())
        return preds