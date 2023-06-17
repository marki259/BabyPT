from torch import nn
import torch 
import numpy as np
import pandas as pd

import arff

# %% 
# Specify model

class Sparse(nn.Module):
    def __init__(self, dim):
        super().__init__()
        init_par = torch.randn((dim, 1))
        self.beta = nn.Parameter(init_par)

    def forward(self, x):
        return torch.matmul(x, self.beta)
    
    def get_betas(self):
        return self.beta
    
class SquaredLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, target):
        n = x.shape[0]
        return (x - target).pow(2.0).sum()/n
    
class Penalty(nn.Module):
    def __init__(self):
        super().__init__()
        self.param_l1 = nn.Parameter(torch.randn(1))
        self.param_l2 = nn.Parameter(torch.randn(1))

    def forward(self, betas):
        p1 = self.param_l1 * betas.abs().sum()
        p2 = self.param_l2 * betas.pow(2.0).sum()

        return p1 + p2

# %% 
# Data prep 
with open("./data/dataset_8_liver-disorders.arff", "r") as f:
    df_diabete = arff.load(f)

# Get into pandas format
col_names = [a[0] for a in df_diabete["attributes"]]
df_diabete = pd.DataFrame(df_diabete["data"], columns=col_names) 
df_diabete = pd.get_dummies(df_diabete, columns=["selector"], drop_first=True)

# %% 
# Define batches
# Should always return splitted train and exclusive validation sets
def get_batches(df: pd.DataFrame, size=64):
    n = df.shape[0]
    idx_train = np.random.choice(range(n), size)
    idx_val = [idx for idx in range(n) if idx not in idx_train]
    idx_val = np.random.choice(idx_val, int(size/2))

    df_train = df.iloc[idx_train, :]
    df_val = df.iloc[idx_val, :]

    y_train = df_train["mcv"]
    x_train = df_train[[c for c in df_train.columns if c != "mcv"]]

    y_val = df_val["mcv"]
    x_val = df_val[[c for c in df_val.columns if c != "mcv"]]

    y_train, x_train = torch.tensor(y_train.to_numpy()).float(), torch.tensor(x_train.to_numpy()).float()
    y_val, x_val = torch.tensor(y_val.to_numpy()).float(), torch.tensor(x_val.to_numpy()).float()

    return y_train, x_train, y_val, x_val

# %% 
# Training round 
y_train, x_train, y_val, x_val = get_batches(df_diabete)
m = x_train.shape[1]
sparse = Sparse(m)
penalty = Penalty()
squared_loss = SquaredLoss()

optimizer_beta = torch.optim.Adam(sparse.parameters(), lr=1e-3)
optimizer_lambda = torch.optim.Adam(penalty.parameters(), lr=1e-2)

def train_epoch():
    y_train, x_train, y_val, x_val = get_batches(df_diabete)

    optimizer_beta.zero_grad()

    x_train = sparse(x_train)
    train_loss = squared_loss(x_train, y_train)
    train_loss.backward()
    optimizer_beta.step()

    optimizer_lambda.zero_grad()

    x_val = sparse(x_val)
    val_loss = squared_loss(x_val, y_val) + penalty(sparse.get_betas())
    val_loss.backward()
    optimizer_lambda.step()

    return train_loss, val_loss

# %% Main
if __name__ == "__main__":
    n_epochs = 2500

    for epoch in range(n_epochs):
        train_loss, val_loss = train_epoch()

        if epoch % 50 == 0:
            print("----------")
            print(f"At epoch: {epoch}")
            print("Train loss: {:.2f}".format(train_loss.item()))
            print("val loss: {:.2f}".format(val_loss.item()))
            print("----------")

    print()