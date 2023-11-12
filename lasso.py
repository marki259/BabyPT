from torch import nn
import torch
import numpy as np
import pandas as pd

import arff


# %%
# Specify model
class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        return self.linear(x)


class Sparse(nn.Module):
    def __init__(self, dim):
        super().__init__()
        init_par = torch.randn((dim, 1))
        self.beta = nn.Parameter(init_par)

    def forward(self, x):
        return torch.matmul(x, self.beta)

    def get_betas(self):
        return self.beta


def squared_loss(x, target):
    n = x.shape[0]
    return (x - target).pow(2.0).sum() / n


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
# Define scaler
def minmax_scaler(df: pd.DataFrame):
    df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
    return df


# Apply
df_diabete_std = minmax_scaler(df_diabete)


# %%
# Define batches
# Should always return splitted train and exclusive validation sets
def get_batches(df: pd.DataFrame, size=64):
    n = df.shape[0]
    idx_train = np.random.choice(range(n), size, replace=False)
    # idx_val = [idx for idx in range(n) if idx not in idx_train]
    # idx_val = np.random.choice(idx_val, int(size/2), replace=False)

    df_train = df.iloc[idx_train, :]
    # df_val = df.iloc[idx_val, :]

    y_train = df_train["mcv"]
    x_train = df_train[[c for c in df_train.columns if c != "mcv"]]
    x_train["intercept"] = 1

    # y_val = df_val["mcv"]
    # x_val = df_val[[c for c in df_val.columns if c != "mcv"]]
    # x_val["intercept"] = 1

    y_train, x_train = (
        torch.tensor(y_train.to_numpy()).float(),
        torch.tensor(x_train.to_numpy()).float(),
    )
    # y_val, x_val = torch.tensor(y_val.to_numpy()).float(), torch.tensor(x_val.to_numpy()).float()

    return y_train[:, None], x_train, torch.tensor(0.0), torch.tensor(0.0)


# %%
# R2
def r_squared(x, target):
    return 1 - (x - target).pow(2.0).sum() / (target.mean() - target).pow(2.0).sum()


# %%
# Training round
y_train, x_train, y_val, x_val = get_batches(df_diabete_std)
m = x_train.shape[1]
sparse = Sparse(m)
penalty = Penalty()

optimizer_beta = torch.optim.SGD(sparse.parameters(), lr=1e-2)
optimizer_lambda = torch.optim.Adam(penalty.parameters(), lr=1e-2)


def train_epoch():
    y_train, x_train, y_val, x_val = get_batches(
        df_diabete_std, size=df_diabete_std.shape[0]
    )

    optimizer_beta.zero_grad()

    x_train = sparse(x_train)
    train_loss = squared_loss(x_train, y_train)
    train_loss.backward()

    optimizer_beta.step()

    # optimizer_lambda.zero_grad()

    # x_val = sparse(x_val)
    # val_loss = squared_loss(x_val, y_val) + penalty(sparse.get_betas())
    # val_loss.backward()
    # optimizer_lambda.step()

    # return train_loss, val_loss
    return train_loss, torch.tensor(0)


# %% Main
if __name__ == "__main__":
    n_epochs = 10000

    y_train, x_train, _, _ = get_batches(df_diabete_std, size=128)
    x_train = sparse(x_train)
    r2_init = r_squared(x_train, y_train)

    for epoch in range(n_epochs):
        train_loss, val_loss = train_epoch()

        if epoch % 50 == 0:
            print("----------")
            print(f"At epoch: {epoch}")
            print("Train loss: {:.2f}".format(train_loss.item()))
            print("val loss: {:.2f}".format(val_loss.item()))
            print("----------")

    # R2 of prediction
    y_train, x_train, _, _ = get_batches(df_diabete_std, size=128)
    x_train = sparse(x_train)
    r2_finish = r_squared(x_train, y_train)

    print("Initial R-Squared: {:.2f}".format(r2_init))
    print("Finish R-Squared: {:.2f}".format(r2_finish))
