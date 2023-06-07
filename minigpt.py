import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

class DummyModel(nn.Module):
    def __init__(self, vocab_sizes):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets=None):
        logits = self.embedding_layer(x)
        
        if targets is None:
            loss = None 
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss


# Estimate loss function
# For train and eval split (output a dict)
# Set model to eval and then back to train 
# Evaluate loss over multiple batch attempts
# model is global 
@torch.no_grad()
def estimate_loss():
    loss = {}
    splits = ["train", "val"]
    model.eval()
    for split in splits:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(split)
            _, l = model(x, y)
            losses.append(l)
        losses = torch.tensor(losses)
        loss[split] = torch.mean(losses).item()
        model.train()
    return loss 

# Build a head of self-attention model 
# Needs to have the key, query and value matrix
# Register a triangular 1 matrix as a buffer (to preserve sequential info)
# Standardize with softmax and use dropout layer on the wei matrix

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size)
        self.key = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        self.register_buffer("tril", torch.tril(torch.ones((block_size, block_size))))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        query = self.query(x)
        key = self.key(x)
        wei = query @ key.transpose(-2, -1) * C ** 0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, -1)
        wei = self.dropout(wei)
        value = self.value(x)
        out = wei @ value
        return out

# Create multi-head module 
# Has a list of heads layer
# Forward concatenate the heads and then dropout on (projection of x)
class MultiHead(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.dropout(self.proj(x))
        return x
    
# Implement two layers feedfoward with non linearity (relu in middle)
# + dropout at the hand (Sequential net)
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout()
        )

    def forward(self, x):
        x = self.net(x)

        return x

# Assemble the war beast
# The block
# head_size: head_size * n_head = n_embd
# The Multi-Head 
# The ff net
# Two normalization layer 
# The original embedding data is passed to along the
# multi-head/ffnet output 
class Block(nn.Module):
    def __init__(self, n_head, n_embd):
        super().__init__()
        head_size = n_embd // n_head 
        self.mh = MultiHead(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.mh(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


# BabyPt model
# one embedding table 
# one positional table 
# n_layer Blocks
# layer norm again
# Linear output layer

# For generation
# Logit of last time step
# Crop provided idx to block size
# Last step concat the predicted idx recursively
# So forward and generate methods

class BabyPT(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_head, n_embd) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.linear_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets=None):
        x = self.token_embedding(x)
        x = x + self.positional_embedding(torch.tensor([T for T in range(block_size)]))
        x = self.blocks(x)
        x = self.ln(x)
        x = self.linear_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = x.shape
            x = x.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(x, targets)

        return x, loss
    
    def generate(self, idx, max_token):
        for _ in range(max_token):
            idx = idx[:, -block_size:]
            x, _ = self.forward(idx)
            # Last time-step (depending of length of idx)
            time_step = (block_size - idx.shape[1] + 1)
            x = x[:, -time_step, :] 
            probs = torch.softmax(x, dim=-1)
            idx_new = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_new), dim=-1)

        return decode(idx.numpy().ravel())
    
# Gotta add a training routine now


if __name__ == "__main__":
    x, y = get_batch("train")
    x, y = x.to("cpu"), y.to("cpu")

    model = BabyPT(n_embd)
    y_hat = model.generate(torch.tensor([3]).reshape(1, 1), 10)

    print(y_hat)