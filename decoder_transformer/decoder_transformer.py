import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(1337)

# hyperparameters
BATCH_SIZE = 64 # num of samples, independent sequences to process in parallel
BLOCK_SIZE = 256 # sample length, max context length for prediction
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_EMBD = 384 # number of embedding dimensions
N_HEADS = 6 # number of heads for the mutli-headed self-attention module in decoder blocks
N_BLOCKS = 6 # number of decoder blocks
DROPOUT = 0.2

# ================================== #
###     MODEL ARCHITECTURE     ###
# ================================== #

# multi-layer perceptron
class FFN(nn.Module):
    """a simple MLP"""

    def __init__(self, N_EMBD) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)

# one head of the self-attention
class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size) -> None:
        super().__init__()

        self.key_vector = nn.Linear(N_EMBD, head_size, bias=False) # (T, HdSz)
        self.query_vector = nn.Linear(N_EMBD, head_size, bias=False) # (T, HdSz)
        self.value_vector = nn.Linear(N_EMBD, head_size, bias=False) # (T, HdSz)
        # (T, T) lower triangular matrix of ones
        self.register_buffer('tril', torch.tril(torch.ones(size=(BLOCK_SIZE, BLOCK_SIZE))))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        # generating key, query and value for x (input)
        key = self.key_vector(x) # (B, T, HdSz)
        query = self.query_vector(x) # (B, T, HdSz)
        # HdSz for normalizing the attention scores
        # print(query.shape)
        B, C, HdSz = query.shape

        ### compute attention scores ("affinities")
        # wei is a matrix where each cell is dot product of a query vector & a key vector
        # rows contain tokens' queries and columns are tokens' key vectors
        #  |         a         |        b           |     c      ...
        # a| query(a) . key(a) | query(a) . key(b)  |  query(a) . key(c) ...
        # b| query(b) . key(a) | query(b) . key(b)  |  query(b) . key(c) ...
        # c| query(c) . key(a) | query(b) . key(b)  |  query(c) . key(c) ...
        # ... 
        wei = query @ key.transpose(-2, -1) * HdSz**-0.5 # (B, T, HdSz) @ (B, HdSz, T) --> (B, T, T)
        # setting future tokens scores/affinities to -inf
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        # apply softmax for smooth distribution
        wei = torch.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        value = self.value_vector(x) # (B, T, HdSz)
        output = wei @ value # (B, T, T) @ (B, T, HdSz) --> (B, T, HdSz)
        return output
    
# multiple heads of self-attention
class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(num_heads * head_size, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.projection(out))

class Block(nn.Module):
    """a block of transformer's decoder: communication followed by computation"""

    def __init__(self, n_embd, heads) -> None:
        # n_embd: embedding dimension of tokens, heads: number of self-attention heads we'd like
        super().__init__()
        head_size = n_embd // heads
        self.multi_heads = MultiHeadAttention(heads, head_size)
        self.ffwd = FFN(n_embd)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.multi_heads(self.layer_norm1(x)) # apply multi-head self-attention (B, T, N_EMBD) [N_EMBD == HdSz]
        return x + self.ffwd(self.layer_norm2(x)) # (B, T, N_EMBD)

# the decoder language model
class DecoderWithoutCrossAttention(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        # lookup table for character embedding vectors
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD) # (vocab_size, N_EMBD)
        # lookup table for position embedding vectors
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD) # (T, N_EMBD)
        # blocks of decoder stacked
        self.blocks = nn.Sequential(*[Block(N_EMBD, N_HEADS) for _ in range(N_BLOCKS)])
        self.layernorm = nn.LayerNorm(N_EMBD) # final layer norm
        # language model's head (output layer)
        self.lm_head = nn.Linear(in_features=N_EMBD, out_features=vocab_size) # (N_EMBD, vocab_size)

    def forward(self, idx, target=None):
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx) # (B, T, N_EMBD)
        pos_embeddings = self.position_embedding_table(torch.arange(start=0, end=T, device=DEVICE)) # (T, N_EMBD)
        x = token_embeddings + pos_embeddings # (B, T, N_EMBD)
        x = self.layernorm(self.blocks(x)) # (B, T, N_EMBD)
        logits = self.lm_head(x) # (B, T, N_EMBD) --> (B, T, vocab_size)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(input=logits, target=target)
        
        return logits, loss
    
    def generate(self, idx, max_token):
        # idx is (B, T) array of indices in the current context
        # generate max_token characters
        for _ in range(max_token):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -BLOCK_SIZE:]
            # get the predictions
            logits, loss = self(idx_cond) # (B, T, C)
            # pick the last character's embedding from each 
            # sample's prediction in the batch
            last_logits = logits[:,-1,:] # (B, C) -> (4, 65)
            # convert it to a prob distribution
            y_prob = F.softmax(input=last_logits, dim=1)
            # one sample from each prob distribution
            next_idx = torch.multinomial(input=y_prob, num_samples=1) # (B, 1)
            # concatenate the prediction with given input
            idx = torch.cat((idx, next_idx), dim=1) # (B, T+1)
        return idx
