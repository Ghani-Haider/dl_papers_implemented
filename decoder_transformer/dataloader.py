import torch

# ================================== #
###     LOAD DATA AND SPLIT     ###
# ================================== #

def encode(text, cti):
    return [cti[i] for i in text]

def decode(lst, itc):
    return ''.join([itc[i] for i in lst])

def dataloader(file):
    # read data
    dataset = file
    with open(dataset, 'r') as f:
        text = f.read()

    # possible characters the model can see and emit
    chars = sorted(list(set(text)))
    # number of characters
    vocab_size = len(chars)
    # tokenizer
    cti = {c:i for i, c in enumerate(chars)}
    itc = {i:c for i, c in enumerate(chars)}

    # tokenize entire dataset
    data = torch.tensor(encode(text, cti))

    # train validate split
    split_pos = int(len(data)*0.9) # 90% for training, 10% for validation
    train_data = data[:split_pos]
    val_data = data[split_pos:]

    return vocab_size, train_data, val_data

# get a batch of training data
def get_batch(split, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, DEVICE):
    # get data
    data = train_data if split == 'train' else val_data
    # get batch
    random_offsets = torch.randint(low=0, high=len(data)-BLOCK_SIZE, size=[BATCH_SIZE])
    x = torch.stack([data[i: i+BLOCK_SIZE] for i in random_offsets])
    y = torch.stack([data[i+1: i+1+BLOCK_SIZE] for i in random_offsets])
    return x.to(DEVICE), y.to(DEVICE)