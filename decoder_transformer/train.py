import torch
from dataloader import dataloader, decode, get_batch
from decoder_transformer import DecoderWithoutCrossAttention, DEVICE, BLOCK_SIZE, BATCH_SIZE

# ================================== #
###     TRAINING SECTION     ###
# ================================== #

MAX_ITERS = 5000 # max time for training loop
EVAL_INTERVALS = 500 # interval for training
EVAL_ITERS = 200 # for batch loss aggregation

# load the data
vocab_size, train_data, val_data = dataloader("data/tiny_shakespeare_data.txt")

# estimate loss over eval_iter
def estimate_loss(model):
    result = {}
    model.eval()
    with torch.inference_mode():
        for split in ['train', 'val']:
            losses = 0
            for i in range(EVAL_ITERS):
                xb, yb = get_batch(split, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
                logits, loss = model(xb, yb)
                losses += loss.item()
            result[split] = losses / EVAL_ITERS
    model.train()
    return result

# create model instance and optimizer
model = DecoderWithoutCrossAttention(vocab_size).to(DEVICE)
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

### training the decoder ###
# training loop
for iter in range(MAX_ITERS):
    # print losses
    if iter % EVAL_INTERVALS == 0:
        eval_loss = estimate_loss(model)
        print(f"step {iter}:\t train loss {eval_loss['train']:.4f} | val loss {eval_loss['val']:.4f}")
        pass
    
    # sample batches
    xb, yb = get_batch('train', train_data, val_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
    
    # eval loss and update
    logits, loss = model(xb ,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

### save model and generate characters ###
# torch.save(model.state_dict(), 'model_multi_head.pth')

# set context (a ' ' character)
context_empty = torch.zeros((1,1), dtype=torch.long, device=DEVICE)
print(f"\n***Model Generated Text***\n")
# generate a given sequence of characters (250 max tokens)
print(decode(model.generate(context_empty, 250)[0].tolist()))