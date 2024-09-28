# Before I ran trained a model on the entire Western Canon, I trained a model on just the KJV Bible.
# I didn't calculate any hyperparameters for this.
# I am not including that in the repository because the model was 500 mb.

import tiktoken
import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.data as dx
from mlx.nn.losses import cross_entropy
from gpt_layer import GPT
import os
from functools import partial

kjv_file_path = "./PlatoV1/PlatoV1Dataset/JustBooks/Bibles/kjv.txt"

eod_token_id = 100257

with open(kjv_file_path, 'r') as kjv_file:
    kjv_string = kjv_file.read()
    #print(kjv_string)


encoding = tiktoken.get_encoding("cl100k_base")

kjv_tokens = encoding.encode(kjv_string)
kjv_tokens.append(eod_token_id)
#print(kjv_tokens)
print("Number of tokens")
print(len(kjv_tokens))
#print(encoding.decode(kjv_tokens))

def pad_length(list_of_tokens, block_size):
    offset = len(list_of_tokens) % block_size
    if offset != 0:
        padding = block_size - offset
        list_of_tokens.extend([220] * padding)
    return list_of_tokens

batch_size = 12
block_size: int = 512 # Aka sequence length, what is fed to one attention head
vocab_size: int = 100608
n_head: int = 8
n_embd: int = 128
n_layer: int = 6
dropout: float = 0.0
bias: bool = False
learning_rate = 6e-4
max_iters = 501 # Idk exact, maybe 60,000? I need to calculate compute optimal.


kjv_tokens = pad_length(kjv_tokens, block_size)
assert len(kjv_tokens) % block_size == 0

number_of_sequences = len(kjv_tokens) // block_size
print("Number of sequences")
print(number_of_sequences)

kjv_offset = kjv_tokens
kjv_array = mx.array(kjv_tokens)
del kjv_offset[0]
kjv_offset.append(220)
kjv_array_offset = mx.array(kjv_offset)

# The kjv_array_offset should be kjv_array shifted by one token
# We'll use this to calculate the cross entropy loss in next token prediction
print(len(kjv_array))
print(len(kjv_array_offset))
print(kjv_array[0:15])
print(kjv_array_offset[0:15])

idx = mx.reshape(kjv_array, (number_of_sequences, block_size))
target = mx.reshape(kjv_array_offset, (number_of_sequences, block_size))

train_samples = []
val_samples = []

for i in range(idx.shape[0]):
    sample = {"sequence": idx[i], "target": target[i]}
    # I need to add the target sequence here.
    if i % 10 == 0:
        val_samples.append(sample)
    else:
        train_samples.append(sample)

train_dset = dx.buffer_from_vector(train_samples)
val_dset = dx.buffer_from_vector(val_samples)

train_dset = (
    train_dset
    .shuffle()
    .to_stream()
    .batch(batch_size)
    .prefetch(8, 4)
    .repeat(10000)
)
val_dset = (
    val_dset
    .shuffle()
    .to_stream()
    .batch(batch_size)
    .prefetch(8, 4)
    .repeat(10000)
)

model = GPT(
    dims=n_embd,
    num_heads=n_head,
    vocab_size=vocab_size,
    block_size=block_size,
    num_layers=n_layer,
    dropout=dropout,
)
model.set_dtype(dtype=mx.bfloat16)
mx.eval(model.parameters())

# print(model)


# Get a function which gives the loss and gradient of the
# loss with respect to the model's trainable parameters
def compute_loss(idx, targets):
    return nn.losses.cross_entropy(model(idx), targets, axis=-1, reduction="mean")

loss_and_grad_fn = nn.value_and_grad(model, compute_loss)

optimizer = optim.AdamW(learning_rate=learning_rate)

state = [model.state, optimizer.state]

@partial(mx.compile, inputs=state, outputs=state)
def step(idx, targets):
    loss_and_grad_fn = nn.value_and_grad(model, compute_loss)
    loss, grads = loss_and_grad_fn(idx, targets)
    optimizer.update(model, grads)
    return loss

for s in range(max_iters):
    batch = next(train_dset)
    print(batch["sequence"].shape)
    #print("SEQUENCE\n\n")
    #print(encoding.decode(batch["sequence"][1]))
    #print("TARGET\n\n")
    #print(encoding.decode(batch["target"][1]))
    idx = mx.array(batch["sequence"])
    targets = mx.array(batch["target"])

    # Logits are shape (12, 512, 100608)
    loss = step(idx, targets)
    # print(targets.shape)
    # print(loss)
    mx.eval(state)

    if s % 9 == 0:
        print(f"Active Memory: {mx.metal.get_active_memory() / (1024 * 1024 * 1024):.2f} GB")
        print(f"Cache Memory: {mx.metal.get_cache_memory() / (1024 * 1024 * 1024):.2f} GB")
        print(f"Peak Memory: {mx.metal.get_peak_memory() / (1024 * 1024 * 1024):.2f} GB")

        val_batch = next(val_dset)
        val_idx = mx.array(val_batch["sequence"])
        val_targets = mx.array(val_batch["target"])

        val_loss, _ = loss_and_grad_fn(val_idx, val_targets)
        print(f"Validation Loss at step {s}: {val_loss}")
        print(f"Training Loss at step {s}: {loss}")

    if s == 100 or s % 500 == 10:
        save_dir = "./TrainingLLMFromScratchMLX/test_safetensors"
        
        weights_file = os.path.join(save_dir, f"model_weights_step_{s}.safetensors")
        
        model.save_weights(weights_file)




# Learning rate decay + warmup cycles
# Calculate Chinchilla optimal parameters. I could speed this up by lowering block size.
# How do I resume training? Keep track of step and my palce in the dataset. Checkpointing.
# Then I just need to scale up the data.
# Maybe make things look a little bit nicer.
# caffeinate -i