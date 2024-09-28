# This is mostly the same as training, except with a lower learning rate. I also trian on my own writings.


import tiktoken
import os
import mlx.data as dx
import mlx.core as mx
import os
from datasets import Dataset
from datasets import load_from_disk
from gpt_layer import GPT
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn.losses import cross_entropy
from functools import partial


essay_file_path = "./TrainingLLMFromScratchMLX/combined_essays.txt"

with open(essay_file_path, 'r') as essay_file:
    essay_string = essay_file.read()
    # print(essay_string)


encoding = tiktoken.get_encoding("cl100k_base")
tokens = encoding.encode(essay_string, allowed_special="all")

print("Number of tokens")
print(len(tokens))

def pad_length(list_of_tokens, block_size):
    offset = len(list_of_tokens) % block_size
    if offset != 0:
        padding = block_size - offset
        list_of_tokens.extend([220] * padding)
    return list_of_tokens

batch_size = 12
block_size: int = 512 # Aka sequence length, what is fed to one attention head
vocab_size: int = 100608

tokens = pad_length(tokens, block_size)
assert len(tokens) % block_size == 0



number_of_sequences = len(tokens) // block_size

targets = tokens
tokens = mx.array(tokens)
del targets[0]
targets.append(220)
targets = mx.array(targets)

idx = mx.reshape(tokens, (number_of_sequences, block_size))
target = mx.reshape(targets, (number_of_sequences, block_size))

# Assuming idx and target are numpy arrays
# Stack idx and target along a new axis
combined = mx.stack((idx, target), axis=-1)

# Initialize lists for validation and training samples

data_dict = {
    "sequence": combined[:, :, 0].tolist(),  # idx data
    "target": combined[:, :, 1].tolist()     # target data
}

dataset = Dataset.from_dict(data_dict)

dataset = dataset.shuffle(seed=42)

# split = dataset.train_test_split(test_size=0.1)




# Apply batching and format to dataset
batched_dataset = dataset.batch(batch_size=batch_size)
dataset_size = len(batched_dataset)
print(dataset_size)

batch_size = 12
block_size: int = 1024 # Aka sequence length, what is fed to one attention head
vocab_size: int = 100608
n_head: int = 8
n_embd: int = 256
n_layer: int = 4
dropout: float = 0.2
bias: bool = False
max_iters = 880 # 80 epochs

model = GPT(
    dims=n_embd,
    num_heads=n_head,
    vocab_size=vocab_size,
    block_size=block_size,
    num_layers=n_layer,
    dropout=dropout,
)
model.set_dtype(dtype=mx.bfloat16)
# mx.eval(model.parameters())

model.load_weights("./TrainingLLMFromScratchMLX/test_safetensors/plato_v1_step_24000.safetensors")



def compute_loss(idx, targets):
    return nn.losses.cross_entropy(model(idx), targets, axis=-1, reduction="mean")

loss_and_grad_fn = nn.value_and_grad(model, compute_loss)

# Learning rate kept constant, during pretraining it decayed from 6e-4 to 6e-5
learning_rate = 1e-6
warmup_iters = 220


warmup = optim.linear_schedule(0, learning_rate, warmup_iters)
scheduler = optim.join_schedules([warmup, learning_rate], [warmup_iters])

optimizer = optim.AdamW(learning_rate=learning_rate)


state = [model.state, optimizer.state]

@partial(mx.compile, inputs=state, outputs=state)
def step(idx, targets):
    loss_and_grad_fn = nn.value_and_grad(model, compute_loss)
    loss, grads = loss_and_grad_fn(idx, targets)
    optimizer.update(model, grads)
    return loss

for s in range(max_iters + 1):
    i = s % dataset_size

    batch = batched_dataset[i]
    #print(f"Beginning step {s}")
    idx = mx.array(batch["sequence"])
    targets = mx.array(batch["target"])

    # Logits are shape (12, 512, 100608)
    loss = step(idx, targets)

    # print(loss)
    mx.eval(state)

    if s % 220 == 0:
        save_dir = "./TrainingLLMFromScratchMLX/test_safetensors"
        
        weights_file = os.path.join(save_dir, f"finetuned_plato_v1_step_{s}.safetensors")
        
        model.save_weights(weights_file)

        print(f"Training loss: {loss}")

        input_tokens = encoding.encode("I will discuss Hegel’s descriptive conception ", allowed_special="all")

        idx = mx.array(input_tokens)
        idx = mx.reshape(idx, (1, len(idx)))
        generated_sequence = model.generate(idx, max_new_tokens=block_size)
        generated_sequence = generated_sequence[0]
        print(generated_sequence)
        output = generated_sequence.tolist()

        for idx, token_id in enumerate(output):
            if token_id >= 100257:
                print(f"Invalid token ID found: {token_id}")
                output[idx] = 100257
        print(encoding.decode(output, errors="replace"))

