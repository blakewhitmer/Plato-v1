import tiktoken
import os
import mlx.data as dx
import mlx.core as mx
import os
from datasets import Dataset
from datasets import load_dataset
from datasets import load_from_disk
from gpt_layer import GPT
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn.losses import cross_entropy
from functools import partial
from mlx.utils import tree_flatten

batched_dataset = load_from_disk("./TrainingLLMFromScratchMLX/dataset")
# split = batched_dataset.train_test_split(test_size=0.1)
# I opted against calculating any validation loss. This was a tough decision to make.
# Ideally, this LLM would understand everything in its dataset.
# I don't want to leave out 10% of the Western Canon, I want to train this LLM on the entire Western Canon.
# I'll need to do more research to figure out how to give LLMs "canonical data", as in, datasets where every line is important.
# Regradless, this model is not meant to be performant.
encoding = tiktoken.get_encoding("cl100k_base")
# test_size = len(split["train"])
# train_size = len(split["test"])
dataset_size = len(batched_dataset)
print(dataset_size)

batch_size = 12
block_size: int = 1024 # Aka sequence length, what is fed to one attention head
vocab_size: int = 100608
n_head: int = 8
n_embd: int = 256
n_layer: int = 4
dropout: float = 0.0
bias: bool = False
max_iters = 24000 # About 2 epochs

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

"""
print(len(tree_flatten(model.parameters())))
total_parameters = 0
for i in tree_flatten(model.trainable_parameters()):
    if i[0] in ["wte.weight", "wpe.weight", "lm_head.weight"]:
        # By Kaplan et al, exclude all vocabulary and positional embeddings
        continue
    total_parameters += i[1].size
    # print(i[0])
    # print(i[1].size)
print(f"{total_parameters:.2e}")
# About 3.15e+06. Target is 3.5 mil, this is fine
"""

def compute_loss(idx, targets):
    return nn.losses.cross_entropy(model(idx), targets, axis=-1, reduction="mean")

loss_and_grad_fn = nn.value_and_grad(model, compute_loss)

# Taken from Karpathy
learning_rate = 6e-4
weight_decay = 1e-1
warmup_iters = 2000
lr_decay_iters = 22000
min_lr = 6e-5


warmup = optim.linear_schedule(0, learning_rate, warmup_iters)
decay = optim.cosine_decay(learning_rate, lr_decay_iters, min_lr)
scheduler = optim.join_schedules([warmup, decay], [warmup_iters])

optimizer = optim.AdamW(learning_rate=scheduler)


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

    if s == 10 or s == 100 or s == 500 or s % 1000 == 0:
        save_dir = "./TrainingLLMFromScratchMLX/test_safetensors"
        
        weights_file = os.path.join(save_dir, f"plato_v1_step_{s}.safetensors")
        
        model.save_weights(weights_file)

        print(f"Training loss: {loss}")

        input_tokens = encoding.encode("Socrates: What is Justice?", allowed_special="all")

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

