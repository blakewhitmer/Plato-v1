import tiktoken
import mlx.core as mx
from gpt_layer import GPT

batch_size = 12
block_size: int = 1024 # Aka sequence length, what is fed to one attention head
vocab_size: int = 100608
n_head: int = 8
n_embd: int = 256
n_layer: int = 4
dropout: float = 0.0
bias: bool = False
max_iters = 600000

model = GPT(
    dims=n_embd,
    num_heads=n_head,
    vocab_size=vocab_size,
    block_size=block_size,
    num_layers=n_layer,
    dropout=dropout,
)
model.set_dtype(dtype=mx.bfloat16)

def pad_length(list_of_tokens, block_size):
    offset = len(list_of_tokens) % block_size
    if offset != 0:
        padding = block_size - offset
        list_of_tokens.extend([220] * padding)
    return list_of_tokens


model.load_weights("./test_safetensors/finetuned_plato_v1_step_880.safetensors")

# print(model)
# print(model.parameters())

encoding = tiktoken.get_encoding("cl100k_base")

input_tokens = encoding.encode("Genesis 1:1 In the beginning, God")
# print(input_tokens)
idx = mx.array(input_tokens)
idx = mx.reshape(idx, (1, len(idx)))
generated_sequence = model.generate(idx, max_new_tokens=500)
generated_sequence = generated_sequence[0]
# print(generated_sequence)
output = generated_sequence.tolist()
# print(output)
print(encoding.decode(output))
print("--------------------------------------------")

input_tokens = encoding.encode("I will discuss Hegel’s descriptive conception")
# print(input_tokens)
idx = mx.array(input_tokens)
idx = mx.reshape(idx, (1, len(idx)))
generated_sequence = model.generate(idx, max_new_tokens=500)
generated_sequence = generated_sequence[0]
# print(generated_sequence)
output = generated_sequence.tolist()
# print(output)
print(encoding.decode(output))
print("--------------------------------------------")

input_tokens = encoding.encode("Amie Thomasson said:")
# print(input_tokens)
idx = mx.array(input_tokens)
idx = mx.reshape(idx, (1, len(idx)))
generated_sequence = model.generate(idx, max_new_tokens=500)
generated_sequence = generated_sequence[0]
# print(generated_sequence)
output = generated_sequence.tolist()
# print(output)
print(encoding.decode(output))

input_tokens = encoding.encode("Socrates: What is Justice?")
# print(input_tokens)
idx = mx.array(input_tokens)
idx = mx.reshape(idx, (1, len(idx)))
generated_sequence = model.generate(idx, max_new_tokens=500)
generated_sequence = generated_sequence[0]
# print(generated_sequence)
output = generated_sequence.tolist()
# print(output)
print(encoding.decode(output))