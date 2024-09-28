# /Users/blakewhitmer/TrainingLLMFromScratchMLX/test_safetensors/model_weights_step_500.safetensors
# The first model I ever trained.
# Not very good. I didn't even use learning rate decay.
# Doesn't actually work because I changed my hyperaparameters
batch_size = 12
block_size: int = 512 # Aka sequence length, what is fed to one attention head
vocab_size: int = 100608
n_head: int = 8
n_embd: int = 128
n_layer: int = 6
dropout: float = 0.0
bias: bool = False
learning_rate = 6e-4
max_iters = 501

# /Users/blakewhitmer/TrainingLLMFromScratchMLX/test_safetensors/test_chinchilla_parameters_weights_step_9010.safetensors
# Trained for about an hour, according to compute optimal chinchilla parameters
# Got through 110714880 of 73584077 tokens, 1.5 epochs.
# Got through 9010 steps. The dataset had 11977 batches. 75% of my dataset.
# Didn't calculate validation loss. Just stopped training based on chinchilla parameters.

batch_size = 12
block_size: int = 1024 # Aka sequence length, what is fed to one attention head
vocab_size: int = 100608
n_head: int = 8
n_embd: int = 256
n_layer: int = 4
dropout: float = 0.0
bias: bool = False
max_iters = 600000

# /Users/blakewhitmer/TrainingLLMFromScratchMLX/test_safetensors/chinchilla_parameters_weights_step_8010.safetensors
# No train/test split.
# No validation loss. I know that's taboo, but I wanted to train on all of this dataset.
# I would be fine if this model overfit. After all, I'm training on texts that people spend their lives trying to memorize...
# Besides, according to Chinchilla parameters, this should underfit, anyways.
# Training loss: array(3.85938, dtype=bfloat16)
# Trained overnight. This should be way past the compute optimal fronteir, but I haven't calcualted the actual number of flops it trained for.
# There was a bug where I couldn't see the number of steps.
# And there was no way my training would reach 600000.

batch_size = 12
block_size: int = 1024 # Aka sequence length, what is fed to one attention head
vocab_size: int = 100608
n_head: int = 8
n_embd: int = 256
n_layer: int = 4
dropout: float = 0.0
bias: bool = False
max_iters = 600000 

learning_rate = 6e-4
weight_decay = 1e-1
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# One more try. I shortened the training to 5 epochs.
# Plato v1.
# /Users/blakewhitmer/TrainingLLMFromScratchMLX/test_safetensors/plato_v1_val_loss_step_11000.safetensors
# End training loss 4.125
# git lfs untrack "test_safetensors/finetuned_plato_v1_step_880.safetensors"
git rm --cached "test_safetensors/finetuned_plato_v1_step_880.safetensors"
batch_size = 12
block_size: int = 1024 # Aka sequence length, what is fed to one attention head
vocab_size: int = 100608
n_head: int = 8
n_embd: int = 256
n_layer: int = 4
dropout: float = 0.0
bias: bool = False
max_iters = 60000

learning_rate = 6e-4
weight_decay = 1e-1
warmup_iters = 2000
lr_decay_iters = 60000
min_lr = 6e-5