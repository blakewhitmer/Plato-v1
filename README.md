Note for Github!
The model itself was over the git lfs limit of 100 mb.



This is a small langauge model trained from scratch. It's about 3 million parameters. It was pretrained on the entire Western Canon--or, at least, 70 million tokens of the Western Canon. I finetuned it on the essays I wrote in college.

I will add the dataset to Hugging Face. It is still a work in progress, but I believe it represents the best open source repository of its kind.



To set up the environment, ensure you have Conda installed. Then, run the following command to create the environment with all necessary dependencies:

conda env create -f environment.yml

To test the model, just run sample.py. The weights are in test_safetensors/finetuned_plato_v1_step_880.safetensors. For the GPT module, go to gpt_layer.py. To see the code I used to run the model, go to train_on_canon.py.

You can skip trimmingcanon.py. It is a collection of scripts I used to remove noise from the datset.

The dataset for this model is on Hugging Face.

https://huggingface.co/datasets/wordgrammer/The_Entire_Western_Canon

This model is not very good. A 3 million parameter model is not large enough to understand philosophy. In fact, most of the frontier models struggle to understand philosophy. I made this primarily as a way to understand the inner workings of a transformer from first principles. I would only recommend building a language model from scratch for this reason. It is much more cost-effective to finetune an open source model. But as an educational experience, training a model from scratch is invaluable.

I will not update this repository further.


Many thanks to the following resources:
- The math behind attention. https://x-dev.pages.jsc.fz-juelich.de/2022/07/13/transformers-matmul.html
- Useful diagram for the parts of a Transformer beyond attention: https://en.m.wikipedia.org/wiki/Generative_pre-trained_transformer
- The Scaling Laws paper Kaplan et al: https://arxiv.org/pdf/2001.08361
- The Chinchilla paper to calculate model hyperparameters: https://arxiv.org/pdf/2203.15556
- Andrej Karpathy's nanoGPT repository, plus his various other tutorials. https://github.com/karpathy/nanoGPT
- The MLX team.
- Open Source ebook providers that wish to remain unnamed.
- I didn’t use this paper, but I wish I had it when I started this project: https://arxiv.org/pdf/2207.09238