# Transformers
In this repository we will be developing all the components required for a large language model (LLM) from scratch using jax. The primary focus is on understanding the model architecture, so we will scale back on training and inferencing to save on compute. The largest model, GPT-2, requires a distributed cluster of GPUs to train, which will add a lot of overhead in the code and cost for renting the cluster. The notebooks will loosely follow Andrej Karpathy's Pytorch implementation in his video [2].

## Implementation Details
We will use the "tiny shakespeare" dataset for training to save on compute and storage. Jax and Flax are used for creating all the models. The tokenizer is a simple single character tokenizer with 65 characters in its vocabulary.

## Models
The models start from the simplest to the most complicated. As you progress through the notebooks, more components of the transformer will be developed, until we can create an LLM transformer model. Then we will combine all the components and scale them up to create GPT-2. 

### Bigram
The bigram model is the simplest autoregressive language model. It isn't a transformer, but it contains important parts of the transformer, such as the token embedding. This makes it ideal for demonstrating how to to develop the minimum infrastructure needed to develop an autoregressive text prediction model. Most of this infrastructure will then be used in the later notebooks for the transformer models.

### Scaled Self-Attention
![Scaled Self-Attention](./images/scaled_self-attention.png)

This is the simplest component of the transformer model which still contains its most imporatant part, the attention mechanism. This notebook demonstrates how the attention mechanism works by using the most fundamental components, the query, key, and value (Q, K, V). They interact with each other following the process shown in the image above. 

### Multi-Headed Self-Attention
![Multi-Head Attention](./images/multi-head-attention.png)

Multi-headed self-attention is a block containing multiple scaled self-attention models in parallel, which are then concatenated and processed through a dense layer, as shown in the image above. Its main advantage over the single-headed self-attention is that it can capture information from different representation subspaces. This enables it to capture more diverse and complex subspaces in the input space.

### Transformer Decoder
![Attention Transformer Block](./images/attention_transformer_block.png)

The transformer decoder combines all the tools demonstrated in the previous notebooks with other innovations in deep learning, like the skip connections [4], and layer normalization [5], into a single block. This block is repeatable and scalable, making it a vital component of large language models. The decoder block refers to the right block in the image above. We don't need the encoder block for our example problem. 

### GPT-2
GPT-2 [6] is a scaled up version of the transformer decoder. It uses multiple decoder block layers, larger embedding size, larger block size, more heads in the multi-headed attention block, and a larger vocabulary size. The vocabulary size is 50257 tokens, but for our example we will stick with the simple 65 token vocabulary. It comes in multiple sized versions, small, medium, large and XL, to accomadate for the varying compute requirements of the user.


## Installation Requirements:
If you have a GPU you can install Jax by running the following first:
```
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
All the other requirements are provided below:
```
pip install --upgrade git+https://github.com/google/flax.git
pip install optax
pip install numpy
pip install jupyter

```

### References:
- [1] [GPT colab notebook](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)
- [2] [Video: Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy)
- [3] [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [4] [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [5] [Layer Normalization](https://arxiv.org/abs/1607.06450)
- [6] [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- [7] [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580)