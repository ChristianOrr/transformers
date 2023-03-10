import jax
from jax import random
import jax.numpy as jnp
import optax
from functools import partial



def get_batch(data, rng_key, batch_size, block_size):
    """
    Extracts a random batch of input and target data
    Args:
        data: An array of all the data's token ID's.
        rng_key: Random number generator key.
        batch_size: Number of parallel batches.
        block_size: Maximum time length for the token sequence.
    Returns:
        Input token ID's and target token ID's.
    """
    ix = random.randint(key=rng_key, shape=(batch_size, ), minval=0, maxval=len(data) - block_size)
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


@partial(jax.jit, static_argnames=['forward_fn', 'vocab_size', 'batch_size', 'block_size', 'max_new_tokens'])
def generate(variables, forward_fn, index_seq, rng_key, vocab_size, batch_size, block_size, max_new_tokens):
    """
    Generates max_new_tokens number of new tokens, 
    given the starting sequence of tokens index_seq 
    Args:
        variables: Bigram models parameters.
        forward_fn: Function that performs a forward pass of the model.
        index_seq: Array of token indices with shape (B, T), 
            where B is the batch size and T is the time steps.
        rng_key: Random number generator key.
        vocab_size: Number of independent tokens in the vocabulary.
        batch_size: Number of parallel batches.
        max_new_tokens: Maximum number of new tokens to generate
    Returns:
        An array of generated indices
    """
    # Batched sampling function
    batched_choice = jax.vmap(jax.random.choice)
    
    for _ in range(max_new_tokens):
        # Crop index_seq to the last block_size tokens
        index_cond = index_seq[:, -block_size:]
        logits = forward_fn(variables, index_cond)
        # Focus only on the last time step
        # Shape changes from (B, T, C) -> (B, C)
        logits = logits[:, -1, :]
        # Convert to probabilities using softmax
        probs = jax.nn.softmax(logits, axis=-1)
        # Sample a token index using probs
        rng_key, subkey = jax.random.split(rng_key)
        batched_key = subkey.reshape(1, -1)
        batched_key = jnp.repeat(batched_key, batch_size, axis=0)
        a = jnp.arange(vocab_size).reshape(1, -1)
        a = jnp.repeat(a, batch_size, axis=0)
        next_indexes = batched_choice(batched_key, a, p=probs)
        next_indexes = next_indexes.reshape(batch_size, -1)
        # Append the sampled index to the running sequence
        index_seq = jnp.concatenate([index_seq, next_indexes], axis=1)
    return index_seq


def masked_fill(mask, a, fill):
    return jax.lax.select(mask, a, jax.lax.broadcast(fill, a.shape))


def loss_fn(variables, forward_fn, index_seq, labels):
    """
    Calculates the cross entropy loss of 
    all batches and time steps, 
    then returns the mean.
    Args:
        variables: Language model parameters.
        forward_fn: Function that performs a forward pass of the model.
        index_seq: Array of token indices with shape (B, T), 
            where B is the batch size and T is the time steps.
        labels: Indexes of the next token in the sequence.
    Returns:
        Cross entropy loss
    """
    logits = forward_fn(variables, index_seq)

    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    # Average loss across all batches and time steps
    loss = loss.mean()
    return loss
