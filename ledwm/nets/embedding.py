# %%
import jax.numpy as jnp
from jax import random
from ledwm.nets.Initializer import Initializer

# from . import nets
# import ninjax as nj
from ledwm import ninjax as nj


class Embedding(nj.Module):
    def __init__(self, num_embeddings, embedding_dim, winit="uniform"):
        self.num_embed = num_embeddings
        self.embed_dim = embedding_dim
        self.winit = winit

    def __call__(self, token_ids):
        return self.get(
            "embedding_matrix",
            Initializer(self.winit),
            (self.num_embed, self.embed_dim),
        )[token_ids]


# state = {}  # Assuming initial state is empty
# rng = nj.rng()


# def embed_tokens(
#     embed_layer,
#     token_ids,
# ):
#     return embed_layer.__call__(token_ids)


# token_ids = jnp.array([0, 23, 506, 102, 0, 0])  # Including padding IDs (0)

# embed_layer = Embedding(
#     num_embeddings=25,
#     embedding_dim=256,
#     padding_id=0,
#     winit="normal",
#     name="entity_embed",
# )

# _embed_tokens = nj.pure(embed_tokens)
# embeddings, new_state = _embed_tokens(state, rng, embed_layer, token_ids)
# print(embeddings)

# ------------

# import jax.numpy as jnp
# from jax import random


# def initialize_embedding(vocab_size, embedding_dim, key):
#     """Initialize the embedding table."""
#     return random.normal(key, (vocab_size, embedding_dim))


# def embedding_lookup(embedding_table, token_ids):
#     """Retrieve embeddings from the table based on token ids."""
#     return embedding_table[token_ids]


# # Parameters
# vocab_size = 10000  # Size of vocabulary including padding token
# embedding_dim = 300  # Size of each embedding vector
# key = random.PRNGKey(0)  # Random key for JAX

# # Initialize embedding table
# embedding_table = initialize_embedding(vocab_size, embedding_dim, key)

# # Example token IDs, with 0 as padding
# token_ids = jnp.array([0, 23, 506, 102, 0, 0])

# # Look up embeddings
# embeddings = embedding_lookup(embedding_table, token_ids)

# print(embeddings)
# print(embeddings.shape)  # Should be (number of tokens, embedding_dim)
