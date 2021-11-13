from mesh_transformer.sampling import nucleaus_sample
import optax

# the goal of this fork is to be able to turn the mesh transformer model into an encoder
# i.e. make it skip the final projection layer and just return its hidden states over the inputted tokens

# small test model
params = {
    "layers": 4,
    "d_model": 256,
    "n_heads": 4,
    "n_vocab": 512,
    "norm": "layernorm",
    "pe": "rotary",
    "pe_rotary_dims": 64,

    "seq": 128,
    "cores_per_replica": 8,
    "per_replica_batch": 1,
    "sampler":nucleaus_sample,
    "optimizer":optax.scale(0)
}

from mesh_transformer.transformer_shard import CausalTransformer
network = CausalTransformer(params)

import numpy as np

tokens = np.random.randint(0, high=params["n_vocab"], size = (1, params["seq"]))

top_p = 0.9
temp = 1.0
output = network.generate(batched_tokens, 1, gen_len,
        {"top_p": np.ones(total_batch) * top_p, "temp": np.ones(total_batch) * temp})

# TODO: Add an encode method to the network that returns all hidden states corresponding to 
# seqeunce (i.e. give output without giving projected)
