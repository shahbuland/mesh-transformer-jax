from mesh_transformer.sampling import nucleaus_sample
import optax
import jax

# the goal of this fork is to be able to turn the mesh transformer model into an encoder
# i.e. make it skip the final projection layer and just return its hidden states over the inputted tokens

# small test model
params = {
    "layers": 4,
    "d_model": 256,
    "n_heads": 16,
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
import numpy as np

# set up devices and resource env
cores_per_replica = 8
tpu_size = jax.device_count()
mesh_shape = (tpu_size // cores_per_replica, cores_per_replica)
devices = np.array(jax.devices()).reshape(mesh_shape)

with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
    network = CausalTransformer(params)
    bs = 16
    tokens = np.random.randint(0, high=params["n_vocab"], size = (bs, params["seq"]))
    enc = network.encode(tokens, 128 * np.ones(bs))

