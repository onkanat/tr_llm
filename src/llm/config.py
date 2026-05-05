# NanoGPT Configuration Stub for Kristal-Vektörel Mimarisi

# The core optimization: instead of a 50,257+ BPE vocab size,
# we use our deterministic morphological dictionary size.
vocab_size = 20500

# Other typical parameters can be smaller due to semantic density
n_layer = 12
n_head = 12
n_embd = 768 # Aligned with our vector DB default
dropout = 0.0

block_size = 1024 # Maximum sequence length of morphemes
