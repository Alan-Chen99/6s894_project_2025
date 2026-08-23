# Random Hadamard transform prototypes

`rht16_triton.py` implements the tiled H16 operation used to prototype the
NVFP4 Wgrad dataflow:

`x -> fixed random sign -> normalized H16 -> block scale -> low-precision output`

The plain RHT emits the input dtype. The Hopper application prototype fuses a
per-row (16-element) amax/scale with native E4M3 output. It is not NVFP4:
H100 cannot execute native NVFP4, and E4M3 differs from NVFP4's E2M1 values and
hierarchical scale representation. Native NVFP4 results require SM100+.

The current comparison is against an eager, separate PyTorch pipeline and is a
fusion ablation, not a comparison with Transformer Engine. Before publication,
add Transformer Engine and torch.compile baselines, randomize order, repeat in
independent processes, and evaluate model-derived tensor distributions.
