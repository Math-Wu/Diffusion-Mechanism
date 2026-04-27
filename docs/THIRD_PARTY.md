# Third-Party References

This project implements a compact research scaffold for the sampler x
architecture CIFAR pilot. The first implementation is intentionally small and
inspectable, while following the public designs below.

- DiT architecture: inspired by `facebookresearch/DiT`, the official PyTorch
  implementation of "Scalable Diffusion Models with Transformers".
- U-ViT architecture: inspired by `baofff/U-ViT`, the official implementation
  of "All are Worth Words: A ViT Backbone for Diffusion Models".
- DPM-Solver++: sampler interface and naming follow `LuChengTHU/dpm-solver`
  (MIT licensed). The local implementation uses the shared VP ODE wrapper in
  this repository for unified NFE accounting.
- UniPC: sampler interface and naming follow `wl-zhao/UniPC` (MIT licensed).
  The local implementation uses the shared VP ODE wrapper in this repository
  for unified NFE accounting.
- DEIS: sampler interface and the `t_ab` exponential-integrator update follow
  the official `qsh-zh/deis` PyTorch implementation. The local implementation
  reimplements the official VPSDE coefficient construction in pure PyTorch so
  it does not require JAX at evaluation time.

Before using this scaffold for paper-grade numbers, replace the remaining
compact sampler kernels with line-by-line vendored official implementations or
add numerical parity tests against those repositories.
