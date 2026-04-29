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
- DEIS: sampler interface and the `rho_ab` exponential-integrator update follow
  the official `qsh-zh/deis` PyTorch implementation. The local implementation
  reimplements the official VPSDE coefficient construction in pure PyTorch so
  it does not require JAX at evaluation time. A `t_ab` implementation is kept
  in the sampler module for parity/debugging, but `rho_ab` is the default
  because it is more stable for this CIFAR cosine-VP setup.
- OpenAI guided-diffusion ImageNet64 adapter: `src/dm/third_party/guided_diffusion`
  vendors the minimal MIT-licensed official modules needed to instantiate
  `64x64_diffusion.pt` for external validation smoke tests.
- U-ViT ImageNet64 adapter: `src/dm/third_party/uvit_official` vendors the
  minimal MIT-licensed official `baofff/U-ViT` modules needed to instantiate
  `imagenet64_uvit_large.pth` for external validation smoke tests.

Before using this scaffold for paper-grade numbers, replace the remaining
compact sampler kernels with line-by-line vendored official implementations or
add numerical parity tests against those repositories.
