# RoMa v2 refiners, without DINOv3

`romav2_refiners.pt` (release `v2.0.1-refiners`) holds only the refiners
of RoMa v2 and their VGG19-BN features, taken from the authors' weights
`romav2.0.1.pt`. It contains **no DINOv3 weights** and no part of RoMa
v2's coarse matcher.

- Source: <https://github.com/Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt>
  (1,095,883,548 bytes, SHA-256
  `1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7`).
- Kept: the 323 tensors under `refiners.*` (ConvRefiners at 1/4, 1/2 and
  full resolution) and `refiner_features.*` (VGG19-BN), 5.1 M parameters,
  float32 (and BatchNorm's integer counters), saved with `torch.save`.
- Dropped: `f.*` (DINOv3 ViT-L/16, 303 M parameters) and `matcher.*`
  (the coarse matcher, 117 M parameters).
- Result: 20,485,732 bytes, SHA-256
  `aee6a6014b89c06e2ddc628c7cdeb6541a44ceb8904b414b2396090baa3cc977`.
- Made with `extract_refiners.py` (in this repository); the output must
  be named `romav2_refiners.pt` to reproduce the checksum (`torch.save`
  stores the file name in the archive).

The refiners take a coarse warp and refine it from VGG features of both
images, so they can follow a coarse matcher other than RoMa v2's own
(for instance RoMa v1's, on DINOv2).

## Licence and credit

RoMa v2 is by Johan Edstedt and co-authors, published under the MIT
License (see `LICENSE`, copied from
<https://github.com/Parskatt/RoMaV2>); these weights are a subset of
their release and keep that licence and notice. `extract_refiners.py` is
released into the public domain (Unlicense).

Edstedt, J., Nordström, D., Zhang, Y., Bökman, G., Astermark, J.,
Larsson, V., Heyden, A., Kahl, F., Wadenbäck, M. and Felsberg, M.:
RoMa v2: Harder Better Faster Denser Feature Matching. arXiv:2511.15706.
