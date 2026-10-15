# SPDX-License-Identifier: Unlicense
"""Extract RoMa v2's refiners from romav2.0.1.pt, without DINOv3.

    python extract_refiners.py romav2.0.1.pt romav2_refiners.pt

Keeps the tensors under refiners.* and refiner_features.* (the ConvRefiners
at 1/4, 1/2 and 1/1 and their VGG19-BN features); drops f.* (the DINOv3
ViT-L/16 backbone) and matcher.* (the coarse matcher, trained on DINOv3
features).
"""

import sys

import torch

src, dst = sys.argv[1:3]
sd = torch.load(src, map_location="cpu", weights_only=True)
keep = {
    k: v.contiguous().clone()
    for k, v in sd.items()
    if k.startswith(("refiners.", "refiner_features."))
}
torch.save(keep, dst)
print(f"{len(keep)} of {len(sd)} tensors kept in {dst}")
