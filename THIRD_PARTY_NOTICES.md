# Third-party code

This repository contains code that its authors did not write. The vendored files are kept as they
were in the experiments of the paper, so that the released code computes what the experiments
computed. The only changes to vendored files are import paths and licence headers, listed below.
`ruff` does not lint or format `src/thinvessel/models/third_party/`.

## Vendored code (`src/thinvessel/models/third_party/`)

### Segment Anything (Meta), via MedSAM and torchtrainer — Apache License 2.0

- Files: `segment_anything/__init__.py`, `segment_anything/build_sam.py`,
  `segment_anything/modeling/{__init__,common,image_encoder,mask_decoder,prompt_encoder,sam,transformer}.py`.
- Original: <https://github.com/facebookresearch/segment-anything>, Copyright (c) Meta Platforms, Inc. and affiliates.
- Path of the copy: Meta's code → MedSAM (<https://github.com/bowang-lab/MedSAM>, `segment_anything/`, Apache-2.0)
  → torchtrainer (<https://github.com/chcomin/torchtrainer>, `torchtrainer/models/medsam/segment_anything/`, MIT,
  Cesar H. Comin) → the private experiments repository of this paper → this repository.
- Licence: Apache License 2.0, text in [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).
- Modifications with respect to MedSAM (`main`, commit `7c5dfff`):
  - torchtrainer reformatted `sam.py` and `build_sam.py` (docstrings, type hints), removed the
    `SamPredictor`/`SamAutomaticMaskGenerator` imports from `__init__.py`, and removed Meta's licence header
    from these three files;
  - the experiments repository moved the automatic checkpoint download of `build_sam.py` inside
    `if checkpoint is not None` and made `image_encoder.get_rel_pos` create its index tensors on the device of
    `rel_pos` (`torch.arange(..., device=rel_pos.device)`; same values);
  - this repository restored Meta's licence header in `__init__.py`, `build_sam.py` and `modeling/sam.py` and
    added a line pointing to this file. No code was changed.
- At run time, `thinvessel.models.medsam` replaces `image_encoder.get_rel_pos` with a version that caches its index
  grid (numerically identical), as the experiments did.

### TinyViT (Microsoft), via MobileSAM, LiteMedSAM and torchtrainer — MIT License

- File: `tiny_vit_sam.py`.
- Original: TinyViT, <https://github.com/microsoft/Cream/tree/main/TinyViT>, Copyright (c) Microsoft Corporation
  (header kept), adapted from LeViT and Swin Transformer; the variant for SAM comes from MobileSAM
  (<https://github.com/ChaoningZhang/MobileSAM>) and LiteMedSAM
  (<https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM>, `tiny_vit_sam.py`).
- Licence: MIT, text in [`LICENSES/MIT-tinyvit.txt`](LICENSES/MIT-tinyvit.txt).
- Modifications with respect to LiteMedSAM (made in torchtrainer): `timm.models.layers` imports replaced by
  `timm.layers`, the unused `register_model`/`Tuple` imports removed, code reformatted. Unchanged here.

### LiteMedSAM wrapper (torchtrainer) — MIT License

- File: `litemedsam.py` (the `LiteMedSAM` module: fixed whole-window box prompt, mask decoder, bilinear upsampling).
- Original: torchtrainer, `torchtrainer/models/litemedsam/litemedsam.py`, Copyright (c) 2024 Cesar Henrique Comin,
  adapted from LiteMedSAM (Apache-2.0).
- Licence: MIT, text in [`LICENSES/MIT-torchtrainer.txt`](LICENSES/MIT-torchtrainer.txt).
- Modifications: one import path (`..medsam.segment_anything.modeling` → `.segment_anything.modeling`) and a header
  naming the source.

## Adapted code (our files, written from these sources)

- `src/thinvessel/models/lwnet.py` — the U-Net blocks of Little W-Net, A. Galdran, <https://github.com/agaldran/lwnet>,
  Copyright (c) 2020 Adrian Galdran, MIT License ([`LICENSES/MIT-lwnet.txt`](LICENSES/MIT-lwnet.txt)); reached us
  through torchtrainer (`torchtrainer/models/unet_lw.py`). Paper: Galdran et al., *Scientific Reports* 12, 2022.
- `src/thinvessel/models/convunext.py` — ConvUNeXt, after Han, Jian and Wang, *Knowledge-Based Systems* 253, 2022
  (doi:10.1016/j.knosys.2022.109512). The implementation was provided to the first author by the corresponding
  author; its exact upstream source is not recorded in the experiments repository.
- `src/thinvessel/models/medsam.py`, `src/thinvessel/models/litemedsam.py` — wrappers around the vendored code
  above, following the MedSAM and LiteMedSAM training code of Ma et al. (Apache-2.0).
- `src/thinvessel/models/convnext.py` — uses `torchvision.models.convnext_large` (BSD-3-Clause, torchvision).

## Model weights (not included)

The fine-tuned runs start from the published MedSAM (`medsam_vit_b.pth`) and LiteMedSAM (`lite_medsam.pth`)
checkpoints of Ma et al. They are not redistributed here; see the README for their sources. Their authors state
that the models may be used for research purposes only.
