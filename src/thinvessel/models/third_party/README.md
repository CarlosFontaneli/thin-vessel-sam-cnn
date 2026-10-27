# Vendored third-party code

Code that this repository did not write, kept as it was used in the paper's experiments so that the models compute
exactly what they computed then. Do not edit these files; the adapters in [`..`](..) wrap them.

| File | Origin | Licence |
|---|---|---|
| `segment_anything/` | Segment Anything (Meta), via MedSAM and torchtrainer | Apache-2.0 |
| `tiny_vit_sam.py` | TinyViT (Microsoft), via MobileSAM, LiteMedSAM and torchtrainer | MIT |
| `litemedsam.py` | LiteMedSAM wrapper of torchtrainer (C. H. Comin) | MIT |

The only changes are one import path and restored licence headers. Sources, licences and every modification are in
[`THIRD_PARTY_NOTICES.md`](../../../../THIRD_PARTY_NOTICES.md). This folder is excluded from `ruff` (lint and format).
