"""Code of "Evaluating SAM-based and Convolutional Architectures in Thin Retinal Vessel Segmentation".

Subpackages:

- ``thinvessel.data``: datasets, splits and the FIVES mask correction (Section 3.1, Appendix A.1).
- ``thinvessel.models``: LW-Net, ConvUNeXt, ConvNeXt, MedSAM and LiteMedSAM (Section 3.2).
- ``thinvessel.training``: Lightning module, soft Dice loss and Optuna search (Section 3.3, Appendix A).
- ``thinvessel.evaluation``: sliding-window and upscaled inference and the metrics (Section 3.4).
- ``thinvessel.analysis``: cost, convergence and search records (the tables in ``results/``).
"""

__version__ = "1.0.0"
