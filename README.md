# Video Action Classification

This project explores video recognition on HMDB51, a 51-class action dataset. The supplied notebooks contain a frame-based ViT baseline and an LS-ViT implementation.

## Notebooks

- **Basic ViT** (`[Code]-Basic-ViT-base-patch16-224.ipynb`) implements a ViT-Base backbone, loads pretrained `timm` weights, processes video frames, and averages frame features before classification. This pooling step does not model temporal order.
- **LS-ViT** (`[Solution]-LSVIT-HMDB51.ipynb`) combines a ViT backbone with spatial-motion interaction (SMIF) and LSViT blocks to model video features. It loads pretrained `vit_base_patch16_224` weights and uses grouped train/validation splits.

The saved notebook outputs report best validation accuracy of **53.80%** for Basic ViT and **63.88%** for LS-ViT. These are validation results stored in the notebooks; the notebooks were not rerun for this README.

## Training configuration

The LS-ViT notebook uses 10 epochs, batch size 4, backbone learning rate `5e-5`, and head learning rate `2.5e-4`. The Basic ViT notebook uses batch size 8 and includes a 10-epoch training output.

## Dataset and runtime

The notebooks expect HMDB51 data extracted under `hmdb51_data`. The dataset is not included here. The Basic ViT notebook references an HMDB51 ZIP in a Google Drive path; update that path for your own Colab or local environment. The LS-ViT notebook expects `HMDB51.zip` in its working directory before extraction.

The notebooks use Python, PyTorch, `timm`, and `tqdm`. A GPU runtime is recommended for training.

## Results reported in the CV

The CV reports **79.6% test accuracy** for VideoMAE-Base and **65.0%** for a ViT-Small baseline, with MixUp (`alpha=0.2`), 25 epochs, and 60-view test-time augmentation. Those models and settings are not implemented in the two supplied notebooks, so these CV figures are reported separately and are not reproduced by the saved notebook outputs above.

