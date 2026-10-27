# Selected hyperparameters of the 36 searches

The selected trial of each search is the trial with the highest validation Dice among the trials that
finished (Appendix A). Every value below is read from the released configuration
`configs/<regime>/<dataset>/<model>.yaml`, which holds the settings of that trial as recorded in its
`trial_<n>/config.yaml`. Every run used AdamW, a cosine schedule over at most 1000 epochs, seed 42, and
early stopping after 50 epochs without improvement (patience 50 validations every epoch, or 5 every 10
epochs). "Val. every" is the validation interval in epochs. MedSAM and LiteMedSAM fine-tuned and from scratch
keep the published architectures; the compact models are those of Table VI of the paper.

| Model | Regime | Dataset | Trial | Learning rate | Weight decay | Batch | Val. every | Patience | Architecture |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| lwnet | CNN | drive | 36 | 2.248e-03 | 1.209e-05 | 4 | 1 | 50 | layers [8, 16, 32] |
| lwnet | CNN | hrf | 4 | 3.520e-02 | 1.967e-05 | 8 | 10 | 5 | layers [8, 16, 32] |
| lwnet | CNN | fives | 11 | 2.785e-03 | 1.291e-04 | 16 | 1 | 50 | layers [8, 16, 32] |
| lwnet | CNN | vessmap | 82 | 4.198e-03 | 1.948e-05 | 4 | 1 | 50 | layers [8, 16, 32] |
| convunext | CNN | drive | 10 | 4.726e-03 | 1.421e-03 | 2 | 1 | 50 | base_c 64 |
| convunext | CNN | hrf | 15 | 9.734e-04 | 3.545e-06 | 4 | 10 | 5 | base_c 32 |
| convunext | CNN | fives | 1 | 5.343e-05 | 2.859e-05 | 32 | 1 | 50 | base_c 32 |
| convunext | CNN | vessmap | 2 | 6.647e-04 | 4.809e-05 | 2 | 1 | 50 | base_c 64 |
| convnext | CNN | drive | 0 | 2.481e-04 | 6.027e-06 | 4 | 1 | 50 | drop-path 0.354 |
| convnext | CNN | hrf | 0 | 2.481e-04 | 6.027e-06 | 4 | 10 | 5 | drop-path 0.010 |
| convnext | CNN | fives | 0 | 2.481e-04 | 6.027e-06 | 4 | 1 | 50 | drop-path 0.485 |
| convnext | CNN | vessmap | 57 | 1.918e-04 | 1.152e-04 | 2 | 1 | 50 | drop-path 0.486 |
| litemedsam | fine-tuned | drive | 21 | 2.484e-04 | 9.071e-06 | 4 | 1 | 50 | published (TinyViT) |
| litemedsam | fine-tuned | hrf | 0 | 2.481e-04 | 6.027e-06 | 4 | 10 | 5 | published (TinyViT) |
| litemedsam | fine-tuned | fives | 2 | 1.462e-05 | 1.146e-03 | 16 | 1 | 50 | published (TinyViT) |
| litemedsam | fine-tuned | vessmap | 47 | 1.019e-04 | 1.566e-05 | 2 | 1 | 50 | published (TinyViT) |
| medsam | fine-tuned | drive | 85 | 6.407e-04 | 7.206e-06 | 2 | 1 | 50 | published (ViT-B) |
| medsam | fine-tuned | hrf | 18 | 3.273e-04 | 1.117e-02 | 4 | 10 | 5 | published (ViT-B) |
| medsam | fine-tuned | fives | 0 | 8.472e-05 | 9.847e-04 | 2 | 10 | 5 | published (ViT-B) |
| medsam | fine-tuned | vessmap | 94 | 2.925e-04 | 2.324e-02 | 2 | 1 | 50 | published (ViT-B) |
| litemedsam | from scratch | drive | 55 | 7.077e-04 | 5.172e-06 | 2 | 1 | 50 | published (TinyViT) |
| litemedsam | from scratch | hrf | 3 | 6.672e-05 | 8.431e-03 | 8 | 10 | 5 | published (TinyViT) |
| litemedsam | from scratch | fives | 31 | 1.651e-04 | 9.695e-02 | 8 | 10 | 5 | published (TinyViT) |
| litemedsam | from scratch | vessmap | 0 | 1.570e-03 | 2.481e-03 | 2 | 1 | 50 | published (TinyViT) |
| medsam | from scratch | drive | 19 | 3.053e-05 | 4.067e-03 | 1 | 1 | 50 | published (ViT-B) |
| medsam | from scratch | hrf | 27 | 2.567e-05 | 2.288e-06 | 1 | 10 | 5 | published (ViT-B) |
| medsam | from scratch | fives | 12 | 7.346e-05 | 1.216e-06 | 2 | 10 | 5 | published (ViT-B) |
| medsam | from scratch | vessmap | 4 | 3.972e-05 | 1.140e-03 | 1 | 1 | 50 | published (ViT-B) |
| litemedsam | compact | drive | 23 | 1.117e-03 | 7.241e-06 | 8 | 10 | 5 | dims [64, 128, 160, 320], depths [2, 2, 6, 2], heads [2, 4, 5, 10], MLP 6.0, windows [7, 7, 7, 7] |
| litemedsam | compact | hrf | 69 | 5.888e-04 | 5.492e-04 | 8 | 10 | 5 | dims [32, 64, 80, 160], depths [1, 1, 3, 1], heads [2, 4, 5, 10], MLP 4.0, windows [7, 7, 14, 7] |
| litemedsam | compact | fives | 39 | 9.846e-05 | 2.690e-04 | 8 | 10 | 5 | dims [64, 128, 160, 320], depths [1, 1, 3, 1], heads [2, 4, 5, 10], MLP 4.0, windows [7, 7, 14, 7] |
| litemedsam | compact | vessmap | 71 | 1.027e-03 | 1.006e-04 | 8 | 10 | 5 | dims [64, 128, 160, 320], depths [1, 1, 3, 1], heads [2, 4, 5, 10], MLP 4.0, windows [7, 7, 14, 7] |
| medsam | compact | drive | 1 | 8.472e-05 | 9.847e-04 | 16 | 10 | 5 | patch 4, dim 256, depth 4, heads 8, MLP 2.0, window 14 |
| medsam | compact | hrf | 0 | 8.472e-05 | 9.847e-04 | 4 | 10 | 5 | patch 4, dim 128, depth 6, heads 4, MLP 2.0, window 14 |
| medsam | compact | fives | 77 | 5.551e-05 | 1.147e-04 | 16 | 10 | 5 | patch 8, dim 256, depth 8, heads 8, MLP 6.0, window 7 |
| medsam | compact | vessmap | 88 | 1.008e-04 | 3.014e-05 | 4 | 10 | 5 | patch 4, dim 384, depth 6, heads 4, MLP 6.0, window 7 |
