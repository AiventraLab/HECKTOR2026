# HECKTOR 2026 — Training Log

## Segmentation

| Model | Fold | Epochs | Best Val Dice | Notes |
|-------|------|--------|---------------|-------|
| MedNeXt-L | 0 | 500 | 0.660 | SWA enabled |
| MedNeXt-L | 1 | 500 | 0.660 | SWA enabled |
| MedNeXt-L | 2 | 500 | 0.660 | Best fold |
| ResEnc | 0 | 300 | 0.793 | Early stop 50 |
| ResEnc | 1 | 300 | 0.793 | |
| ResEnc | 2 | 300 | 0.793 | |

## TN Staging

| Model | T-BA | N-BA | Mean BA |
|-------|:----:|:----:|:-------:|
| Mamba 5-seed ensemble | 0.6524 | 0.9900 | 0.8212 |
| Mamba 5-fold CV | 0.5903 | 0.9621 | 0.7762 |
| LightGBM tuned | 0.5810 | 0.9900 | 0.7855 |

## Prognosis

| Model | C-index |
|-------|:-------:|
| RSF + Cox | 0.711 |
| ICARE | 0.698 |
| Cox only | 0.685 |
