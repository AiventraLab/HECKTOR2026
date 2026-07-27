# HECKTOR 2026 Unified Pipeline

MICCAI-standard repository combining:
- **Segmentation**: MedNeXt-L + ResEnc ensemble with deep supervision and TTA
- **TN Staging**: Mamba FeatureGroupMamba ensemble (LightGBM/CORN fallback)
- **Prognosis**: RSF + Cox + GBS ensemble (rank-weighted)

## Repository structure

```
HECKTOR2026/
├── src/
│   ├── shared/        # columns, io_utils, metrics, splits
│   ├── segmentation/  # models, data, losses, utils, scripts
│   ├── staging/       # Mamba TN staging + radiomics
│   └── prognosis/     # RSF/Cox ensemble
├── submission/Task/   # Grand Challenge entrypoint
├── configs/           # YAML hyperparameter configs
├── pipeline.py        # End-to-end inference script
└── work/              # splits, outputs, caches
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional dependencies for full feature parity:
```bash
pip install mamba-ssm icare sksurv pycox torchtuples coral-pytorch
```

## Model weights

Pre-trained weights for segmentation and TN staging are available at:
```
https://huggingface.co/datasets/rabin-hackathon/HECKTOR2026-weights
```

Download and place them under `work/models/`:
```bash
work/models/
├── nnunet/
├── segresnet_best.pt
├── mamba_tn_ensemble.pth
├── prognosis_ensemble.pkl
└── seg_weights.json
```

## Training

```bash
# 1. Create splits
python src/shared/splits.py --csv data/train.csv --out work/splits_final.json

# 2. Train segmentation
python src/segmentation/scripts/train.py --config mednext --fold 0
python src/segmentation/scripts/train.py --config resenc --fold 0

# 3. Extract features
python src/staging/train.py --features work/features.csv --csv data/train.csv --split work/splits_final.json --out work

# 4. Train TN staging
python src/staging/train.py --features work/features.csv --csv data/train.csv --split work/splits_final.json --out work

# 5. Train prognosis
python src/prognosis/train.py --features work/features.csv --csv data/train.csv --split work/splits_final.json --out work
```

## Inference

```bash
# Full pipeline
python pipeline.py --ct case/CT.nii.gz --pet case/PT.nii.gz --ehr '{"Age": 55}'

# Grand Challenge
python submission/Task/inference.py
```

## Subtasks and weights

| Subtask | Weight | Module |
|---------|--------|--------|
| Segmentation (DSCagg) | 25 % | src/segmentation |
| TN Staging (BA) | 35 % | src/staging |
| Prognosis (C-index) | 40 % | src/prognosis |

## Citation

If you use this code, please cite:

```bibtex
@article{hecktor2026,
  title   = {HECKTOR 2026: Head and Neck Tumor Segmentation and Outcome Prediction},
  journal = {arXiv preprint arXiv:2509.00367},
  year    = {2026}
}
```

## Key references

- MedNeXt: Roy et al., arXiv:2303.09975
- nnU-Net v2: Isensee et al., Nature Methods 2021
- Mamba: Gu & Dao, arXiv:2312.00752
- HECKTOR 2026 overview: arXiv:2509.00367
