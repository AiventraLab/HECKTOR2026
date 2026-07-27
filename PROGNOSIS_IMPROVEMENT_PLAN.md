# Prognosis Improvement Plan

## Current Best
- RSF + Cox rank-average ensemble: C-index 0.711

## Experiments
1. Deep-MTLR (pycox): C-index 0.698 — discarded
2. Feature engineering (+24 features): no improvement
3. Noise augmentation: C-index dropped to 0.703
4. Predicted mask features: C-index 0.698

## Next Steps
- [ ] Add delta-GTVp (treatment response)
- [ ] Try DeepSurv
- [ ] 3-way ensemble: ICARE + RSF + Cox
- [ ] Add GLCM/GLRLM texture features
