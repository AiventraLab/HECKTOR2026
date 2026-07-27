# HPO Plan

## Segmentation
- Phase 1: Arch search (20 trials, 50 epochs, fold 0)
- Phase 2: Train HPO (30 trials, 50 epochs, fold 0)
- Phase 3: Full 5-fold CV with best params

## TN Staging
- T-stage: Optuna 40 trials (LightGBM vs CORN)
- N-stage: Optuna 80 trials (LightGBM)

## Prognosis
- No HPO: fixed RSF + Cox ensemble
