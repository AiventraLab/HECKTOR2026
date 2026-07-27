# Ensemble Strategy

## Segmentation

- **10-model softmax average**: 5-fold MedNeXt-L + 5-fold ResEnc
- **TTA**: 8-flip per-model, then average
- **Weights**: uniform (0.1 each)
- **Fallback**: if one arm is empty, trust the other

## TN Staging

- **5-seed Mamba ensemble**: average logits before argmax
- **Seeds**: [42, 123, 252, 378, 456]
- **Saturation**: 5 seeds sufficient (10 seeds gave same result)

## Prognosis

- **RSF + Cox rank-average**: RSF 0.6 + Cox 0.4
- **Orientation**: higher score = higher risk
