# Rabin vs Prabesh Comparison

## Segmentation
- Rabin: nnU-Net ResEnc-M + SegResNetDS
- Prabesh: MedNeXt-L + ResEnc ensemble
- Winner: Prabesh (MedNeXt-L is SOTA)

## TN Staging
- Rabin: LightGBM + CORN ordinal (T-BA 0.5810)
- Prabesh: FeatureGroupMamba 5-seed ensemble (T-BA 0.6524)
- Winner: Prabesh

## Prognosis
- Rabin: RSF + Cox + GBS ensemble (C-index 0.711)
- Prabesh: ICARE only (C-index 0.698)
- Winner: Rabin

## Final
Use Rabin prognosis + Prabesh segmentation + Prabesh TN staging.
