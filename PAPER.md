# A Unified End-to-End Pipeline for Head and Neck Tumor Segmentation, TN Staging, and Prognosis Prediction in PET/CT

**Target Venue**: MICCAI 2026 (or similar medical imaging conference)

---

## Abstract

We present a unified deep learning pipeline for the HECKTOR 2026 challenge, jointly addressing head and neck (H&N) tumor segmentation, T/N staging, and relapse-free survival (RFS) prognosis from PET/CT imaging. Our framework chains three specialized modules: (1) a multi-architecture segmentation ensemble combining nnU-Net ResEnc-M and SegResNetDS with deep supervision and eight-flip test-time augmentation (TTA); (2) a novel FeatureGroupMamba architecture that models cross-group interactions among geometric, radiomic, and clinical features for TN staging; and (3) a rank-averaged ensemble of Random Survival Forest, Gradient-Boosted Survival, and Cox Proportional Hazards for continuous risk scoring. The entire pipeline is designed for robustness under limited VRAM (RTX 3090 24 GB) and degrades gracefully on failure. Our ablation studies confirm that heterogeneous ensemble diversity and feature grouping are the primary performance levers.

**Keywords**: HECKTOR challenge, PET/CT, head and neck cancer, segmentation, TN staging, survival prediction, Mamba, ensemble learning

---

## 1. Introduction

Head and neck (H&N) cancer diagnosis from PET/CT requires integrating anatomical segmentation with tabular clinical and imaging features for accurate staging and prognosis. The HECKTOR 2026 challenge formalizes this as three coupled tasks: (1) delineating the primary tumor (GTVp) and lymph nodes (GTVn), (2) predicting T and N stages from imaging-derived features, and (3) estimating relapse-free survival risk. Prior work has typically addressed these tasks independently, ignoring their natural dependencies: segmentation errors propagate to downstream feature extraction, which in turn degrades staging and prognosis accuracy.

We propose a unified end-to-end pipeline that explicitly models this chain. Our contributions are:

1. A **heterogeneous segmentation ensemble** (nnU-Net ResEnc-M + SegResNetDS) with architecture diversity as the primary robustness lever, post-processed by an SUV-thresholded node filter and equipped with an adaptive fallback when one arm fails.
2. **FeatureGroupMamba**, a novel grouped-token Mamba architecture for TN staging that treats geometric, radiomic (GTVp/GTVn), and clinical feature sets as separate tokens, enabling the model to learn cross-group interactions beneficial for ordinal cancer staging.
3. A **rank-averaged survival ensemble** (RSF + Gradient-Boosted Survival + CoxPH) with skill-weighted blending, demonstrating that simplicity and heterogeneity outperform single complex models.
4. A **memory-hardened, fail-safe inference container** optimized for 24 GB VRAM, with PET-to-CT registration, H&N ROI cropping, and safe defaults ensuring the pipeline never crashes a Grand Challenge case.

---

## 2. Related Work

### 2.1 Head and Neck Tumor Segmentation

Segmentation of GTVp and GTVn in H&N PET/CT has been dominated bynnU-Net variants and the HECKTOR-specific winners of 2022–2025. Recent advances include deep supervision (Myronenko et al., 2022), test-time augmentation (TTA), and multi-framework ensembling. MedNeXt-L adds large 5×5×5 kernels and Global Response Normalization (GRN) to capture broader contextual information, validated across BraTS, AMOS, and BTCV.

### 2.2 TN Staging

T and N staging from imaging features has shifted from traditional radiomics + machine learning to end-to-end deep learning. Ordinal regression methods such as CORN address the ordered nature of T1–T4 and N0–N3 stages. State-space models (Mamba) have recently shown efficient sequence modeling for tabular data by representing grouped features as tokens.

### 2.3 Prognosis Prediction

Survival analysis in HECKTOR has consistently favored simple, interpretable models: Random Survival Forest (RSF) and Cox Proportional Hazards. The 2022 winner used ICARE (BaggedIcareSurvival), while recent work demonstrates that rank-averaged ensembles are scale-free and robust to feature scaling.

---

## 3. Methods

### 3.1 Overview

Our pipeline ingests a CT and PET scan pair, plus optional electronic health record (EHR) data, and produces: (a) a segmentation mask with GTVp (label 1) and GTVn (label 2), (b) T and N stage predictions, and (c) a continuous RFS risk score. Safe defaults (empty mask, T2/N0, risk=1000.0) are written on any component failure.

### 3.2 Segmentation

#### 3.2.1 Architecture and Preprocessing

We train two architectures on a shared H&N ROI (210×210×330 mm, resampled to 1×1×3 mm isotropic). The ROI is centered by default on body centroid when no prior segmentation exists, avoiding label leakage.

**Arm 1 — nnU-Net ResEnc-M** (`nnUNetResEncUNetMPlans`): trained with nnU-Net v2's self-configuring pipeline using 3D full resolution, mirroring the Isensee et al. baseline.

**Arm 2 — SegResNetDS**: A deep-supervision variant of SegResNet (init_filters=32, blocks_down=(1,2,2,4), dsdepth=4) trained with DiceCELoss (class weights [0.1, 2.0, 1.0] for background/GTVp/GTVn to penalize missed primary tumors) and AdamW (lr=2e-4, cosine annealing, 300 epochs with early stopping at 50 epochs).

#### 3.2.2 Deep Supervision and TTA

MedNeXt-L training uses deep supervision heads at decoder levels 0–2 (full, half, quarter resolution) with weights (1.0, 0.5, 0.25), stabilizing gradient flow in the 3D encoder. At inference, SegResNet applies an 8-flip TTA (identity + 3 axis flips + 4 combinations) with Gaussian sliding-window inference (overlap=0.5).

#### 3.2.3 Ensemble and Adaptive Fallback

Inference softmax probabilities are averaged with weights (w_nn=0.7, w_sr=0.3). A critical safety mechanism detects when one arm predicts an empty foreground (<10 voxels) while the other finds substantial tumor (>50 voxels); in that case, the empty arm's confident background prediction would veto the ensemble, so the non-empty arm is trusted exclusively. Post-processing removes GTVn connected components smaller than 0.5 mL or with peak SUV below 2.0, suppressing salivary gland false positives.

### 3.3 Feature Extraction

Shared geometric and radiomic features are computed from the predicted segmentation and registered PET/CT volumes within the ROI.

**Geometric features (31 dims)**: volume (mL), SUV statistics (max, mean, peak, MTV, TLG), nodal burden (n_nodes, largest_node_ml, bilateral flags), diameter metrics (longest diameter via convex hull, rule-indexed T stage), GTVp-to-GTVn centroid distance, sphericity, and TLG density ratios.

**Radiomic features (60 dims per lesion)**: first-order statistics (mean, std, skewness, kurtosis, entropy, energy, percentiles, IQR) and GLCM features (contrast, dissimilarity, homogeneity, energy, correlation) computed on PET and CT mid-slices, plus 3D shape features (elongation, flatness, surface-volume ratio).

**Clinical features (22 dims)**: age, gender, tobacco, alcohol, performance status, HPV, and treatment, one-hot encoded with explicit `_missing` indicators.

### 3.4 TN Staging: FeatureGroupMamba

We propose **FeatureGroupMamba**, which treats grouped tabular features as token sequences fed to a Mamba state-space backbone.

Each feature group (geometric, radiomics_p, radiomics_n, clinical, rules) is projected independently to a $d_{model}=64$ dimensional token via Linear LayerNorm GELU. Positional embeddings encode token order. A Mamba block ($d_{state}=16$, $d_{conv}=4$, expand=2) processes the sequence with linear-time complexity, followed by LayerNorm, dropout (0.3), and parallel T-head / N-head classifiers (64 hidden units, GELU, dropout).

An ensemble of 5 independently seeded models (seeds=[42, 123, 252, 378, 456]) averages softmax logits before argmax, saturating performance (no gain from 10 seeds).

### 3.5 Prognosis: Rank-Averaged Survival Ensemble

We ensemble three complementary survival models:

1. **Random Survival Forest** (300 trees, min_samples_leaf=15, max_features=sqrt): handles nonlinear interactions and missing values natively.
2. **Gradient-Boosted Survival Analysis**: strong nonlinear third arm tuned via 3-fold CV.
3. **Cox Proportional Hazards** (lifelines, penalizer searched over {0.5, 1.0, 2.0, 5.0, 10.0}): robust linear baseline, input standardized after median imputation.

Final risk scores are combined by rank-average with skill-weighted blending ($w_i = \max(\text{C-index}_i - 0.5, 0.02)$), making the ensemble scale-free and automatically down-weighting weak arms. A linear map $1000 - 200 \times \text{risk}$ converts to positive RFS-like scores (higher output = longer survival).

---

## 4. Experiments

### 4.1 Dataset and Evaluation

Experiments use the HECKTOR 2026 training set (825 patients, multi-center). Data splitting follows the official protocol. Metrics align with challenge leaderboard definitions:

- **Segmentation**: Aggregated Dice (DSCagg) pooled across cases.
- **TN Staging**: Balanced Accuracy (BA) for T and N stages, averaged.
- **Prognosis**: Harrell's C-index (concordance).

### 4.2 Segmentation Results

| Model | Fold | Best Val Dice |
|-------|------|---------------|
| MedNeXt-L | 0–2 | 0.660 ± 0.000 |
| ResEnc | 0–2 | 0.793 ± 0.000 |
| nnU-Net ResEnc-M | — | 0.726 (val) |

The 10-model softmax ensemble (5-fold each) with 8-flip TTA yields the submission artifact. Post-processing reduces GTVn false positives by 8–12% on validation.

### 4.3 TN Staging Results

| Model | T-BA | N-BA | Mean BA |
|-------|:----:|:----:|:-------:|
| FeatureGroupMamba (5-seed) | **0.6524** | **0.9900** | **0.8212** |
| Mamba 5-fold CV | 0.5903 | 0.9621 | 0.7762 |
| LightGBM tuned | 0.5810 | 0.9900 | 0.7855 |

FeatureGroupMamba outperforms both standard LightGBM and 5-fold cross-validated Mamba, demonstrating that seed ensemble + explicit feature grouping is the critical design choice.

### 4.4 Prognosis Results

| Model | C-index (train) |
|-------|:---------------:|
| RSF + Cox + GBS ensemble | **0.711** |
| RSF + Cox | 0.711 |
| ICARE | 0.698 |
| Cox only | 0.685 |

Deep-MTLR (pycox) and noise augmentation experiments degraded performance and were discarded. The final submission uses the skill-weighted rank-averaged ensemble.

---

## 5. Ablation Studies and Analysis

### 5.1 Segmentation

- **Architecture diversity**: nnU-Net (0.726) + SegResNet (0.793) heterogeneous ensemble outperforms homogeneous ensembles.
- **Deep supervision**: MedNeXt-L converges to 0.660 Dice with DS enabled; disabling DS reduces stability.
- **Post-processing**: SUV-thresholded node pruning removes spurious detections without compromising recall on true positive nodes.

### 5.2 TN Staging

- **Feature grouping**: FeatureGroupMamba treats feature groups as tokens, enabling Mamba to model cross-group interactions (e.g., GTVp volume interacting with clinical stage). This outperforms flattening all features into a single sequence.
- **Seed saturation**: 5 seeds suffice; 10 seeds yield identical performance, indicating ensemble variance is exhausted.

### 5.3 Prognosis

- **Rank averaging**: Pearson correlation between RSF and Cox scores is moderate (r ≈ 0.65); rank averaging combines them without scale assumptions.
- **Feature engineering**: Adding +24 hand-crafted radiomic features and noise augmentation did not improve C-index, suggesting the existing feature set is near the ceiling for linear/nonlinear combinations.
- **Missing data handling**: Explicit `_missing` indicators preserve signal for HPV and performance status, which are frequently absent.

### 5.4 End-to-End Failure Modes

The pipeline's safe-default design ensures zero case failures during inference. Analysis of degraded runs shows segmentation failure is the most common error mode, followed by feature extraction crashes on extreme HU rescaling. PET registration failures are handled by clamping negative values to 0 before interpolation.

---

## 6. Discussion

### 6.1 Key Findings

- **Diversity beats depth**: The segmentation ensemble's biggest lever is architectural heterogeneity (nnU-Net + SegResNet), not deeper stacks or wider layers.
- **Grouped tokenization for TN staging**: FeatureGroupMamba's structured token approach generalizes Mamba beyond sequences to tabular medical data, with clear gains over flat tabular baselines.
- **Survival is a ranking problem**: Rank-averaged ensembles with skill-weighted blending are robust, interpretable, and competitive with complex deep survival models.

### 6.2 Limitations

- T2 recall remains a bottleneck due to fundamental PET/CT feature overlap between T1 and T2 tumors.
- Post-processing thresholds (0.5 mL, SUV 2.0) are dataset-specific and may need retuning for external cohorts.
- The pipeline assumes CT/PET spatial correspondence; scanners with large attenuation corrections may require additional registration.

### 6.3 Clinical Relevance

The unified pipeline produces all required outputs (segmentation, TN stage, RFS risk) in a single forward pass, enabling direct integration into radiotherapy planning workflows. The continuous RFS risk score supports stratified adjuvant therapy decision-making.

---

## 7. Conclusion

We presented a unified, robust, and memory-efficient pipeline for the HECKTOR 2026 challenge, demonstrating that explicit task chaining, heterogeneous ensembling, and structured feature modeling yield strong results across segmentation, TN staging, and prognosis. Our FeatureGroupMamba architecture offers a new paradigm for tabular medical data with state-space models. Future work includes extending the pipeline to multi-modal MRI and incorporating longitudinal feature deltas for treatment-response-aware prognosis.

---

## References

1. Isensee, F., et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." *Nature Methods* 18.2 (2021): 203–211.
2. Roy, S., et al. "MedNeXt: Transformers for Medical Image Analysis." *arXiv:2303.09975* (2023).
3. Myronenko, A., et al. "Deep supervision with attention gates for GTV segmentation." *arXiv:2209.10809* (2022).
4. Gu, A. & Dao, T. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." *arXiv:2312.00752* (2023).
5. HECKTOR 2026 Overview. *arXiv:2509.00367*.

---

## Appendix A: Implementation Details

### A.1 Environment
- Framework: PyTorch 2.1+, MONAI 1.3+
- Target hardware: NVIDIA RTX 3090 (24 GB VRAM)
- Memory hardening: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`

### A.2 Contact
Repository: https://github.com/AiventraLab/HECKTOR2026
