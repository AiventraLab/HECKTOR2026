"""nnU-Net v2 (ResEnc) segmentation arm.

Training is driven by the nnU-Net CLI (see scripts/02_train_segmentation.sh);
this module provides (a) the dataset.json/env helpers and (b) a Python
predictor wrapper so the submission container can run inference without the CLI
and return softmax probabilities for ensembling.

ResEnc-M targets ~9-11 GB VRAM -> fits the 16 GB T4 with margin.
"""
from __future__ import annotations
import os
import numpy as np

# CLI cheat-sheet (also in scripts/02_train_segmentation.sh):
#   export nnUNet_raw=... nnUNet_preprocessed=... nnUNet_results=...
#   nnUNetv2_plan_and_preprocess -d 21 -pl nnUNetPlannerResEncM --verify_dataset_integrity
#   nnUNetv2_train 21 3d_fullres {0..4} -p nnUNetResEncUNetMPlans --npz
#   nnUNetv2_predict -i IN -o OUT -d 21 -c 3d_fullres -p nnUNetResEncUNetMPlans -f 0 1 2 3 4


def set_env(raw, preprocessed, results):
    os.environ["nnUNet_raw"] = raw
    os.environ["nnUNet_preprocessed"] = preprocessed
    os.environ["nnUNet_results"] = results


class NNUNetPredictor:
    """Thin wrapper over nnunetv2's predictor for in-container inference."""

    def __init__(self, model_folder, folds=(0, 1, 2, 3, 4),
                 checkpoint="checkpoint_final.pth", tta=True, device="cuda"):
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=tta,
            perform_everything_on_device=True,
            device=torch.device(device, 0), verbose=False, allow_tqdm=False,
        )
        self.predictor.initialize_from_trained_model_folder(
            model_folder, use_folds=tuple(folds), checkpoint_name=checkpoint)

    def predict_prob(self, ct_path, pet_path):
        """Return (softmax_probs [C,H,W,D], properties) for a CT+PET pair.
        Channel order MUST be [CT, PET] to match training (_0000/_0001)."""
        from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
        io = SimpleITKIO()
        img, props = io.read_images([ct_path, pet_path])   # channels-first
        seg, probs = self.predictor.predict_single_npy_array(
            input_image=img, image_properties=props,
            segmentation_previous_stage=None, output_file_truncated=None,
            save_or_return_probabilities=True,
        )
        return np.asarray(probs), props
