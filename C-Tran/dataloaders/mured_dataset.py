"""
MuReD (Multi-label Retinal Diseases) dataset loader for C-Tran.

Follows the preprocessing / augmentation described in
Rodriguez et al., "Multi-Label Retinal Disease Classification Using
Transformers" (IEEE JBHI 2023):
  - Field-of-view (FOV) extraction / background removal (crop black border)
  - Resize to 384x384
  - Albumentations training augmentations (paper Table III)
  - ImageNet normalisation

It exposes the same sample dict interface as the other C-Tran dataloaders
(`image`, `labels`, `mask`, `imageIDs`) so it can be dropped into the existing
LMT (label-mask-training) machinery.
"""
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(0)  # avoid cv2/DataLoader worker thread contention


# ---------------------------------------------------------------------------
# Field-of-view extraction (background removal), a la Kulkarni et al. /
# the RIADD reference `crop_image_from_gray`.
# ---------------------------------------------------------------------------
def crop_image_from_gray(img, tol=7):
    """Crop the black border around the retina using a grayscale threshold."""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray_img > tol
    check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
    if check_shape == 0:  # image is all black; return as-is
        return img
    img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
    img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
    img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
    return np.stack([img1, img2, img3], axis=-1)


# ---------------------------------------------------------------------------
# Label-mask-training (LMT) unknown-index sampling, replicating
# dataloaders/data_utils.get_unk_mask_indices.
# ---------------------------------------------------------------------------
def get_unk_mask_indices(num_labels, known_labels, testing, seed_val=None):
    if testing:
        # Deterministic per-image mask for reproducibility across epochs.
        if seed_val is not None:
            random.seed(seed_val)
        unk_mask_indices = random.sample(range(num_labels), (num_labels - int(known_labels)))
    else:
        # Sample a random number of KNOWN labels during training:
        #   0 .. 75% known  <=>  25% .. 100% unknown  (paper's LMT scheme).
        if known_labels > 0:
            random.seed()
            num_known = random.randint(0, int(num_labels * 0.75))
        else:
            num_known = 0
        unk_mask_indices = random.sample(range(num_labels), (num_labels - num_known))
    return unk_mask_indices


class MuredDataset(Dataset):
    def __init__(self, data_frame, img_root, transform=None, num_labels=20,
                 known_labels=0, testing=False, fov_crop=True):
        """
        data_frame : pandas DataFrame whose first column is the image ID and
                     the remaining columns are the 20 binary label columns.
        img_root   : directory containing the image files.
        transform  : albumentations transform (expects `image=` kwarg).
        known_labels : >0 enables LMT partial-label sampling during training.
        testing    : if True, use the deterministic (all-unknown) mask.
        """
        self.df = data_frame.reset_index(drop=True)
        self.img_root = img_root
        self.transform = transform
        self.num_labels = num_labels
        self.known_labels = known_labels
        self.testing = testing
        self.fov_crop = fov_crop
        self.label_cols = list(self.df.columns[1:])
        assert len(self.label_cols) == num_labels, \
            f"expected {num_labels} label cols, got {len(self.label_cols)}"
        self.epoch = 1

    def __len__(self):
        return len(self.df)

    def _load_image(self, image_id):
        path = os.path.join(self.img_root, image_id)
        img = cv2.imread(path)  # BGR, HWC uint8
        if img is None:
            raise FileNotFoundError(f"could not read image: {path}")
        if self.fov_crop:
            img = crop_image_from_gray(img)
        img = img[:, :, ::-1]  # BGR -> RGB
        return np.ascontiguousarray(img)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        image_id = row.iloc[0]
        image = self._load_image(image_id)
        if self.transform is not None:
            image = self.transform(image=image)['image']

        labels = torch.Tensor(row[self.label_cols].values.astype(np.float32))

        seed_val = int(hash(image_id) & 0xFFFFFFFF) if self.testing else None
        unk_mask_indices = get_unk_mask_indices(
            self.num_labels, self.known_labels, self.testing, seed_val)

        mask = labels.clone()
        mask.scatter_(0, torch.Tensor(unk_mask_indices).long(), -1)

        return {
            'image': image,
            'labels': labels,
            'mask': mask,
            'imageIDs': image_id,
        }
