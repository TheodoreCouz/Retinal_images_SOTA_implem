"""Inference latency benchmark: KAMATALAB paths 2+3 (path 1 dropped), untrained weights.

Mirrors Pytorch-RIADD/dump_riadd_paths.py's inference pattern exactly (same model
classes, image sizes, RiaddDataSet + get_riadd_test_transforms, batch size 8) but:
  - skips path 1 (multi-stage, 4 heads) entirely, per the request.
  - uses freshly-initialized (pretrained=False) weights standing in for the missing
    fold checkpoints -- same architecture/resolution, so identical FLOPs/latency.
  - times the forward pass instead of accumulating probabilities.

"10 models x 3 queries" = path2 (tf_efficientnet_b5_ns@960) x 5 folds
                         + path3 (tf_efficientnet_b6_ns@768) x 5 folds
                         each run TTA=3x (upstream's CFG['tta']), exactly as
                         head_probs() in dump_riadd_paths.py does per real fold ckpt.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Pytorch-RIADD'))
from timm.models import create_model
from timm.data import get_riadd_test_transforms, RiaddDataSet

SRC = '/globalsc/ucl/ingi/cousint/ISBI_datasets/RFMiD'
IMG_DIR = f'{SRC}/Test'
CSV = f'{SRC}/testing_labels_29.csv'
N_IMAGES = 40
N_FOLDS = 5
TTA = 3
BS = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PATHS = {
    'p2': ('tf_efficientnet_b5_ns', 960),
    'p3': ('tf_efficientnet_b6_ns', 768),
}


def make_loader(img_size):
    df = pd.read_csv(CSV).iloc[:N_IMAGES].reset_index(drop=True)
    ds = RiaddDataSet(image_ids=df, baseImgPath=IMG_DIR)
    ds.transform = get_riadd_test_transforms({'img_size': img_size})
    return DataLoader(ds, batch_size=BS, shuffle=False, num_workers=4, pin_memory=True)


@torch.no_grad()
def timed_pass(model, loader):
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    for x, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type='cuda', enabled=(DEVICE == 'cuda')):
            model(x).sigmoid()
        n += x.shape[0]
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt, n


def main():
    print(f'device={DEVICE} images={N_IMAGES} folds/path={N_FOLDS} tta={TTA} batch_size={BS}',
          flush=True)
    if DEVICE == 'cuda':
        print('gpu:', torch.cuda.get_device_name(0), flush=True)

    all_rows = []
    for path, (model_name, img_size) in PATHS.items():
        loader = make_loader(img_size)
        for fold in range(N_FOLDS):
            model = create_model(model_name, pretrained=False, num_classes=29)
            model = model.to(DEVICE).eval()
            # warm-up (cudnn autotune / lazy init) -- not timed, matches steady-state
            # latency rather than first-call compile overhead.
            timed_pass(model, loader)
            for q in range(TTA):
                dt, n = timed_pass(model, loader)
                per_image_ms = dt / n * 1000
                all_rows.append(dict(path=path, model=model_name, img_size=img_size,
                                      fold=fold, query=q, batch_s=dt, n_images=n,
                                      per_image_ms=per_image_ms))
                print(f'{path} fold{fold} query{q}: {dt*1000:8.2f} ms total, '
                      f'{per_image_ms:6.2f} ms/image', flush=True)
            del model
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'riadd_p23_latency.csv')
    df.to_csv(out_csv, index=False)
    print(f'\nwrote {out_csv}', flush=True)

    print('\n=== summary (mean +/- std over the 3 queries, per fold) ===')
    for path in PATHS:
        sub = df[df.path == path]
        print(f'{path} ({PATHS[path][0]}@{PATHS[path][1]}): '
              f'{sub.per_image_ms.mean():.2f} +/- {sub.per_image_ms.std():.2f} ms/image  '
              f'({sub.batch_s.mean()*1000:.1f} +/- {sub.batch_s.std()*1000:.1f} ms for {N_IMAGES} images)')

    print(f'\n=== overall across all {len(df)} queries (10 models x {TTA}) ===')
    print(f'{df.per_image_ms.mean():.2f} +/- {df.per_image_ms.std():.2f} ms/image')
    print(f'{df.batch_s.mean()*1000:.1f} +/- {df.batch_s.std()*1000:.1f} ms per {N_IMAGES}-image pass')


if __name__ == '__main__':
    main()
