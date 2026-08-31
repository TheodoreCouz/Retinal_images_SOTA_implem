"""Evaluate the STRATIFIED-split MURED CI run (5 seeds x 5-fold ensemble) under
the 9-metric Fiber protocol, with a t-based 95% CI across seeds. Mirrors
eval_ci.py but targets the MURED-STRAT checkpoints + stratified val/test CSVs.

    CUDA_VISIBLE_DEVICES=2 .venv/bin/python eval_mured_strat.py
"""
import glob, os, json
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from types import SimpleNamespace
from scipy import stats
from timm.models import create_model
from timm.data import get_riadd_valid_transforms, PrismDataSet
from metrics9 import compute_all, METRIC_KEYS

MURED = "/storage2/cousin/datasets/MURED"
IMGDIR = f"{MURED}/images/images"
VAL_CSV = f"{MURED}/val_labels_stratified.csv"
TEST_CSV = f"{MURED}/test_labels_stratified.csv"
PREFIX = "ckpt/tf_efficientnet_b6_ns-MURED-STRAT-SGD1E-1-seed"
SEEDS = [42, 43, 44, 45, 46]
NC = 20
OUT_DIR = "MURED/Performance"
PRED_DIR = f"{OUT_DIR}/preds"


def fold_ckpts(seed):
    paths = []
    for f in range(5):
        hits = sorted(glob.glob(f"{PREFIX}{seed}fold_{f}/train/*/model_best.pth.tar"))
        if not hits:
            return None
        paths.append(hits[-1])
    return paths


def load_model(ckpt):
    m = create_model("tf_efficientnet_b6_ns", pretrained=False, num_classes=NC)
    ck = torch.load(ckpt, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in ck["state_dict"].items()}
    m.load_state_dict(sd, strict=True)
    return m.cuda().eval()


@torch.no_grad()
def predict(model, loader):
    out = []
    for x, _ in loader:
        x = x.cuda(non_blocking=True)
        with torch.cuda.amp.autocast():
            out.append(model(x).sigmoid().float().cpu().numpy())
    return np.concatenate(out)


def ensemble_probs(seed, split, csv):
    os.makedirs(PRED_DIR, exist_ok=True)
    cache = f"{PRED_DIR}/mured_strat_seed{seed}_{split}.npy"
    df = pd.read_csv(csv)
    y = df[df.columns[1:]].values.astype(np.int64)
    if os.path.exists(cache):
        return np.load(cache), y
    ckpts = fold_ckpts(seed)
    assert ckpts, f"missing folds for seed {seed}"
    ds = PrismDataSet(image_ids=df, baseImgPath=IMGDIR)
    ds.transform = get_riadd_valid_transforms(SimpleNamespace(img_size=960))
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=6, pin_memory=True)
    fold_p = []
    for cp in ckpts:
        m = load_model(cp)
        fold_p.append(predict(m, loader))
        del m
        torch.cuda.empty_cache()
    ens = np.mean(fold_p, axis=0)
    np.save(cache, ens)
    return ens, y


def ci(vals):
    a = np.asarray(vals, float); n = len(a); mean = float(a.mean())
    if n < 2:
        return dict(mean=mean, std=None, n=n, ci95_lo=None, ci95_hi=None, half_width=None)
    sd = float(a.std(ddof=1)); se = sd / np.sqrt(n); h = float(stats.t.ppf(0.975, n - 1) * se)
    return dict(mean=mean, std=sd, n=n, ci95_lo=mean - h, ci95_hi=mean + h, half_width=h)


def main():
    per_seed = []
    for s in SEEDS:
        vp, vy = ensemble_probs(s, "val", VAL_CSV)
        tp, ty = ensemble_probs(s, "test", TEST_CSV)
        m = compute_all(vp, vy, tp, ty)
        per_seed.append(dict(seed=s, metrics={k: m[k] for k in METRIC_KEYS}))
        print(f"seed {s}: " + "  ".join(f"{k}={m[k]:.4f}" for k in METRIC_KEYS))
    agg = {k: ci([r["metrics"][k] for r in per_seed]) for k in METRIC_KEYS}
    n_test = len(pd.read_csv(TEST_CSV)); n_val = len(pd.read_csv(VAL_CSV))
    result = dict(dataset="MURED", split="stratified", n_train=1504, n_val=n_val,
                  n_test=n_test, n_classes=NC, seeds=SEEDS, model="5-fold ensemble x5 seeds",
                  per_seed=per_seed, test_ci=agg)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/mured_strat_ci.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n== test CI (n=5 seeds) ==")
    for k in METRIC_KEYS:
        c = agg[k]
        print(f"  {k:18s} {c['mean']:.4f}  95% CI [{c['ci95_lo']:.4f}, {c['ci95_hi']:.4f}]")
    print(f"\nwrote {OUT_DIR}/mured_strat_ci.json")


if __name__ == "__main__":
    main()
