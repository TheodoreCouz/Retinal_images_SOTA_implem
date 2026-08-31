"""Compute the 9-metric set (metrics.txt) for MURED/RFMiD/PRISM from cached preds.

CPU-only (reads performance/preds/*.npy) so it never touches the GPUs. Tunes
per-class thresholds on validation, applies to test. Writes performance/all_metrics.json.
"""
import json
import numpy as np
import pandas as pd
from metrics9 import compute_all, METRIC_KEYS

MURED = "/storage2/cousin/datasets/MURED"
PRISM = "/home/cousin/research/Fiber_dino/PRISM v1"
RFMID = "/storage2/cousin/datasets/RFMiD"

# name -> (val_npy, val_csv, test_npy, test_csv, disease_start)
CFG = {
    "mured": ("performance/preds/mured_val.npy",  f"{MURED}/val_labels.csv",
              "performance/preds/mured_test.npy", f"{MURED}/test_labels.csv", 0),
    "rfmid": ("performance/preds/rfmid_val_ensemble.npy",  f"{RFMID}/validation_labels_29.csv",
              "performance/preds/rfmid_test_ensemble.npy", f"{RFMID}/testing_labels_29.csv", 1),
    "prism": ("performance/preds/prism_val.npy",  f"{PRISM}/val_labels.csv",
              "performance/preds/prism_test.npy", f"{PRISM}/test_labels.csv", 0),
}


def load(npy, csv, d0):
    df = pd.read_csv(csv)
    cols = list(df.columns[1:])[d0:]
    y = df[df.columns[1:]].values.astype(np.int64)[:, d0:]
    p = np.load(npy)[:, d0:]
    return p, y, cols


def main():
    out = {}
    for name, (vnpy, vcsv, tnpy, tcsv, d0) in CFG.items():
        vp, vy, cols = load(vnpy, vcsv, d0)
        tp, ty, _ = load(tnpy, tcsv, d0)
        m = compute_all(vp, vy, tp, ty)
        m["classes"] = cols
        out[name] = m
        print(f"=== {name}  (C={m['n_classes']}, test N={m['n_images']}, "
              f"eval-B {m['n_classes_evaluable_B']}/{m['n_classes']}, "
              f"Recall@n imgs {m['n_images_recall_at_n']}/{m['n_images']}) ===")
        for k in METRIC_KEYS:
            print(f"   {k:20s} {m[k]:.4f}")
        tc = np.array(m["thresholds"])
        print(f"   thresholds t_c: min={tc.min():.3f} median={np.median(tc):.3f} max={tc.max():.3f}")
    with open("performance/all_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote performance/all_metrics.json")


if __name__ == "__main__":
    main()
