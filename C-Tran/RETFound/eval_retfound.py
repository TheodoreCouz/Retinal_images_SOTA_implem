"""Evaluate a trained C-Tran/RETFound run under the Fiber 9-metric protocol
(metrics.txt), FLAG-EXCLUSIVE class set — identical to eval_performance_fundus.py
so the numbers sit directly beside the DenseNet C-Tran / RETFound / RIADD rows:
    MuReD 20 (unchanged) | RFMiD 28 (drop Disease_Risk) | PRISM 31 (drop NORMAL).

Reads the cached best_preds.npz (val+test probs/targets), drops the screening
flag column by name, computes the 9 metrics (per-class F1-optimal thresholds on
val), and writes a markdown + json report.

  python eval_retfound.py results/<dataset>/<name> [...]
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CTRAN_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _CTRAN_ROOT)

from dataloaders.fundus_prep import PREP                          # noqa: E402
from utils.fiber_metrics import compute_fiber_metrics, METRIC_KEYS  # noqa: E402

DROP_FLAG = {"mured": None, "rfmid": "Disease_Risk", "prism": "NORMAL"}
PERF = os.path.join(_HERE, "performance")


def drop_flag(probs, targets, label_cols, flag):
    if flag is None:
        return probs, targets, label_cols
    assert flag in label_cols, f"flag '{flag}' not in {label_cols}"
    fi = label_cols.index(flag)
    keep = [i for i in range(len(label_cols)) if i != fi]
    return probs[:, keep], targets[:, keep], [label_cols[i] for i in keep]


def evaluate_run(run_dir):
    dataset = os.path.basename(os.path.dirname(run_dir))
    flag = DROP_FLAG.get(dataset)
    z = np.load(os.path.join(run_dir, "best_preds.npz"), allow_pickle=True)
    lc = [str(c) for c in z["label_cols"]]
    # sanity: label cols match the deterministic PREP order
    prep_cols = [str(c) for c in PREP[dataset]()[3]]
    assert prep_cols == lc, "label-col mismatch vs PREP"
    vp, vt, _ = drop_flag(z["val_pred"], z["val_true"], lc, flag)
    tp, tt, red = drop_flag(z["test_pred"], z["test_true"], lc, flag)
    m = compute_fiber_metrics(vt, vp, tt, tp)
    m["_n_classes"], m["_flag"], m["_dataset"] = len(red), flag, dataset
    m["_n_test"] = int(tt.shape[0])
    return m


def write_report(run_dir, m):
    os.makedirs(PERF, exist_ok=True)
    ds = m["_dataset"]
    tag = f"{ds}_ctran_retfound"
    L = [f"# {ds.upper()} — C-Tran with RETFound (ViT-L/16) backbone",
         "",
         f"C-Tran methodology (label queries + LMT + transformer + diagonal read-out) "
         f"with the `best_retfound_lse_select{'' if ds=='prism' else '_'+ds}` encoder. "
         f"Fiber 9-metric protocol, flag-exclusive: **{m['_n_classes']} classes** "
         f"(dropped `{m['_flag']}`) · test **{m['_n_test']}** · per-class F1-optimal "
         f"thresholds tuned on val.",
         "",
         "| Metric | Value |", "|---|---|"]
    for key, lbl, _ in METRIC_KEYS:
        L.append(f"| {lbl} | {m[key]:.4f} |")
    open(os.path.join(PERF, f"{tag}.md"), "w").write("\n".join(L))
    json.dump({k: (m[k] if not k.startswith("_") else m[k]) for k in m
               if not isinstance(m[k], np.ndarray)},
              open(os.path.join(PERF, f"{tag}.json"), "w"), indent=2, default=float)
    return os.path.join(PERF, f"{tag}.md")


def main():
    runs = [x for x in sys.argv[1:] if not x.startswith("--")]
    for run_dir in runs:
        run_dir = run_dir.rstrip("/")
        m = evaluate_run(run_dir)
        print(f"\n{run_dir}  ({m['_n_classes']} classes, dropped {m['_flag']}, test {m['_n_test']})")
        for key, lbl, _ in METRIC_KEYS:
            print(f"  {lbl:16s} {m[key]:.4f}")
        p = write_report(run_dir, m)
        print(f"  -> wrote {p}")


if __name__ == "__main__":
    main()
