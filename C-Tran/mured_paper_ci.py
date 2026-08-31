"""
MuReD-only: aggregate the 5 seeds under the PAPER's own metrics and report
mean +/- 95% CI, then check whether the paper's point estimates fall inside.
Writes performance/ctran_poly_constlr_bs32_mured_paper_ci.md.
"""
import os
import numpy as np
from scipy import stats
from utils.mured_metrics import compute_mured_metrics

SEEDS = [1, 43, 44, 45, 46]
KEYS = ['ML_F1', 'ML_mAP', 'ML_AUC', 'ML_Score', 'Bin_AUC', 'Bin_F1', 'Model_Score']
PAPER = dict(ML_F1=0.573, ML_mAP=0.685, ML_AUC=0.962, ML_Score=0.824,
             Bin_AUC=0.976, Bin_F1=0.824, Model_Score=0.900)


def run_dir(seed):
    return f"results/mured/ctran_poly_constlr_bs32_seed{seed}"


def ci95(vals):
    vals = np.asarray(vals, float)
    n = len(vals)
    mean = float(vals.mean())
    if n < 2:
        return mean, mean, mean
    h = stats.t.ppf(0.975, n - 1) * vals.std(ddof=1) / np.sqrt(n)
    return mean, mean - h, mean + h


def main():
    per_seed = {}
    for s in SEEDS:
        p = os.path.join(run_dir(s), 'best_preds.npz')
        if not os.path.exists(p):
            print(f"[warn] seed {s}: {p} missing, skipping")
            continue
        z = np.load(p, allow_pickle=True)
        per_seed[s] = compute_mured_metrics(z['test_true'], z['test_pred'],
                                            list(z['label_cols']))
    seeds = sorted(per_seed)
    n = len(seeds)
    print(f"MuReD paper-metric CI over seeds {seeds} (n={n})\n")
    lines = [
        "# MuReD reproduction - paper-metric 95% CI (5 seeds)",
        "",
        f"Config `ctran_poly_constlr_bs32`, seeds {seeds} (n={n}), TEST split, "
        "**paper metrics** (section V.A). Consistency check: does the paper's "
        "value fall inside the reproduction's 95% CI?",
        "",
        "| Metric | Repro mean (95% CI) | Paper | Paper inside CI? |",
        "|---|---|---|---|",
    ]
    for k in KEYS:
        mean, lo, hi = ci95([per_seed[s][k] for s in seeds])
        inside = "yes" if lo <= PAPER[k] <= hi else "no"
        print(f"  {k:12s} {mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
              f"paper {PAPER[k]:.3f}  inside={inside}")
        lines.append(f"| {k} | {mean:.4f} ([{lo:.4f}, {hi:.4f}]) | "
                     f"{PAPER[k]:.3f} | {inside} |")
    lines += ["", "## Per-seed", "",
              "| Seed | " + " | ".join(KEYS) + " |",
              "|---|" + "---|" * len(KEYS)]
    for s in seeds:
        lines.append(f"| {s} | " + " | ".join(f"{per_seed[s][k]:.4f}" for k in KEYS) + " |")
    lines.append("")
    os.makedirs('performance', exist_ok=True)
    out = 'performance/ctran_poly_constlr_bs32_mured_paper_ci.md'
    with open(out, 'w') as f:
        f.write("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
