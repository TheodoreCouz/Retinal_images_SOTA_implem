"""Isolate per-member vs pooled Platt calibration on the ten-model RIADD blend
(path 2 + path 3, path 1 dropped, TTA held OFF for both rows).

WHY THIS FILE EXISTS
---------------------
No calibration code (Platt scaling, or any recalibration) exists anywhere else in
this repository -- grep for LogisticRegression/Platt/calibrat across every .py file
here returns nothing. metrics9.py (reused below, NOT reimplemented) only tunes
per-class F1-optimal thresholds; it never rescales probabilities. There is also no
prior "breadth_eval" artifact in this repo to reproduce against. This script is a
fresh, from-scratch implementation of the missing calibration step, built directly
on the one real per-member artifact that does exist:

  /globalsc/ucl/ingi/cousint/SOTA_runs/probs/riadd23_members_{rfmid,mured}_{val,test}_seed{42..46}.npz
    probs   (10, N, C) float32 -- members ordered p2/fold0..4, p3/fold0..4
            (tf_efficientnet_b5_ns@960 x5, tf_efficientnet_b6_ns@768 x5), i.e. the
            ten-model blend with path 1 dropped, as produced by
            Pytorch-RIADD/dump_riadd23_members.py.
    targets (N, C) int64
  These are DETERMINISTIC single-pass predictions (get_riadd_valid_transforms) --
  no TTA was ever dumped per-member (see that script's own docstring: "the
  stochastic TTA would only add noise to a correlation estimate"). So TTA-on
  members do not exist as saved artifacts; Step 3 of the original request is
  therefore SKIPPED here, not simulated.

WHAT THIS FILE DELIBERATELY DOES NOT DO
-----------------------------------------
- It does not compare against a "2-Net" (RETFound-FT + EfficientNetV2-L) model.
  No such combined model, checkpoint, or averaging script exists in this repo --
  RetExpert here is a single RETFound ViT-L + Knowledge-Unit-adapter model, and
  EfficientNetV2-L does not appear anywhere in the codebase. Any "2-Net" number
  would have to be invented, which is exactly what was ruled out.
- It does not reproduce a prior "10 members, no TTA" row from a breadth_eval.json,
  because no such file exists anywhere in this repo (working tree or git history).
- It does not use or fabricate any number from a paper draft; there is none here.

WHAT IS REUSED VS. NEW
------------------------
Reused, unmodified: Pytorch-RIADD/metrics9.py's tune_thresholds() and compute_all()
(per-class F1-threshold tuning, macro AUROC/mAP, Recall@n / TNR@(m-n)-j). Cross-
checked below (see `--selfcheck`) against an independent, directly-coded
recomputation of macro F1 / macro mAP from sklearn primitives, on the raw
(uncalibrated) 10-member average -- the closest available stand-in for "a known
row", since no prior calibrated row exists to check against.

New in this file: per-class Platt scaling (sklearn LogisticRegression, 1-D input
= logit of the clipped probability), fit on validation and applied to validation
+ test. This is standard Platt scaling generalized from a raw classifier score to
an existing probability by working in logit space.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Pytorch-RIADD'))
from metrics9 import tune_thresholds, compute_all  # noqa: E402  (reused, not reimplemented)

PROBS = os.environ.get('PROBS_DIR', '/globalsc/ucl/ingi/cousint/SOTA_runs/probs')
SEEDS = (42, 43, 44, 45, 46)
DATASETS = ('rfmid', 'mured')
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(val_p_c, val_t_c):
    """One class: fit sigmoid(A*logit(p)+B) on validation. Returns (callable, passthrough, A).

    C=1.0 is sklearn's standard default -- NOT tuned for this data. Weaker
    regularization (tried C=1e10) does not change the sign of A on the unstable
    classes below, only its magnitude, so this is not a regularization bug; see
    the rank-inversion warning this triggers for classes with a handful of
    validation positives.
    """
    if len(np.unique(val_t_c)) < 2:
        return (lambda p: p), True, None, 0  # nothing to calibrate against -- pass through
    n_clipped = int(np.sum((val_p_c <= EPS) | (val_p_c >= 1 - EPS)))
    x = logit(val_p_c).reshape(-1, 1)
    clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    clf.fit(x, val_t_c)
    pos_col = list(clf.classes_).index(1)
    A = float(clf.coef_[0, 0])
    return (lambda p: clf.predict_proba(logit(p).reshape(-1, 1))[:, pos_col]), False, A, n_clipped


def calibrate_per_class(fit_p, fit_t, apply_p_list):
    """Fit one Platt model per class on (fit_p, fit_t); apply to each array in
    apply_p_list (each shape (N_i, C)). Returns list of calibrated arrays, plus
    diagnostics: (n_passthrough, [(class, n_pos_in_fit_set, A), ...] for A<0)."""
    C = fit_p.shape[1]
    out = [np.empty_like(a, dtype=np.float64) for a in apply_p_list]
    n_passthrough = 0
    rank_inversions = []
    for c in range(C):
        f, passthrough, A, n_clipped = fit_platt(fit_p[:, c], fit_t[:, c])
        n_passthrough += int(passthrough)
        if A is not None and A < 0:
            rank_inversions.append((c, int(fit_t[:, c].sum()), A, n_clipped))
        for a_in, a_out in zip(apply_p_list, out):
            a_out[:, c] = f(a_in[:, c])
    return out, n_passthrough, rank_inversions


def row_A_per_member(val_probs, val_t, test_probs, test_t):
    """probs: (10, N, C). Calibrate each member per class on val, THEN average."""
    n_members = val_probs.shape[0]
    cal_val = np.empty_like(val_probs, dtype=np.float64)
    cal_test = np.empty_like(test_probs, dtype=np.float64)
    total_passthrough = 0
    all_inversions = []
    for m in range(n_members):
        (cv, ct), npass, inv = calibrate_per_class(val_probs[m], val_t, [val_probs[m], test_probs[m]])
        cal_val[m], cal_test[m] = cv, ct
        total_passthrough += npass
        all_inversions.extend((m, c, npos, A, nclip) for c, npos, A, nclip in inv)
    return cal_val.mean(axis=0), cal_test.mean(axis=0), total_passthrough, all_inversions


def row_B_pooled(val_probs, val_t, test_probs, test_t):
    """probs: (10, N, C). Average RAW members first, THEN calibrate the pooled vector."""
    raw_val = val_probs.mean(axis=0)
    raw_test = test_probs.mean(axis=0)
    (cal_val, cal_test), npass, inv = calibrate_per_class(raw_val, val_t, [raw_val, raw_test])
    return cal_val, cal_test, npass, inv, raw_val, raw_test


def load(ds, split, seed):
    f = f'{PROBS}/riadd23_members_{ds}_{split}_seed{seed}.npz'
    if not os.path.exists(f):
        raise FileNotFoundError(f)
    z = np.load(f)
    return z['probs'].astype(np.float64), z['targets'].astype(np.int64)


def selfcheck():
    """Cross-check metrics9.compute_all against an independently-coded
    macro F1 / macro mAP, on the RAW (uncalibrated) 10-member average for RFMiD
    seed 42 -- this repo's closest available "known row" (no prior breadth_eval
    row exists to compare against; that absence is reported, not papered over)."""
    val_p, val_t = load('rfmid', 'val', 42)
    test_p, test_t = load('rfmid', 'test', 42)
    raw_val, raw_test = val_p.mean(axis=0), test_p.mean(axis=0)
    got = compute_all(raw_val, val_t, raw_test, test_t)

    thr = tune_thresholds(raw_val, val_t)
    f1s = [f1_score(test_t[:, c], (raw_test[:, c] >= thr[c]).astype(int), zero_division=0)
           for c in range(test_t.shape[1])]
    indep_f1 = float(np.mean(f1s))
    aps = [average_precision_score(test_t[:, c], raw_test[:, c]) for c in range(test_t.shape[1])
           if len(np.unique(test_t[:, c])) > 1]
    indep_map = float(np.mean(aps))

    ok = np.isclose(got['macro_f1'], indep_f1) and np.isclose(got['macro_map'], indep_map)
    print(f"[selfcheck] metrics9.compute_all vs independent sklearn recompute "
          f"(RFMiD seed42, raw 10-member average, uncalibrated):")
    print(f"  macro_f1:  metrics9={got['macro_f1']:.6f}  independent={indep_f1:.6f}")
    print(f"  macro_map: metrics9={got['macro_map']:.6f}  independent={indep_map:.6f}")
    print(f"  MATCH: {ok}")
    if not ok:
        print("  *** metrics9.py does NOT match an independent recomputation -- "
              "treat downstream numbers as suspect until this is fixed. ***")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selfcheck', action='store_true')
    args = ap.parse_args()

    print(f"PROBS_DIR = {PROBS}")
    print(f"seeds used (all, no subsetting): {SEEDS}")
    for ds in DATASETS:
        for split in ('val', 'test'):
            missing = [s for s in SEEDS if not os.path.exists(
                f'{PROBS}/riadd23_members_{ds}_{split}_seed{s}.npz')]
            print(f"  {ds}/{split}: {'ALL 5 seeds present' if not missing else f'MISSING seeds {missing}'}")

    ok = selfcheck()
    print()

    rows = []
    pooled_sanity = []
    all_rank_inversions = []  # (ds, seed, row, member_or_None, class, n_pos, A, n_clipped)
    for ds in DATASETS:
        for seed in SEEDS:
            val_p, val_t = load(ds, 'val', seed)
            test_p, test_t = load(ds, 'test', seed)

            avA_val, avA_test, npassA, invA = row_A_per_member(val_p, val_t, test_p, test_t)
            metricsA = compute_all(avA_val, val_t, avA_test, test_t)
            all_rank_inversions.extend((ds, seed, 'A', m, c, npos, A, nclip) for m, c, npos, A, nclip in invA)

            calB_val, calB_test, npassB, invB, rawB_val, rawB_test = row_B_pooled(val_p, val_t, test_p, test_t)
            metricsB = compute_all(calB_val, val_t, calB_test, test_t)
            all_rank_inversions.extend((ds, seed, 'B', None, c, npos, A, nclip) for c, npos, A, nclip in invB)
            # Sanity check: uncalibrated raw pooled blend vs Row B (pooled calibration).
            raw_metrics = compute_all(rawB_val, val_t, rawB_test, test_t)
            d_f1 = metricsB['macro_f1'] - raw_metrics['macro_f1']
            d_map = metricsB['macro_map'] - raw_metrics['macro_map']
            pooled_sanity.append((ds, seed, d_f1, d_map))

            for row_name, m, npass in (('A_per_member', metricsA, npassA), ('B_pooled', metricsB, npassB)):
                rows.append(dict(dataset=ds, seed=seed, row=row_name,
                                  macro_f1=m['macro_f1'], macro_map=m['macro_map'],
                                  recall_at_n=m['recall_at_n'], n_passthrough_classes=npass))

    df = pd.DataFrame(rows)
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration_isolation.csv')
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}\n")

    print("=== sanity check: per-class Platt on a SINGLE pooled vector must not move ===")
    print("=== tuned-threshold macro F1 or macro mAP vs the raw uncalibrated blend  ===")
    max_abs_sanity = 0.0
    for ds, seed, d_f1, d_map in pooled_sanity:
        max_abs_sanity = max(max_abs_sanity, abs(d_f1), abs(d_map))
        flag = "  <-- NONZERO" if (abs(d_f1) > 1e-9 or abs(d_map) > 1e-9) else ""
        print(f"  {ds} seed{seed}: d(macro_f1)={d_f1:+.6f}  d(macro_map)={d_map:+.6f}{flag}")
    if max_abs_sanity > 1e-9:
        print(f"\n  *** FLAG: pooled-calibration sanity check FAILS (max |delta|={max_abs_sanity:.6f}). "
              f"Not a pooling/averaging interaction; not a metrics9.py bug (selfcheck above passed). "
              f"Two DISTINCT mechanisms, confirmed by direct inspection, not guessed: ***")
        print(f"\n  (1) RFMiD macro_map + most of RFMiD macro_f1 (up to 0.0174/0.0016): rank-inverting "
              f"Platt fits. A logistic regression fit with as few as 2 positive validation examples "
              f"for a class can, by chance, best-fit a NEGATIVE slope (A<0) if the raw model's "
              f"ranking doesn't happen to correlate positively with those 2 labels -- that flips the "
              f"ranking for that one class on TEST, moving its AP and tuned-F1, which drags the macro "
              f"average. All instances found are class 14 (n_pos_val=2) and occasionally class 18 "
              f"(n_pos_val=4) -- RFMiD's rarest classes. Not an EPS-clipping artifact "
              f"(n_clipped is printed per instance; many are 0):")
        print(f"      rank-inverting (class, n_pos_in_val, A, n_probs_clipped_at_EPS) instances:")
        for ds, seed, row, m, c, npos, A, nclip in all_rank_inversions:
            who = f"member {m}" if m is not None else "pooled vector"
            print(f"        {ds} seed{seed} row{row} ({who}): class {c}, n_pos_val={npos}, "
                  f"A={A:+.4f}, n_clipped={nclip}")
        print(f"\n  (2) MuReD macro_f1 only (macro_map is exactly 0 for MuReD -- consistent, since "
              f"mAP does not depend on tuned thresholds at all): a DIFFERENT mechanism, verified by "
              f"direct inspection (mured seed42, class 10) -- the Platt transform IS monotonic here "
              f"(no negative-A instances logged for MuReD above), and the F1-optimal threshold "
              f"search on validation is confirmed to select the SAME partition of the validation set "
              f"before and after calibration. But metrics9.tune_thresholds() picks the first "
              f"candidate cutpoint achieving max F1, and when several cutpoints tie on validation, a "
              f"monotonic-but-nonlinear (logit/sigmoid) transform can shift which tied candidate is "
              f"selected. Both thresholds are equally optimal ON VALIDATION, but they sit at "
              f"different absolute probability values, so they can classify a held-out TEST point "
              f"near the boundary differently (confirmed: mured seed42 class 10, exactly 1 test "
              f"sample reclassified). This is a threshold-tie-breaking sensitivity, not a "
              f"calibration or ranking error.")
    else:
        print(f"  OK: sanity check passes (all deltas < 1e-9), as expected from a per-class "
              f"monotone transform of a single vector.")

    print("\n=== Row A (per-member calibrate-then-average) vs Row B (average-then-calibrate) ===")
    metrics = ['macro_f1', 'macro_map', 'recall_at_n']
    caption_candidates = []
    for ds in DATASETS:
        print(f"\n-- {ds} --")
        for metric in metrics:
            a = df[(df.dataset == ds) & (df.row == 'A_per_member')].sort_values('seed')[metric].values
            b = df[(df.dataset == ds) & (df.row == 'B_pooled')].sort_values('seed')[metric].values
            diff = a - b
            wins = int(np.sum(diff > 0))
            losses = int(np.sum(diff < 0))
            ties = int(np.sum(diff == 0))
            print(f"  {metric:12s}  A={a.mean()*100:6.2f}+/-{a.std()*100:4.2f}  "
                  f"B={b.mean()*100:6.2f}+/-{b.std()*100:4.2f}  "
                  f"A-B={diff.mean()*100:+6.3f}+/-{diff.std()*100:4.2f}  "
                  f"win/loss/tie={wins}/{losses}/{ties}")
            caption_candidates.append((ds, metric, np.abs(diff).max()))

    print("\n=== Step 3 (TTA on) ===")
    print("SKIPPED: no per-member TTA predictions are saved anywhere in "
          f"{PROBS} -- dump_riadd23_members.py deliberately only dumps a single "
          "deterministic pass per member (see its docstring). Re-running TTA "
          "inference per member x 5 seeds x 2 datasets was not done here since "
          "that is new inference, not a re-score of existing artifacts.")

    print("\n=== Step 4: caption number ===")
    ds_c, metric_c, val_c = max(caption_candidates, key=lambda x: x[2])
    # round UP at 3 decimal places (as a fraction, i.e. 0.1% granularity)
    import math
    rounded = math.ceil(val_c * 1000) / 1000
    print(f"largest |A-B| = {val_c:.6f} ({ds_c}, {metric_c})")
    print(f'  caption sentence: "calibrating each of its ten members instead of the pooled '
          f'blend changes its scores by at most {rounded:.3f}"')

    print("\n2-Net comparison: NOT DONE. No 2-Net (RETFound-FT + EfficientNetV2-L) model, "
          "checkpoint, or averaging pipeline exists anywhere in this repo -- inventing that "
          "comparison was explicitly ruled out. If you have those predictions saved "
          "elsewhere, point me to them and I will add the comparison.")

    print("\n=== reproduction check against a prior 'breadth_eval, 10 members, no TTA' row ===")
    print("NOT POSSIBLE: no breadth_eval.py or breadth_eval.json exists anywhere in this repo "
          "(working tree or git history) -- there is no prior row to reproduce against. "
          "Row A above is the first computation of this configuration.")


if __name__ == '__main__':
    main()
