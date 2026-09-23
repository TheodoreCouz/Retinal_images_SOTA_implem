"""Best-effort reconstruction of KAMATALAB's undocumented post-processing.

The paper (Appendix A.1) says only this about it, with no formulas:

    "The authors applied post-processing mainly on two aspects: minor classes
    and correlation between two sub-challenges. The post-processing applied
    are, average blending for major classes only, providing meta-Information,
    replacement with min-max values, and rank voting. Moreover, the risk score
    of sub-challenge one depends on whether the image has any disease in
    sub-challenge two. Decrease predictions with a low-risk score in
    sub-challenge two gave a reasonable raise in performance."

None of this is precise enough to reproduce exactly -- there is no released
code for it. What follows is an engineering GUESS at what these four minor-
class techniques could mean, built to test one specific hypothesis: that the
41% macro recall we measured (vs. precision 57%) is a symptom of thresholding
rare classes at a flat 0.5 the way "average blending for major classes only"
implies was NOT done for minor classes upstream.

Design choices, each a guess:

  * "major" / "minor" split: reused directly from KAMATALAB's OWN multi-stage
    split (G9 = the >65-occurrence tier = major, G8+G11 = minor). This is the
    one part of this file that isn't invented -- it is their own documented
    split, reused for a second purpose they never described doing.

  * "average blending for major classes only" -> major-class columns are left
    exactly as the plain equal-weight path blend, threshold 0.5. Unchanged
    from the existing REPRODUCTION mode.

  * "rank voting" for minor classes -> per class, flag the top-K highest-
    probability test images as positive, instead of thresholding at 0.5. K is
    NOT read off the test labels (that would leak test information no real
    competitor has); it is estimated from the class's positive rate on the
    validation split, scaled to the test set size: K_c = round(prevalence_c *
    N_test). This is the standard meaning of "rank voting" in a Kaggle/
    challenge context: convert an absolute-probability decision into a
    relative-rank decision, which sidesteps the fact that a poorly-calibrated
    rare-class head may never emit any raw score above 0.5 at all.

  * "replacement with min-max values" for minor classes -> per-class min-max
    rescaling of the blended probability over the test set, THEN threshold at
    0.5 on the rescaled value. Implemented and reported separately from rank
    voting for comparison: min-max is a monotonic (rank-preserving) rescaling,
    so combined with a fixed threshold it changes WHERE the cut falls but not
    the ranking, whereas rank voting fixes the cut's absolute count. They are
    alternative guesses at the same paragraph, not additive -- combining them
    with the current rank-voting rule is a no-op (verified empirically).

  * "providing meta-Information" -> NOT reconstructed. No hypothesis in this
    file attempts it; the paper gives no hook for what the meta-information
    was (image quality flags? camera metadata? the RFMiD CSVs don't obviously
    carry any candidate). Left out rather than guessed at random.

  * SC1<->SC2 cross-adjustment -> NOT applied by default (see
    `apply_sc1_sc2_adjustment`, provided but off) because it targets DR AUC /
    SC1 F1, not the SC2 recall bottleneck this file was built to address, and
    folding it in would muddy which change caused which effect.

None of this is verified against KAMATALAB's actual code -- there isn't any
to check against. Treat every number produced downstream of this module as
"what a plausible reconstruction gets," not "the reproduction."
"""
import numpy as np

# CSV iloc indices from the vendored RiaddDataSet*Classes, minus 1 (dump_riadd_paths.py).
G9 = [i - 1 for i in (2, 3, 4, 5, 6, 7, 8, 13, 17)]            # major (>65 train occurrences)
G8 = [i - 1 for i in (10, 12, 14, 18, 23, 24, 26, 29)]          # minor
G11 = [i - 1 for i in (9, 11, 15, 16, 19, 20, 21, 22, 25, 27, 28)]  # minor
MAJOR = sorted(G9)
MINOR = sorted(G8 + G11)


def rank_vote(test_probs, val_probs, val_targets, cols=MINOR):
    """Top-K-by-rank binary decision for `cols`, K estimated from val prevalence."""
    n_test = test_probs.shape[0]
    prevalence = val_targets.mean(axis=0)
    pred = (test_probs >= 0.5).astype(int)  # untouched columns keep the flat-0.5 decision
    for c in cols:
        k = max(1, round(float(prevalence[c]) * n_test))
        order = np.argsort(-test_probs[:, c])
        col = np.zeros(n_test, dtype=int)
        col[order[:k]] = 1
        pred[:, c] = col
    return pred


def minmax_threshold(test_probs, cols=MINOR):
    """Per-class min-max rescale of `cols` over the test set, then threshold 0.5."""
    pred = (test_probs >= 0.5).astype(int)
    for c in cols:
        col = test_probs[:, c]
        rng = col.max() - col.min()
        rescaled = (col - col.min()) / rng if rng > 0 else col
        pred[:, c] = (rescaled >= 0.5).astype(int)
    return pred


def apply_sc1_sc2_adjustment(dr_probs, disease_probs, alpha=0.3):
    """Guess at "decrease SC1 predictions with a low-risk score in SC2".

    Off by default -- provided for completeness, not wired into the default
    reconstruction (see module docstring). alpha is unfit/untuned; this is
    the least-grounded guess in the file.
    """
    sc2_risk = disease_probs.max(axis=1)  # image's own highest disease probability
    return dr_probs * (1 - alpha) + dr_probs * sc2_risk * alpha
