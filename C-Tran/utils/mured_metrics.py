"""
MuReD / RIADD evaluation metrics, matching
Rodriguez et al., "Multi-Label Retinal Disease Classification Using
Transformers" (IEEE JBHI 2023), Section V.A.

Let T = the set of DISEASE labels = all labels EXCEPT "NORMAL".

    ML_mAP  = mean average-precision over T
    ML_F1   = mean F1 (thr 0.5) over T
    ML_AUC  = mean ROC-AUC over T
    ML_Score    = (ML_mAP + ML_AUC) / 2
    Bin_AUC = ROC-AUC of the NORMAL class  (disease detection / screening)
    Bin_F1  = F1  (thr 0.5) of the NORMAL class
    Model_Score = (ML_Score + Bin_AUC) / 2   <-- the headline number (0.900)

The Bin_* metrics being exactly the NORMAL-class column was confirmed against
the paper's own Table XII (NORMAL: AUC 0.976, F1 0.824 == Table XI Bin AUC/F1).
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, \
    precision_score, recall_score


def _safe_auc(yt, yp):
    return roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else None


def _safe_ap(yt, yp):
    return average_precision_score(yt, yp) if yt.sum() > 0 else None


def compute_mured_metrics(y_true, y_prob, label_cols,
                          normal_name='NORMAL', threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    normal_idx = label_cols.index(normal_name)
    disease_idx = [i for i in range(len(label_cols)) if i != normal_idx]

    per_class = {}
    f1s, aps, aucs = [], [], []
    for i in range(len(label_cols)):
        yt, yp, yhat = y_true[:, i], y_prob[:, i], y_pred[:, i]
        auc = _safe_auc(yt, yp)
        ap = _safe_ap(yt, yp)
        f1 = f1_score(yt, yhat, zero_division=0)
        prec = precision_score(yt, yhat, zero_division=0)
        rec = recall_score(yt, yhat, zero_division=0)
        per_class[label_cols[i]] = dict(
            n_pos=int(yt.sum()), precision=prec, recall=rec, f1=f1,
            auc=auc, ap=ap)
        if i in disease_idx:
            f1s.append(f1)                      # F1 always defined (0 if none)
            if auc is not None:
                aucs.append(auc)
            if ap is not None:
                aps.append(ap)

    ML_F1 = float(np.mean(f1s))
    ML_mAP = float(np.mean(aps))
    ML_AUC = float(np.mean(aucs))
    ML_Score = (ML_mAP + ML_AUC) / 2.0

    yt_n, yp_n, yhat_n = y_true[:, normal_idx], y_prob[:, normal_idx], y_pred[:, normal_idx]
    Bin_AUC = _safe_auc(yt_n, yp_n)
    Bin_AUC = float(Bin_AUC) if Bin_AUC is not None else float('nan')
    Bin_F1 = float(f1_score(yt_n, yhat_n, zero_division=0))
    Model_Score = (ML_Score + Bin_AUC) / 2.0

    return dict(
        ML_F1=ML_F1, ML_mAP=ML_mAP, ML_AUC=ML_AUC, ML_Score=ML_Score,
        Bin_AUC=Bin_AUC, Bin_F1=Bin_F1, Model_Score=Model_Score,
        n_disease_auc=len(aucs), n_disease_total=len(disease_idx),
        per_class=per_class)


def format_metrics(m):
    lines = []
    lines.append(
        "  ML_F1={ML_F1:.3f}  ML_mAP={ML_mAP:.3f}  ML_AUC={ML_AUC:.3f}  "
        "ML_Score={ML_Score:.3f}  |  Bin_AUC={Bin_AUC:.3f}  Bin_F1={Bin_F1:.3f}"
        "  ||  Model_Score={Model_Score:.4f}".format(**m))
    return "\n".join(lines)


def format_per_class(m):
    lines = ["  {:8s} {:>5s} {:>6s} {:>6s} {:>6s} {:>6s}".format(
        "class", "n_pos", "Prec", "Rec", "F1", "AUC")]
    for c, d in m['per_class'].items():
        auc = "  -  " if d['auc'] is None else "{:.3f}".format(d['auc'])
        lines.append("  {:8s} {:5d} {:6.3f} {:6.3f} {:6.3f} {:>6s}".format(
            c, d['n_pos'], d['precision'], d['recall'], d['f1'], auc))
    return "\n".join(lines)
