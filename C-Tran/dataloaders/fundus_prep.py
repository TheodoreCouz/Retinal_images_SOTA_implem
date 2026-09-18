"""
Dataset preparation for the multi-dataset C-Tran fundus experiments.

Each prep function returns (train_df, val_df, test_df, label_cols) where every
df has:
  - column 0 = ABSOLUTE path to the image file (so MuredDataset can be used
    with img_root='')
  - remaining columns = the binary label columns (in `label_cols` order)

Datasets:
  mured  : 20 classes, *_labels_stratified.csv (paper reproduction)
  rfmid  : 29 reduced RIADD classes (Disease_Risk + 27 diseases + OTHER),
           per RFMiD/training_classes.txt
  prism  : signs.csv (only signs with >=100 total occurrences) + NORMAL from
           is_normal.csv; splits from split_manifest new_split column
"""
import os
import pandas as pd

# Dataset roots. Overridable per-site with the MURED_DIR / RFMID_DIR / PRISM_DIR
# environment variables, so the same code runs against a different data mount
# without editing the file.
MURED_DIR = os.environ.get("MURED_DIR", "/storage2/cousin/datasets/MURED")
RFMID_DIR = os.environ.get("RFMID_DIR", "/storage2/cousin/datasets/RFMiD")
PRISM_DIR = os.environ.get("PRISM_DIR", "/home/cousin/research/Fiber_dino/PRISM v1")

RFMID_CLASSES = ("Disease_Risk,DR,ARMD,MH,DN,MYA,BRVO,TSLN,ERM,LS,MS,CSR,ODC,"
                 "CRVO,TV,AH,ODP,ODE,ST,AION,PT,RT,RS,CRS,EDN,RPEC,MHL,RP,"
                 "OTHER").split(',')


def _abspath_df(df, id_to_path, label_cols):
    out = pd.DataFrame()
    out['path'] = df.iloc[:, 0].map(id_to_path)
    for c in label_cols:
        out[c] = df[c].astype(int).values
    return out.reset_index(drop=True)


def prep_mured():
    img_dir = os.path.join(MURED_DIR, 'images', 'images')
    dfs = {}
    for split, f in [('train', 'train_labels_stratified.csv'),
                     ('val', 'val_labels_stratified.csv'),
                     ('test', 'test_labels_stratified.csv')]:
        df = pd.read_csv(os.path.join(MURED_DIR, f))
        dfs[split] = df
    label_cols = list(dfs['train'].columns[1:])
    prep = {k: _abspath_df(v, lambda i: os.path.join(img_dir, i), label_cols)
            for k, v in dfs.items()}
    return prep['train'], prep['val'], prep['test'], label_cols


def prep_rfmid():
    label_cols = RFMID_CLASSES
    out = {}
    for split, f, d in [('train', 'train_labels.csv', 'Training'),
                        ('val', 'validation_labels_29.csv', 'Validation'),
                        ('test', 'testing_labels_29.csv', 'Test')]:
        df = pd.read_csv(os.path.join(RFMID_DIR, f))
        assert list(df.columns[1:]) == label_cols, f"{f} column mismatch"
        img_dir = os.path.join(RFMID_DIR, d)
        out[split] = _abspath_df(
            df, lambda i, _d=img_dir: os.path.join(_d, f"{i}.png"), label_cols)
    return out['train'], out['val'], out['test'], label_cols


def prep_prism(min_occ=100):
    signs = pd.read_csv(os.path.join(PRISM_DIR, 'signs.csv'))
    isn = pd.read_csv(os.path.join(PRISM_DIR, 'is_normal.csv'))
    man = pd.read_csv(os.path.join(PRISM_DIR,
                                   'split_manifest_20260619_005524.csv'))
    signs = signs.rename(columns={signs.columns[0]: 'filename'})
    isn = isn.rename(columns={isn.columns[0]: 'filename'})

    sign_cols = list(signs.columns[1:])
    counts = signs[sign_cols].sum()
    kept_signs = [c for c in sign_cols if counts[c] >= min_occ]
    label_cols = kept_signs + ['NORMAL']  # signs first, NORMAL last

    m = signs[['filename'] + kept_signs].merge(
        isn[['filename', 'NORMAL']], on='filename', how='inner').merge(
        man[['filename', 'ext', 'new_split']], on='filename', how='inner')

    def path(row):
        return os.path.join(PRISM_DIR, row['new_split'],
                            f"{row['filename']}{row['ext']}")
    m['path'] = m.apply(path, axis=1)

    out = {}
    for split in ['train', 'val', 'test']:
        sub = m[m['new_split'] == split].reset_index(drop=True)
        cols = ['path'] + label_cols
        out[split] = sub[cols].copy()
        for c in label_cols:
            out[split][c] = out[split][c].astype(int)
    return out['train'], out['val'], out['test'], label_cols


PREP = {'mured': prep_mured, 'rfmid': prep_rfmid, 'prism': prep_prism}
