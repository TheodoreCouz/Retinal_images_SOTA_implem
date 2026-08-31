"""
Label-Powerset Random OverSampling (LP-ROS), Charte et al. 2015.

The paper ("Multi-Label Retinal Disease Classification Using Transformers")
selected LP-ROS with a 10% resampling ratio as the best imbalance strategy.

LP-ROS transforms the multi-label problem into a multi-class one via the
label-powerset (each distinct label combination is a "class"), then clones
random samples from the minority combinations (those below the mean bag size)
until `P%` extra samples have been added.
"""
import numpy as np
import pandas as pd


def lp_ros(df, percentage=10, seed=1):
    """
    df         : training DataFrame (col 0 = image id, rest = binary labels).
    percentage : how many extra samples to add, as a % of the dataset size.
    Returns a new (shuffled) DataFrame with the cloned rows appended.
    """
    rng = np.random.RandomState(seed)
    label_cols = list(df.columns[1:])
    n = len(df)
    samples_to_clone = int(round(n * percentage / 100.0))
    if samples_to_clone <= 0:
        return df.reset_index(drop=True)

    # Group row indices by their label-combination (label powerset).
    labelset_keys = [tuple(r) for r in df[label_cols].values.astype(int)]
    bags = {}
    for i, key in enumerate(labelset_keys):
        bags.setdefault(key, []).append(i)

    mean_size = n / float(len(bags))
    min_bags = [idxs for idxs in bags.values() if len(idxs) < mean_size]
    if not min_bags:
        return df.reset_index(drop=True)

    mean_increment = samples_to_clone // len(min_bags)
    remainder = samples_to_clone - mean_increment * len(min_bags)

    clone_indices = []
    for b, bag in enumerate(min_bags):
        # Distribute the rounding remainder over the first `remainder` bags.
        quota = mean_increment + (1 if b < remainder else 0)
        if quota <= 0:
            continue
        clone_indices.extend(rng.choice(bag, size=quota, replace=True).tolist())

    cloned = df.iloc[clone_indices]
    out = pd.concat([df, cloned], ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out
