"""Purged & embargoed cross-validation splitter. No random/shuffled splitting."""
import numpy as np


def get_purged_embargoed_splits(n_samples: int, n_splits: int, purge: int, embargo: int):
    """Yield (train_idx, val_idx) positional int arrays.

    [0, n_samples) is tiled into n_splits contiguous blocks (remainder goes to
    the last block); fold i validates on block i. Train excludes any index
    inside [v0 - purge, v1 + embargo] around that block, both sides allowed.
    """
    block = n_samples // n_splits
    starts = [i * block for i in range(n_splits)]
    ends = [s + block - 1 for s in starts]
    ends[-1] = n_samples - 1

    idx = np.arange(n_samples)
    for v0, v1 in zip(starts, ends):
        val_idx = idx[v0:v1 + 1]
        train_mask = (idx < v0 - purge) | (idx > v1 + embargo)
        train_idx = idx[train_mask]
        assert train_idx.size > 0, f"fold [{v0},{v1}] has an empty train set"
        assert val_idx.size > 0, f"fold [{v0},{v1}] has an empty val set"
        yield train_idx, val_idx
