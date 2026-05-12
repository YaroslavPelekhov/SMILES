"""
splitting.py — Stratified K-fold split with held-out validation slice.

For each of ``K`` folds we hold one stratified slice out as the test split,
carve a stratified validation slice out of the remainder, and use the rest
for training.

Two desirable properties of this scheme:
    1. Metrics in ``run_evaluation`` are averaged across K folds, reducing
       variance over a 689-sample dataset.
    2. The union of all (train ∪ val) indices across folds covers every
       sample, so ``solution.py``'s final probe is fit on all 689 labelled
       points before being applied to ``test.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


N_FOLDS: int = 5
VAL_FRACTION_OF_TRAIN: float = 0.15  # fraction of the non-test pool used for val


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,    # ignored — kept for API compatibility
    val_size: float = 0.15,     # ignored — kept for API compatibility
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Stratified K-fold split with a per-fold validation slice.

    For each of ``N_FOLDS`` folds:
        * ``idx_test``   — that fold's stratified held-out slice.
        * ``idx_val``    — a stratified ``VAL_FRACTION_OF_TRAIN`` slice
                            carved from the remaining (non-test) samples.
        * ``idx_train``  — everything else.

    Returns:
        ``N_FOLDS`` tuples of ``(idx_train, idx_val, idx_test)``.
    """
    y = np.asarray(y)
    idx = np.arange(len(y))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)
    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []

    for fold_idx, (idx_non_test, idx_test) in enumerate(skf.split(idx, y)):
        idx_train, idx_val = train_test_split(
            idx_non_test,
            test_size=VAL_FRACTION_OF_TRAIN,
            random_state=random_state + fold_idx,
            stratify=y[idx_non_test],
        )
        splits.append((idx_train, idx_val, idx_test))

    return splits
