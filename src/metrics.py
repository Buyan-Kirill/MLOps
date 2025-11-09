import numpy as np
from scipy.stats import spearmanr
from typing import Set, List, Union


def jaccard(set1: Set, set2: Set) -> float:
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def spearman_rank_correlation(
    y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]
) -> float:
    if len(set(y_true)) < 2 or len(set(y_pred)) < 2:
        return np.nan
    corr, _ = spearmanr(y_true, y_pred)
    return corr
