from __future__ import annotations

import numpy as np


def native_importance(best, n_features: int):
    est = best
    if hasattr(best, "named_steps"):
        est = best.named_steps.get("clf", best)
    if hasattr(est, "feature_importances_"):
        return np.asarray(est.feature_importances_)
    if hasattr(est, "coef_"):
        return np.abs(np.asarray(est.coef_)).ravel()
    if hasattr(est, "estimators_"):
        trees = np.ravel(est.estimators_)
        if hasattr(trees[0], "feature_importances_"):
            return np.mean([t.feature_importances_ for t in trees], axis=0)
    return None
