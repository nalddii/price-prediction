import numpy as np
import pandas as pd


def fit_global_bias(anchor_true, anchor_pred):
    """
    Estimate a single multiplicative bias correction from anchors.
    Returns the median log-residual: log(true / pred).
    """
    at = np.asarray(anchor_true, dtype=float)
    ap = np.asarray(anchor_pred, dtype=float)
    valid = np.isfinite(at) & np.isfinite(ap) & (at > 0) & (ap > 0)
    if valid.sum() < 5:
        return 0.0
    return float(np.median(np.log(at[valid] / ap[valid])))


def fit_per_cat_bias(anchor_true, anchor_pred, anchor_cat,
                     global_bias, k_shrink=5.0):
    """
    Estimate per-category log-residual biases with empirical-Bayes shrinkage
    toward the global bias.
    """
    at  = np.asarray(anchor_true, dtype=float)
    ap  = np.asarray(anchor_pred, dtype=float)
    cat = np.asarray(anchor_cat)

    valid = np.isfinite(at) & np.isfinite(ap) & (at > 0) & (ap > 0)
    if valid.sum() < 5:
        return {}

    log_resid = np.log(at[valid] / ap[valid])
    df = pd.DataFrame({"cat": cat[valid], "lr": log_resid})

    cat_map = {}
    for c, g in df.groupby("cat"):
        n  = len(g)
        mu = np.median(g["lr"])
        if np.isfinite(mu):
            cat_map[c] = (n * mu + k_shrink * global_bias) / (n + k_shrink)
    return cat_map


def calibrate(anchor_true, anchor_pred, prediction_output,
              method="global",
              anchor_cat=None, target_cat=None,
              k_shrink=5.0):
    """
    Calibrate raw model predictions using anchor signals.

    Parameters
    ----------
    anchor_true       : true prices on anchor rows
    anchor_pred       : raw model predictions on anchor rows
    prediction_output : raw model predictions on target rows (to be calibrated)
    method            : "global" or "global_plus_cat"
    anchor_cat        : category ids for each anchor row
    target_cat        : category ids for each target row
    k_shrink          : EB shrinkage strength for per-category bias

    Returns
    -------
    calibrated_prediction : np.ndarray
    diagnostics           : dict with global_bias, n_cat, method
    """
    raw = np.asarray(prediction_output, dtype=float)

    # Layer 1: global bias (always applied)
    global_bias = fit_global_bias(anchor_true, anchor_pred)

    if method == "global":
        calibrated = raw * np.exp(global_bias)
        return calibrated, {"method": method,
                            "global_bias": global_bias,
                            "n_cat": 0}

    elif method == "global_plus_cat":
        if anchor_cat is None or target_cat is None:
            raise ValueError("method='global_plus_cat' requires anchor_cat and target_cat")

        cat_map = fit_per_cat_bias(anchor_true, anchor_pred, anchor_cat,
                                   global_bias=global_bias, k_shrink=k_shrink)

        target_cat = np.asarray(target_cat)
        bias_arr = np.array([cat_map.get(c, global_bias) for c in target_cat])
        calibrated = raw * np.exp(bias_arr)

        return calibrated, {"method": method,
                            "global_bias": global_bias,
                            "n_cat": len(cat_map)}

    else:
        raise ValueError(f"Unknown method: {method!r}. "
                         "Use 'global' or 'global_plus_cat'.")