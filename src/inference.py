import pickle
import numpy as np
import pandas as pd

from src.cleaning_data import KEY
from src.features_builder import (
    build_history_features, inject_anchor_state, add_temporal_features,
)
from src.calibration import calibrate

# ---------------------------------------------------------------------------
# Feature definitions per approach
# ---------------------------------------------------------------------------
A1_NUM_FEATS = [
    "dow", "dom", "month", "is_weekend", "is_double_date",
    "day_avg_discount", "day_promo_rate", "day_free_ship"
]
A1_CAT_FEATS = ["shopId", "itemId", "modelId"]
A1_ALL_FEATS = A1_NUM_FEATS + A1_CAT_FEATS

A2_GROUPING    = ["shopId", "itemId", "modelId"]
A2_RIDGE_FEATS = [
    "dow", "dom", "month", "is_weekend", "is_double_date",
    "day_avg_discount", "day_promo_rate", "day_free_ship", "cat_disc_shrunk",
]

A3_NUM_FEATS = [
    "last_price", "price_mean_all", "price_median_all", "price_std_all",
    "price_min_all", "price_max_all", "price_cv_all",
    "p_last5_mean", "p_last5_median", "p_last5_std", "p_last5_min", "p_last5_max",
    "p_last20_mean", "p_last20_median", "p_last20_std",
    "last_vs_mean", "last_vs_min", "momentum_short", "momentum_long",
    "days_since_last", "n_obs_all",
    "disc_freq", "disc_mean", "disc_max", "last_discount",
    "shop_n_listings", "shop_price_median", "cat_median", "cat_mean",
    "dow", "dom", "month", "is_weekend", "is_double_date",
    "day_avg_discount", "day_promo_rate", "day_free_ship", "cat_disc_shrunk",
]
A3_CAT_FEATS = ["shopId", "itemId", "modelId", "cat_id_hist", "brand_hist"]
A3_ALL_FEATS = A3_NUM_FEATS + A3_CAT_FEATS


# ---------------------------------------------------------------------------
# Shared per-day inference loop
#   Each predict function returns (raw_prices, cat_id_hist) for a group of
#   rows.  The loop handles splitting anchors/targets, calibrating, and
#   assembling the final submission.
# ---------------------------------------------------------------------------
def _run_inference(test_df, train_df, predict_fn):
    test_df = test_df.copy()
    test_df["day"] = test_df["capturedAt"].dt.floor("D")
    out = []

    for day, day_rows in test_df.groupby("day"):
        anchors = day_rows[day_rows["price"].notna()].copy()
        targets = day_rows[day_rows["price"].isna()].copy()
        if len(anchors) == 0 or len(targets) == 0:
            continue
        print(f"  [{day.date()}] anchors={len(anchors)}, targets={len(targets):,}")

        raw_t, t_cat = predict_fn(targets, train_df, anchors)
        raw_a, a_cat = predict_fn(anchors, train_df, anchors)

        cal_t, diag = calibrate(
            anchor_true=anchors["price"].values, anchor_pred=raw_a,
            prediction_output=raw_t, method="global_plus_cat",
            anchor_cat=a_cat, target_cat=t_cat,
        )

        targets_out = targets.copy()
        targets_out["price"] = cal_t
        out.append(targets_out)
        print(f"     global_bias={diag['global_bias']:+.3f}, n_cat={diag['n_cat']}")

    anchors_all = test_df[test_df["price"].notna()].copy()
    sub = pd.concat([anchors_all] + out, ignore_index=True)
    return (sub.drop(columns=["day"], errors="ignore")
               .sort_values(["capturedAt"] + KEY)
               .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Approach 1 — Naive Global LightGBM
#   Features: IDs + temporal + anchor day-state.
#   Predicts log(price) directly.
# ---------------------------------------------------------------------------
def _prepare_a1_features(df, train_df, anchors):
    df = df.copy()
    if "cat_id_hist" not in df.columns and "cat_id" in df.columns:
        df = df.rename(columns={"cat_id": "cat_id_hist"})
    df = inject_anchor_state(df, anchors, train_df)
    df = add_temporal_features(df)
    for c in A1_ALL_FEATS:
        if c not in df.columns:
            df[c] = np.nan
    for c in A1_CAT_FEATS:
        df[c] = df[c].astype("category")
    return df


def predict_a1(test_df, train_df, model):
    def _predict(df, train, anchors):
        feat = _prepare_a1_features(df, train, anchors)
        raw = np.exp(model.predict(feat[A1_ALL_FEATS],
                                    num_iteration=model.best_iteration))
        return raw, feat["cat_id_hist"].values
    return _run_inference(test_df, train_df, _predict)


# ---------------------------------------------------------------------------
# Approach 2 — Per-Product Ridge
#   models is a dict: (shopId, itemId, modelId) -> (Ridge, p_min, p_max)
#   Falls back to NaN for products without a fitted Ridge.
# ---------------------------------------------------------------------------
def _prepare_a2_features(df, train_df, anchors):
    df = df.copy()
    if "cat_id_hist" not in df.columns and "cat_id" in df.columns:
        df = df.rename(columns={"cat_id": "cat_id_hist"})
    df = add_temporal_features(df)
    df = inject_anchor_state(df, anchors, train_df)
    return df


def _predict_with_ridges(df, models):
    preds = np.full(len(df), np.nan)
    feat_arr = df[A2_RIDGE_FEATS].fillna(0).values
    keys_arr = df[A2_GROUPING].values

    for i in range(len(df)):
        k = tuple(keys_arr[i])
        if k not in models:
            continue                                 # truly unseen → NaN
        entry = models[k]
        if entry[0] == "median":
            preds[i] = entry[1]                      # constant fallback
        else:                                        # entry[0] == "ridge"
            _, m, p_min, p_max = entry
            pred = float(np.exp(m.predict(feat_arr[i:i+1])[0]))
            preds[i] = np.clip(pred, p_min * 0.5, p_max * 2.0)
    return preds


def predict_a2(test_df, train_df, models):
    def _predict(df, train, anchors):
        feat = _prepare_a2_features(df, train, anchors)
        raw = _predict_with_ridges(feat, models)
        return raw, feat["cat_id_hist"].values
    return _run_inference(test_df, train_df, _predict)

# ---------------------------------------------------------------------------
# Approach 3 — Full Pipeline (history + anchor state, log-residual target)
# ---------------------------------------------------------------------------
def _prepare_a3_features(df, train_df, anchors):
    feat = build_history_features(train_df, df)
    feat = inject_anchor_state(feat, anchors, train_df)
    feat = add_temporal_features(feat)
    mask = feat["last_price"].isna() | (feat["last_price"] <= 0)
    feat.loc[mask, "last_price"] = (
        feat.loc[mask, "shop_price_median"]
            .fillna(feat.loc[mask, "cat_median"])
            .fillna(train_df["price"].median())
    )
    for c in A3_CAT_FEATS:
        feat[c] = feat[c].astype("category")
    return feat


def predict_a3(test_df, train_df, model):
    def _predict(df, train, anchors):
        feat = _prepare_a3_features(df, train, anchors)
        raw = feat["last_price"].values * np.exp(
            model.predict(feat[A3_ALL_FEATS], num_iteration=model.best_iteration))
        return raw, feat["cat_id_hist"].values
    return _run_inference(test_df, train_df, _predict)


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------
def load_model(model_path):
    """Load a saved model. Supports .lgb (LightGBM) and .pkl (Ridge dict)."""
    import lightgbm as lgb
    suffix = model_path.suffix
    if suffix == ".lgb":
        return lgb.Booster(model_file=str(model_path))
    if suffix == ".pkl":
        with open(model_path, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unknown model format: {suffix}")


# Map of approach prefix → (file extension, predictor function)
APPROACHES = {
    "a1": (".lgb", predict_a1),
    "a2": (".pkl", predict_a2),
    "a3": (".lgb", predict_a3),
}


def resolve_approach(model_name):
    """Pick the (extension, predictor) pair based on the model name prefix."""
    for prefix, (ext, fn) in APPROACHES.items():
        if model_name.startswith(prefix):
            return ext, fn
    raise ValueError(
        f"Unknown model name: {model_name!r}. "
        f"Must start with one of: {list(APPROACHES)}"
    )