import pandas as pd
import numpy as np

from src.cleaning_data import KEY


def add_temporal_features(df):
    df = df.copy()
    t = df["capturedAt"]
    df["dow"]            = t.dt.dayofweek.astype("int8")
    df["dom"]            = t.dt.day.astype("int8")
    df["month"]          = t.dt.month.astype("int8")
    df["is_weekend"]     = (t.dt.dayofweek >= 5).astype("int8")
    df["is_double_date"] = (t.dt.day == t.dt.month).astype("int8")
    return df


def build_history_features(history_df, target_df, key=None):
    if key is None:
        key = KEY

    history_df = history_df.sort_values("capturedAt").reset_index(drop=True)
    target_df  = target_df.copy().reset_index(drop=True)
    target_df["_orig_idx"] = np.arange(len(target_df))

    # (a) asof — last snapshot per product
    asof_left = target_df[["capturedAt"] + key + ["_orig_idx"]].sort_values("capturedAt")
    asof_right = (history_df[["capturedAt"] + key + ["price","show_discount",
                              "promotionId","cat_id","brand"]]
                  .rename(columns={"capturedAt":"last_capturedAt", "price":"last_price",
                                   "show_discount":"last_discount",
                                   "promotionId":"last_promotionId",
                                   "cat_id":"cat_id_hist", "brand":"brand_hist"})
                  .sort_values("last_capturedAt"))
    asof = pd.merge_asof(asof_left, asof_right,
                         left_on="capturedAt", right_on="last_capturedAt",
                         by=key, allow_exact_matches=False, direction="backward")
    asof["days_since_last"] = (asof["capturedAt"] - asof["last_capturedAt"]).dt.total_seconds()/86400
    asof_feats = asof.drop(columns=["capturedAt"] + key)

    # (b) all-time aggregates
    agg = (history_df.groupby(key)["price"]
           .agg(price_mean_all="mean", price_median_all="median", price_std_all="std",
                price_min_all="min", price_max_all="max", n_obs_all="count")
           .reset_index())

    # (c) windowed last-N
    def last_n(df, n, prefix):
        g = (df.groupby(key, group_keys=False)
               .apply(lambda x: x.nlargest(n, "capturedAt")["price"]
                                  .agg(["mean","median","std","min","max"])))
        return g.reset_index().rename(columns={
            "mean":f"{prefix}_mean","median":f"{prefix}_median",
            "std":f"{prefix}_std","min":f"{prefix}_min","max":f"{prefix}_max"})
    last5  = last_n(history_df, 5,  "p_last5")
    last20 = last_n(history_df, 20, "p_last20")

    # (d) discount cadence
    disc = (history_df.assign(_on_sale=(history_df["show_discount"].fillna(0) > 0).astype(int))
            .groupby(key)
            .agg(disc_freq=("_on_sale","mean"), disc_mean=("show_discount","mean"),
                 disc_max=("show_discount","max"))
            .reset_index())

    # (e) shop & cat stats
    shop = (history_df.groupby("shopId")
            .agg(shop_n_listings=("itemId","nunique"), shop_price_median=("price","median"))
            .reset_index())
    cat = (history_df.groupby("cat_id")["price"]
           .agg(cat_median="median", cat_mean="mean").reset_index()
           .rename(columns={"cat_id":"cat_id_hist"}))

    out = (target_df.merge(asof_feats, on="_orig_idx", how="left")
                    .merge(agg,    on=key,           how="left")
                    .merge(last5,  on=key,           how="left")
                    .merge(last20, on=key,           how="left")
                    .merge(disc,   on=key,           how="left")
                    .merge(shop,   on="shopId",      how="left")
                    .merge(cat,    on="cat_id_hist", how="left"))

    out["price_cv_all"]   = out["price_std_all"] / (out["price_mean_all"] + 1e-6)
    out["last_vs_mean"]   = out["last_price"]    / (out["price_mean_all"] + 1e-6)
    out["last_vs_min"]    = out["last_price"]    / (out["price_min_all"]  + 1e-6)
    out["momentum_short"] = out["last_price"]    / (out["p_last5_mean"]   + 1e-6)
    out["momentum_long"]  = out["p_last5_mean"]  / (out["p_last20_mean"]  + 1e-6)

    return out.sort_values("_orig_idx").drop(columns=["_orig_idx"]).reset_index(drop=True)


def inject_anchor_state(target_df, anchors_df, history_df, key=None, k_shrink=2, include_cat_disc=True):
    if key is None:
        key = KEY

    target_df = target_df.copy()
    if "cat_id_hist" not in target_df.columns:
        cat_lookup = (history_df.sort_values("capturedAt")
                      .groupby(key)["cat_id"].last().reset_index()
                      .rename(columns={"cat_id":"cat_id_hist"}))
        target_df = target_df.merge(cat_lookup, on=key, how="left")

    day_avg_disc = anchors_df["show_discount"].mean()
    target_df["day_avg_discount"] = day_avg_disc
    target_df["day_promo_rate"]   = (anchors_df["promotionId"].fillna(0) != 0).mean()
    target_df["day_free_ship"]    = anchors_df["is_free_shipping"].mean()

    if include_cat_disc:
        cat = (anchors_df.groupby("cat_id")
       .agg(cat_disc=("show_discount","mean"), cat_n=("show_discount","size"))
       .reset_index().rename(columns={"cat_id":"cat_id_hist"}))

        cat_disc_target = cat["cat_disc"].median()

        cat["cat_disc_shrunk"] = (
            cat["cat_n"] * cat["cat_disc"].fillna(cat_disc_target)
            + k_shrink * cat_disc_target
        ) / (cat["cat_n"] + k_shrink)

        target_df = target_df.merge(cat[["cat_id_hist","cat_disc_shrunk"]],
                                    on="cat_id_hist", how="left")
        target_df["cat_disc_shrunk"] = target_df["cat_disc_shrunk"].fillna(day_avg_disc)
    return target_df