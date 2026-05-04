import numpy as np
import pandas as pd

BOOL_COLS = ["is_free_shipping", "is_pre_order", "is_official_shop",
             "is_verified", "is_preferred_plus_seller"]
KEY = ["shopId", "itemId", "modelId"]


def basic_clean(df):
    df = df.copy()
    df["capturedAt"] = pd.to_datetime(df["capturedAt"], errors="coerce", utc=True)
    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].map({"t": 1, "f": 0, True: 1, False: 0})
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "priceBeforeDiscount" in df.columns:
        df.loc[df["priceBeforeDiscount"] == 0, "priceBeforeDiscount"] = np.nan
    return df


def detect_price_outliers_per_product(df, key=None,
                                       z_thresh=5.0,
                                       min_obs=5,
                                       min_cluster=3,
                                       min_pct_deviation=0.30,
                                       max_recurrence=2,
                                       end_run_min=2):
    """
    Flag genuinely anomalous price observations.

    A row is flagged only if all of the following hold:
      1. STATISTICALLY EXTREME — MAD z-score > z_thresh.
      2. TEMPORALLY ISOLATED — not part of a run of >= min_cluster.
      3. MAGNITUDE-EXTREME — deviation > min_pct_deviation of median.
      4. NON-RECURRING — the deviating value doesn't recur > max_recurrence times.
      5. NOT AN END-OF-SERIES RUN — if the trailing observations form a
         consecutive run of >= end_run_min flagged points, they are
         protected (likely a change-point starting). A single isolated
         spike at the very end is still flagged.

    Parameters
    ----------
    end_run_min : int
        Minimum length of a consecutive flagged run at the END of the
        series for the points to be protected from flagging.
    """
    if key is None:
        key = KEY

    flags = pd.Series(False, index=df.index)
    summary_rows = []

    work = df[key + ["capturedAt", "price"]].copy()
    work["_idx"] = df.index
    work = work.sort_values(key + ["capturedAt"])

    for keys, g in work.groupby(key, sort=False):
        if len(g) < min_obs:
            continue

        prices = g["price"].values
        median = np.median(prices)
        mad    = np.median(np.abs(prices - median))

        # (1) statistical
        if mad == 0:
            stat_mask = np.abs(prices - median) > 0.01 * abs(median)
        else:
            z = 0.6745 * (prices - median) / mad
            stat_mask = np.abs(z) > z_thresh

        # (2) temporal isolation
        isolated_mask = np.zeros_like(stat_mask)
        i = 0
        while i < len(stat_mask):
            if stat_mask[i]:
                j = i
                while j < len(stat_mask) and stat_mask[j]:
                    j += 1
                if (j - i) < min_cluster:
                    isolated_mask[i:j] = True
                i = j
            else:
                i += 1

        # (3) magnitude
        if abs(median) > 0:
            magnitude_mask = np.abs(prices - median) / abs(median) > min_pct_deviation
        else:
            magnitude_mask = np.zeros_like(stat_mask)

        # (4) non-recurrence
        recurrence_mask = np.ones_like(stat_mask)
        for k in np.where(isolated_mask & magnitude_mask)[0]:
            target = prices[k]
            tol = max(abs(target) * 0.01, 1.0)
            n_similar = np.sum(np.abs(prices - target) <= tol)
            if n_similar > max_recurrence:
                recurrence_mask[k] = False

        # Provisional flags before end-of-series protection
        provisional_mask = isolated_mask & magnitude_mask & recurrence_mask

        # (5) END-OF-SERIES PROTECTION
        end_safe_mask = np.ones_like(provisional_mask)
        n = len(provisional_mask)
        if n > 0 and provisional_mask[-1]:
            run_start = n - 1
            while run_start > 0 and provisional_mask[run_start - 1]:
                run_start -= 1
            run_len = n - run_start
            if run_len >= end_run_min:
                end_safe_mask[run_start:] = False

        final_mask = provisional_mask & end_safe_mask

        n_out = int(final_mask.sum())
        if n_out > 0:
            flags.loc[g["_idx"].values[final_mask]] = True
            row = {f"{k}": v for k, v in zip(key, keys if isinstance(keys, tuple) else (keys,))}
            row.update({
                "n_obs":             len(g),
                "median_price":      float(median),
                "mad":               float(mad),
                "n_stat":            int(stat_mask.sum()),
                "n_after_isolation": int(isolated_mask.sum()),
                "n_after_magnitude": int((isolated_mask & magnitude_mask).sum()),
                "n_provisional":     int(provisional_mask.sum()),
                "n_outliers":        n_out,
            })
            summary_rows.append(row)

    summary = (pd.DataFrame(summary_rows).sort_values("n_outliers", ascending=False)
                                          .reset_index(drop=True)
               if summary_rows else
               pd.DataFrame(columns=key + ["n_obs","median_price","mad",
                                            "n_stat","n_after_isolation",
                                            "n_after_magnitude","n_provisional",
                                            "n_outliers"]))
    return summary, flags


def remove_flagged_outliers(df, flags, verbose=False):
    """
    Drop rows flagged as outliers and return a clean copy.
    """
    flags = flags.reindex(df.index, fill_value=False)

    n_before = len(df)
    n_dropped = int(flags.sum())
    cleaned_df = df.loc[~flags].reset_index(drop=True)
    n_after = len(cleaned_df)

    if verbose:
        pct = (n_dropped / n_before * 100) if n_before else 0.0
        print(f"Rows before:  {n_before:,}")
        print(f"Rows dropped: {n_dropped:,} ({pct:.4f}%)")
        print(f"Rows after:   {n_after:,}")

    return cleaned_df