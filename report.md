
## Draft `REPORT.md`

The full writeup, structured to map directly to the 5 evaluation criteria. I'll give you the skeleton with prompts for the content you should fill in:

```markdown
# Technical Report — Price Prediction Pipeline

## 1. Problem framing

Brief restatement (3–4 sentences) of the outage-day reconstruction task and
why it's specifically a *change-detection* problem rather than a forecasting
problem. Reference the 87% price-stability finding.

## 2. Exploratory data analysis

Key findings from `notebooks/01_eda.ipynb`, with the figures embedded:

- **Price stability**: 87% of products have constant price across all
  observations. Median `range_ratio` = 1.00, 99th percentile = 1.76.
  → motivates `last_price` as the dominant feature and log-residual target.
- **Wide price scale**: prices range from ~1k to 100M+ IDR (5+ orders of
  magnitude). → motivates log-space modelling and MAPE/MedAPE over MAE/RMSE.
- **Category heterogeneity**: median q90/q10 within a category = 8.5×.
  → motivates per-category calibration on top of any global correction.
- **Currency**: appears to be IDR. (Show the price distribution figure.)

[insert figure: log-price distribution + per-product range ratio]

## 3. Outlier detection

Document the iterative outlier story — this is one of the strongest pieces
of the writeup because it shows analytical rigour:

1. **Naïve IQR** flagged 28% of rows. Inspection revealed the rule fails on
   bounded variables (`review_rating`) and heavy-tailed counts (`cmt_count`).
2. **Per-product MAD z-score** flagged ~5%. Visual inspection showed nearly
   all flags were *sustained* price changes (real sales/relistings).
3. **Added temporal isolation filter** — flag only short runs.
   Now flagging short recurring discount events as anomalies.
4. **Added magnitude + non-recurrence filters** — only flag deviations
   >30% from median that don't recur. Now mostly correct, but flags
   end-of-series points where change-point and anomaly are
   indistinguishable.
5. **Added end-of-series rule** — protect a trailing run of ≥2 flagged
   points as a candidate change-point; isolated single trailing spikes
   remain flagged.

Final outlier rate: ~Y% of training rows. These were removed before
training. Validation and test sets were not modified.

[insert figure: top-10 flagged products with the final detector]

## 4. Feature engineering

### Temporal features
`dow`, `dom`, `month`, `is_weekend`, `is_double_date`. The "double date"
feature (`day == month`) captures Indonesian e-commerce flash-sale events
(11.11, 12.12, etc.) which are visible in the price data.

### Categorical identifiers
LightGBM handles `shopId`, `itemId`, `modelId`, `cat_id`, `brand` natively
as categoricals. For the per-product Ridge in Approach 2, the entity grouping
itself encodes identity, so no explicit categorical encoding is needed.

### History-derived features (Approach 3 only)
- `last_price` — the dominant predictor
- All-time aggregates: mean, median, std, min, max, count
- Last-N windows: 5 and 20 most recent observations
- Discount cadence: frequency, mean depth, max depth
- Shop and category medians as cold-start fallbacks
- Derived ratios: `last_vs_mean`, `momentum_short`, `momentum_long`

All history features use leak-free `merge_asof` with
`allow_exact_matches=False`.

### Anchor-derived features (all approaches)
The pipeline's signature mechanism. From the 100 same-day anchors:

- `day_avg_discount`, `day_promo_rate`, `day_free_ship`: scalar
  marketplace mood indicators
- `cat_disc_shrunk`: per-category discount level with empirical-Bayes
  shrinkage toward the *median across categories* (not the mean — the
  show_discount distribution is bimodal).

These are stamped onto every target row of the same day, giving the model
a real-time signal even though target rows have no observable features.

### Cold-start handling
Products with no history use a fallback chain:
`last_price` → `shop_price_median` → `cat_median` → global median.

## 5. Modelling approach

### Approach 1 — Naive Global LightGBM
Single model on IDs + temporal + anchor state. Predicts `log(price)`
directly. Tests: *what's the marketplace-wide baseline given only the
features available at inference time?*

Critical design choice: **train and inference share the same feature
distribution by construction**. We never train on features (like
`show_discount` or `stock`) that are NaN at test time.

### Approach 2 — Per-Product Ridge
Same feature set as A1, but one Ridge per `(shopId, itemId, modelId)`.
Tests: *holding features constant, does per-product modelling beat a global
model?*

Why Ridge, not LightGBM? Per-product sample sizes are tiny (median ~15
observations). Tree models can't split meaningfully on so few rows; Ridge
with L2 shrinkage handles small-n gracefully.

Why deliberately exclude `last_price`? To isolate the personalisation
question. Mixing history features into A2 would confound "did per-entity
modelling help?" with "did history features help?". The latter is what A3
tests.

### Approach 3 — Full Pipeline
Adds history-derived features. Target is `log(price / last_price)` (a
log-residual against last seen price). Tests: *does adding history-derived
features help?*

### Why LightGBM (not deep learning, not Prophet/ARIMA)
- **vs. deep learning**: 306k rows is small for sequence models; gains from
  representation learning don't justify the complexity for tabular data
  this clean.
- **vs. time-series methods (Prophet/ARIMA)**: prices are mostly constant,
  so trend/seasonality decomposition adds noise. The task is per-row
  point estimation conditional on side information, not univariate
  forecasting.
- **LightGBM with L1 objective**: native categorical support, robust to
  outliers, handles wide feature distributions, fast to iterate.

## 6. Anchor calibration

Three-layer post-hoc correction applied to all three approaches:

- **L1 — global multiplicative bias**: median log-residual on anchors.
  Captures marketplace-wide shifts (currency adjustment, broad promo days).
- **L2 — per-category bias**: log-residuals grouped by `cat_id` with
  empirical-Bayes shrinkage toward the global bias. Captures category-level
  effects that the global bias averages over.
- **L3 — isotonic regression** (Approach 3 only): monotone non-linear
  correction fit on anchor `(pred, true)` pairs.

Each layer is fit *only on the anchor rows of the same day* and applied
*only to that day's targets*.

## 7. Validation methodology

- **Time-based holdout**: last 3 days of training held out as a simulated
  outage. Each day, sample 100 random rows as fake anchors; predict the
  rest; report metrics.
- **Per-day metric reporting**: averaged across the 3 holdout days to
  reduce single-day variance.
- **Metrics**: MAPE, **MedAPE** (primary — robust to the wide price scale),
  MAE, RMSE.

Why time-based and not random? The brief explicitly simulates an outage
day; a random split would leak future information into training.

## 8. Results

### Validation hierarchy

| Method | MAPE | MedAPE | MAE | RMSE | n |
|---|---:|---:|---:|---:|---:|
| A1 raw | | | | | |
| A1 + calib | | | | | |
| A2 raw | | | | | |
| A2 + calib | | | | | |
| A3 raw | | | | | |
| A3 + L1 | | | | | |
| A3 + L2 | | | | | |
| A3 + L3 (final) | | | | | |

[insert bar chart: MedAPE per method]

### Approach 1 vs Approach 2

[Tell the comparison story honestly. Likely: A2 covers fewer rows than
A1 (per-product models need ≥5 obs); on covered rows, A2 is comparable
to or slightly better/worse than A1 depending on grouping. The conclusion
is that personalisation alone — without history features — doesn't
clearly beat a global model.]

### Effect of anchor calibration

[Show the L0 → A1 raw → A1 + calib → A3 + L3 progression. The marginal
contribution of each calibration layer.]

## 9. Insights and unexpected findings

Pick 3–5 of these to write up properly:

- The IQR rule failure on bounded/heavy-tailed columns (28% false positives)
- The change-point vs. anomaly ambiguity at the end of a series
- `cat_disc_shrunk` shrinkage target: mean was wrong because of bimodal
  distribution; median is the right choice
- 87% price stability → the task is fundamentally change detection
- Approach 2 with `last_price` would dominate, but excluding it is the
  honest test of personalisation alone

## 10. Limitations and future work

- Per-product Ridges have <100% coverage on test products with very short
  history. A1 fallback is used in production.
- The detector cannot resolve change-point vs anomaly at the end of the
  series; we delegate this to the anchor-calibration mechanism, which is
  the right architectural choice but worth noting.
- We did not tune `k_shrink` for `cat_disc_shrunk` exhaustively; a grid
  search on validation could squeeze a small additional gain.
- Could explore a hybrid blend (per-product Ridge prediction × A3
  prediction, weighted by history depth) — left for future iteration.

## 11. Reproducibility

```bash
pip install -r requirements.txt
jupyter notebook notebooks/03_training.ipynb   # retrain
python make_predictions.py <test_file> <model_name>