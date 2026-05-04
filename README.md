# Price Intelligence — MrScraper Take-Home

Reconstruct e-commerce product prices on days when scraping fails, using
historical scrape data and 100 manually-collected "anchor" prices per
outage day.

## Problem at a glance

- **Train**: 306k historical scrapes
- **Test**: 16 days × ~4,800 rows/day, of which 100/day are anchors with true price
- **Task**: predict price for the remaining ~4,700 rows/day given only `(capturedAt, shopId, itemId, modelId)` and the day's 100 anchors

## Headline results

| Approach | MedAPE | MAPE | MAE (IDR) | Notes |
|---|---:|---:|---:|---|
| **A1** Global LightGBM (IDs + temporal + anchor state) | X.XX% | X.X% | X | Marketplace-wide baseline |
| **A2** Per-product Ridge (same features, one model per product) | X.XX% | X.X% | X | Tests personalisation alone |
| **A3** Full pipeline (history features + anchor state + 3-layer calib) | **X.XX%** | **X.X%** | **X** | Best |

Validation: time-based split, last 3 days of training held out as a simulated outage.

## Key insight

The task is **change detection**, not forecasting. 87% of products never
change price across the entire training window, so the dominant predictor
is `last_price`. The 100 daily anchors are critical — not as evaluation
data, but as a **same-day calibration signal** that reveals which
categories are deviating today.

## Quickstart

```bash
pip install -r requirements.txt

# train (or skip — pre-trained models are committed in models/)
jupyter notebook notebooks/03_training.ipynb

# predict on a test file
python make_predictions.py data/test/your_test.csv a3_global