import pandas as pd
from pathlib import Path

from src.cleaning_data import basic_clean, detect_price_outliers_per_product, remove_flagged_outliers


def load_and_clean_train(train_path):
    """Load training data, clean it, and remove outliers."""
    train = basic_clean(pd.read_csv(train_path))
    _, prod_flags = detect_price_outliers_per_product(train, z_thresh=5.0)
    train = remove_flagged_outliers(train, prod_flags)
    return train


def load_and_clean_test(test_path):
    """Load test data and apply basic cleaning."""
    return basic_clean(pd.read_csv(test_path))
