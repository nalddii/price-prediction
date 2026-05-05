import sys
import time
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_and_clean_train, load_and_clean_test
from src.inference import resolve_approach, load_model


def main():
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    test_arg = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) >= 3 else cfg["default_model"]
    if len(sys.argv) < 3:
        print(f"Using default model: {model_name}")

    # Decide which approach + file extension
    try:
        ext, predictor = resolve_approach(model_name)
    except ValueError as e:
        sys.exit(str(e))

    # Resolve paths
    test_path = Path(test_arg)
    if not test_path.is_file():
        test_path = PROJECT_ROOT / "data" / "test" / test_arg

    model_path = PROJECT_ROOT / cfg["models_dir"] / f"{model_name}{ext}"
    train_path = PROJECT_ROOT / cfg["train_path"]

    if not test_path.is_file():
        sys.exit(f"Test file not found: {test_arg}")
    if not model_path.is_file():
        sys.exit(f"Model not found: {model_path}")

    print(f"  Test file : {test_path}")
    print(f"  Model     : {model_path.name}")

    t0 = time.time()
    train_df = load_and_clean_train(str(train_path))
    test_df  = load_and_clean_test(str(test_path))

    print(f"Loading model: {model_path.name}")
    model = load_model(model_path)

    print("Running predictions...")
    submission = predictor(test_df, train_df, model)

    out_dir = PROJECT_ROOT / "data" / "prediction_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}-prediction-{test_path.name}"
    submission.to_csv(out_path, index=False)

    print("DONE")
    print(f"  Output    : {out_path}")
    print(f"  Runtime   : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()