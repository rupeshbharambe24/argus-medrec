"""Train the DDI contextual severity XGBoost regressor.

Uses a synthetic labeling function (transparent; documented in the submission)
over features that *would* be available at inference time. No patient data is
required — features are sampled from plausible distributions and labels are
deterministic + small noise.

This is HACKATHON-GRADE training — the real production equivalent would use
real outcome labels from an EHR + ADE registry. That is beyond scope.

Run:
    python scripts/train_ddi_model.py --n-samples 20000 --out argus/ml/artifacts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from argus.ml.ddi_severity import FEATURE_COLUMNS, SEVERITY_TO_SCORE
from argus.schemas import Severity


def _label_function(features: np.ndarray) -> np.ndarray:
    """Deterministic + mild-noise labeling function.

    Feature order matches FEATURE_COLUMNS:
        0 base_severity_score
        1 age
        2 is_female
        3 egfr_est
        4 potassium
        5 inr
        6 qtc
        7 has_ckd
        8 has_hepatic
        9 has_cardiac
       10 coadministered_med_count
    """
    y = features[:, 0].copy()  # start from base severity

    # Age amplification
    age = features[:, 1]
    y += np.clip((age - 65) / 60, 0, 0.5)
    y += np.clip((age - 80) / 40, 0, 0.3)

    # Renal
    egfr = features[:, 3]
    y += np.where(egfr < 30, 0.5, np.where(egfr < 60, 0.25, 0))

    # Hepatic impairment
    y += features[:, 8] * 0.3

    # Cardiac + long QTc
    qtc = features[:, 6]
    y += np.where(qtc > 470, 0.35, 0)
    y += features[:, 9] * 0.15

    # Hypokalemia increases DDI risk for QTc-relevant interactions (approx)
    k = features[:, 4]
    y += np.where(k < 3.5, 0.2, 0)

    # High INR + interaction severity compound
    inr = features[:, 5]
    y += np.where(inr > 3.5, 0.25, 0)

    # Polypharmacy
    n_meds = features[:, 10]
    y += np.clip((n_meds - 8) / 20, 0, 0.3)

    # CKD flag (beyond eGFR which may be missing)
    y += features[:, 7] * 0.15

    # Noise
    rng = np.random.default_rng(42)
    y += rng.normal(0, 0.15, size=y.shape)

    return np.clip(y, 0.0, 5.0)


def _sample_features(n: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)

    base_sev_scores = np.array(list(SEVERITY_TO_SCORE.values()))
    base = rng.choice(base_sev_scores, size=n, p=[0.1, 0.2, 0.35, 0.25, 0.10])

    age = np.clip(rng.normal(68, 16, n), 18, 100)
    is_female = rng.integers(0, 2, n).astype(float)
    egfr = np.clip(rng.normal(65, 25, n), 5, 150)
    potassium = np.clip(rng.normal(4.1, 0.5, n), 2.5, 6.5)
    inr = np.clip(rng.gamma(2.5, 0.8, n), 0.8, 8.0)
    qtc = np.clip(rng.normal(440, 35, n), 350, 600)
    has_ckd = (egfr < 60).astype(float) * (rng.random(n) > 0.3).astype(float)
    has_hepatic = (rng.random(n) < 0.08).astype(float)
    has_cardiac = (rng.random(n) < 0.25).astype(float)
    n_meds = rng.poisson(7, n).astype(float)

    return np.column_stack(
        [base, age, is_female, egfr, potassium, inr, qtc,
         has_ckd, has_hepatic, has_cardiac, n_meds]
    )


def train(n_samples: int, out_dir: Path, seed: int = 0) -> dict:
    try:
        import xgboost as xgb
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Install with: pip install -e '.[dev]'")
        return {"error": str(exc)}

    X = _sample_features(n_samples, seed=seed)
    y = _label_function(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "ddi_severity.xgb"
    schema_path = out_dir / "feature_schema.json"
    model.save_model(str(model_path))
    with schema_path.open("w") as f:
        json.dump({"features": FEATURE_COLUMNS}, f, indent=2)

    metrics = {
        "n_train": len(y_train),
        "n_test": len(y_test),
        "mae": mae,
        "r2": r2,
        "feature_columns": FEATURE_COLUMNS,
    }
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ Model saved to {model_path}")
    print(f"  MAE={mae:.3f}   R²={r2:.3f}   (n_test={len(y_test)})")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="Train DDI severity model")
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--out", type=Path, default=Path("argus/ml/artifacts"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    result = train(args.n_samples, args.out, seed=args.seed)
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
