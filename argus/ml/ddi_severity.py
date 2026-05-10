"""DDI severity model — XGBoost regressor with SHAP explanations.

Inference returns (score, factors_with_shap_values) in the same shape as the
heuristic fallback so the caller is indifferent to which path produced the score.

If the artifact file is missing, `score_interaction` returns `None` and the
caller uses the heuristic.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from argus.config import get_settings
from argus.logging_setup import get_logger
from argus.schemas import InteractionFactor, Severity

log = get_logger(__name__)

_MODEL: Any = None                  # XGBRegressor
_EXPLAINER: Any = None              # shap.TreeExplainer
_FEATURE_ORDER: list[str] = []
_LOAD_LOCK = Lock()


FEATURE_COLUMNS = [
    "base_severity_score",
    "age",
    "is_female",
    "egfr_est",
    "potassium",
    "inr",
    "qtc",
    "has_ckd",
    "has_hepatic",
    "has_cardiac",
    "coadministered_med_count",
]

SEVERITY_TO_SCORE = {
    Severity.TRIVIAL: 0.5,
    Severity.MINOR: 1.5,
    Severity.MODERATE: 2.5,
    Severity.MAJOR: 4.0,
    Severity.CRITICAL: 4.8,
}


def _model_path() -> Path:
    return Path(get_settings().model_dir) / "ddi_severity.xgb"


def _load_model() -> bool:
    """Load model + SHAP explainer. Returns True on success."""
    global _MODEL, _EXPLAINER, _FEATURE_ORDER
    path = _model_path()
    if not path.exists():
        return False

    try:
        import shap
        import xgboost as xgb
    except ImportError as exc:
        log.warning("ml.ddi.imports_missing", error=str(exc))
        return False

    try:
        model = xgb.XGBRegressor()
        model.load_model(str(path))
    except Exception as exc:  # noqa: BLE001
        log.warning("ml.ddi.load_failed", error=str(exc))
        return False

    try:
        explainer = shap.TreeExplainer(model)
    except Exception as exc:  # noqa: BLE001
        log.warning("ml.ddi.explainer_failed", error=str(exc))
        return False

    _MODEL = model
    _EXPLAINER = explainer
    _FEATURE_ORDER = list(FEATURE_COLUMNS)
    log.info("ml.ddi.loaded", path=str(path))
    return True


def _ensure_loaded() -> bool:
    global _MODEL
    if _MODEL is not None:
        return True
    with _LOAD_LOCK:
        if _MODEL is not None:
            return True
        return _load_model()


def score_interaction(
    base: Severity, ctx: dict[str, Any]
) -> tuple[float, list[InteractionFactor]] | None:
    """Predict contextual severity; returns None if model not available."""
    if not _ensure_loaded():
        return None
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return None

    features = _build_feature_vector(base, ctx)
    x = np.array([features], dtype=float)

    pred = float(_MODEL.predict(x)[0])
    pred = max(0.0, min(5.0, pred))

    shap_values = _EXPLAINER.shap_values(x)[0]
    factors: list[InteractionFactor] = []
    for name, value in zip(_FEATURE_ORDER, shap_values, strict=True):
        if abs(value) < 0.02:
            continue
        direction = "increases_risk" if value > 0 else "decreases_risk"
        factors.append(
            InteractionFactor(factor=name, direction=direction, shap_value=float(value))
        )
    factors.sort(key=lambda f: -abs(f.shap_value or 0.0))
    return pred, factors[:6]


def _build_feature_vector(base: Severity, ctx: dict[str, Any]) -> list[float]:
    from argus.tools._common import egfr_ckd_epi_2021

    cr = ctx.get("creatinine_mg_dl")
    age = ctx.get("age")
    sex = (ctx.get("sex") or "").lower()

    egfr_est = 90.0
    if cr is not None and age is not None and sex:
        try:
            egfr_est = egfr_ckd_epi_2021(cr, int(age), sex)
        except Exception:  # noqa: BLE001
            pass

    return [
        SEVERITY_TO_SCORE[base],
        float(age or 60),
        1.0 if sex.startswith("f") else 0.0,
        float(egfr_est),
        float(ctx.get("potassium_meq_l") or 4.0),
        float(ctx.get("inr") or 1.0),
        float(ctx.get("qtc_ms") or 420),
        1.0 if ctx.get("has_ckd") else 0.0,
        1.0 if ctx.get("has_hepatic") else 0.0,
        1.0 if ctx.get("has_cardiac") else 0.0,
        float(len(ctx.get("condition_codes") or [])),
    ]
