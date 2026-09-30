from pathlib import Path

import yaml
import numpy as np
from jinja2 import Template

from .features import feature_matrix

ROOT = Path(__file__).resolve().parents[2]
REASONS = ROOT / "config" / "reason_codes.yaml"

BORROWER_TEMPLATE = Template(
    "Based on the information used for this prototype decision, "
    "the factors most affecting the modeled risk were:\n"
    "{% for r in reasons %}- {{r.text}}\n{% endfor %}"
    "This is an automated prototype assessment using synthetic data "
    "and does not constitute a credit decision for a real person."
)


def _shap_factors(model, Xrow, feature_names, k=3):
    """Try genuine SHAP TreeExplainer."""
    import shap

    base = model.calibrated_classifiers_[0].estimator
    values = shap.TreeExplainer(base).shap_values(Xrow)

    values = np.asarray(values)

    # SHAP can return different shapes depending on version/model.
    if values.ndim > 1:
        values = values.reshape(-1)

    idx = np.argsort(np.abs(values))[::-1][:k]

    return [
        (feature_names[i], float(values[i]))
        for i in idx
    ]


def _fallback_factors(model, Xrow, feature_names, k=3):
    """
    Windows-safe deterministic fallback.

    Uses local one-feature perturbation against the model's prediction.
    This is used only when SHAP/numba cannot load.
    """
    base_pd = float(model.predict_proba(Xrow)[0, 1])

    factors = []

    for feature in feature_names:
        perturbed = Xrow.copy()

        value = perturbed.iloc[0][feature]

        if not np.isfinite(value):
            continue

        # Apply a deterministic relative perturbation.
        delta = max(abs(float(value)) * 0.10, 0.01)

        low = perturbed.copy()
        high = perturbed.copy()

        low.loc[:, feature] = value - delta
        high.loc[:, feature] = value + delta

        low_pd = float(model.predict_proba(low)[0, 1])
        high_pd = float(model.predict_proba(high)[0, 1])

        # Approximate local sensitivity.
        impact = (high_pd - low_pd) / 2.0

        factors.append((feature, impact))

    factors.sort(key=lambda x: abs(x[1]), reverse=True)

    return factors[:k]


def top_factors(model, Xrow, feature_names, k=3):
    """
    Prefer SHAP.

    If SHAP cannot load because of a platform-specific native dependency
    restriction, use deterministic local sensitivity instead.
    """
    try:
        return _shap_factors(model, Xrow, feature_names, k)
    except Exception:
        return _fallback_factors(model, Xrow, feature_names, k)


def explain(model, row, kind="alternative", k=3):
    X = feature_matrix(row.to_frame().T, kind)
    names = list(X.columns)

    factors = top_factors(
        model,
        X,
        names,
        k
    )

    codes = yaml.safe_load(
        REASONS.read_text(encoding="utf-8")
    )["codes"]

    reasons = []

    for feature, impact in factors:
        if feature in codes and abs(impact) > 1e-8:
            reasons.append(
                {
                    "code": codes[feature]["code"],
                    "text": codes[feature]["text"],
                    "feature": feature,
                    "impact": float(impact),
                }
            )

    if not reasons:
        reasons = [
            {
                "code": "R00",
                "text": (
                    "The model identified multiple interacting factors; "
                    "manual review is required for a more detailed assessment."
                ),
                "feature": "interaction",
                "impact": 0.0,
            }
        ]

    return {
        "reason_codes": reasons,
        "borrower_message": BORROWER_TEMPLATE.render(
            reasons=reasons
        ),
    }