"""
Explainable AI - why was this crop recommended?

Uses SHAP for tree models (RandomForest / XGBoost) to get per-feature
contributions for the single prediction. Falls back to the model's global
feature importances if SHAP is unavailable or the model is ELM.
"""

import numpy as np
from crops_meta import FEATURES


def explain_instance(bundle, x_row):
    """
    bundle: the loaded crop_model.pkl dict
    x_row : 1-D array of the 7 feature values (unscaled)
    Returns list of (feature, contribution) sorted by |contribution| desc.
    """
    model = bundle["model"]
    x = np.asarray(x_row, dtype=float).reshape(1, -1)

    # try SHAP for tree models
    try:
        import shap
        if bundle["model_name"] in ("Random Forest", "XGBoost", "Decision Tree"):
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(x)
            # predicted class index
            pred = model.predict(x)[0]
            classes = list(getattr(model, "classes_", bundle["classes"]))
            if bundle["model_name"] == "XGBoost" and bundle.get("label_encoder") is not None:
                # xgboost predicts encoded ints
                classes = list(bundle["label_encoder"].classes_)
                pred_idx = int(pred)
            else:
                pred_idx = classes.index(pred)

            # sv shape varies by shap/model version -> normalize to (features,)
            contrib = _extract_contrib(sv, pred_idx)
            if contrib is not None:
                pairs = list(zip(FEATURES, contrib))
                pairs.sort(key=lambda p: abs(p[1]), reverse=True)
                return pairs
    except Exception:
        pass

    # fallback: global feature importances
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        pairs = list(zip(FEATURES, imp))
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        return pairs

    return None


def _extract_contrib(sv, pred_idx):
    arr = np.asarray(sv)
    # (classes, 1, features)
    if arr.ndim == 3 and arr.shape[0] > pred_idx and arr.shape[2] == len(FEATURES):
        return arr[pred_idx, 0, :]
    # (1, features, classes)
    if arr.ndim == 3 and arr.shape[1] == len(FEATURES):
        return arr[0, :, pred_idx]
    # (1, features)
    if arr.ndim == 2 and arr.shape[1] == len(FEATURES):
        return arr[0, :]
    # list-like already handled by np.asarray; last resort
    try:
        return np.asarray(sv[pred_idx])[0]
    except Exception:
        return None
