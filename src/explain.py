"""Explainability: SHAP for the tree model, standardized coefficients for linear.

SHAP (SHapley Additive exPlanations) is the method that attributes a model's
prediction to each input feature — i.e. how much each parameter contributed,
expressed as additive scores. We use it on the chosen tree model and pair it
with the directly interpretable coefficients of the linear model.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def shap_tree(model, X_train, X_explain, out_png: str):
    """Compute SHAP values for a fitted tree model and save a summary plot.

    Returns a DataFrame ranking features by mean |SHAP| (contribution score).
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)

    plt.figure()
    shap.summary_plot(shap_values, X_explain, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": X_explain.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def linear_contributions(pipeline, feature_names) -> pd.DataFrame:
    """Standardized linear coefficients = per-feature contribution scores."""
    coef = pipeline.named_steps["reg"].coef_
    return (
        pd.DataFrame({"feature": feature_names, "coefficient": coef})
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .reset_index(drop=True)
    )


def plot_importance_bar(rank_df, value_col, title, out_png, top=15):
    """Generic horizontal bar chart of feature contributions."""
    d = rank_df.head(top).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(d["feature"], d[value_col])
    plt.xlabel(value_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
