from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import shap


def generate_explanations(model, X, output_dir: Path):
    """Generate SHAP plots for a fitted tree-based model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    X_sample = X.sample(
        n=min(100, len(X)),
        random_state=42,
    )

    transformed = preprocessor.transform(X_sample)
    transformed = np.asarray(transformed)

    feature_names = preprocessor.get_feature_names_out()

    if classifier.__class__.__name__ != "XGBClassifier":
        return False

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(transformed)

    plt.figure()
    shap.summary_plot(
        shap_values.values,
        transformed,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "shap_summary.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close("all")

    return True
