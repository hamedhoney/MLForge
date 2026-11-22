import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

from mlforge.models.pipeline import build_models


def optimize_models(X, y, n_trials=10):
    """Tune candidate models using training data only."""
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    def objective(trial):
        model_name = trial.suggest_categorical(
            "model_name",
            ["logistic_regression", "xgboost"],
        )

        models = build_models(X)
        model = models[model_name]

        if model_name == "logistic_regression":
            params = {
                "classifier__C": trial.suggest_float("C", 0.001, 100, log=True),
            }
        else:
            params = {
                "classifier__n_estimators": trial.suggest_int(
                    "n_estimators", 100, 400, step=100
                ),
                "classifier__max_depth": trial.suggest_int("max_depth", 2, 6),
                "classifier__learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "classifier__subsample": trial.suggest_float("subsample", 0.7, 1.0),
            }

        model.set_params(**params)

        scores = cross_val_score(
            model,
            X,
            y,
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
            error_score="raise",
        )

        return float(scores.mean())

    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="mlforge-optimization",
    )

    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params.copy()
    best_name = best_params.pop("model_name")

    best_model = build_models(X)[best_name]

    if best_name == "logistic_regression":
        best_model.set_params(classifier__C=best_params["C"])
    else:
        best_model.set_params(
            **{f"classifier__{key}": value for key, value in best_params.items()}
        )

    return best_model, study
