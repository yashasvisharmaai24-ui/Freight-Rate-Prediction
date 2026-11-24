from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyRegressor


ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
TARGET = "posted_rate"
CATEGORICAL_COLUMNS = ["pickup", "delivery", "equipment"]
BASE_NUMERIC_COLUMNS = ["distance", "weight", "month", "day", "day_of_week", "day_of_year_sin", "day_of_year_cos"]
RICH_NUMERIC_COLUMNS = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "market_index",
    "quote_signal",
]


def build_features(frame: pd.DataFrame, include_rich_features: bool) -> pd.DataFrame:
    """Create only features available in the selected prediction scenario."""
    required = ["pickup", "delivery", "distance", "equipment", "weight", "date"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(missing)}")

    features = frame[required + (RICH_NUMERIC_COLUMNS if include_rich_features else [])].copy()
    dates = pd.to_datetime(features.pop("date"), errors="coerce")
    if dates.isna().any():
        raise ValueError("Input contains invalid or missing dates.")

    features["month"] = dates.dt.month.astype(float)
    features["day"] = dates.dt.day.astype(float)
    features["day_of_week"] = dates.dt.dayofweek.astype(float)
    day_angle = 2 * np.pi * dates.dt.dayofyear / 365.25
    features["day_of_year_sin"] = np.sin(day_angle)
    features["day_of_year_cos"] = np.cos(day_angle)

    for column in features.columns:
        if column in CATEGORICAL_COLUMNS:
            features[column] = features[column].astype("string")
        else:
            features[column] = pd.to_numeric(features[column], errors="coerce")

    # Non-positive shipment weights are implausible and treated as missing,
    # rather than allowing corrupted values to distort the learned relationship.
    features.loc[features["weight"] <= 0, "weight"] = np.nan
    return features


def make_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    categorical = [column for column in CATEGORICAL_COLUMNS if column in feature_columns]
    numeric = [column for column in feature_columns if column not in categorical]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_candidates() -> dict[str, Any]:
    return {
        "Median baseline": DummyRegressor(strategy="median"),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=140,
            min_samples_leaf=2,
            max_features=0.9,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=110,
            min_samples_leaf=2,
            max_features=0.9,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=160,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }


def make_pipeline(model: Any, feature_columns: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(feature_columns)),
            ("model", model),
        ]
    )


def evaluate_regime(
    name: str,
    features: pd.DataFrame,
    target: pd.Series,
    train_mask: pd.Series,
    holdout_mask: pd.Series,
) -> tuple[dict[str, Any], Pipeline]:
    feature_columns = list(features.columns)
    results: list[dict[str, Any]] = []
    candidates = make_candidates()
    best_pipeline: Pipeline | None = None
    best_name: str | None = None
    best_mae = math.inf

    for model_name, estimator in candidates.items():
        pipeline = make_pipeline(estimator, feature_columns)
        started = time.perf_counter()
        pipeline.fit(features.loc[train_mask], target.loc[train_mask])
        predicted = np.maximum(pipeline.predict(features.loc[holdout_mask]), 0.01)
        actual = target.loc[holdout_mask]
        mae = float(mean_absolute_error(actual, predicted))
        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        r2 = float(r2_score(actual, predicted))
        results.append(
            {
                "model": model_name,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "fit_seconds": round(time.perf_counter() - started, 2),
            }
        )
        print(f"{name} | {model_name}: MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.4f}")
        if mae < best_mae:
            best_mae = mae
            best_name = model_name
            best_pipeline = pipeline

    if best_pipeline is None or best_name is None:
        raise RuntimeError(f"No candidate model completed for {name}.")

    selected_estimator = make_candidates()[best_name]
    final_pipeline = make_pipeline(selected_estimator, feature_columns)
    final_pipeline.fit(features, target)
    return (
        {
            "name": name,
            "features": feature_columns,
            "selected_model": best_name,
            "selection_metric": "lowest chronological holdout MAE",
            "models": results,
            "holdout_rows": int(holdout_mask.sum()),
            "refit_rows": int(len(features)),
        },
        final_pipeline,
    )


def build_eda(train: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric = train.select_dtypes(include=np.number)
    dates = pd.to_datetime(train["date"], errors="coerce")
    monthly_rate = (
        train.assign(_month=dates.dt.to_period("M").astype(str))
        .groupby("_month", as_index=False)[TARGET]
        .median()
    )
    correlations = numeric.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(
        key=lambda values: values.abs(), ascending=False
    )
    numeric_statistics: dict[str, dict[str, float]] = {}
    iqr_outlier_counts: dict[str, int] = {}
    for column in numeric.columns:
        values = pd.to_numeric(train[column], errors="coerce").dropna()
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        numeric_statistics[column] = {
            "count": int(values.count()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "q1": q1,
            "median": float(values.median()),
            "q3": q3,
            "max": float(values.max()),
        }
        iqr_outlier_counts[column] = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(train[TARGET], bins=55, color="#176B73", edgecolor="white")
    axes[0, 0].set(title="Posted-rate distribution", xlabel="Posted rate ($)", ylabel="Rows")
    sample = train.sample(min(5000, len(train)), random_state=RANDOM_STATE)
    axes[0, 1].scatter(sample["distance"], sample[TARGET], s=7, alpha=0.2, color="#176B73")
    axes[0, 1].set(title="Rate vs. shipment distance", xlabel="Distance (miles)", ylabel="Posted rate ($)")
    axes[1, 0].plot(monthly_rate["_month"], monthly_rate[TARGET], marker="o", color="#C06C3E")
    axes[1, 0].set(title="Monthly median posted rate", xlabel="Month", ylabel="Median rate ($)")
    axes[1, 0].tick_params(axis="x", rotation=35)
    colors = ["#176B73" if value >= 0 else "#C06C3E" for value in correlations]
    axes[1, 1].barh(correlations.index, correlations.values, color=colors)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set(title="Numeric feature correlation with rate", xlabel="Pearson correlation")
    figure.tight_layout()
    figure.savefig(output_dir / "eda_overview.png", dpi=170, bbox_inches="tight")
    plt.close(figure)

    distribution_columns = ["distance", "weight", "market_index", "quote_signal"]
    figure, axes = plt.subplots(2, 2, figsize=(10, 6))
    for axis, column in zip(axes.flat, distribution_columns):
        values = pd.to_numeric(train[column], errors="coerce").dropna()
        axis.hist(values, bins=40, color="#176B73", edgecolor="white")
        axis.set(title=f"{column} distribution", xlabel=column, ylabel="Rows")
    figure.tight_layout()
    figure.savefig(output_dir / "numeric_distributions.png", dpi=170, bbox_inches="tight")
    plt.close(figure)

    equipment = train.groupby("equipment")[TARGET].median().sort_values()
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(equipment.index, equipment.values, color="#176B73")
    axis.set(title="Median posted rate by equipment", ylabel="Median posted rate ($)")
    figure.tight_layout()
    figure.savefig(output_dir / "rate_by_equipment.png", dpi=170, bbox_inches="tight")
    plt.close(figure)

    return {
        "rows": int(len(train)),
        "columns": list(train.columns),
        "dtypes": {key: str(value) for key, value in train.dtypes.items()},
        "missing_values": {key: int(value) for key, value in train.isna().sum().items()},
        "duplicate_load_ids": int(train["load_id"].duplicated().sum()),
        "duplicate_full_rows": int(train.duplicated().sum()),
        "unique_pickups": int(train["pickup"].nunique()),
        "unique_deliveries": int(train["delivery"].nunique()),
        "equipment_counts": {key: int(value) for key, value in train["equipment"].value_counts().items()},
        "categorical_value_counts": {
            column: {
                str(key): int(value)
                for key, value in train[column].fillna("__missing__").astype(str).value_counts().items()
            }
            for column in CATEGORICAL_COLUMNS
        },
        "numeric_statistics": numeric_statistics,
        "iqr_outlier_counts": iqr_outlier_counts,
        "negative_weights": int((train["weight"] < 0).sum()),
        "nonpositive_weights": int((train["weight"] <= 0).sum()),
        "target": {
            "min": float(train[TARGET].min()),
            "q1": float(train[TARGET].quantile(0.25)),
            "median": float(train[TARGET].median()),
            "mean": float(train[TARGET].mean()),
            "q3": float(train[TARGET].quantile(0.75)),
            "max": float(train[TARGET].max()),
        },
        "date_min": str(dates.min().date()),
        "date_max": str(dates.max().date()),
        "monthly_median_rate": {
            str(row["_month"]): float(row[TARGET]) for _, row in monthly_rate.iterrows()
        },
        "numeric_target_correlations": {str(key): float(value) for key, value in correlations.items()},
        "equipment_median_rate": {str(key): float(value) for key, value in equipment.items()},
    }


def plot_model_comparison(comparisons: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, metric, title in zip(axes, ["mae", "rmse"], ["Holdout MAE (lower is better)", "Holdout RMSE (lower is better)"]):
        labels: list[str] = []
        values: list[float] = []
        for comparison in comparisons:
            for result in comparison["models"]:
                labels.append(f"{comparison['name']}\n{result['model']}")
                values.append(result[metric])
        axis.bar(range(len(values)), values, color="#176B73")
        axis.set(title=title, ylabel=f"{metric.upper()} ($)")
        axis.set_xticks(range(len(labels)), labels, rotation=70, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def run() -> None:
    data_dir = ROOT / "data"
    train_path = data_dir / "train_test.csv"
    validation_path = data_dir / "validation.csv"
    template_path = data_dir / "validation_predictions_template.csv"
    december_path = data_dir / "december_chart_inputs.csv"
    for path in [train_path, validation_path, template_path, december_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required assessment input is missing: {path}")

    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    template = pd.read_csv(template_path)
    december = pd.read_csv(december_path)
    if TARGET not in train or train[TARGET].isna().any():
        raise ValueError("Training data must include a complete posted_rate target.")
    if train["load_id"].duplicated().any():
        raise ValueError("Training data contains duplicate load_id values.")
    if validation["load_id"].duplicated().any() or template["load_id"].duplicated().any():
        raise ValueError("Validation or template contains duplicate load_id values.")
    if set(validation["load_id"]) != set(template["load_id"]):
        raise ValueError("Validation IDs do not match the prediction template.")

    train_dates = pd.to_datetime(train["date"], errors="coerce")
    if train_dates.isna().any():
        raise ValueError("Training data contains invalid dates.")
    unique_dates = sorted(train_dates.unique())
    cutoff = pd.Timestamp(unique_dates[int(len(unique_dates) * 0.8)])
    train_mask = train_dates < cutoff
    holdout_mask = ~train_mask
    if train_mask.sum() == 0 or holdout_mask.sum() == 0:
        raise ValueError("Chronological holdout split produced an empty partition.")
    target = pd.to_numeric(train[TARGET], errors="raise").astype(float)
    if (target <= 0).any() or not np.isfinite(target).all():
        raise ValueError("Training target must contain finite, positive rates.")

    output_dirs = {name: ROOT / name for name in ["reports", "models", "scorer_results", "data"]}
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    eda = build_eda(train, output_dirs["reports"])
    print(f"Chronological holdout: training dates before {cutoff.date()}; holdout from {cutoff.date()}.")

    comparisons: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    for regime, include_rich in [("validation_full_features", True), ("december_available_features", False)]:
        features = build_features(train, include_rich)
        comparison, pipeline = evaluate_regime(regime, features, target, train_mask, holdout_mask)
        comparisons.append(comparison)
        fitted_models[regime] = pipeline
        dump(pipeline, output_dirs["models"] / f"{regime}.joblib")

    validation_features = build_features(validation, include_rich_features=True)
    validation_predictions = np.maximum(
        fitted_models["validation_full_features"].predict(validation_features), 0.01
    )
    predictions = pd.DataFrame(
        {
            "load_id": template["load_id"],
            "predicted_rate": pd.Series(
                dict(zip(validation["load_id"], validation_predictions))
            ).reindex(template["load_id"]).to_numpy(),
        }
    )
    if predictions["predicted_rate"].isna().any() or not np.isfinite(predictions["predicted_rate"]).all():
        raise ValueError("Validation prediction generation produced invalid values.")
    predictions.to_csv(ROOT / "validation_predictions.csv", index=False, float_format="%.6f")

    december_features = build_features(december, include_rich_features=False)
    december_predictions = np.maximum(
        fitted_models["december_available_features"].predict(december_features), 0.01
    )
    if not np.isfinite(december_predictions).all():
        raise ValueError("December prediction generation produced invalid values.")
    december["predicted_rate"] = december_predictions
    december.to_csv(december_path, index=False, float_format="%.6f")
    december.to_csv(output_dirs["data"] / "december_chart_inputs.csv", index=False, float_format="%.6f")

    plot_model_comparison(comparisons, output_dirs["reports"] / "model_comparison.png")
    validation_date = pd.to_datetime(validation["date"], errors="coerce")
    metadata = {
        "python_version": sys.version.split()[0],
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "data": eda,
        "split": {
            "strategy": "chronological 80/20 split by distinct training dates",
            "cutoff_date": str(cutoff.date()),
            "train_rows": int(train_mask.sum()),
            "holdout_rows": int(holdout_mask.sum()),
            "train_date_end": str(pd.Timestamp(train_dates[train_mask].max()).date()),
            "holdout_date_start": str(pd.Timestamp(train_dates[holdout_mask].min()).date()),
        },
        "comparisons": comparisons,
        "validation": {
            "rows": int(len(validation)),
            "date_min": str(validation_date.min().date()),
            "date_max": str(validation_date.max().date()),
            "prediction_min": float(predictions["predicted_rate"].min()),
            "prediction_median": float(predictions["predicted_rate"].median()),
            "prediction_max": float(predictions["predicted_rate"].max()),
            "unseen_pickups": sorted(set(validation["pickup"]) - set(train["pickup"])),
            "unseen_deliveries": sorted(set(validation["delivery"]) - set(train["delivery"])),
        },
        "december": {
            "rows": int(len(december)),
            "date_min": str(december["date"].min()),
            "date_max": str(december["date"].max()),
            "prediction_min": float(december["predicted_rate"].min()),
            "prediction_median": float(december["predicted_rate"].median()),
            "prediction_max": float(december["predicted_rate"].max()),
            "available_features_only": True,
        },
    }
    (output_dirs["reports"] / "results.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("Selected models:")
    for comparison in comparisons:
        selected = next(row for row in comparison["models"] if row["model"] == comparison["selected_model"])
        print(
            f"  {comparison['name']}: {comparison['selected_model']} "
            f"(holdout MAE={selected['mae']:.2f}, RMSE={selected['rmse']:.2f}, R2={selected['r2']:.4f})"
        )
    print("Created validation_predictions.csv, updated December CSV, models, and EDA/model comparison charts.")


if __name__ == "__main__":
    run()
