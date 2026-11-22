"""Validate a freight rate model on later months and generate assessment predictions."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs"
CATEGORICAL = ["pickup", "delivery", "equipment"]


def read_input(filename, *, has_date=True):
    # The assessment README uses underscores; downloaded CSVs may use hyphens.
    for name in (filename, filename.replace("_", "-")):
        path = DATA / name
        if path.is_file():
            return pd.read_csv(path, parse_dates=["date"] if has_date else None)
    raise FileNotFoundError(f"Place {filename} in {DATA}. See README.md for the input files.")


def make_features(shipments):
    features = shipments[CATEGORICAL + ["distance", "weight"]].copy()
    features["weekday"] = shipments["date"].dt.dayofweek
    features["day_of_month"] = shipments["date"].dt.day
    features["month"] = shipments["date"].dt.month
    return features


def prepare_inputs(shipments, training_rows):
    features = make_features(shipments)
    valid_weights = training_rows.loc[training_rows["weight"] > 0, "weight"]
    clean_weights = features["weight"].where(features["weight"] > 0)
    features["weight"] = clean_weights.fillna(valid_weights.median())

    # The same training labels encode both datasets. Unseen cities become missing.
    for column in CATEGORICAL:
        known_values = training_rows[column].dropna().unique()
        features[column] = pd.Categorical(features[column], categories=known_values)
    return features


def fit_model(training_rows):
    model = HistGradientBoostingRegressor(
        categorical_features="from_dtype",
        loss="absolute_error",
        max_iter=120,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )
    price_per_mile = training_rows["posted_rate"] / training_rows["distance"]
    return model.fit(prepare_inputs(training_rows, training_rows), price_per_mile)


def predict_prices(model, shipments, training_rows):
    per_mile = model.predict(prepare_inputs(shipments, training_rows))
    return per_mile * shipments["distance"]


def validate_by_month(loads):
    print("Test month | Training loads | Test loads | Baseline MAE | Model MAE")
    for start in pd.date_range("2025-08-01", periods=3, freq="MS"):
        end = start + pd.offsets.MonthBegin(1)
        history = loads.loc[loads["date"] < start]
        held_out = loads.loc[(loads["date"] >= start) & (loads["date"] < end)]

        train_rate = history["posted_rate"] / history["distance"]
        typical_rate = train_rate.groupby(history["equipment"]).median()
        baseline = held_out["equipment"].map(typical_rate) * held_out["distance"]

        predictions = predict_prices(fit_model(history), held_out, history)
        baseline_mae = mean_absolute_error(held_out["posted_rate"], baseline)
        model_mae = mean_absolute_error(held_out["posted_rate"], predictions)
        print(
            f"{start:%B} | {len(history):,} | {len(held_out):,} | "
            f"${baseline_mae:.2f} | ${model_mae:.2f}"
        )


def main():
    loads = read_input("train_test.csv")
    validation = read_input("validation.csv")
    template = read_input("validation_predictions_template.csv", has_date=False)
    december = read_input("december_chart_inputs.csv")
    validate_by_month(loads)

    final_model = fit_model(loads)
    prices = validation[["load_id"]].copy()
    prices["predicted_rate"] = predict_prices(final_model, validation, loads).round(2)
    submission = template[["load_id"]].merge(
        prices, on="load_id", how="left", validate="one_to_one"
    )
    if len(submission) != 12_000 or len(validation) != len(submission):
        raise ValueError("The template and validation file must each contain 12,000 rows.")
    if submission["load_id"].isna().any() or submission["predicted_rate"].isna().any():
        raise ValueError("The template must match one shipment ID to each predicted rate.")

    december["predicted_rate"] = predict_prices(final_model, december, loads).round(2)
    for data in (submission, december):
        rates = data["predicted_rate"]
        if not np.isfinite(rates).all() or (rates <= 0).any():
            raise ValueError("Every predicted rate must be finite and positive.")

    OUTPUT.mkdir(exist_ok=True)
    submission.to_csv(OUTPUT / "validation_predictions.csv", index=False)
    december.to_csv(OUTPUT / "december_predictions.csv", index=False, date_format="%Y-%m-%d")
    print(
        f"Saved {len(submission):,} shipment predictions and "
        f"{len(december)} December predictions in outputs/."
    )


if __name__ == "__main__":
    main()
