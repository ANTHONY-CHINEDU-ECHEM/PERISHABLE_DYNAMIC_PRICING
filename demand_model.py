"""Fits a log-log demand elasticity model per SKU category:
log(units_sold) = a + elasticity * log(price/base_price) + b * expiry_urgency
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def fit_category(df_cat):
    X = np.column_stack([
        np.log(df_cat["price"] / df_cat["base_price"]),
        np.clip(4 - df_cat["days_to_expiry"], 0, None),
    ])
    y = np.log(df_cat["units_sold"].clip(lower=0.1))
    model = LinearRegression().fit(X, y)
    return model


def main():
    df = pd.read_csv(ROOT / "data" / "price_demand_history.csv")
    models = {}
    summary = {}
    for cat, grp in df.groupby("category"):
        m = fit_category(grp)
        models[cat] = m
        summary[cat] = {"elasticity_coef": round(float(m.coef_[0]), 3),
                         "expiry_urgency_coef": round(float(m.coef_[1]), 3),
                         "intercept": round(float(m.intercept_), 3), "r2": round(float(m.score(
                             np.column_stack([np.log(grp["price"] / grp["base_price"]),
                                               np.clip(4 - grp["days_to_expiry"], 0, None)]),
                             np.log(grp["units_sold"].clip(lower=0.1)))), 4)}
    joblib.dump(models, MODEL_DIR / "elasticity_models.joblib")
    with open(MODEL_DIR / "elasticity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
