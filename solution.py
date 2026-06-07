"""
Traffic Demand Forecasting
Team: Aaryan Dwivedi, Devansh Bhardwaj, Rahul Das, Aagman Khantwal
OOF R² = 0.999581  |  Score = 99.9581 / 100
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import lightgbm as lgb

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

train = pd.read_csv(os.path.join(DATA, "train.csv"))
test  = pd.read_csv(os.path.join(DATA, "test.csv"))

print(f"Train: {train.shape}  |  Test: {test.shape}")


def ts_to_slot(ts):
    h, m = int(ts.split(":")[0]), int(ts.split(":")[1])
    return h * 4 + m // 15


def get_hour(ts):
    return int(ts.split(":")[0])


for df in [train, test]:
    df["hour"] = df["timestamp"].apply(get_hour)
    df["slot"] = df["timestamp"].apply(ts_to_slot)

train48 = train[train["day"] == 48].copy()
train49 = train[train["day"] == 49].copy()

EARLY_SLOTS  = set(range(9))
GLOBAL_MEAN  = train["demand"].mean()
GLOBAL_TEMP  = train["Temperature"].median()


# ── Lookup tables ─────────────────────────────────────────────────────────────

slot_lookup = train48.groupby(["geohash", "slot"])["demand"].mean().rename("d48_slot")
hour_lookup = train48.groupby(["geohash", "hour"])["demand"].mean().rename("d48_hr")

geo_stats48 = train48.groupby("geohash")["demand"].agg(["mean", "std", "median", "max", "min"])
geo_stats48.columns = ["d48_gmean", "d48_gstd", "d48_gmed", "d48_gmax", "d48_gmin"]

slot_mean_global = train48.groupby("slot")["demand"].mean().rename("slot_mean")

geo_mean48 = train48.groupby("geohash")["demand"].mean()
train48["nd"] = train48["demand"] / (train48["geohash"].map(geo_mean48) + 1e-9)
norm_slot48 = train48.groupby(["geohash", "slot"])["nd"].mean().rename("norm_slot_d48")

d48_morn = train48[train48["slot"].isin(EARLY_SLOTS)].groupby("geohash")["demand"].mean()
d49_morn = train49.groupby("geohash")["demand"].mean()

global_sc = d49_morn.sum() / (d48_morn.reindex(d49_morn.index).fillna(0).sum() + 1e-9)
geo_scale = (d49_morn / (d48_morn.reindex(d49_morn.index).fillna(0) + 1e-9)).rename("scale")

geo_d49_pivot = train49.pivot_table(index="geohash", columns="slot", values="demand", aggfunc="mean")
geo_d49_pivot.columns = [f"d49_s{c}" for c in geo_d49_pivot.columns]

d49_stats = train49.groupby("geohash")["demand"].agg(["mean", "std", "max"])
d49_stats.columns = ["d49_gmean", "d49_gstd", "d49_gmax"]

d49_hr = train49.groupby(["geohash", "hour"])["demand"].mean().rename("d49_hr_morn")


def morning_slope(row):
    v = row.dropna().values
    if len(v) < 2:
        return 0.0
    return sp_stats.linregress(np.arange(len(v)), v)[0]


geo_d49_slope = geo_d49_pivot.apply(morning_slope, axis=1).rename("d49_slope")


def make_features(df):
    df = df.copy()
    df = df.join(slot_lookup,    on=["geohash", "slot"], how="left")
    df = df.join(hour_lookup,    on=["geohash", "hour"], how="left")
    df = df.join(geo_stats48,    on="geohash",           how="left")
    df = df.join(norm_slot48,    on=["geohash", "slot"], how="left")
    df = df.join(geo_scale,      on="geohash",           how="left")
    df = df.join(d49_stats,      on="geohash",           how="left")
    df = df.join(geo_d49_pivot,  on="geohash",           how="left")
    df = df.join(geo_d49_slope,  on="geohash",           how="left")
    df = df.join(d49_hr,         on=["geohash", "hour"], how="left")

    df["slot_mean"] = df["slot"].map(slot_mean_global).fillna(GLOBAL_MEAN)
    df["scale"]     = df["scale"].fillna(global_sc)

    df["pred_slot"]  = (df["d48_slot"]  * df["scale"]).clip(0, 1)
    df["pred_hr"]    = (df["d48_hr"]    * df["scale"]).clip(0, 1)
    df["pred_gmean"] = (df["d48_gmean"] * df["scale"]).clip(0, 1)
    df["pred_norm"]  = (df["norm_slot_d48"] * df["d49_gmean"]).clip(0, 1)

    df["pred_slot"]  = df["pred_slot"].fillna(df["pred_hr"]).fillna(df["pred_gmean"]).fillna(df["slot_mean"])
    df["pred_hr"]    = df["pred_hr"].fillna(df["pred_gmean"]).fillna(df["slot_mean"])
    df["pred_norm"]  = df["pred_norm"].fillna(df["pred_slot"])

    df["RT"]   = df["RoadType"].map({"Residential": 0, "Street": 1, "Highway": 2}).fillna(-1)
    df["WE"]   = df["Weather"].map({"Sunny": 0, "Rainy": 1, "Foggy": 2, "Snowy": 3}).fillna(-1)
    df["LV"]   = (df["LargeVehicles"] == "Allowed").astype(int)
    df["LM"]   = (df["Landmarks"] == "Yes").astype(int)
    df["Temp"] = df["Temperature"].fillna(GLOBAL_TEMP)

    df["hsin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hcos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["ssin"] = np.sin(2 * np.pi * df["slot"] / 96)
    df["scos"] = np.cos(2 * np.pi * df["slot"] / 96)
    df["rush"] = df["hour"].isin([7, 8, 9]).astype(int)

    fill_cols = [
        "d48_slot", "d48_hr", "d48_gmean", "d48_gstd", "d48_gmed", "d48_gmax", "d48_gmin",
        "norm_slot_d48", "d49_gmean", "d49_gstd", "d49_gmax", "d49_slope", "d49_hr_morn",
        "pred_slot", "pred_hr", "pred_gmean", "pred_norm",
    ] + [f"d49_s{i}" for i in range(9)]

    for c in fill_cols:
        if c in df.columns:
            med = df[c].median()
            df[c] = df[c].fillna(med if not np.isnan(med) else 0)

    return df


FEATURES = [
    "slot", "hour", "day", "hsin", "hcos", "ssin", "scos", "rush",
    "RT", "WE", "LV", "LM", "NumberofLanes", "Temp",
    "d48_slot", "d48_hr", "d48_gmean", "d48_gstd", "d48_gmed", "d48_gmax", "d48_gmin",
    "norm_slot_d48", "scale", "d49_gmean", "d49_gstd", "d49_gmax", "d49_slope", "d49_hr_morn",
    "slot_mean", "pred_slot", "pred_hr", "pred_gmean", "pred_norm",
] + [f"d49_s{i}" for i in range(9)]


# ── Build training data (simulate test scenario on day-48) ────────────────────

train48_morning = train48[train48["slot"].isin(EARLY_SLOTS)].copy()
train48_daytime = train48[~train48["slot"].isin(EARLY_SLOTS)].copy()

_saved = (geo_scale.copy(), d49_stats.copy(), geo_d49_pivot.copy(),
          geo_d49_slope.copy(), d49_hr.copy(), global_sc)

geo_scale      = pd.Series(1.0, index=train48["geohash"].unique(), name="scale")
global_sc      = 1.0
d49_stats      = train48_morning.groupby("geohash")["demand"].agg(["mean", "std", "max"])
d49_stats.columns = ["d49_gmean", "d49_gstd", "d49_gmax"]
geo_d49_pivot  = train48_morning.pivot_table(index="geohash", columns="slot", values="demand", aggfunc="mean")
geo_d49_pivot.columns = [f"d49_s{c}" for c in geo_d49_pivot.columns]
geo_d49_slope  = geo_d49_pivot.apply(morning_slope, axis=1).rename("d49_slope")
d49_hr         = train48_morning.groupby(["geohash", "hour"])["demand"].mean().rename("d49_hr_morn")

X_train = make_features(train48_daytime)[FEATURES]
y_train = train48_daytime["demand"].values

geo_scale, d49_stats, geo_d49_pivot, geo_d49_slope, d49_hr, global_sc = _saved

X_test = make_features(test)[FEATURES]

print(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")
print(f"NaN check - train: {X_train.isna().sum().sum()}  test: {X_test.isna().sum().sum()}")


# ── LightGBM 5-fold CV ────────────────────────────────────────────────────────

LGB_PARAMS = dict(
    objective        = "regression",
    metric           = "rmse",
    learning_rate    = 0.2,
    num_leaves       = 128,
    max_depth        = 7,
    min_child_samples= 15,
    feature_fraction = 0.85,
    bagging_fraction = 0.85,
    bagging_freq     = 5,
    reg_alpha        = 0.03,
    reg_lambda       = 0.03,
    n_jobs           = -1,
    verbose          = -1,
    random_state     = 42,
)

kf         = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds  = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

print("\n" + "-" * 55)
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train.iloc[va_idx], y_train[va_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    model = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round = 10000,
        valid_sets      = [dval],
        callbacks       = [
            lgb.early_stopping(300, verbose=False),
            lgb.log_evaluation(99999),
        ],
    )

    oof_preds[va_idx] = model.predict(X_va,    num_iteration=model.best_iteration)
    test_preds       += model.predict(X_test,  num_iteration=model.best_iteration) / 5

    fold_r2 = r2_score(y_va, oof_preds[va_idx])
    print(f"  Fold {fold + 1}/5  |  R² = {fold_r2:.6f}  |  trees = {model.best_iteration}")

oof_r2 = r2_score(y_train, oof_preds)
print("-" * 55)
print(f"  OOF R²  = {oof_r2:.6f}")
print(f"  Score   = {max(0.0, 100 * oof_r2):.4f} / 100")
print("-" * 55)


# ── Save submission ───────────────────────────────────────────────────────────

test_preds = np.clip(test_preds, 0.0, 1.0)
submission = pd.DataFrame({"Index": test["Index"], "demand": test_preds})

out_path = os.path.join(BASE, "submission.csv")
submission.to_csv(out_path, index=False)

print(f"\nSaved -> {out_path}")
print(f"Rows: {len(submission)}  |  demand range: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
print("\nPreview:")
print(submission.head(10).to_string(index=False))
