#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json, time, warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
import joblib
from datetime import timedelta

# sklearn
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401





from sklearn.ensemble import HistGradientBoostingRegressor as HGBR





from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ========== CONFIG ==========
DATA_PATH = r"C:\Users\ASUS\skripsi_carin\data\dataset.xlsx"
OUTDIR    = Path(r"C:\Users\ASUS\Documents\skripsi mancing\models_hgbr60_carin_only")
TEST_DAYS = 60
N_SPLITS_CV = 3
SEED = 42

PARAM_GRID = {
    "learning_rate":      [0.05, 0.03, 0.01],
    "max_depth":          [1, 2, 3],
    "max_iter":           [150, 300, 450],
    "l2_regularization":  [0.0, 1e-3],
    "min_samples_leaf":   [10, 30],
}
EARLY_STOP_KW = dict(early_stopping=True, validation_fraction=0.15, n_iter_no_change=20)

# daftar kota (sesbuaikan kalau perlu)
# CITY_NAMES = ["Kota Bandung","Kota Bekasi","Kota Bogor","Kota Depok","Kota Sukabumi", "Kab. Sragen","Kab. Banyumas","Kab Boyolali","Kab. Bulukomba","Kab. Bulungan", "Kab.Bungo","Kab. Cirebon","Kab. Jember"]
CITY_NAMES = [
   "Kota Pekanbaru","Kota Pematang Siantar","Kab. Polewali Mandar",
    "Kota Pontianak","Kota Probolinggo","Kota Samarinda","Kota Sampit","Kota Semarang",
    "Kota Serang","Kota Sibolga","Kota Singkawang","Kota Sorong","Kab Sragen","Kota Sukabumi",
    "Kab. Sukoharjo","Kab. Sumba Timur","Kab. Sumbawa","Kab. Sumenep","Kota Surabaya",
    "Kota Surakarta (Solo)","Kota Tanggerang","Kota Tanjung","Kota Tanjung Pandan",
    "Kota Tanjung Pinang","Kota Tarakan","Kota Tasikmalaya","Kota Sumenep",
    "Kota Tembilahan","Kota Ternate","Kota Tual","Kota Watampone","Kab. Wonogiri","Kota Yogyakarta"
][:105]

# 12 eksperimen fitur
FEATURE_EXPERIMENTS = [
    {"name":"lag1"        , "LAGS":[]   , "ROLLS":[]  },
    {"name":"lag7"        , "LAGS":[7]  , "ROLLS":[]  },
    {"name":"lag14"       , "LAGS":[14] , "ROLLS":[]  },
    {"name":"lag30"       , "LAGS":[30] , "ROLLS":[]  },
    {"name":"lag1_roll7"  , "LAGS":[]   , "ROLLS":[7] },
    {"name":"lag7_roll7"  , "LAGS":[7]  , "ROLLS":[7] },
    {"name":"lag14_roll7" , "LAGS":[14] , "ROLLS":[7] },
    {"name":"lag30_roll7" , "LAGS":[30] , "ROLLS":[7] },
    {"name":"lag1_roll30" , "LAGS":[]   , "ROLLS":[30]},
    {"name":"lag7_roll30" , "LAGS":[7]  , "ROLLS":[30]},
    {"name":"lag14_roll30", "LAGS":[14] , "ROLLS":[30]},
    {"name":"lag30_roll30", "LAGS":[30] , "ROLLS":[30]},
]

# ========== HELPERS ==========
EID_DATES = ["2020-05-24","2021-05-13","2022-05-02","2023-04-22","2024-04-10","2025-03-31"]
def make_flags(idx: pd.DatetimeIndex) -> pd.DataFrame:
    flags = pd.DataFrame(index=idx)
    flags["month"] = idx.month
    flags["dayofweek"] = idx.dayofweek
    flags["time_index"] = (idx - idx.min()).days
    flags["is_end_of_year"] = ((idx.month == 12) & (idx.day >= 22)).astype(int)
    flags["is_new_year"] = ((idx.month == 1) & (idx.day <= 7)).astype(int)
    eid = np.zeros(len(idx), dtype=int)
    for d in EID_DATES:
        t = pd.Timestamp(d)
        eid |= ((idx >= t - pd.Timedelta(days=14)) & (idx <= t + pd.Timedelta(days=7))).astype(int)
    flags["is_eid_window"] = eid
    return flags

@dataclass
class FeatureCfg:
    add_lags: list
    rolls: list

def build_features_level_target(y: pd.Series, cfg: FeatureCfg):
    y = pd.Series(pd.to_numeric(y, errors="coerce"), index=pd.DatetimeIndex(y.index)).astype(float)
    X = pd.DataFrame(index=y.index)
    X["lag_1"] = y.shift(1)
    for L in (cfg.add_lags or []):
        if L == 1: continue
        X[f"lag_{L}"] = y.shift(L)
    for W in (cfg.rolls or []):
        X[f"rollmean_{W}"] = y.shift(1).rolling(W, min_periods=max(1, W//2)).mean()
    X = X.join(make_flags(X.index))
    df = pd.concat([X, y.rename("y")], axis=1).dropna()
    y_target = df.pop("y")
    return df, y_target

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True); return p

def fmt_hhmmss(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape_pct(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    denom = np.clip(np.abs(y_true), 1e-9, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

# ========== PROCESSING ==========
def process_city(df_all: pd.DataFrame, city_name: str, shared_best_dir: Path) -> dict:
    if city_name not in df_all.columns:
        print(f"[SKIP] {city_name}: column not found."); return {"city": city_name, "ok": False}

    series_full = (
        pd.Series(pd.to_numeric(df_all[city_name], errors="coerce").values,
                  index=pd.DatetimeIndex(df_all["date"]))
          .asfreq("D").ffill().bfill()
    )
    if len(series_full) <= TEST_DAYS + 200:
        print(f"[SKIP] {city_name}: series too short (len={len(series_full)})."); return {"city": city_name, "ok": False}

    city_safe = city_name.replace("/", "-").replace("\\", "-").replace(" ", "_")[:180]
    # use the shared folder for all cities
    best_dir = shared_best_dir

    best_record = {"r2": -1e9, "exp_name": None, "model": None, "feature_cols": None, "metrics": None}

    for exp in FEATURE_EXPERIMENTS:
        add_lags = exp["LAGS"]; rolls = exp["ROLLS"]; exp_name = exp["name"]

        X_all, y_all = build_features_level_target(series_full, FeatureCfg(add_lags=add_lags, rolls=rolls))
        if len(X_all) <= TEST_DAYS + 50:
            print(f"[SKIP] {city_name} | {exp_name}: too short after dropna (len={len(X_all)}).")
            continue

        X_train, X_test = X_all.iloc[:-TEST_DAYS], X_all.iloc[-TEST_DAYS:]
        y_train, y_test = y_all.iloc[:-TEST_DAYS], y_all.iloc[-TEST_DAYS:]

        max_lookback = int(max([1, *add_lags, *(rolls or [0])]))
        try:
            tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV, gap=max_lookback)
            gap_used = max_lookback
        except TypeError:
            tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
            gap_used = 0

        base = HGBR(loss="squared_error", random_state=SEED, **EARLY_STOP_KW)
        gs = GridSearchCV(estimator=base, param_grid=PARAM_GRID, scoring="r2",
                          refit=True, cv=tscv, n_jobs=-1, verbose=0, return_train_score=False)
        t0 = time.perf_counter()
        gs.fit(X_train, y_train)
        train_secs = time.perf_counter() - t0
        train_hms = fmt_hhmmss(train_secs)

        best_model = gs.best_estimator_
        cv_best_r2 = float(gs.best_score_)

        y_pred = best_model.predict(X_test)
        r2_val = float(r2_score(y_test, y_pred))
        mae_val = float(mean_absolute_error(y_test, y_pred))
        rmse_val = float(rmse(y_test, y_pred))
        mse_val = float(mean_squared_error(y_test, y_pred))
        mape_val = float(mape_pct(y_test, y_pred))

        metrics = dict(
            city=city_name, exp=exp_name,
            lags=[1, *add_lags], rolls=rolls,
            r2=r2_val, mae=mae_val, rmse=rmse_val, mse=mse_val, mape=mape_val,
            cv_best_r2=cv_best_r2, cv_refit_metric="r2",
            n_total=int(len(X_all)), n_train=int(len(X_train)), n_test=int(len(X_test)),
            train_time_seconds=float(train_secs), train_time_hhmmss=train_hms, cv_gap=int(gap_used),
            best_params=gs.best_params_
        )

        if r2_val > best_record["r2"]:
            best_record.update(
                r2=r2_val,
                exp_name=exp_name,
                model=best_model,
                feature_cols=list(X_train.columns),
                metrics=metrics
            )

        print(f"{city_name} | {exp_name} | TEST R2={r2_val:.4f} | CV r2={cv_best_r2:.4f} | time={train_hms}")

    # save only the pack (joblib) for best model into the shared folder
    if best_record["model"] is not None:
        pack = {
            "model": best_record["model"],
            "feature_cols": best_record["feature_cols"],
            "best_config": {
                "mode": "level",
                "transform": "none",
                "alpha_blend": 1.0,
                "train_until": None,
                "exp_name": best_record["exp_name"],
            },
            "metrics": best_record["metrics"],
        }
        # filename includes city_safe and exp_name so there are no collisions


        pack_path = best_dir / f"{city_safe}__{best_record['exp_name']}__best_pack.joblib"
        joblib.dump(pack, pack_path)
        
        
        
        print(f"[SAVED PACK] {city_name} -> {pack_path}")
        return {"city": city_name, "ok": True, "best_r2": float(best_record["r2"]), "pack_path": str(pack_path)}
    else:
        print(f"[NO MODEL] {city_name}: nothing saved.")
        return {"city": city_name, "ok": False}

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    # ensure top-level OUTDIR exists
    OUTDIR.mkdir(parents=True, exist_ok=True)
    # create a single shared folder for all best packs
    shared_best_dir = ensure_dir(OUTDIR / "best_model")

    df = pd.read_excel(DATA_PATH, engine="openpyxl")
    if df.columns[0].lower() != "date":
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    for city in CITY_NAMES:
        print("\n" + "#"*60)
        print(f"### Processing: {city}")
        print("#"*60)
        process_city(df, city, shared_best_dir)

if __name__ == "__main__":
    main()
