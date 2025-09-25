#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trainer TOP 30 kota (mode=diff) — kompatibel dgn app.py kamu.

Fitur utama:
- NUM_CITIES=30
- GridSearchCV n_jobs=1 (stabil, hemat RAM)
- Simpan model hanya jika R² test >= SAVE_R2_THRESH
- Skip kota yang sudah punya pack .joblib (resume)
- Simpan progress incremental ke CSV
- Simpan ringkasan final ke CSV & Excel
"""

import os
# Batasi thread BLAS/OpenMP biar gak makan RAM berlebihan
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import re, json, gc
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, make_scorer

# ================== CONFIG ==================
# Path dataset & folder models backend (samakan dgn app.py -> MODELS_DIR)
DATA_PATH     = r"C:\Users\ASUS\skripsi_carin\data\dataset_filled_ffill_bfill.xlsx"
MODELS_OUTDIR = Path(r"C:\Users\ASUS\skripsi_carin\models")  # <— backend kamu baca dari sini

NUM_CITIES      = 30          # <— top 30 dulu
TEST_DAYS       = 180         # evaluasi di ekor
HORIZON_DAYS    = 120         # optional forecast simpan CSV
SAVE_R2_THRESH  = 0.60        # simpan model hanya kalau R² test >= ini

# Fitur utama — DISINKRONKAN dengan app.py::make_features_entity
USE_LAG7        = True
USE_LAG14       = True
USE_ROLL_MEAN7  = True
USE_ROLL_MEAN14 = True

# Event (Idul Fitri, akhir tahun, krisis) — MATIKAN dulu supaya 100% match backend
# (Kalau mau ON, backend (app.py) harus ditambah kolom fitur yg sama saat inference)
USE_EVENTS      = False

EID_DATES    = ["2020-05-24","2021-05-13","2022-05-02","2023-04-22","2024-04-10","2025-03-31"]
CRISIS_START = "2021-01-01"
CRISIS_END   = "2022-03-31"

# Grid untuk GBM (delta)
PARAM_GRID = {
    "loss": ["huber"],
    "random_state": [42],
    "subsample": [1.0],
    "learning_rate": [0.03, 0.05, 0.01],
    "n_estimators": [400, 600, 800, 1000],
    "max_depth": [1, 2, 3],
    "min_samples_leaf": [1, 3],
}
# ===========================================

def rmse(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat)**2)))

def mape_pct(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    return float(np.mean(np.abs((y - yhat) / np.clip(np.abs(y), 1e-9, None))) * 100)

def neg_rmse(ytrue, yhat):  # for GridSearchCV
    return -rmse(ytrue, yhat)

def find_date_col(df: pd.DataFrame):
    for c in df.columns:
        if str(c).strip().lower() in {"date", "tanggal", "tgl"}:
            return c
    for c in df.columns:
        try:
            pd.to_datetime(df[c]); return c
        except Exception:
            pass
    raise ValueError("Kolom tanggal tidak ditemukan (coba: date/tanggal/tgl).")

def to_entity_from_excel_col(name: str) -> str:
    # match transform di app.py: lower + spasi → underscore (titik dibiarkan)
    return re.sub(r"\s+", "_", name.strip().lower())

def make_features_like_backend(y: pd.Series):
    """
    Subset fitur yang DIJAMIN ada di app.py::make_features_entity:
      - lag_1, lag_7, lag_14
      - roll7_mean, roll14_mean  (rolling berbasis shift(1) utk anti-leak)
      - dayofweek, month
    Target = delta (y_t - y_{t-1})
    """
    y = y.copy(); y.index = pd.DatetimeIndex(y.index)
    lag1 = y.shift(1)

    X = pd.DataFrame(index=y.index)
    # lags
    X["lag_1"] = lag1
    if USE_LAG7:  X["lag_7"]  = y.shift(7)
    if USE_LAG14: X["lag_14"] = y.shift(14)

    # rolling ala backend (pakai shift(1) di dalam window)
    s7  = y.shift(1).rolling(7,  min_periods=3)
    s14 = y.shift(1).rolling(14, min_periods=3)
    if USE_ROLL_MEAN7:  X["roll7_mean"]  = s7.mean()
    if USE_ROLL_MEAN14: X["roll14_mean"] = s14.mean()

    # calendar
    idx = X.index
    X["dayofweek"] = idx.dayofweek
    X["month"]     = idx.month

    if USE_EVENTS:
        X["is_end_of_year"] = ((idx.month == 12) & (idx.day >= 20)).astype(int)
        X["is_new_year"]    = ((idx.month == 1)  & (idx.day <= 10)).astype(int)
        is_eid = np.zeros(len(idx), dtype=int)
        for d in EID_DATES:
            d = pd.Timestamp(d)
            is_eid |= ((idx >= d - pd.Timedelta(days=14)) & (idx <= d + pd.Timedelta(days=7))).astype(int)
        X["is_eid_window"] = is_eid
        X["is_crisis_2021"] = ((idx >= pd.Timestamp(CRISIS_START)) &
                               (idx <= pd.Timestamp(CRISIS_END))).astype(int)

    y_delta = (y - lag1)

    data = pd.concat([X, y_delta.rename("y_delta"), lag1.rename("lag1_only")], axis=1).dropna()
    w = np.where(data["y_delta"].abs() > 0, 3.0, 1.0)  # weight perubahan

    X_final   = data.drop(columns=["y_delta", "lag1_only"])
    y_final   = data["y_delta"]
    lag1_only = data["lag1_only"]
    w_series  = pd.Series(w, index=data.index)
    return X_final, y_final, lag1_only, w_series

def forecast_iterative_delta(model, hist: pd.Series, horizon: int, feature_order: list) -> pd.DataFrame:
    s = hist.copy(); s.index = pd.DatetimeIndex(s.index)
    rows = []
    for _ in range(horizon):
        d_next = s.index[-1] + pd.Timedelta(days=1)
        row = {}
        row["lag_1"]  = s.iloc[-1]
        row["lag_7"]  = s.iloc[-7]  if ("lag_7"  in feature_order and len(s) >= 7)  else row["lag_1"]
        row["lag_14"] = s.iloc[-14] if ("lag_14" in feature_order and len(s) >= 14) else row["lag_1"]

        # rolling ala backend (shift(1))
        if "roll7_mean" in feature_order:
            row["roll7_mean"] = float(s.shift(1).rolling(7,  min_periods=3).mean().iloc[-1]) if len(s) >= 2 else float(s.iloc[-1])
        if "roll14_mean" in feature_order:
            row["roll14_mean"] = float(s.shift(1).rolling(14, min_periods=3).mean().iloc[-1]) if len(s) >= 2 else float(s.iloc[-1])

        row["dayofweek"] = d_next.dayofweek
        row["month"]     = d_next.month

        if USE_EVENTS:
            row["is_end_of_year"] = int((d_next.month == 12) and (d_next.day >= 20))
            row["is_new_year"]    = int((d_next.month == 1)  and (d_next.day <= 10))
            is_eid = 0
            for d in EID_DATES:
                d = pd.Timestamp(d)
                if (d_next >= d - pd.Timedelta(days=14)) and (d_next <= d + pd.Timedelta(days=7)):
                    is_eid = 1; break
            row["is_eid_window"]  = is_eid
            row["is_crisis_2021"] = int(pd.Timestamp(CRISIS_START) <= d_next <= pd.Timestamp(CRISIS_END))

        Xn = pd.DataFrame([row])
        for c in feature_order:
            if c not in Xn.columns:
                Xn[c] = 0.0
        Xn = Xn[feature_order]

        delta_hat = float(model.predict(Xn)[0])
        delta_hat = float(np.clip(delta_hat, -500, 500))  # clamp opsional
        y_hat = row["lag_1"] + delta_hat

        s.loc[d_next] = y_hat
        rows.append((d_next, y_hat))
    return pd.DataFrame(rows, columns=["date", "y_hat"])

def train_one_city(df: pd.DataFrame, date_col: str, city_col: str) -> dict:
    entity = to_entity_from_excel_col(city_col)

    # Skip kalau sudah ada model (resume)
    pack_path = MODELS_OUTDIR / f"{entity}_gbm_delta.pack.joblib"
    if pack_path.exists():
        return {"city": city_col, "entity": entity, "status": "skipped_exists", "model_path": str(pack_path)}

    series = pd.Series(pd.to_numeric(df[city_col], errors="coerce").values,
                       index=pd.to_datetime(df[date_col], errors="coerce")).dropna()
    y = series.asfreq("D").ffill().bfill()

    # fitur & target
    X, ydel, lag1, w = make_features_like_backend(y)
    if len(X) <= TEST_DAYS + 30:
        return {"city": city_col, "entity": entity, "status": "short_data"}

    split_point = X.index[-TEST_DAYS]
    X_tr, X_te = X.loc[X.index < split_point],  X.loc[X.index >= split_point]
    y_tr, y_te = ydel.loc[ydel.index < split_point], ydel.loc[ydel.index >= split_point]
    lag1_tr, lag1_te = lag1.loc[lag1.index < split_point], lag1.loc[lag1.index >= split_point]
    w_tr = w.loc[w.index < split_point]

    # baseline lag1 pada test (level)
    y_true_level = y.loc[X_te.index].values
    yhat_base = lag1_te.values
    base_R2   = r2_score(y_true_level, yhat_base)
    base_RMSE = rmse(y_true_level, yhat_base)
    base_MAE  = mean_absolute_error(y_true_level, yhat_base)
    base_MAPE = mape_pct(y_true_level, yhat_base)

    # GBM delta + CV (hemat RAM)
    tscv = TimeSeriesSplit(n_splits=5)
    gs = GridSearchCV(
        GradientBoostingRegressor(),
        param_grid=PARAM_GRID,
        scoring={"neg_rmse": make_scorer(neg_rmse), "r2": "r2"},
        refit="r2",
        cv=tscv,
        n_jobs=1,                 # stabil (hindari worker crash)
        pre_dispatch="2*n_jobs",
        return_train_score=False, # hemat memori
        verbose=0
    )
    gs.fit(X_tr, y_tr, sample_weight=w_tr)
    best = gs.best_estimator_

    # pred test: delta → level
    delta_hat_te = best.predict(X_te)
    yhat_model = lag1_te.values + delta_hat_te

    # blend pakai ekor train
    tail = min(90, len(X_tr))
    delta_tail  = best.predict(X_tr.iloc[-tail:])
    y_tail_true = y.loc[X_tr.index[-tail:]].values
    y_tail_model = lag1_tr.iloc[-tail:].values + delta_tail
    y_tail_base  = lag1_tr.iloc[-tail:].values
    best_alpha, best_e = 1.0, float("inf")
    for a in np.linspace(0, 1, 21):
        e = rmse(y_tail_true, a*y_tail_model + (1-a)*y_tail_base)
        if e < best_e: best_e, best_alpha = e, float(a)
    yhat_blend = best_alpha * yhat_model + (1 - best_alpha) * yhat_base

    R2   = r2_score(y_true_level, yhat_blend)
    RMSE = rmse(y_true_level, yhat_blend)
    MAE  = mean_absolute_error(y_true_level, yhat_blend)
    MAPE = mape_pct(y_true_level, yhat_blend)

    # simpan pack kalau lolos threshold
    MODELS_OUTDIR.mkdir(parents=True, exist_ok=True)
    saved = False
    model_path = ""
    status = "not_saved"
    if R2 >= SAVE_R2_THRESH:
        pack = {
            "model": best,
            "feature_cols": list(X_tr.columns),
            "best_config": {
                "mode": "diff",              # <— penting, backend kamu paham ini
                "transform": "none",
                "train_until": str((split_point - pd.Timedelta(days=1)).date()),
                "alpha_blend": best_alpha
            },
            "metrics": {
                "r2_test": float(R2),
                "rmse_test": float(RMSE),
                "mae_test": float(MAE),
                "mape_test_pct": float(MAPE),
                "r2_baseline": float(base_R2),
                "rmse_baseline": float(base_RMSE),
                "mae_baseline": float(base_MAE),
                "mape_baseline_pct": float(base_MAPE),
                "cv_best_params": gs.best_params_
            }
        }
        model_fpath = MODELS_OUTDIR / f"{entity}_gbm_delta.pack.joblib"  # cocok dgn app.py glob
        joblib.dump(pack, model_fpath)
        model_path = str(model_fpath)
        saved = True
        status = "saved"

        # optional: forecast 120 hari
        try:
            fc = forecast_iterative_delta(best, y, HORIZON_DAYS, feature_order=list(X_tr.columns))
            fc.to_csv(MODELS_OUTDIR / f"{entity}_forecast_{HORIZON_DAYS}d.csv", index=False)
        except Exception:
            pass

    # bersihin memori
    del X_tr, X_te, y_tr, y_te, lag1_tr, lag1_te, w_tr, gs, best
    gc.collect()

    return {
        "city": city_col,
        "entity": entity,
        "n": int(len(y)),
        "r2": float(R2), "rmse": float(RMSE), "mae": float(MAE), "mape_pct": float(MAPE),
        "r2_base": float(base_R2), "rmse_base": float(base_RMSE),
        "saved": saved, "status": status, "model_path": model_path,
        "best_params": json.dumps(PARAM_GRID) if not saved else json.dumps(pack["metrics"]["cv_best_params"]),
    }

def pick_first_n_cities(df: pd.DataFrame, date_col: str, n: int) -> list[str]:
    # ambil n kolom numeric pertama selain kolom tanggal (urut sesuai file)
    cols = []
    for c in df.columns:
        if c == date_col: continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= 0.3:
            cols.append(c)
        if len(cols) >= n: break
    return cols

def main():
    MODELS_OUTDIR.mkdir(parents=True, exist_ok=True)

    # incremental summary path
    incr_csv = MODELS_OUTDIR / "training_summary_top30_incremental.csv"

    df = pd.read_excel(DATA_PATH)
    dcol = find_date_col(df)
    df = df.dropna(subset=[dcol]).sort_values(dcol).reset_index(drop=True)

    cities = pick_first_n_cities(df, dcol, NUM_CITIES)
    print(f"[INFO] Melatih {len(cities)} kolom kota pertama: {cities[:5]}{' ...' if len(cities)>5 else ''}")

    results = []
    saved_cnt = 0
    for i, city in enumerate(cities, 1):
        print(f"[{i}/{len(cities)}] Training: {city} …")
        try:
            res = train_one_city(df, dcol, city)
        except Exception as e:
            res = {"city": city, "entity": "", "status": f"error: {e}"}

        results.append(res)
        # tulis incremental supaya aman kalau proses berhenti
        pd.DataFrame([res]).to_csv(
            incr_csv, index=False, mode="a", header=not incr_csv.exists(), encoding="utf-8"
        )

        tag = res.get("status", "")
        if tag == "saved":
            saved_cnt += 1
            print(f"   -> saved | R2={res.get('r2', 0):.3f}")
        elif tag == "skipped_exists":
            print(f"   -> skipped (already exists)")
        else:
            print(f"   -> {tag} | R2={res.get('r2', float('nan')):.3f}")

    # ringkasan final
    summary = pd.DataFrame(results)
    cols_order = ["city","entity","n","r2","rmse","mae","mape_pct","r2_base","rmse_base","saved","status","model_path","best_params"]
    for col in cols_order:
        if col not in summary.columns: summary[col] = ""
    summary = summary[cols_order].sort_values(["saved","r2"], ascending=[False, False])

    csv_path  = MODELS_OUTDIR / "training_summary_top30.csv"
    xlsx_path = MODELS_OUTDIR / "training_summary_top30.xlsx"
    summary.to_csv(csv_path, index=False, encoding="utf-8")
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as wr:
        summary.to_excel(wr, index=False, sheet_name="summary")

    print(f">> Saved: {csv_path}")
    print(f">> Saved: {xlsx_path}")
    print(f">> Done. Models saved: {saved_cnt}/{len(cities)}")

if __name__ == "__main__":
    main()
