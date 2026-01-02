#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import matplotlib.dates as mdates

import time, warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# sklearn
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ================= CONFIG =================
DATA_PATH = r"C:\Users\ASUS\skripsi_carin\data\dataset.xlsx"
OUTDIR = Path(r"C:\Users\ASUS\Documents\skripsi mancing\models_final_visual")

TRAIN_START = "2020-01-01"
TRAIN_END   = "2024-06-30"
TEST_START  = "2024-07-01"
TEST_END    = "2025-07-01"

SEED = 42
N_SPLITS_CV = 3

CITY_NAMES = [
    "Kota Surabaya",
    "Kota Depok",
    "Kota Yogyakarta",
    "Kotamobagu",
    "Kota Metro",
    "Kab. Sumenep",
    "Kota Bogor",
]

# 👉 Mapping nama kota untuk tampilan grafik
CITY_DISPLAY_NAME = {
    "Kota Surabaya": "Surabaya",
    "Kota Depok": "Depok",
    "Kota Yogyakarta": "Yogyakarta",
    "Kotamobagu": "Kotamobagu",
    "Kota Metro": "Metro",
    "Kab. Sumenep": "Sumenep",
    "Kota Bogor": "Bogor",
}

PARAM_GRID = {
    "learning_rate": [0.05, 0.03, 0.01],
    "max_depth": [1, 2, 3],
    "max_iter": [150, 300],
    "l2_regularization": [0.0, 1e-3],
    "min_samples_leaf": [10, 30],
}

EARLY_STOP = dict(
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20
)

# ================= FEATURE ENGINEERING =================
EID_DATES = [
    "2020-05-24","2021-05-13","2022-05-02",
    "2023-04-22","2024-04-10","2025-03-31"
]

def make_flags(idx):
    df = pd.DataFrame(index=idx)
    df["month"] = idx.month
    df["dayofweek"] = idx.dayofweek
    df["time_index"] = (idx - idx.min()).days

    eid = np.zeros(len(idx), dtype=int)
    for d in EID_DATES:
        t = pd.Timestamp(d)
        eid |= ((idx >= t - pd.Timedelta(days=14)) &
                (idx <= t + pd.Timedelta(days=7))).astype(int)

    df["is_eid_window"] = eid
    return df

@dataclass
class FeatureCfg:
    lags: list
    rolls: list

def build_features(y, cfg: FeatureCfg):
    y = y.astype(float)
    X = pd.DataFrame(index=y.index)

    X["lag_1"] = y.shift(1)

    for L in cfg.lags:
        if L != 1:
            X[f"lag_{L}"] = y.shift(L)

    for W in cfg.rolls:
        X[f"rollmean_{W}"] = (
            y.shift(1)
            .rolling(W, min_periods=W // 2)
            .mean()
        )

    X = X.join(make_flags(X.index))
    df = pd.concat([X, y.rename("y")], axis=1).dropna()
    return df.drop(columns="y"), df["y"]

# ================= METRICS =================
def rmse(y, yp):
    return np.sqrt(mean_squared_error(y, yp))

def mape(y, yp):
    y, yp = np.array(y), np.array(yp)
    return np.mean(np.abs((y - yp) / np.clip(np.abs(y), 1e-9, None))) * 100


def plot_actual_vs_pred(dates, y_actual, y_pred, city_key, outdir):
    
    display_city = CITY_DISPLAY_NAME.get(city_key, city_key)

    # smoothing ONLY for visualization
    y_actual_s = pd.Series(y_actual).rolling(7, center=True).mean()
    y_pred_s   = pd.Series(y_pred).rolling(7, center=True).mean()

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        dates,
        y_actual_s,
        label="Actual Price",
        color="#00FF66",
        linewidth=3.5,
        solid_capstyle="round"
    )

    ax.plot(
        dates,
        y_pred_s,
        label="Predicted Price",
        color="#7A1FA2",
        linewidth=3.5,
        linestyle="--"
    )

    # ===== AXIS LABELS =====
    ax.set_xlabel("Date", fontsize=13, labelpad=8)
    ax.set_ylabel("Price (Indonesian Rupiah)", fontsize=13, labelpad=10)

    # ===== TITLE =====
    ax.set_title(
        f"{display_city}",
        fontsize=14,
        weight="bold",
        pad=12
    )

    # ===== TICK SIZE + LEBIH BANYAK TICKS =====
    ax.tick_params(axis="both", which="major", labelsize=11, width=1.5, length=6)
    ax.tick_params(axis="both", which="minor", labelsize=9, width=1, length=3)
    
    # ===== FORMAT SUMBU X (TANGGAL LEBIH RAPAT) =====
    # Tentukan interval berdasarkan range data
    date_range = (dates.max() - dates.min()).days
    
    if date_range > 300:  # lebih dari 1 tahun
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # tiap 2 bulan
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
    else:  # kurang dari 1 tahun
        ax.xaxis.set_major_locator(mdates.MonthLocator())  # tiap bulan
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=2))
    
    # ===== FORMAT SUMBU Y (HARGA LEBIH RAPAT) =====
    # Bikin tick Y lebih banyak dan rapat
    y_min = min(y_actual_s.min(), y_pred_s.min())
    y_max = max(y_actual_s.max(), y_pred_s.max())
    y_range = y_max - y_min
    
    # Tentukan interval yang masuk akal
    if y_range > 5000:
        y_step = 1000  # interval Rp 1000
    elif y_range > 2000:
        y_step = 500
    elif y_range > 1000:
        y_step = 200
    else:
        y_step = 100
    
    # Buat tick locations
    y_ticks = np.arange(
        np.floor(y_min / y_step) * y_step,
        np.ceil(y_max / y_step) * y_step + y_step,
        y_step
    )
    ax.set_yticks(y_ticks)
    
    # Minor ticks untuk Y
    ax.yaxis.set_minor_locator(plt.MultipleLocator(y_step / 2))

    # ===== GRID (LEBIH JELAS) =====
    ax.grid(True, which='major', linestyle='-', alpha=0.3, linewidth=1)
    ax.grid(True, which='minor', linestyle=':', alpha=0.15, linewidth=0.5)

    # ===== LEGEND =====
    ax.legend(
        fontsize=11,
        loc="upper left",
        frameon=True,
        fancybox=True,
        shadow=True
    )

    # ===== DEMPETIN KE DATA (INI PENTING!) =====
    ax.margins(x=0.005, y=0.02)  # lebih kecil = lebih dempet
    
    # Set limit Y yang lebih ketat
    ax.set_ylim(y_min * 0.995, y_max * 1.005)

    # ===== ROTASI LABEL X (BIAR GA KETUMPUK) =====
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

    # ===== FINAL TOUCH =====
    fig.tight_layout()
    
    # Adjust margins biar maksimal padat
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12)

    fname = display_city.replace(" ", "_")
    fig.savefig(outdir / f"{fname}_actual_vs_predicted.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

# ================= PROCESS =================

def plot_two_cities_side_by_side(dates1, y_actual1, y_pred1, city_key1,
                                   dates2, y_actual2, y_pred2, city_key2, outdir):
    """
    Plot 2 kota dalam 1 gambar (kiri-kanan)
    """
    
    display_city1 = CITY_DISPLAY_NAME.get(city_key1, city_key1)
    display_city2 = CITY_DISPLAY_NAME.get(city_key2, city_key2)

    # Smoothing
    y_actual1_s = pd.Series(y_actual1).rolling(7, center=True).mean()
    y_pred1_s   = pd.Series(y_pred1).rolling(7, center=True).mean()
    y_actual2_s = pd.Series(y_actual2).rolling(7, center=True).mean()
    y_pred2_s   = pd.Series(y_pred2).rolling(7, center=True).mean()

    # BIKIN FIGURE DENGAN 2 SUBPLOT (1 baris, 2 kolom)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # ========== PLOT KIRI (City 1) ==========
    ax1.plot(dates1, y_actual1_s, label="Actual Price", 
             color="#00FF66", linewidth=5, solid_capstyle="round")
    ax1.plot(dates1, y_pred1_s, label="Predicted Price",
             color="#7A1FA2", linewidth=5, linestyle="--")
    
    ax1.set_xlabel("Date", fontsize=22, labelpad=12, weight='bold')
    ax1.set_ylabel("Price (Indonesian Rupiah)", fontsize=22, labelpad=14, weight='bold')
    ax1.set_title(f"Actual vs Predicted – {display_city1}", 
                  fontsize=24, weight="bold", pad=20)
    
    # Format X axis
    date_range1 = (dates1.max() - dates1.min()).days
    total_months1 = date_range1 / 30
    month_interval1 = max(1, int(np.ceil(total_months1 / 5)))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax1.xaxis.set_minor_locator(plt.NullLocator())
    
    # Format Y axis
    y_min1 = min(y_actual1_s.min(), y_pred1_s.min())
    y_max1 = max(y_actual1_s.max(), y_pred1_s.max())
    y_range1 = y_max1 - y_min1
    
    if y_range1 > 5000:
        y_step1 = 1000
    elif y_range1 > 2000:
        y_step1 = 500
    elif y_range1 > 1000:
        y_step1 = 200
    else:
        y_step1 = 100
    
    y_ticks1 = np.arange(
        np.floor(y_min1 / y_step1) * y_step1,
        np.ceil(y_max1 / y_step1) * y_step1 + y_step1,
        y_step1
    )
    ax1.set_yticks(y_ticks1)
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(y_step1 / 2))
    
    ax1.tick_params(axis="both", which="major", labelsize=18, width=2.5, length=10)
    ax1.grid(True, which='major', linestyle='-', alpha=0.4, linewidth=1.2)
    ax1.legend(fontsize=20, loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax1.margins(x=0, y=0.01)
    ax1.set_ylim(y_min1 * 0.998, y_max1 * 1.002)
    ax1.set_xlim(dates1.min(), dates1.max())

    # ========== PLOT KANAN (City 2) ==========
    ax2.plot(dates2, y_actual2_s, label="Actual Price",
             color="#00FF66", linewidth=5, solid_capstyle="round")
    ax2.plot(dates2, y_pred2_s, label="Predicted Price",
             color="#7A1FA2", linewidth=5, linestyle="--")
    
    ax2.set_xlabel("Date", fontsize=22, labelpad=12, weight='bold')
    ax2.set_ylabel("Price (Indonesian Rupiah)", fontsize=22, labelpad=14, weight='bold')
    ax2.set_title(f"Actual vs Predicted – {display_city2}",
                  fontsize=24, weight="bold", pad=20)
    
    # Format X axis
    date_range2 = (dates2.max() - dates2.min()).days
    total_months2 = date_range2 / 30
    month_interval2 = max(1, int(np.ceil(total_months2 / 5)))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax2.xaxis.set_minor_locator(plt.NullLocator())
    
    # Format Y axis
    y_min2 = min(y_actual2_s.min(), y_pred2_s.min())
    y_max2 = max(y_actual2_s.max(), y_pred2_s.max())
    y_range2 = y_max2 - y_min2
    
    if y_range2 > 5000:
        y_step2 = 1000
    elif y_range2 > 2000:
        y_step2 = 500
    elif y_range2 > 1000:
        y_step2 = 200
    else:
        y_step2 = 100
    
    y_ticks2 = np.arange(
        np.floor(y_min2 / y_step2) * y_step2,
        np.ceil(y_max2 / y_step2) * y_step2 + y_step2,
        y_step2
    )
    ax2.set_yticks(y_ticks2)
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(y_step2 / 2))
    
    ax2.tick_params(axis="both", which="major", labelsize=18, width=2.5, length=10)
    ax2.grid(True, which='major', linestyle='-', alpha=0.4, linewidth=1.2)
    ax2.legend(fontsize=20, loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax2.margins(x=0, y=0.01)
    ax2.set_ylim(y_min2 * 0.998, y_max2 * 1.002)
    ax2.set_xlim(dates2.min(), dates2.max())

    # ===== FINAL TOUCH =====
    fig.tight_layout(pad=3.0)
    
    fname1 = display_city1.replace(" ", "_")
    fname2 = display_city2.replace(" ", "_")
    fig.savefig(outdir / f"{fname1}_and_{fname2}_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


# ===== MODIFIKASI FUNGSI process_city =====
# Ubah fungsi process_city agar return data test-nya:

def process_city_with_return(df, city, outdir):
    """Versi process_city yang return data untuk di-merge"""
    print(f"\n### Processing: {city}")

    y = pd.Series(df[city].values, index=df["date"]).asfreq("D").ffill().bfill()
    X_all, y_all = build_features(y, FeatureCfg(lags=[7, 14, 30], rolls=[7, 30]))

    train_mask = (X_all.index >= TRAIN_START) & (X_all.index <= TRAIN_END)
    test_mask  = (X_all.index >= TEST_START) & (X_all.index <= TEST_END)

    X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
    X_test, y_test   = X_all.loc[test_mask],  y_all.loc[test_mask]

    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    model = HistGradientBoostingRegressor(loss="squared_error", random_state=SEED, **EARLY_STOP)
    gs = GridSearchCV(model, PARAM_GRID, scoring="r2", cv=tscv, n_jobs=-1)
    
    t0 = time.perf_counter()
    gs.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": rmse(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "mape": mape(y_test, y_pred),
        "best_params": gs.best_params_,
        "train_time_sec": train_time
    }

    pack = {"model": best_model, "features": list(X_train.columns), "metrics": metrics}
    
    display_city = CITY_DISPLAY_NAME.get(city, city)
    fname = display_city.replace(" ", "_")
    joblib.dump(pack, outdir / f"{fname}_best_model.joblib")
    
    # Plot individual
    plot_actual_vs_pred(X_test.index, y_test, y_pred, city, outdir)
    
    print(f"R2={metrics['r2']:.4f} | MAPE={metrics['mape']:.2f}%")
    
    # RETURN DATA BUAT DI-MERGE!
    return {
        "dates": X_test.index,
        "y_actual": y_test,
        "y_pred": y_pred,
        "city_key": city
    }

# ================= RUN =================
# ===== MODIFIKASI FUNGSI main() =====
def main():
    warnings.filterwarnings("ignore")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(DATA_PATH)
    df.columns = ["date"] + list(df.columns[1:])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Simpan hasil semua kota
    results = {}
    for city in CITY_NAMES:
        results[city] = process_city_with_return(df, city, OUTDIR)

    # ===== BIKIN GRAFIK GABUNGAN (CONTOH: SURABAYA + DEPOK) =====
    print("\n### Creating merged plots...")
    
    # Contoh 1: Surabaya + Depok
    plot_two_cities_side_by_side(
        results["Kota Surabaya"]["dates"], 
        results["Kota Surabaya"]["y_actual"],
        results["Kota Surabaya"]["y_pred"],
        "Kota Surabaya",
        results["Kota Depok"]["dates"],
        results["Kota Depok"]["y_actual"],
        results["Kota Depok"]["y_pred"],
        "Kota Depok",
        OUTDIR
    )
    
    # Contoh 2: Yogyakarta + Bogor (bisa ditambah sesuai kebutuhan)
    plot_two_cities_side_by_side(
        results["Kota Yogyakarta"]["dates"],
        results["Kota Yogyakarta"]["y_actual"],
        results["Kota Yogyakarta"]["y_pred"],
        "Kota Yogyakarta",
        results["Kota Bogor"]["dates"],
        results["Kota Bogor"]["y_actual"],
        results["Kota Bogor"]["y_pred"],
        "Kota Bogor",
        OUTDIR
    )
    
    print("✅ All plots created successfully!")

if __name__ == "__main__":
    main()
