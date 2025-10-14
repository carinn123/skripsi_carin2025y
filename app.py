# app.py
import os, json, warnings, re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd, time
from flask import Flask, request, jsonify, send_from_directory, abort, render_template
from flask_cors import CORS
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import math
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
import traceback
from io import StringIO

# =======================
# KONFIGURASI DASAR
# =======================
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH   = BASE_DIR / "data" / "dataset_filled_ffill_bfill.xlsx"
MODELS_DIR  = BASE_DIR / "packs"

ENTITY_PROV_PATH = BASE_DIR / "static" / "entity_to_province.json"
CITY_COORDS_PATH = BASE_DIR / "static" / "city_coords.json"
EVAL_XLSX = BASE_DIR / "models" / "training_summary_all_cities.xlsx"
TOPN_XLSX = BASE_DIR / "data" / "topn_per_tahun_per_kota.xlsx"
REGIONAL_XLSX = BASE_DIR / "data" / "regional_correlation_summary.xlsx"
_REGION_CACHE = {"mtime": None, "df": None}

# Cache evaluasi di memory
_EVAL_CACHE = None


FILL_METHOD = "ffill_bfill"  # atau "interpolate"
RELOAD_TOKEN = os.getenv("RELOAD_TOKEN", "secret123")

# =======================
# MAPPING PROVINSI → PULAU
# =======================
PROVINCE_TO_ISLAND = {
    "DKI Jakarta": "Jawa",
    "Jawa Barat": "Jawa",
    "Jawa Tengah": "Jawa",
    "DI Yogyakarta": "Jawa",
    "Jawa Timur": "Jawa",
    "Banten": "Jawa",
    "Aceh": "Sumatra",
    "Sumatera Utara": "Sumatra",
    "Sumatera Barat": "Sumatra",
    "Riau": "Sumatra",
    "Kepulauan Riau": "Sumatra",
    "Jambi": "Sumatra",
    "Sumatera Selatan": "Sumatra",
    "Kepulauan Bangka Belitung": "Sumatra",
    "Bali": "Bali–NT",
    "Nusa Tenggara Barat": "Bali–NT",
    "Nusa Tenggara Timur": "Bali–NT",
    "Kalimantan Barat": "Kalimantan",
    "Kalimantan Tengah": "Kalimantan",
    "Kalimantan Selatan": "Kalimantan",
    "Kalimantan Timur": "Kalimantan",
    "Kalimantan Utara": "Kalimantan",
    "Sulawesi Utara": "Sulawesi",
    "Sulawesi Tengah": "Sulawesi",
    "Sulawesi Selatan": "Sulawesi",
    "Sulawesi Tenggara": "Sulawesi",
    "Gorontalo": "Sulawesi",
    "Sulawesi Barat": "Sulawesi",
    "Maluku": "Maluku",
    "Maluku Utara": "Maluku",
    "Papua": "Papua",
    "Papua Barat": "Papua",
    "Papua Barat Daya": "Papua",
    "Papua Tengah": "Papua",
    "Papua Pegunungan": "Papua",
    "Papua Selatan": "Papua",
}

# =======================
# SLUG → ENTITY (opsional; sisanya heuristik)
# =======================
CITY_SLUG_TO_ENTITY = {
    # "banjarmasin": "kota_banjarmasin",
    # "jakarta-pusat": "kota_administrasi_jakarta_pusat",
    # "jakarta-selatan": "kota_administrasi_jakarta_selatan",
    # "jakarta-barat": "kota_administrasi_jakarta_barat",
    # "jakarta-timur": "kota_administrasi_jakarta_timur",
    # "jakarta-utara": "kota_administrasi_jakarta_utara",
    # "bandung": "kota_bandung",
    # "surabaya": "kota_surabaya",
    # "medan": "kota_medan",
    # "semarang": "kota_semarang",
    # "makassar": "kota_makassar",
    # "palembang": "kota_palembang",
    # "pontianak": "kota_pontianak",
    # "manado": "kota_manado",
    # "denpasar": "kota_denpasar",
    # "mataram": "kota_mataram",
    # "kupang": "kota_kupang",
    # "ambon": "kota_ambon",
    # "jayapura": "kota_jayapura",
    # "biak": "kab._biak_numfor",
    # "bogor": "kota_bogor",
}

# =======================
# APP
# =======================
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

# Kurangi cache static agar cepat terlihat perubahan saat dev
from datetime import timedelta
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = timedelta(seconds=0)
# =======================
# UTIL DATA
# =======================

# ===== Kalender flags sederhana (SAMAKAN dengan training) =====
EID_DATES = ["2020-05-24","2021-05-13","2022-05-02","2023-04-22","2024-04-10","2025-03-31"]


import joblib, time
from dataclasses import dataclass

# local fallbacks (use global if defined)
UPLOAD_TEST_DAYS = globals().get("TEST_DAYS", 365)
UPLOAD_PARAM_GRID = globals().get("PARAM_GRID", {
    "learning_rate": [0.05, 0.01],
    "max_depth": [1, 2],
    "max_iter": [150],
    "l2_regularization": [0.0],
    "min_samples_leaf": [10]
})
UPLOAD_FEATURE_EXPS = globals().get("FEATURE_EXPERIMENTS", [
    {"name":"lag1","LAGS":[],"ROLLS":[]},
    {"name":"lag7","LAGS":[7],"ROLLS":[]},
    {"name":"lag30","LAGS":[30],"ROLLS":[]}
])
UPLOAD_EID_DATES = globals().get("EID_DATES", ["2020-05-24","2021-05-13","2022-05-02","2023-04-22","2024-04-10","2025-03-31"])
UPLOAD_OUTDIR = Path("./models_packs_uploads")
UPLOAD_OUTDIR.mkdir(parents=True, exist_ok=True)
UPLOAD_N_SPLITS = globals().get("N_SPLITS_CV", 3)
UPLOAD_SEED = globals().get("SEED", 42)
UPLOAD_EARLY_STOP = globals().get("EARLY_STOP_KW", dict(early_stopping=True, validation_fraction=0.15, n_iter_no_change=20))

# local helper to avoid relying on global ensure_dir
def upload_ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

# small feature cfg for upload
@dataclass
class upload_FeatureCfg:
    add_lags: list
    rolls: list

# simple calendar flags (upload-local)
def upload_make_flags(idx):
    di = pd.DatetimeIndex(pd.to_datetime(idx, errors="coerce")).tz_localize(None).normalize()
    f = pd.DataFrame(index=di)
    f["month"] = di.month
    f["dayofweek"] = di.dayofweek
    f["time_index"] = (di - di.min()).days
    f["is_end_of_year"] = ((di.month == 12) & (di.day >= 22)).astype(int)
    f["is_new_year"] = ((di.month == 1) & (di.day <= 7)).astype(int)
    eid = np.zeros(len(di), dtype=int)
    for d in UPLOAD_EID_DATES:
        t = pd.Timestamp(d)
        eid |= ((di >= t - pd.Timedelta(days=14)) & (di <= t + pd.Timedelta(days=7))).astype(int)
    f["is_eid_window"] = eid
    return f

def upload_build_features_level_target(y: pd.Series, cfg: upload_FeatureCfg):
    y = pd.Series(pd.to_numeric(y, errors="coerce"), index=pd.DatetimeIndex(y.index)).astype(float)
    X = pd.DataFrame(index=y.index)
    X["lag_1"] = y.shift(1)
    for L in (cfg.add_lags or []):
        if L == 1: continue
        X[f"lag_{L}"] = y.shift(L)
    for W in (cfg.rolls or []):
        X[f"rollmean_{W}"] = y.shift(1).rolling(W, min_periods=max(1, W//2)).mean()
    X = X.join(upload_make_flags(X.index))
    df = pd.concat([X, y.rename("y")], axis=1).dropna()
    y_target = df.pop("y")
    return df, y_target

def upload_build_feature_row_for_date(feature_cols, history_series: pd.Series, target_date: pd.Timestamp):
    idx_min = history_series.index.min()
    row = {}
    for col in feature_cols:
        if col.startswith('lag_'):
            L = int(col.split('_')[1])
            lookup = target_date - pd.Timedelta(days=L)
            row[col] = float(history_series.loc[lookup]) if lookup in history_series.index else float(history_series.iloc[-1])
        elif col.startswith('rollmean_'):
            W = int(col.split('_')[1])
            end = target_date - pd.Timedelta(days=1)
            window = history_series.loc[:end].iloc[-W:] if len(history_series.loc[:end])>0 else history_series.iloc[-W:]
            row[col] = float(window.mean()) if len(window)>0 else float(history_series.iloc[-1])
        else:
            flags = upload_make_flags([target_date]).iloc[0].to_dict()
            row[col] = flags.get(col, 0)
    return row

def upload_iterative_forecast_from_pack(pack: dict, series: pd.Series, horizons=[1,7,10]):
    model = pack['model']
    feature_cols = pack['feature_cols']
    last_date = series.index.max()
    history = series.copy().astype(float)
    max_h = max(horizons)
    preds = {}
    for step in range(1, max_h+1):
        target_date = last_date + pd.Timedelta(days=step)
        row = upload_build_feature_row_for_date(feature_cols, history, target_date)
        Xrow = pd.DataFrame([row], columns=feature_cols)
        # align columns to model if possible
        if hasattr(model, "feature_names_in_"):
            for c in model.feature_names_in_:
                if c not in Xrow.columns:
                    Xrow[c] = 0.0
            Xrow = Xrow.reindex(columns=list(model.feature_names_in_), fill_value=0.0)
        yhat = float(model.predict(Xrow)[0])
        history.loc[target_date] = yhat
        if step in horizons:
            preds[str(step)] = {'date': target_date.strftime('%Y-%m-%d'), 'value': yhat}
    return preds

# training wrapper (upload-local)
def upload_train_one_city(series_full: pd.Series, city_name: str, outdir: Path, test_days=None):
    """
    Train model untuk satu kota dari series harga harian.
    Sekarang: test_days otomatis = 20% dari panjang data (dibulatkan ke atas), minimal 7 hari.
    """

    n_total = len(series_full.dropna())

    # === Tentukan test_days otomatis (20% dibulatkan ke atas, min=7)
    if test_days is None:
        test_days = max(7, int(np.ceil(n_total * 0.2)))

    # Jika test_days terlalu besar, kurangi jadi max 40% data
    if test_days >= n_total * 0.4:
        test_days = int(np.floor(n_total * 0.3))

    print(f"[TRAIN] {city_name}: total={n_total}, test_days={test_days}")

    # === Validasi panjang data
    if len(series_full) <= test_days + 50:
        return {
            "city": city_name,
            "ok": False,
            "reason": f"series too short (len={len(series_full)}, need>{test_days + 200})",
            "n": int(len(series_full))
        }

    best_record = {
        "r2": -1e9,
        "exp_name": None,
        "model": None,
        "feature_cols": None,
        "metrics": None
    }

    # === Loop fitur eksperimen ===
    for exp in UPLOAD_FEATURE_EXPS:
        add_lags = exp["LAGS"]
        rolls = exp["ROLLS"]
        exp_name = exp["name"]

        X_all, y_all = upload_build_features_level_target(
            series_full,
            upload_FeatureCfg(add_lags=add_lags, rolls=rolls)
        )

        if len(X_all) <= test_days + 50:
            print(f"[SKIP] {city_name} • {exp_name}: data < test_days+50 ({len(X_all)})")
            continue

        # Split train-test
        X_train, X_test = X_all.iloc[:-test_days], X_all.iloc[-test_days:]
        y_train, y_test = y_all.iloc[:-test_days], y_all.iloc[-test_days:]

        max_lookback = int(max([1, *add_lags, *(rolls or [0])]))

        try:
            tscv = TimeSeriesSplit(n_splits=UPLOAD_N_SPLITS, gap=max_lookback)
        except TypeError:
            tscv = TimeSeriesSplit(n_splits=UPLOAD_N_SPLITS)

        base = HGBR(loss="squared_error", random_state=UPLOAD_SEED, **UPLOAD_EARLY_STOP)
        gs = GridSearchCV(
            estimator=base,
            param_grid=UPLOAD_PARAM_GRID,
            scoring="r2",
            refit=True,
            cv=tscv,
            n_jobs=1,
            verbose=0
        )

        print(f"[GRIDSEARCH] {city_name} • {exp_name}: training {len(X_train)} / testing {len(X_test)} ...")
        t0 = time.perf_counter()
        gs.fit(X_train, y_train)
        train_secs = time.perf_counter() - t0

        best_model = gs.best_estimator_
        y_pred = best_model.predict(X_test)

        r2_val = float(r2_score(y_test, y_pred))
        mae_val = float(mean_absolute_error(y_test, y_pred))

        metrics = dict(
            city=city_name,
            exp=exp_name,
            lags=[1, *add_lags],
            rolls=rolls,
            r2=r2_val,
            mae=mae_val,
            cv_best_r2=float(gs.best_score_),
            train_time_seconds=float(train_secs),
            best_params=gs.best_params_
        )

        print(f"[RESULT] {city_name} • {exp_name}: R²={r2_val:.3f}, MAE={mae_val:.1f}, best={gs.best_params_}")

        if r2_val > best_record["r2"]:
            best_record.update(
                r2=r2_val,
                exp_name=exp_name,
                model=best_model,
                feature_cols=list(X_train.columns),
                metrics=metrics
            )

    # === Jika tidak ada model terbaik ditemukan
    if best_record["model"] is None:
        return {
            "city": city_name,
            "ok": False,
            "reason": "no model found (semua eksperimen gagal)",
            "n": int(len(series_full))
        }

    # === Simpan model terbaik
    city_safe = city_name.replace("/", "-").replace("\\", "-").replace(" ", "_")[:180]
    best_dir = upload_ensure_dir(outdir / city_safe / "best_model")
    pack = {
        "model": best_record["model"],
        "feature_cols": best_record["feature_cols"],
        "best_config": {"exp_name": best_record["exp_name"]},
        "metrics": best_record["metrics"]
    }

    pack_path = best_dir / f"{city_safe}__{best_record['exp_name']}__best_pack.joblib"
    joblib.dump(pack, pack_path)

    preds = upload_iterative_forecast_from_pack(pack, series_full, horizons=[1, 7, 10])

    return {
        "city": city_name,
        "ok": True,
        "best_r2": float(best_record["r2"]),
        "pack_path": str(pack_path),
        "predictions": preds,
        "metrics": best_record["metrics"],
        "test_days": test_days,
        "n_total": n_total
    }


# route
@app.route("/api/upload_file", methods=["POST"])
def upload_file_endpoint():
    # debug helper: return traceback when exception occurs
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file"}), 400
        f = request.files['file']
        mode = request.form.get('mode', 'full').lower()

        # read file robustly and report errors
        try:
            fname = (f.filename or "").lower()
            if fname.endswith(".csv"):
                df = pd.read_csv(f)
            else:
                # openpyxl needed for .xlsx; this may raise if not installed or bad file
                df = pd.read_excel(f, engine="openpyxl")
        except Exception as e:
            tb = traceback.format_exc()
            print("[UPLOAD READ ERROR]", tb)
            return jsonify({"error": "read error", "detail": str(e), "trace": tb}), 400

        # validate date column
        if df.shape[1] == 0:
            return jsonify({"error": "Empty file or no columns read"}), 400

        if df.columns[0].lower() != 'date':
            df = df.rename(columns={df.columns[0]: 'date'})
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception as e:
            tb = traceback.format_exc()
            print("[DATE PARSE ERROR]", tb)
            return jsonify({"error": "date parse error", "detail": str(e), "trace": tb}), 400

        df = df.sort_values('date').reset_index(drop=True)

        # collect non-empty value columns
        value_cols = [c for c in df.columns if c.lower() != 'date' and not df[c].dropna().empty]
        if not value_cols:
            return jsonify({"error": "No valid city/value columns found"}), 400

        # choose first column (for now)
        col = value_cols[0]
        series_full = (
            pd.Series(pd.to_numeric(df[col], errors='coerce').values, index=pd.DatetimeIndex(df['date']))
            .asfreq('D')
            .ffill().bfill()
        )

        print(f"[UPLOAD] city={col}, mode={mode}, len={len(series_full)}")

        # QUICK mode: no training heavy (safe for debugging)
        if mode == 'quick':
            s = series_full.dropna()
            stats = {
                'n_points': int(s.shape[0]),
                'avg': float(s.mean()) if len(s)>0 else None,
                'min': float(s.min()) if len(s)>0 else None,
                'max': float(s.max()) if len(s)>0 else None
            }
            last_date = series_full.index.max()
            preds = {
                '1': {'date': (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d'), 'value': float(series_full.iloc[-1])},
                '7': {'date': (last_date + pd.Timedelta(days=7)).strftime('%Y-%m-%d'),
                       'value': float(series_full.shift(1).rolling(7, min_periods=1).mean().iloc[-1])},
                '30': {'date': (last_date + pd.Timedelta(days=30)).strftime('%Y-%m-%d'),
                       'value': float(series_full.shift(1).rolling(30, min_periods=1).mean().iloc[-1])},
            }
            if len(s) > 180: s = s.iloc[-180:]
            trend = {'dates': [d.strftime('%Y-%m-%d') for d in s.index], 'values': [float(x) for x in s.values]}
            pred_series = {'dates': trend['dates'], 'actual': trend['values'], 'pred': [None] + trend['values'][:-1]}
            return jsonify({'mode': 'quick', 'column': col, 'stats': stats, 'predictions': preds, 'trend': trend, 'pred_series': pred_series})

        # FULL training path (wrap in try and return trace on error)
        try:
            res = upload_train_one_city(series_full, col, UPLOAD_OUTDIR)
        except Exception as e:
            tb = traceback.format_exc()
            print("[TRAIN ERROR]", tb)
            return jsonify({"error": "train error", "detail": str(e), "trace": tb}), 500

        if not res.get("ok"):
            return jsonify({"error": res.get("reason", "train failed"), "detail": res}), 400

        s = series_full.dropna()
        stats = {'n_points': int(s.shape[0]), 'avg': float(s.mean()), 'min': float(s.min()), 'max': float(s.max())}
        if len(s) > 180: s = s.iloc[-180:]
        trend = {'dates': [d.strftime('%Y-%m-%d') for d in s.index], 'values': [float(x) for x in s.values]}

        return jsonify({
            'mode': 'full',
            'city': col,
            'stats': stats,
            'predictions': res.get('predictions'),
            'metrics': res.get('metrics'),
            'trend': trend,
            'best_r2': res.get('best_r2'),
            'pack_path': res.get('pack_path')
        })
    except Exception as e:
        tb = traceback.format_exc()
        print("[UPLOAD ENDPOINT ERROR]", tb)
        return jsonify({"error": "unexpected server error", "detail": str(e), "trace": tb}), 500

def _make_calendar_flags(idx: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=idx)
    f["month"] = idx.month
    f["dayofweek"] = idx.dayofweek
    # penting: nama kolom HARUS sama dgn training
    f["time_index"] = (idx - idx.min()).days
    f["is_end_of_year"] = ((idx.month == 12) & (idx.day >= 22)).astype(int)
    f["is_new_year"] = ((idx.month == 1) & (idx.day <= 7)).astype(int)
    eid = np.zeros(len(idx), dtype=int)
    for d in EID_DATES:
        t = pd.Timestamp(d)
        eid |= ((idx >= t - pd.Timedelta(days=14)) & (idx <= t + pd.Timedelta(days=7))).astype(int)
    f["is_eid_window"] = eid
    return f

def to_numeric_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.replace({"-": np.nan, "": np.nan})
    s = s.str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def build_wide_long(path_xlsx: str, fill_method: str):
    raw = pd.read_excel(path_xlsx)
    date_col   = raw.columns[0]
    value_cols = raw.columns[1:]

    # >>> samakan dengan training: TANPA dayfirst
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")

    raw = raw.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    # >>> pakai parser angka yang robust gaya Indonesia/Inggris
    def _parse_num_id_en(x):
        if pd.isna(x): return np.nan
        s = str(x).strip()
        # buang semua pemisah ribuan umum: spasi, koma, titik
        # lalu pulihkan titik desimal jika memang desimal (jarang dipakai di dataset harga harian)
        s = re.sub(r"[^\d,.\-]", "", s)
        # kasus ribuan ID: 17.750 -> 17750
        if s.count(".") == 1 and s.split(".")[1].isdigit() and len(s.split(".")[1]) == 3:
            s = s.replace(".", "")
        # kasus ribuan EN: 18,000 -> 18000
        s = s.replace(",", "")
        try: return float(s)
        except: return np.nan

    for c in value_cols:
        raw[c] = raw[c].apply(_parse_num_id_en)

    full_idx = pd.date_range(raw[date_col].min(), raw[date_col].max(), freq="D")
    wide = (raw.set_index(date_col).reindex(full_idx).rename_axis("date").sort_index())

    if fill_method == "ffill_bfill":
        wide[value_cols] = wide[value_cols].ffill().bfill()
    elif fill_method == "interpolate":
        wide[value_cols] = wide[value_cols].interpolate(method="time", limit_direction="both").ffill().bfill()
    else:
        raise ValueError("FILL_METHOD harus 'ffill_bfill' atau 'interpolate'.")

    long_df = (wide.reset_index()
                    .melt(id_vars="date", var_name="entity", value_name="value")
                    .dropna(subset=["value"])
                    .sort_values(["entity","date"])
                    .reset_index(drop=True))
    long_df["entity"] = (long_df["entity"].astype(str).str.strip().str.lower()
                         .str.replace(r"\s+", "_", regex=True))
    return wide, long_df


def _compute_last_actual_dates(path_xlsx: str) -> dict:
    """Cari last tanggal NON-NaN per kolom di file mentah (bukan hasil ffill penuh)."""
    raw = pd.read_excel(path_xlsx)
    date_col = raw.columns[0]
    raw[date_col] = pd.to_datetime(raw[date_col], dayfirst=True, errors="coerce")
    raw = raw.dropna(subset=[date_col]).sort_values(date_col)
    value_cols = raw.columns[1:]

    cutoff = pd.Timestamp("2025-07-01")  # batas akhir data nyata
    last = {}
    for c in value_cols:
        ent = re.sub(r"\s+", "_", str(c).strip().lower())
        s = pd.to_numeric(raw[c], errors="coerce")
        valid_dates = raw.loc[s.notna(), date_col]
        valid_dates = valid_dates[valid_dates <= cutoff]  # pastikan tidak lewat Juli
        if not valid_dates.empty:
            last[ent] = valid_dates.max().normalize()
    return last


def add_fourier(df: pd.DataFrame, date_col: str, periods, K=2, prefix="fyr"):
    t = (df[date_col] - df[date_col].min()).dt.days.astype(float)
    for P in periods:
        for k in range(1, K+1):
            df[f"{prefix}_sin_P{P}_k{k}"] = np.sin(2*np.pi*k*t/P)
            df[f"{prefix}_cos_P{P}_k{k}"] = np.cos(2*np.pi*k*t/P)

def _series_stats(list_of_dict):
    """
    list_of_dict: [{'date':'YYYY-MM-DD','value':float}, ...]
    return: dict {n, avg, min, min_date, max, max_date, start, end, change_pct}
    """
    vals = [(pd.to_datetime(d['date']).date(), float(d['value'])) 
            for d in list_of_dict if d.get('value') is not None]
    if not vals:
        return dict(n=0, avg=None, min=None, min_date=None,
                    max=None, max_date=None, start=None, end=None, change_pct=None)

    n = len(vals)
    s = sum(v for _,v in vals)
    avg = s / n
    # urut tanggal untuk start-end & robust
    vals_sorted = sorted(vals, key=lambda x: x[0])
    start = vals_sorted[0][1]
    end   = vals_sorted[-1][1]
    min_date, min_val = min(vals, key=lambda x: x[1])
    max_date, max_val = max(vals, key=lambda x: x[1])
    change_pct = None if start == 0 else (end - start) / start * 100.0

    return dict(
        n=n, avg=avg,
        min=min_val, min_date=min_date.isoformat(),
        max=max_val, max_date=max_date.isoformat(),
        start=start, end=end,
        change_pct=change_pct
    )

# Kalender flags sederhana (samakan dengan training)
EID_DATES = ["2020-05-24","2021-05-13","2022-05-02","2023-04-22","2024-04-10","2025-03-31"]


def make_flags(idx) -> pd.DataFrame:
    di = pd.DatetimeIndex(pd.to_datetime(idx, errors="coerce")).tz_localize(None).normalize()
    f = pd.DataFrame(index=di)
    f["month"] = di.month
    f["dayofweek"] = di.dayofweek
    # time index berbasis awal seri
    f["time_index"] = (di - di.min()).days
    f["is_end_of_year"] = ((di.month == 12) & (di.day >= 22)).astype(int)
    f["is_new_year"]  = ((di.month == 1) & (di.day <= 7)).astype(int)

    eid = np.zeros(len(di), dtype=int)
    for d in EID_DATES:
        t = pd.Timestamp(d)
        eid |= ((di >= t - pd.Timedelta(days=14)) & (di <= t + pd.Timedelta(days=7))).astype(int)
    f["is_eid_window"] = eid
    return f



def make_features_entity(dfe: pd.DataFrame, horizon=1):
    df = dfe.sort_values("date").copy()

    # === hanya fitur yang dipakai saat training lag30 ===
    df["lag_1"]  = df["value"].shift(1)
    df["lag_30"] = df["value"].shift(30)

    # flags kalender (TIDAK bikin month/dayofweek lagi di luar flags)
    flags = make_flags(df["date"])
    df = pd.concat([df.set_index("date"), flags], axis=1).reset_index()

    # target untuk one-step eval
    df["y_next"] = df["value"].shift(-1)
    df["y_diff"] = df["y_next"] - df["value"]

    # drop baris yang belum punya lags
    df = df.dropna(subset=["lag_1","lag_30","y_next"]).reset_index(drop=True)

    # urutan fitur persis seperti model.feature_names_in_
    feature_cols = ["lag_1","lag_30","month","dayofweek","time_index",
                    "is_end_of_year","is_new_year","is_eid_window"]
    return df, feature_cols



def _week_of_month_int(dt: pd.Timestamp) -> int:
    return int(((dt.day - 1) // 7) + 1)

def _basic_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"n": 0, "min": None, "max": None, "mean": None, "vol_pct": None}
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "vol_pct": float(np.std(arr, ddof=0) / np.mean(arr) * 100.0) if np.mean(arr) != 0 else None
    }


def _slugify_city(name: str) -> str:
    s = (name or "").strip().lower()
    # anggap spasi atau '-' sama-sama dipetakan ke underscore
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^\w_]", "", s)
    return s

def _load_eval_metrics():
    """Baca Excel evaluasi (Kota, MAE, RMSE, MAPE, R2) -> dict per slug kota."""
    global _EVAL_CACHE
    if _EVAL_CACHE is not None:
        return _EVAL_CACHE

    p = EVAL_XLSX
    if not p.exists():
        _EVAL_CACHE = {}
        return _EVAL_CACHE

    df = pd.read_excel(p)

    # Toleransi variasi nama kolom
    col_map = {c.lower(): c for c in df.columns}
    col_kota  = col_map.get("kota") or col_map.get("city") or col_map.get("kab/kota") or col_map.get("kabupaten/kota")
    col_mae   = col_map.get("mae")
    col_rmse  = col_map.get("rmse")
    col_mape  = col_map.get("mape")
    col_r2    = col_map.get("r2") or col_map.get("r²") or col_map.get("r2 score")

    if not col_kota:
        # minimal perlu identitas kota
        _EVAL_CACHE = {}
        return _EVAL_CACHE

    data = {}
    for _, row in df.iterrows():
        label = str(row[col_kota]).strip()
        if not label:
            continue
        slug = _slugify_city(label)

        mae  = float(row[col_mae])  if col_mae  in df.columns and not pd.isna(row[col_mae])  else None
        rmse = float(row[col_rmse]) if col_rmse in df.columns and not pd.isna(row[col_rmse]) else None
        mape = float(row[col_mape]) if col_mape in df.columns and not pd.isna(row[col_mape]) else None
        r2   = float(row[col_r2])   if col_r2   in df.columns and not pd.isna(row[col_r2])   else None

        mse = None
        if rmse is not None and not math.isnan(rmse):
            mse = rmse * rmse

        data[slug] = {
            "label": label,
            "mae": mae,
            "mape": mape,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
        }

    _EVAL_CACHE = data
    return _EVAL_CACHE
# =======================
# LOAD DATA SEKALI
# =======================
print(">> Starting app.py")
print(">> Expecting Excel at:", DATA_PATH)
WIDE, LONG_DF = build_wide_long(str(DATA_PATH), FILL_METHOD)
print(">> Dataset loaded:", len(LONG_DF), "rows")
ENTITIES = set(LONG_DF["entity"].unique())
LAST_ACTUAL = _compute_last_actual_dates(str(DATA_PATH))
print(">> LAST_ACTUAL computed (sample):", list(LAST_ACTUAL.items())[:3])

# =======================
# LOAD JSON PENDUKUNG
# =======================
def _safe_load_json(p: Path, default: dict) -> dict:
    if not p.exists():
        warnings.warn(f"File tidak ditemukan: {p}. Lanjut dengan default kosong.")
        return default
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        warnings.warn(f"Gagal load {p}: {e}. Lanjut dengan default.")
        return default

ENTITY_TO_PROVINCE = _safe_load_json(ENTITY_PROV_PATH, {})
CITY_COORDS        = _safe_load_json(CITY_COORDS_PATH, {})

PROVINCE_TO_ISLAND = {k.upper(): v for k, v in PROVINCE_TO_ISLAND.items()}

missing_etp = sorted(e for e in ENTITIES if e not in ENTITY_TO_PROVINCE)
if missing_etp:
    print(f"⚠️ Belum terpetakan ke provinsi ({len(missing_etp)}):", ", ".join(missing_etp[:15]), "...")

missing_coords = sorted(e for e in ENTITIES if e not in CITY_COORDS)
if missing_coords:
    print(f"⚠️ Belum ada koordinat kota ({len(missing_coords)}):", ", ".join(missing_coords[:15]), "...")

# =======================
# MODEL CACHE
# =======================
_MODEL_CACHE = {}  # entity -> { model, feature_cols, config, metrics, smear, mode, transform, alpha }

def _normalize_to_slug(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r'^(kota administrasi|kota|kab\.?|kabupaten)\s+', '', text)
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')

def _slug_to_entity(s: str) -> str:
    s0 = (s or "").strip()
    s_lower = s0.lower()
    if s_lower in ENTITIES:
        return s_lower

    slug = _normalize_to_slug(s0)
    if slug in CITY_SLUG_TO_ENTITY:
        ent = CITY_SLUG_TO_ENTITY[slug]
        if ent in ENTITIES: return ent

    cand = f"kota_{slug.replace('-', '_')}"
    if cand in ENTITIES: return cand
    cand = f"kab._{slug.replace('-', '_')}"
    if cand in ENTITIES: return cand

    for ent, meta in CITY_COORDS.items():
        label = (meta.get("label") or ent)
        if _normalize_to_slug(label) == slug and ent in ENTITIES:
            return ent

    raise ValueError(f"Mapping untuk '{s}' tidak ditemukan. Periksa city_coords.json / nama kolom Excel.")

def _predict_with_safe_names(model, X_df: pd.DataFrame):
    """
    Selaraskan X_df ke fitur yang dipakai saat fit:
    - Tambah kolom yang hilang (isi 0.0)
    - Urutkan persis sesuai model.feature_names_in_
    - Drop kolom ekstra
    """
    feat_in = getattr(model, "feature_names_in_", None)
    if feat_in is None:
        # Estimator lama -> pakai urutan array
        return model.predict(X_df.to_numpy())

    # Tambah kolom yang hilang
    for c in feat_in:
        if c not in X_df.columns:
            X_df[c] = 0.0

    # Reindex persis urutan saat fit
    X_aligned = X_df.reindex(columns=list(feat_in), fill_value=0.0)
    return model.predict(X_aligned)

def _apply_rolling(series_list, window=30, minp=None):
    if not series_list or window <= 1:
        return series_list

    # samakan dengan training: min_periods=1 (tidak buang data awal)
    if minp is None:
        minp = 1

    df = pd.DataFrame(series_list)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # rolling non-centered, anti-leakage seperti training
    df["value"] = df["value"].rolling(window, min_periods=minp).mean()

    # JANGAN drop awal; biarkan nilai awal sudah tersmooth (tidak NaN karena min_periods=1)
    out = [{"date": d.date().isoformat(), "value": float(v)}
           for d, v in zip(df["date"], df["value"]) if pd.notna(v)]
    return out

def _load_model_for_entity(entity: str):
    if entity in _MODEL_CACHE:
        return _MODEL_CACHE[entity]

    # 1) Cari pack terlebih dulu
    files = sorted(MODELS_DIR.glob(f"{entity}*best_pack.joblib"))
    # 2) Fallback: file joblib apa pun yang cocok
    if not files:
        files = sorted(MODELS_DIR.glob(f"{entity}*.joblib"))
    if not files:
        raise FileNotFoundError(f"Model file untuk '{entity}' tidak ditemukan di {MODELS_DIR}")

    pack = joblib.load(files[0])

    # --- Normalisasi isi pack ---
    if isinstance(pack, dict) and "model" in pack and "feature_cols" in pack:
        model        = pack["model"]
        feature_cols = list(pack["feature_cols"])
        best_cfg     = pack.get("best_config", {"mode": "level", "transform": "none", "train_until": None, "alpha_blend": 1.0})
        metrics      = pack.get("metrics", {})
    else:
        model        = pack
        feature_cols = list(getattr(model, "feature_names_in_", []))
        best_cfg     = {"mode": "level", "transform": "none", "train_until": None, "alpha_blend": 1.0}
        metrics      = {}
        print(f">> WARNING: '{files[0].name}' bukan pack. Memakai raw estimator.")

    alpha      = float(best_cfg.get("alpha_blend", 1.0))
    mode       = best_cfg.get("mode", "level")
    transform  = best_cfg.get("transform", "none")

    # === Reconstruct smearing (kalau level+log) ===
    dfe = LONG_DF.loc[LONG_DF["entity"] == entity, ["entity","date","value"]].copy()
    df_feat, _ = make_features_entity(dfe, horizon=1)

    train_until_raw = best_cfg.get("train_until")
    train_until = pd.to_datetime(train_until_raw) if train_until_raw else df_feat["date"].max()
    df_train = df_feat.loc[df_feat["date"] <= train_until].copy()

    use_log = (mode == "level" and transform == "log")
    smear = 1.0  # default aman

    if use_log:
        if df_train.empty:
            # fallback: pakai 80% awal bila cukup panjang; kalau tidak, smear=1.0
            n = len(df_feat)
            if n >= 20:
                cut = max(10, int(0.8 * n))
                df_train = df_feat.iloc[:cut].copy()
                warnings.warn(f"[{entity}] TRAIN subset kosong (train_until={train_until_raw}); "
                              f"fallback {cut}/{n} baris pertama untuk smearing.")
            else:
                warnings.warn(f"[{entity}] Data terlalu pendek ({n}); set smear=1.0.")

        if not df_train.empty:
            # pilih kolom aman
            if feature_cols:
                cols = [c for c in feature_cols if c in df_train.columns]
            else:
                cols = list(getattr(model, "feature_names_in_", []))
                cols = [c for c in cols if c in df_train.columns]
            if not cols:
                cols = [c for c in df_train.columns if c not in ("entity","date","value","y_next","y_diff")]

            X_tr = df_train[cols].dropna(axis=0)
            if X_tr.empty:
                warnings.warn(f"[{entity}] X_tr kosong setelah dropna; smear=1.0.")
                smear = 1.0
            else:
                y_train = df_train.loc[X_tr.index, "y_next"].values
                yhat_tr = _predict_with_safe_names(model, X_tr)
                resid_log = np.log(y_train) - yhat_tr
                smear = float(np.mean(np.exp(resid_log)))

    _MODEL_CACHE[entity] = {
        "model": model,
        "feature_cols": feature_cols,
        "config": best_cfg,
        "metrics": metrics,
        "smear": smear,
        "mode": mode,
        "transform": transform,
        "alpha": alpha,
    }
    print(f">> Loaded model for {entity} | smear={smear:.6f} | mode={mode}/{transform} | file={files[0].name}")
    return _MODEL_CACHE[entity]

def _one_step_predict_series(entity: str) -> pd.DataFrame:
    """
    Prediksi one-step-ahead (historis) dengan feature builder yang IDENTIK dgn training.
    Return: DataFrame [date (t+1), pred(level)].
    """
    b = _load_model_for_entity(entity)
    model     = b["model"]
    mode      = b["mode"]
    transform = b["transform"]
    smear     = b["smear"]
    alpha     = b.get("alpha", 1.0)

    dfe = LONG_DF.loc[LONG_DF["entity"] == entity, ["entity","date","value"]].copy()
    dfe["date"] = pd.to_datetime(dfe["date"]).dt.normalize()

    df_feat = _build_features_training_like(dfe, add_targets=True)

    # Ambil semua kolom fitur (tanpa entity/date/value/targets)
    X = df_feat.drop(columns=["entity","date","value","y_next","y_diff"], errors="ignore")
    # Filter baris valid (tanpa NaN)
    mask_valid = ~X.isna().any(axis=1)
    X = X.loc[mask_valid]
    dfv = df_feat.loc[mask_valid]

    if X.empty:
        raise RuntimeError("Fitur historis kosong setelah filter NaN (histori kurang).")

    yhat_tr = _predict_with_safe_names(model, X)

    # Kembalikan ke level
    if mode == "level":
        yhat_level = np.exp(yhat_tr) * smear if transform == "log" else yhat_tr
    else:
        base_val   = dfv["value"].to_numpy()
        yhat_level = base_val + (alpha * yhat_tr)

    dates = (dfv["date"] + pd.Timedelta(days=1)).dt.normalize()
    out = pd.DataFrame({"date": dates, "pred": yhat_level})
    return out.sort_values("date").reset_index(drop=True)


def make_features_for_next(dfe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Generator fitur untuk prediksi recursive H+1 yang
    *persis* sama dengan fitur training 'lag30':
      ['lag_1','lag_30','month','dayofweek','time_index',
       'is_end_of_year','is_new_year','is_eid_window']
    """
    df = dfe.sort_values("date").copy()

    # Lags yang dipakai saat training
    df["lag_1"]  = df["value"].shift(1)
    df["lag_30"] = df["value"].shift(30)

    # Kalender/flags (jangan buat month/dayofweek dua kali)
    flags = make_flags(df["date"])  # berisi month, dayofweek, time_index, is_* dll
    df = pd.concat([df.set_index("date"), flags], axis=1).reset_index()

    # Kolom fitur final & urutan default (akan tetap disejajarkan lagi di _predict_with_safe_names)
    feature_cols = [
        "lag_1","lag_30","month","dayofweek","time_index",
        "is_end_of_year","is_new_year","is_eid_window"
    ]
    return df, feature_cols


def _recursive_predict(entity: str, days: int):
    if days <= 0:
        return []

    b = _load_model_for_entity(entity)
    model     = b["model"]
    mode      = b["mode"]
    transform = b["transform"]
    smear     = b["smear"]
    alpha     = b.get("alpha", 1.0)

    dfe = LONG_DF.loc[LONG_DF["entity"] == entity, ["entity","date","value"]].copy()
    dfe["date"] = pd.to_datetime(dfe["date"]).dt.normalize()
    dfe = dfe.sort_values("date")

    last_actual_dt = LAST_ACTUAL.get(entity, dfe["date"].max().normalize())
    dfe = dfe[dfe["date"] <= last_actual_dt].copy()
    if dfe.empty:
        raise RuntimeError(f"Data kosong untuk {entity} sebelum {last_actual_dt}")

    preds = []
    for _ in range(days):
        # Build fitur training-like sampai tanggal TERAKHIR aktual/prediksi saat ini
        df_feat = _build_features_training_like(dfe, add_targets=False)

        # Ambil baris TERAKHIR sebagai fitur untuk memprediksi t+1
        # (di training, baris t dipakai untuk menebak y_{t+1})
        X_last = df_feat.drop(columns=["entity","date","value"], errors="ignore").iloc[[-1]]
        if X_last.isna().any(axis=None):
            raise RuntimeError("Fitur terakhir masih NaN (histori kurang).")

        yhat_tr = float(_predict_with_safe_names(model, X_last)[0])
        last_date = dfe["date"].max().normalize()
        last_val  = float(dfe["value"].iloc[-1])

        if mode == "level":
            y_next = float(np.exp(yhat_tr) * smear) if transform == "log" else float(yhat_tr)
        else:
            delta  = float(alpha * yhat_tr)
            delta  = float(np.clip(delta, -500, 500))
            y_next = float(last_val + delta)

        next_date = last_date + pd.Timedelta(days=1)
        if preds and pd.to_datetime(preds[-1]["date"]) >= next_date:
            next_date = pd.to_datetime(preds[-1]["date"]) + pd.Timedelta(days=1)

        preds.append({"date": next_date.date().isoformat(), "pred": round(y_next, 4)})

        # Append prediksi supaya step berikutnya bisa pakai sebagai lag_1, lag_30, dll.
        dfe = pd.concat(
            [dfe, pd.DataFrame([{"entity": entity, "date": next_date, "value": y_next}])],
            ignore_index=True
        )

    return preds


_TOPN_CACHE = {"mtime": None, "df": None}
def _parse_id_number(x):
    """
    Terima string angka gaya ID ('12,392' atau '19.400' atau '12 650'),
    buang pemisah ribuan, lalu parse ke float.
    """
    if pd.isna(x): return np.nan
    s = str(x).strip()
    # buang semua non-digit/non-minus/non-dot
    # (koma/dot/spasi sebagai pemisah ribuan akan hilang)
    import re
    s = re.sub(r"[^\d\.\-]", "", s)
    # kalau masih ada lebih dari satu titik (kasus ribuan '.'), buang semua titik:
    if s.count(".") > 1:
        s = s.replace(".", "")
    # jika formatnya cuma ribuan '19.400' (satu titik), ini juga harus jadi 19400
    # heuristik: kalau setelah titik sisa 3 digit → buang titik
    if s.count(".") == 1:
        left, right = s.split(".")
        if len(right) == 3 and right.isdigit():
            s = left + right
    try:
        return float(s)
    except:
        return np.nan
def _load_topn_df():
    """
    Baca hanya Sheet1 dari Excel topn_per_tahun_per_kota.xlsx
    Kolom minimal: year, province (boleh kosong), city, avg, min, max, n
    """
    if not TOPN_XLSX.exists():
        raise FileNotFoundError(f"File TopN tidak ditemukan: {TOPN_XLSX}")

    mtime = TOPN_XLSX.stat().st_mtime
    if _TOPN_CACHE["df"] is not None and _TOPN_CACHE["mtime"] == mtime:
        return _TOPN_CACHE["df"]

    # Pakai sheet pertama saja
    df = pd.read_excel(TOPN_XLSX, sheet_name=0)

    # Normalisasi nama kolom (lowercase, strip)
    colmap = {c.lower().strip(): c for c in df.columns}
    col_year = colmap.get("year") or colmap.get("tahun")
    col_city = colmap.get("city") or colmap.get("city_label") or colmap.get("kota") or colmap.get("kab/kota") or colmap.get("kabupaten/kota")
    col_prov = colmap.get("province") or colmap.get("provinsi")
    col_avg  = colmap.get("avg") or colmap.get("rata") or colmap.get("rata2") or colmap.get("mean")
    col_min  = colmap.get("min")
    col_max  = colmap.get("max")
    col_n    = colmap.get("n")   or colmap.get("count") or colmap.get("jumlah")

    required = [col_year, col_city, col_avg]
    if any(c is None for c in required):
        raise ValueError("Sheet1 wajib punya kolom: year, city, avg (province/min/max/n opsional).")

    out = pd.DataFrame({
        "year":     pd.to_numeric(df[col_year], errors="coerce"),
        "city":     df[col_city].astype(str),
        "province": df[col_prov].astype(str) if col_prov else "",
        "avg":      df[col_avg].apply(_parse_id_number),
        "min":      df[col_min].apply(_parse_id_number) if col_min else np.nan,
        "max":      df[col_max].apply(_parse_id_number) if col_max else np.nan,
        "n":        pd.to_numeric(df[col_n], errors="coerce") if col_n else np.nan,
    })

    out = out.dropna(subset=["year", "city", "avg"]).copy()
    out["year"] = out["year"].astype(int)
    _TOPN_CACHE.update({"mtime": mtime, "df": out})
    return out
# =======================
# ROUTES
# =======================
MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"Mei",6:"Jun",7:"Jul",8:"Agu",9:"Sep",10:"Okt",11:"Nov",12:"Des"}
from flask import make_response
INDEX_HTML = "index.html"
@app.route("/")
def index():
    resp = make_response(send_from_directory("static", INDEX_HTML))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp
@app.route("/api/islands")
def api_islands():
    return jsonify(["Semua Pulau","Jawa","Sumatra","Kalimantan","Sulawesi","Bali–NT","Maluku","Papua"])

@app.route("/api/cities_full")
def api_cities_full():
    out = []
    source = CITY_COORDS if CITY_COORDS else {e: {} for e in ENTITIES}
    for ent in sorted(source.keys()):
        meta  = CITY_COORDS.get(ent, {})
        label = meta.get("label") or ent.replace("_", " ").title().replace("Kab. ", "Kabupaten ")
        slug  = _normalize_to_slug(label)
        out.append({"entity": ent, "slug": slug, "label": label})
    out.sort(key=lambda x: x["label"].lower())
    return jsonify(out)

@app.route("/api/cities")
def api_cities():
    if CITY_COORDS:
        labels = []
        for ent, meta in CITY_COORDS.items():
            label = meta.get("label") or ent.replace("_", " ").title()
            labels.append(label)
        labels = sorted(set(labels), key=lambda x: x.lower())
        return jsonify(labels)
    labels = []
    for ent in ENTITIES:
        labels.append(ent.replace("_", " ").title())
    labels = sorted(set(labels), key=lambda x: x.lower())
    return jsonify(labels)

@app.route("/api/history")
def api_history():
    slug = request.args.get("city", "").strip().lower()
    year = request.args.get("year", "").strip()
    if not slug:
        return jsonify({"error": "param ?city= wajib"}), 400
    try:
        entity = _slug_to_entity(slug)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    df = LONG_DF.loc[LONG_DF["entity"] == entity, ["date","value"]].copy()
    if year:
        try:
            y = int(year)
            df = df[df["date"].dt.year == y]
        except:
            pass
    data = [{"date": d.date().isoformat(), "value": float(v)} for d, v in zip(df["date"], df["value"])]
    return jsonify({"city": slug, "entity": entity, "points": data})

@app.route("/api/price_at")
def api_price_at():
    slug = request.args.get("city", "").strip().lower()
    date_str = request.args.get("date", "").strip()
    if not slug or not date_str:
        return jsonify({"error": "param ?city= & ?date=YYYY-MM-DD wajib"}), 400
    try:
        entity = _slug_to_entity(slug)
        target_date = pd.to_datetime(date_str).normalize()
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    dfe = LONG_DF.loc[LONG_DF["entity"] == entity, ["date","value"]].copy()
    if dfe.empty:
        return jsonify({"error": "data kota kosong"}), 400

    last_dt = LAST_ACTUAL.get(entity, dfe["date"].max().normalize())
    row = dfe.loc[dfe["date"] == target_date]
    if not row.empty:
        val = float(row["value"].iloc[0])
        return jsonify({"city": slug, "entity": entity, "status":"actual", "date": target_date.date().isoformat(), "value": val})

    if target_date > last_dt:
        try:
            days_ahead = int((target_date - last_dt).days)
            preds = _recursive_predict(entity, days=days_ahead)
            if not preds:
                return jsonify({"error": "gagal memprediksi"}), 500
            val = preds[-1]["pred"]
            return jsonify({
                "city": slug, "entity": entity, "status":"predicted",
                "from_last_actual": last_dt.date().isoformat(),
                "date": target_date.date().isoformat(), "steps": days_ahead, "value": val
            })
        except FileNotFoundError:
            return jsonify({"error": "model untuk kota ini belum tersedia"}), 404
        except Exception as e:
            return jsonify({"error": f"gagal memprediksi: {e}"}), 500

    return jsonify({"city": slug, "entity": entity, "status": "no_data",
                    "date": target_date.date().isoformat(),
                    "message": "tanggal tidak ada di dataset (bukan titik observasi)"})


@app.route("/api/choropleth")
def api_choropleth():
    island = request.args.get("island","Semua Pulau").strip()
    year   = request.args.get("year","").strip()
    month  = request.args.get("month","").strip()
    week   = request.args.get("week","").strip()

    if not year: return jsonify({"error":"param ?year= wajib"}), 400
    try:
        year  = int(year)
        month = int(month) if month else None
        week  = int(week)  if week  else None
    except:
        return jsonify({"error":"format year/month/week tidak valid"}), 400

    df = LONG_DF[["entity","date","value"]].copy()
    df["date"] = df["date"].dt.normalize()
    df["year"] = df["date"].dt.year
    df = df[df["year"] == year]
    if month:
        df["month"] = df["date"].dt.month
        df = df[df["month"] == month]
    if week and month:
        df["week_in_month"] = df["date"].apply(_week_of_month_int)
        df = df[df["week_in_month"] == week]

    if df.empty:
        return jsonify({"data": [], "buckets": None, "last_actual": None})

    df["province"] = df["entity"].map(ENTITY_TO_PROVINCE)
    df = df.dropna(subset=["province"]).copy()
    df["island"]   = df["province"].str.upper().map(PROVINCE_TO_ISLAND)

    if island and island != "Semua Pulau":
        df = df[df["island"] == island]
    if df.empty:
        return jsonify({"data": [], "buckets": None, "last_actual": None})

    grp = df.groupby("province")["value"].mean().reset_index()
    vals = grp["value"].values
    q1, q2 = np.quantile(vals, [1/3, 2/3])

    def cat(v):
        if v <= q1: return "low"
        if v <= q2: return "mid"
        return "high"

    data = [{"province": r.province, "value": float(r.value), "category": cat(r.value)}
            for r in grp.itertuples(index=False)]
    last_actual = str(LONG_DF["date"].max().date())
    return jsonify({"last_actual": last_actual, "buckets": {"low": float(q1), "mid": float(q2)}, "data": data})

@app.route("/api/city_points")
def api_city_points():
    island = (request.args.get("island") or "Semua Pulau").strip()
    year_s = (request.args.get("year") or "").strip()
    month_s = (request.args.get("month") or "").strip()
    week_s  = (request.args.get("week") or "").strip()
    band_s  = (request.args.get("band") or "0.05").strip()

    if not year_s:
        return jsonify({"error": "param ?year= wajib"}), 400
    try:
        year  = int(year_s)
        month = int(month_s) if month_s else None
        week  = int(week_s)  if week_s  else None
        band  = float(band_s)
        if band < 0 or band > 0.5: band = 0.05
    except Exception:
        return jsonify({"error": "format query tidak valid"}), 400

    df = LONG_DF[["entity", "date", "value"]].copy()
    df["date"] = df["date"].dt.normalize()
    df["year"] = df["date"].dt.year
    df = df[df["year"] == year]
    if month:
        df["month"] = df["date"].dt.month
        df = df[df["month"] == month]
    if week and month:
        df["week_in_month"] = df["date"].apply(_week_of_month_int)
        df = df[df["week_in_month"] == week]
    if df.empty:
        return jsonify({"last_actual": None, "mean_ref": None, "band_pct": band, "points": []})

    df["province"] = df["entity"].map(ENTITY_TO_PROVINCE)
    df = df.dropna(subset=["province"]).copy()
    df["island"] = df["province"].str.upper().map(PROVINCE_TO_ISLAND)
    if island and island != "Semua Pulau":
        df = df[df["island"] == island]
    if df.empty:
        return jsonify({"last_actual": None, "mean_ref": None, "band_pct": band, "points": []})

    agg = (df.groupby(["entity", "province", "island"])["value"].mean().reset_index()
             .rename(columns={"value": "avg_value"}))
    mean_ref = float(agg["avg_value"].mean())
    lo = mean_ref * (1 - band); hi = mean_ref * (1 + band)

    def cat(val: float) -> str:
        if val <= lo: return "low"
        if val >= hi: return "high"
        return "mid"

    points = []
    for r in agg.itertuples(index=False):
        coords = CITY_COORDS.get(r.entity)
        if not coords: continue
        lat = coords.get("lat"); lng = coords.get("lng")
        if lat is None or lng is None: continue
        points.append({
            "entity": r.entity,
            "label": coords.get("label") or r.entity.replace("_", " ").title(),
            "province": r.province, "island": r.island,
            "value": float(r.avg_value),
            "lat": float(lat), "lng": float(lng),
            "category": cat(float(r.avg_value)),
        })
    last_actual = str(LONG_DF["date"].max().date())
    return jsonify({"last_actual": last_actual, "mean_ref": mean_ref, "band_pct": band, "points": points})

@app.route("/api/trend")
def api_trend():
    slug = request.args.get("city", "").strip().lower()
    year_str = request.args.get("year", "").strip()
    month_str = request.args.get("month", "").strip()
    week_str = request.args.get("week", "").strip()

    if not slug or not year_str:
        return jsonify({"error": "param ?city= dan ?year= wajib"}), 400
    try:
        year = int(year_str)
    except:
        return jsonify({"error": "param ?year= harus angka"}), 400
    try:
        entity = _slug_to_entity(slug)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    df = LONG_DF.loc[LONG_DF["entity"] == entity, ["date", "value"]].copy()
    if df.empty:
        return jsonify({"error": f"data untuk '{entity}' kosong"}), 404

    df = df[df["date"].dt.year == year]
    if df.empty:
        return jsonify({"city": slug, "entity": entity, "year": year,
                        "granularity": "yearly", "series": [], "stats": _basic_stats([])})

    month = None
    if month_str:
        try:
            m = int(month_str)
            if 1 <= m <= 12: month = m
            else: return jsonify({"error": "param ?month= harus 1..12"}), 400
        except:
            return jsonify({"error": "param ?month= tidak valid"}), 400

    week = None
    if week_str:
        try:
            w = int(week_str)
            if 1 <= w <= 5: week = w
            else: return jsonify({"error": "param ?week= harus 1..5"}), 400
        except:
            return jsonify({"error": "param ?week= tidak valid"}), 400

    if month is None:
        df["mon"] = df["date"].dt.month
        grp = df.groupby("mon")["value"].mean().reset_index().sort_values("mon")
        series = [{"label": MONTH_NAMES.get(int(r.mon), str(int(r.mon))), "value": float(r.value)} for r in grp.itertuples(index=False)]
        stats = _basic_stats(df["value"].values)
        return jsonify({"city": slug, "entity": entity, "year": year, "month": None, "week": None,
                        "granularity": "yearly", "series": series, "stats": stats})

    dfm = df[df["date"].dt.month == month].copy()
    if dfm.empty:
        return jsonify({"city": slug, "entity": entity, "year": year, "month": month,
                        "granularity": "monthly", "series": [], "stats": _basic_stats([])})

    if week is None:
        dfm["week_in_month"] = dfm["date"].apply(_week_of_month_int)
        grp = dfm.groupby("week_in_month")["value"].mean().reset_index().sort_values("week_in_month")
        series = [{"label": f"Minggu {int(r.week_in_month)}", "value": float(r.value)} for r in grp.itertuples(index=False)]
        stats = _basic_stats(dfm["value"].values)
        return jsonify({"city": slug, "entity": entity, "year": year, "month": month, "week": None,
                        "granularity": "monthly", "series": series, "stats": stats})

    dfm["week_in_month"] = dfm["date"].apply(_week_of_month_int)
    dfd = dfm[dfm["week_in_month"] == week].copy().sort_values("date")
    series = [{"label": d.date().isoformat(), "value": float(v)} for d, v in zip(dfd["date"], dfd["value"])]
    stats = _basic_stats(dfd["value"].values)
    return jsonify({"city": slug, "entity": entity, "year": year, "month": month, "week": week,
                    "granularity": "daily", "series": series, "stats": stats})

@app.route("/api/metrics")
def api_metrics():
    slug = request.args.get("city", "").strip().lower()
    if not slug:
        return jsonify({"error": "param ?city= wajib"}), 400
    try:
        entity = _slug_to_entity(slug)
        pack = _load_model_for_entity(entity)
    except FileNotFoundError:
        return jsonify({"error": "model untuk kota ini belum tersedia"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({"city": slug, "entity": entity, "best_config": pack["config"], "metrics": pack["metrics"]})

@app.route("/api/cities_search")
def api_cities_search():
    q = (request.args.get("q") or "").strip().lower()
    try:
        limit = int((request.args.get("limit") or "15").strip())
    except:
        limit = 15

    source = CITY_COORDS if CITY_COORDS else {e: {} for e in ENTITIES}
    items = []
    for ent in source.keys():
        meta  = CITY_COORDS.get(ent, {})
        label = meta.get("label") or ent.replace("_", " ").title().replace("Kab. ", "Kabupaten ")
        slug  = _normalize_to_slug(label)
        items.append({"entity": ent, "slug": slug, "label": label})

    if not q:
        return jsonify(sorted(items, key=lambda x: x["label"].lower())[:limit])

    def rank(it):
        lab = it["label"].lower()
        if lab.startswith(q): return (0, lab)   # depan cocok dulu
        if q in lab:         return (1, lab)   # lalu contains
        return (2, lab)

    filtered = [it for it in items if it["label"].lower().startswith(q) or q in it["label"].lower()]
    filtered.sort(key=rank)
    return jsonify(filtered[:limit])


@app.route("/api/predict_range")
def api_predict_range():
    """
    Query:
      - city, start, end (wajib)
      - hide_actual=1|0        : sembunyikan garis aktual (opsional)
      - future_only=1|0        : jika 1, TIDAK kirim one-step historis (opsional)
      - naive_fallback=1|0     : bila model tak ada, isi flat last-actual (opsional)
    """
    slug = request.args.get("city", "").strip().lower()
    start_str = request.args.get("start", "").strip()
    end_str   = request.args.get("end", "").strip()

    hide_actual    = request.args.get("hide_actual", "0").lower() in ("1","true","yes")
    future_only    = request.args.get("future_only", "0").lower() in ("1","true","yes")
    naive_fallback = request.args.get("naive_fallback", "0").lower() in ("1","true","yes")

    if not slug or not start_str or not end_str:
        return jsonify({"error": "param ?city=, ?start=, ?end= wajib"}), 400

    try:
        entity   = _slug_to_entity(slug)
        start_dt = pd.to_datetime(start_str).normalize()
        end_dt   = pd.to_datetime(end_str).normalize()
        if end_dt < start_dt:
            return jsonify({"error": "end harus >= start"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # data mentah untuk entity
    dfe = LONG_DF.loc[LONG_DF["entity"] == entity, ["date","value"]].copy()
    if dfe.empty:
        return jsonify({"error": f"data '{entity}' kosong"}), 404
    dfe["date"] = dfe["date"].dt.normalize()

    # cutoff aktual terakhir (mis. 2025-07-01 / 2025-07-07, dll)
    last_actual_dt = LAST_ACTUAL.get(entity, dfe["date"].max().normalize())

    print(f"DEBUG /api/predict_range entity={entity} start={start_dt.date()} end={end_dt.date()} last_actual={last_actual_dt.date()} future_only={future_only} naive_fallback={naive_fallback}")

    # ====== AKTUAL (hanya sampai cutoff) ======
    actual_series = []
    if not hide_actual:
        hist_end_for_actual = min(end_dt, last_actual_dt)
        mask_hist = (dfe["date"] >= start_dt) & (dfe["date"] <= hist_end_for_actual)
        for d, v in zip(dfe.loc[mask_hist, "date"], dfe.loc[mask_hist, "value"]):
            actual_series.append({"date": d.date().isoformat(), "value": float(v)})

    predicted_series = []

    # ====== ONE-STEP (prediksi di area historis) ======
    # Catatan: hanya sampai cutoff (<= last_actual_dt), jadi tidak overlap dengan future.
    if not future_only and start_dt <= last_actual_dt:
        try:
            one_step = _one_step_predict_series(entity)  # kolom: date, pred
            mask_hist_pred = (one_step["date"] >= start_dt) & (one_step["date"] <= min(end_dt, last_actual_dt))
            for d, v in zip(one_step.loc[mask_hist_pred, "date"], one_step.loc[mask_hist_pred, "pred"]):
                predicted_series.append({"date": d.date().isoformat(), "value": float(v), "pred": float(v)})
            print(f"DEBUG one-step added: {mask_hist_pred.sum()}")
        except FileNotFoundError:
            print("DEBUG one-step: no model file")
        except Exception as e:
            print(f"DEBUG one-step error: {e}")
            return jsonify({"error": f"gagal memprediksi (historis): {e}"}), 500

    # ====== FUTURE (prediksi > cutoff, recursive) ======
    if end_dt > last_actual_dt:
        # mulai prediksi dari H+1 setelah cutoff
        anchor = last_actual_dt
        days_need = int((end_dt - anchor).days)  # jumlah hari yang perlu diprediksi
        print(f"DEBUG future anchor={anchor.date()} days_need={days_need}")
        try:
            preds = _recursive_predict(entity, days=days_need)  # list {date:'YYYY-MM-DD', pred:float}
            cnt = 0
            for p in preds:
                p_dt = pd.to_datetime(p["date"]).normalize()
                if start_dt <= p_dt <= end_dt:
                    val = float(p["pred"])
                    predicted_series.append({"date": p_dt.date().isoformat(), "value": val, "pred": val})
                    cnt += 1
            print(f"DEBUG future added: {cnt}")
        except FileNotFoundError:
            print("DEBUG future: no model file")
            if naive_fallback:
                last_val = float(dfe.loc[dfe["date"] == last_actual_dt, "value"].iloc[0])
                cur = max(last_actual_dt + pd.Timedelta(days=1), start_dt)
                naive_cnt = 0
                while cur <= end_dt:
                    predicted_series.append({"date": cur.date().isoformat(), "value": last_val, "pred": last_val})
                    cur += pd.Timedelta(days=1)
                    naive_cnt += 1
                print(f"DEBUG naive fallback added: {naive_cnt}")
        except Exception as e:
            print(f"DEBUG future error: {e}")
            return jsonify({"error": f"gagal memprediksi (future): {e}"}), 500

    # ====== Rapikan (sort & dedup di predicted) ======
    predicted_series.sort(key=lambda x: x["date"])
    seen = set()
    unique_pred = []
    for p in predicted_series:
        if p["date"] in seen:
            continue
        seen.add(p["date"])
        unique_pred.append(p)
    predicted_series = unique_pred

    viz_roll = int((request.args.get("viz_roll") or "0").strip() or 0)  # default 0 (tanpa smoothing)

    if viz_roll > 1:
        actual_series    = _apply_rolling(actual_series, window=viz_roll)
        predicted_series = _apply_rolling(predicted_series, window=viz_roll)

    print(f"DEBUG result actual={len(actual_series)} predicted={len(predicted_series)}")
    summary_pred = _series_stats(predicted_series)
    summary_act  = _series_stats(actual_series)

    return jsonify({
        "city": slug,
        "entity": entity,
        "range": {"start": start_dt.date().isoformat(), "end": end_dt.date().isoformat()},
        "last_actual": last_actual_dt.date().isoformat(),
        "actual": actual_series,
        "predicted": predicted_series,
        "summary": {
        "actual": summary_act,
        "predicted": summary_pred
    }
    })

# === FEATURE BUILDER (identik dgn training: lag_1, lag_30 + kalender flags) ===
def _build_features_training_like(dfe: pd.DataFrame, add_targets: bool = False) -> pd.DataFrame:
    df = dfe.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date")

    # seri level
    y = pd.to_numeric(df["value"], errors="coerce").astype(float)

    # siapkan frame fitur
    X = pd.DataFrame(index=y.index)

    # selalu sediakan lag_1 (anchor)
    X["lag_1"] = y.shift(1)

    # BACA kebutuhan fitur langsung dari model
    needed = set(getattr(_load_model_for_entity(dfe["entity"].iloc[0])["model"], "feature_names_in_", []))

    # lags tambahan (hanya bila diminta model)
    for L in (7, 14, 30):
        col = f"lag_{L}"
        if col in needed:
            X[col] = y.shift(L)

    # rolling mean (anti-leakage, shift(1))
    for W in (7, 30):
        col = f"rollmean_{W}"
        if col in needed:
            X[col] = y.shift(1).rolling(W, min_periods=max(1, W//2)).mean()

    # flags kalender, persis spt training
    flags = make_flags(X.index)
    for c in ["month","dayofweek","time_index","is_end_of_year","is_new_year","is_eid_window"]:
        if c in needed:
            X[c] = flags[c]

    # gabung & bersihkan
    out = X.join(y.rename("value"))
    if add_targets:
        out["y_next"] = out["value"].shift(-1)
        out["y_diff"] = out["y_next"] - out["value"]

    out = out.reset_index().rename(columns={"index":"date"})
    # drop baris yang belum punya semua fitur yang dibutuhkan model
    must_have = [c for c in needed if c in out.columns]
    out = out.dropna(subset=must_have).reset_index(drop=True)
    return out


@app.route("/api/reload", methods=["POST"])
def api_reload():
    token = request.args.get("token", "")
    if token != RELOAD_TOKEN:
        abort(403)
    global WIDE, LONG_DF, ENTITIES, _MODEL_CACHE, ENTITY_TO_PROVINCE, CITY_COORDS, LAST_ACTUAL
    WIDE, LONG_DF = build_wide_long(str(DATA_PATH), FILL_METHOD)
    ENTITIES = set(LONG_DF["entity"].unique())
    _MODEL_CACHE.clear()
    ENTITY_TO_PROVINCE = _safe_load_json(ENTITY_PROV_PATH, {})
    CITY_COORDS        = _safe_load_json(CITY_COORDS_PATH, {})
    LAST_ACTUAL = _compute_last_actual_dates(str(DATA_PATH))
    return jsonify({"ok": True, "entities": sorted(list(ENTITIES)), "rows": int(len(LONG_DF))})

@app.route("/api/trend_compare")
def api_trend_compare():
    # ?city_a=...&city_b=...&year=2024[&month=&week=]
    a = (request.args.get("city_a") or "").strip().lower()
    b = (request.args.get("city_b") or "").strip().lower()
    year = request.args.get("year","").strip()
    month = request.args.get("month","").strip()
    week = request.args.get("week","").strip()
    if not a or not b or not year: return jsonify({"error":"param city_a, city_b, year wajib"}), 400

    def call(city):
        with app.test_request_context(
            f"/api/trend?city={city}&year={year}&month={month}&week={week}"
        ):
            return api_trend().json

    return jsonify({"a": call(a), "b": call(b)})

@app.route("/api/predict_compare")
def api_predict_compare():
    # ?city_a=...&city_b=...&start=YYYY-MM-DD&end=YYYY-MM-DD
    a = (request.args.get("city_a") or "").strip().lower()
    b = (request.args.get("city_b") or "").strip().lower()
    start = request.args.get("start","").strip()
    end   = request.args.get("end","").strip()
    if not a or not b or not start or not end: 
        return jsonify({"error":"param city_a, city_b, start, end wajib"}), 400

    def call(city):
        with app.test_request_context(
            f"/api/predict_range?city={city}&start={start}&end={end}&naive_fallback=1"
        ):
            return api_predict_range().json

    return jsonify({"a": call(a), "b": call(b)})

def _candidates_for_eval_slug(s: str):
    """Given 'banyuwangi', return plausible eval keys we may have stored from Excel."""
    base = _slugify_city(s)                 # 'banyuwangi' or already like 'kab_banyuwangi'
    # If user already passed kab_/kota_ keep it; else try common prefixes
    cands = {base}
    if not base.startswith(("kab_", "kota_", "kota_administrasi_")):
        cands.update({
            f"kab_{base}",
            f"kota_{base}",
            f"kota_administrasi_{base}",
        })
    # Also accept the dotted entity variant that can appear on the model side
    cands.add(f"kab._{base}")
    return list(cands)

@app.route("/api/eval_summary")
def api_eval_summary():
    """
    Ambil evaluasi dari Excel untuk satu kota (slug/label longgar):
    /api/eval_summary?city=<slug_or_label>
    """
    q = (request.args.get("city", "") or "").strip()
    if not q:
        return jsonify({"error": "parameter 'city' wajib"}), 400

    data = _load_eval_metrics()
    if not data:
        return jsonify({"error": "file evaluasi tidak ditemukan atau kosong"}), 404

    # 1) try several slug candidates (plain, kab_, kota_, kota_administrasi_, dotted)
    for key in _candidates_for_eval_slug(q):
        rec = data.get(key)
        if rec:
            return jsonify({"ok": True, "city": rec["label"], "slug": key, "metrics": rec})

    # 2) fallback: exact label match (case-insensitive, ignoring dots/hyphens)
    q_norm = re.sub(r"[.\-]", " ", q.lower()).strip()
    for slug, val in data.items():
        if re.sub(r"[.\-]", " ", val["label"].lower()).strip() == q_norm:
            return jsonify({"ok": True, "city": val["label"], "slug": slug, "metrics": val})

    # 3) soft fallback: contains match (useful when label is 'Kab. Banyuwangi')
    for slug, val in data.items():
        if q_norm in val["label"].lower():
            return jsonify({"ok": True, "city": val["label"], "slug": slug, "metrics": val})

    return jsonify({"error": f"Evaluasi untuk kota '{q}' tidak ditemukan di Excel"}), 404

# --- REPLACE endpoint lama /api/top5_cities dengan endpoint baru ini ---
@app.route("/api/top_cities")
def api_top_cities():
    """
    Ambil Top-N dari Excel ringkasan:
      /api/top_cities?year=2024&order=desc&limit=5
        - year  : wajib (angka)
        - order : 'desc' (tertinggi) atau 'asc' (terendah). Default: desc
        - limit : jumlah baris (default 5). Bisa > jumlah data tersedia.
    """
    year_s  = (request.args.get("year") or "").strip()
    order   = (request.args.get("order") or "desc").strip().lower()
    limit_s = (request.args.get("limit") or "5").strip()

    if not year_s:
        return jsonify({"error": "parameter year wajib"}), 400
    try:
        year = int(year_s)
    except:
        return jsonify({"error": "parameter year harus angka"}), 400

    try:
        limit = int(limit_s)
        if limit <= 0: limit = 5
    except:
        limit = 5

    asc = (order == "asc")

    try:
        df = _load_topn_df()
    except Exception as e:
        return jsonify({"error": f"Gagal membaca Excel TopN: {e}"}), 500

    sub = df[df["year"] == year].copy()
    if sub.empty:
        return jsonify({"year": year, "order": order, "limit": limit, "data": []})

    sub = sub.sort_values("avg", ascending=asc)
    # hitung rank (1..n) sesuai urutan
    sub["rank"] = range(1, len(sub) + 1)
    if limit:
        sub = sub.head(limit)

    # formatting output
    out = []
    for r in sub.itertuples(index=False):
        out.append({
            "rank": int(r.rank),
            "city": r.city,
            "province": r.province,
            "avg": None if pd.isna(r.avg) else float(r.avg),
            "min": None if pd.isna(r.min) else float(r.min),
            "max": None if pd.isna(r.max) else float(r.max),
            "n":   None if pd.isna(r.n)   else int(r.n),
        })

    return jsonify({
        "year": year,
        "order": order,
        "limit": limit,
        "count": len(out),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data": out
    })

# ===== Loader Excel regional (ringkasan korelasi) =====
def _norm_colmap_regional(df: pd.DataFrame):
    cmap = {c.lower().strip(): c for c in df.columns}
    def pick(*xs):
        for x in xs:
            if x in cmap: return cmap[x]
        return None
    return dict(
        city  = pick("city","kota","kab/kota","kabupaten/kota","entity","label"),
        prov  = pick("province","provinsi"),
        isl   = pick("island","pulau"),
        vmin  = pick("min_value","harga_terendah","min"),
        dmin  = pick("min_date","tanggal_harga_terendah","min_tanggal","min_date_iso"),
        vmax  = pick("max_value","harga_tertinggi","max"),
        dmax  = pick("max_date","tanggal_harga_tertinggi","max_tanggal","max_date_iso"),
        hi    = pick("avg_month_high","rata2_bulan_tertinggi","avg_high"),
        hi_m  = pick("avg_month_high_month","bulan_rata2_tertinggi","avg_high_month"),
        hi_y  = pick("avg_month_high_year","tahun_rata2_tertinggi","avg_high_year"),
        lo    = pick("avg_month_low","rata2_bulan_terendah","avg_low"),
        lo_m  = pick("avg_month_low_month","bulan_rata2_terendah","avg_low_month"),
        lo_y  = pick("avg_month_low_year","tahun_rata2_terendah","avg_low_year"),
    )

def _num_id(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    s = re.sub(r"[^\d.\-]", "", s)
    if s.count(".") == 1:
        L, R = s.split(".")
        if R.isdigit() and len(R) == 3: s = L + R
    try: return float(s)
    except: return np.nan

def _date_iso(x):
    if pd.isna(x): return None
    try: return pd.to_datetime(x).date().isoformat()
    except: return str(x)

def _load_regional_df():
    if not REGIONAL_XLSX.exists():
        raise FileNotFoundError(f"Regional summary tidak ditemukan: {REGIONAL_XLSX}")
    mtime = REGIONAL_XLSX.stat().st_mtime
    if _REGION_CACHE["df"] is not None and _REGION_CACHE["mtime"] == mtime:
        return _REGION_CACHE["df"]

    src = pd.read_excel(REGIONAL_XLSX, sheet_name=0)
    cm = _norm_colmap_regional(src)

    # Fallback: pakai city_label atau city_key kalau "city" nggak ketemu
    if not cm.get("city"):
        if "city_label" in src.columns:
            cm["city"] = "city_label"
        elif "city_key" in src.columns:
            cm["city"] = "city_key"

    if not cm.get("city") or not cm.get("prov"):
        raise ValueError("Excel perlu minimal kolom City & Province (nama fleksibel).")

    df = pd.DataFrame({
        "city":     src[cm["city"]].astype(str).str.strip(),
        "province": src[cm["prov"]].astype(str).str.strip(),
        "island":   src[cm["isl"]].astype(str).str.strip() if cm["isl"] else "",
        "min_value": src[cm["vmin"]].apply(_num_id) if cm["vmin"] else np.nan,
        "min_date":  src[cm["dmin"]].apply(_date_iso) if cm["dmin"] else None,
        "max_value": src[cm["vmax"]].apply(_num_id) if cm["vmax"] else np.nan,
        "max_date":  src[cm["dmax"]].apply(_date_iso) if cm["dmax"] else None,
        "avg_month_high":        src[cm["hi"]].apply(_num_id) if cm["hi"] else np.nan,
        "avg_month_high_month":  pd.to_numeric(src[cm["hi_m"]], errors="coerce").astype("Int64") if cm["hi_m"] else pd.Series([pd.NA]*len(src), dtype="Int64"),
        "avg_month_high_year":   pd.to_numeric(src[cm["hi_y"]], errors="coerce").astype("Int64") if cm["hi_y"] else pd.Series([pd.NA]*len(src), dtype="Int64"),
        "avg_month_low":         src[cm["lo"]].apply(_num_id) if cm["lo"] else np.nan,
        "avg_month_low_month":   pd.to_numeric(src[cm["lo_m"]], errors="coerce").astype("Int64") if cm["lo_m"] else pd.Series([pd.NA]*len(src), dtype="Int64"),
        "avg_month_low_year":    pd.to_numeric(src[cm["lo_y"]], errors="coerce").astype("Int64") if cm["lo_y"] else pd.Series([pd.NA]*len(src), dtype="Int64"),
    })

    # isi island kalau kosong pakai PROVINCE_TO_ISLAND
    def _isl_from_prov(p):
        return PROVINCE_TO_ISLAND.get(str(p).upper())
    df.loc[(~df["province"].isna()) & ((df["island"]=="") | (df["island"].isna())), "island"] = \
        df["province"].map(lambda p: _isl_from_prov(p) or "")

    df["island"] = df["island"].replace({"Bali-NT": "Bali–NT"}).fillna("")
    _REGION_CACHE.update({"mtime": mtime, "df": df})
    return df

@app.route("/api/provinces")
def api_provinces():
    try:
        df = _load_regional_df()
        provs = sorted({p for p in df["province"].dropna().astype(str).str.strip() if p})
        return jsonify(provs)
    except Exception as e:
        return jsonify({"error": f"Gagal baca provinsi: {e}"}), 500

@app.route("/api/region_summary")
def api_region_summary():
    mode  = (request.args.get("mode") or "island").strip().lower()   # 'island' | 'province'
    value = (request.args.get("value") or "").strip()
    if mode not in ("island","province"):
        return jsonify({"error":"mode harus 'island' atau 'province'"}), 400
    if not value:
        return jsonify({"error":"parameter 'value' wajib"}), 400
    try:
        df = _load_regional_df()
        if mode == "island":
            sub = df[df["island"].str.casefold() == value.casefold()].copy()
        else:
            sub = df[df["province"].str.casefold() == value.casefold()].copy()

        sub = sub.sort_values(["province","city"], kind="stable")
        rows = []
        for i, r in enumerate(sub.itertuples(index=False), start=1):
            rows.append({
                "no": i,
                "city": r.city,
                "province": r.province,
                "island": r.island or None,
                "min_value": None if pd.isna(r.min_value) else float(r.min_value),
                "min_date":  r.min_date,
                "max_value": None if pd.isna(r.max_value) else float(r.max_value),
                "max_date":  r.max_date,
                "avg_month_high": None if pd.isna(r.avg_month_high) else float(r.avg_month_high),
                "avg_month_high_month": None if pd.isna(r.avg_month_high_month) else int(r.avg_month_high_month),
                "avg_month_high_year":  None if pd.isna(r.avg_month_high_year)  else int(r.avg_month_high_year),
                "avg_month_low": None if pd.isna(r.avg_month_low) else float(r.avg_month_low),
                "avg_month_low_month": None if pd.isna(r.avg_month_low_month) else int(r.avg_month_low_month),
                "avg_month_low_year":  None if pd.isna(r.avg_month_low_year)  else int(r.avg_month_low_year),
            })
        return jsonify({"mode": mode, "value": value, "count": len(rows), "rows": rows})
    except Exception as e:
        return jsonify({"error": f"Gagal memproses: {e}"}), 500




# ---------- route ----------

# =======================
# MAIN
# =======================
if __name__ == "__main__":
    print(">> Starting server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)

