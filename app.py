# app.py
from statistics import mode
import os, json, warnings, re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd, time
from flask import Flask, request, jsonify, send_from_directory, abort, render_template, current_app
from flask_cors import CORS
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import math
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import time
import os
from datetime import date

GOOGLE_API_KEY = os.getenv("AIzaSyCA1O5IxSkSg_uHpB5ITbk14CUqhnA_YBk")


from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import numpy as np

import pandas as pd

from io import StringIO
from werkzeug.utils import secure_filename
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import traceback

import time
import traceback


# =======================
# KONFIGURASI DASAR
# =======================
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = Path(os.getenv("DATA_PATH", str(BASE_DIR / "data" / "dataset.xlsx")))

# MODELS dirs: bisa dioverride lewat env
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "packs")))
MODELS_DIR_PACKS = Path(os.getenv("MODELS_DIR_PACKS", str(BASE_DIR / "300_test")))
MODELS_30 = Path(os.getenv("MODELS_DIR_PACKS", str(BASE_DIR / "30 test")))

ENTITY_PROV_PATH = BASE_DIR / "static" / "entity_to_province.json"
CITY_COORDS_PATH = BASE_DIR / "static" / "city_coords.json"
EVAL_XLSX = BASE_DIR / "models" / "summary_all_cities.xlsx"
TOPN_XLSX = BASE_DIR / "data" / "topn_per_tahun_per_kota.csv"
REGIONAL_XLSX = BASE_DIR / "data" / "regional_correlation_summary.xlsx"
_REGION_CACHE = {"mtime": None, "df": None}

with open("static/city_coords.json", "r", encoding="utf-8") as f:
    CITY_COORDS = json.load(f)
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
    "Bangka Belitung": "Sumatra",
    "Bali": "Bali–NT",
    "Bengkulu" : "Sumatra",
    "Lampung": "Sumatra",
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
    "DAERAH ISTIMEWA YOGYAKARTA": "Jawa",
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
EID_DATES = ["2020-05-24","2021-05-13","2022-05-02","2023-04-22","2024-04-10","2025-03-31"]

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

    cutoff = pd.Timestamp("2025-12-14")  # batas akhir data nyata
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


# Kalender flags sederhana (samakan dengan training)

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

import math

import math
import numpy as np

def _safe_float_or_none(obj):
    """
    Convert value to float safely. Return None if it's NaN, Inf, None, or not convertible.
    """
    import numpy as np
    import pandas as pd
    import math

    try:
        # handle None / NaN-like
        if obj is None:
            return None

        # handle pandas/numpy NaN and empty arrays
        if isinstance(obj, (np.ndarray, pd.Series)):
            # empty array/series -> None
            if obj.size == 0:
                return None
            # if it's scalar-like array, take first value
            obj = obj.item() if obj.size == 1 else float(obj.mean())

        # handle pd.NA or NaN
        if hasattr(pd, "isna") and pd.isna(obj):
            return None

        # convert to float
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None
def _sanitize_for_json(o):
    if isinstance(o, dict):
        return {k: _sanitize_for_json(v) for k,v in o.items()}
    if isinstance(o, list):
        return [_sanitize_for_json(x) for x in o]
    if pd.isna(o):
        return None
    # numpy scalars
    try:
        import numpy as np
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
    except Exception:
        pass
    return o


def _find_non_json_numbers(obj, path=''):
    """
    Recursively find locations of NaN/Inf/non-serializable numeric-like values.
    Returns list of (path, value_repr).
    """
    problems = []
    if obj is None:
        return problems
    # check numpy/pandas scalars
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            problems.append((path or '/', v))
        return problems
    if isinstance(obj, np.integer):
        return problems
    if isinstance(obj, dict):
        for k, v in obj.items():
            problems += _find_non_json_numbers(v, f"{path}/{k}" if path else str(k))
        return problems
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            problems += _find_non_json_numbers(v, f"{path}[{i}]")
        return problems
    # pandas stuff
    try:
        import pandas as pd
        if isinstance(obj, pd.Timestamp):
            return problems
    except Exception:
        pass
    return problems


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


import re

def _slugify_city(name: str) -> str:
    s = (name or "").strip().lower()

    # NORMALISASI FORMAT FILE PREDICTED
    # kab._banyumas -> kab_banyumas
    # kota._padang  -> kota_padang
    s = s.replace("kab._", "kab_").replace("kota._", "kota_")

    # samakan separator
    s = re.sub(r"[\s\-]+", "_", s)

    # hapus karakter aneh
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

import math
import re
import pandas as pd
import numpy as np
from pathlib import Path

# contoh: EVAL_XLSX = Path("data/eval_summary.xlsx")
# global cache variable exists in your code: _EVAL_CACHE

def _normalize_colname(c: str) -> str:
    """Lowercase, strip, remove percent/paren, replace spaces and special chars."""
    if c is None:
        return ""
    s = str(c).strip().lower()
    s = s.replace("%", "pct")
    s = re.sub(r"[\(\)\[\]\-\/\\]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
def _load_eval_metrics(city: str = None):
    """
    Baca Excel evaluasi model.
    Jika `city` diberikan, cari langsung baris yang cocok di kolom 'Kota' atau 'Kab/Kota'.
    Jika tidak, kembalikan semua hasil evaluasi dalam bentuk dict per kota.
    """
    global _EVAL_CACHE
    p = EVAL_XLSX

    if not p.exists():
        print("DEBUG eval file not found:", p)
        return None if city else {}

    df = pd.read_excel(p)
    col_map = {c.lower(): c for c in df.columns}
    col_kota = (
        col_map.get("kota")
        or col_map.get("city")
        or col_map.get("kab/kota")
        or col_map.get("kabupaten/kota")
    )
    col_mae  = col_map.get("mae")
    col_rmse = col_map.get("rmse")
    col_mape = col_map.get("mape") or col_map.get("mape(%)")
    col_r2   = col_map.get("r2") or col_map.get("r²") or col_map.get("r2 score")

    if not col_kota:
        print("⚠️ Kolom 'Kota' tidak ditemukan di file evaluasi.")
        return None if city else {}

    # ======== JIKA CITY DIKIRIM, CARI LANGSUNG BARISNYA ========
    if city:
        city_norm = _slugify_city(city)  # contoh: "Kab. Banyumas" → "kab_banyumas"

        # Normalisasi kolom nama kota dari Excel juga
        df["_slug"] = df[col_kota].astype(str).apply(lambda x: _slugify_city(x))

        # Cari exact match slug dulu
        match = df[df["_slug"] == city_norm]
        if match.empty:
            # fallback: cari baris yang mengandung nama kota (lebih fleksibel)
            match = df[df[col_kota].astype(str).str.lower().str.contains(city.lower())]

        if not match.empty:
            row = match.iloc[0]  # ambil baris pertama yang cocok
            label = str(row[col_kota]).strip()

            mae  = _to_float_safe(row.get(col_mae))
            rmse = _to_float_safe(row.get(col_rmse))
            mape = _to_float_safe(row.get(col_mape))
            r2   = _to_float_safe(row.get(col_r2))
            if mape and mape > 1.5:
                mape = mape / 100.0
            mse = rmse ** 2 if rmse is not None else None

            return {
                "label": label,
                "mae": mae,
                "rmse": rmse,
                "mse": mse,
                "mape": mape,
                "r2": r2
            }

        # kalau gak ada yang cocok sama sekali
        print(f"⚠️ City '{city}' tidak ditemukan di file evaluasi.")
        return None

    # ======== JIKA TIDAK ADA PARAM CITY, LOAD SEMUA ========
    data = {}
    for _, row in df.iterrows():
        label = str(row[col_kota]).strip()
        if not label:
            continue
        slug = _slugify_city(label)

        mae  = _to_float_safe(row.get(col_mae))
        rmse = _to_float_safe(row.get(col_rmse))
        mape = _to_float_safe(row.get(col_mape))
        r2   = _to_float_safe(row.get(col_r2))
        if mape and mape > 1.5:
            mape = mape / 100.0
        mse = rmse ** 2 if rmse is not None else None

        data[slug] = {
            "label": label,
            "mae": mae,
            "rmse": rmse,
            "mse": mse,
            "mape": mape,
            "r2": r2
        }

    _EVAL_CACHE = data
    return _EVAL_CACHE


def _to_float_safe(v):
    try:
        return float(v)
    except Exception:
        return None




print(">> Starting app.py")
print(">> Expecting Excel at:", DATA_PATH)
WIDE, LONG_DF = build_wide_long(str(DATA_PATH), FILL_METHOD)
print(">> Dataset loaded:", len(LONG_DF), "rows")
ENTITIES = set(LONG_DF["entity"].unique())
LAST_ACTUAL = _compute_last_actual_dates(str(DATA_PATH))
print(">> LAST_ACTUAL computed (sample):", list(LAST_ACTUAL.items())[:3])


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

DEFAULT_TEST_DAYS = 365

def _load_model_for_entity(entity: str, mode: str = "test"):
    """
    Load model pack for given entity. `mode` controls which folder to load from:
      - mode == "real" -> MODELS_DIR_PACKS
      - otherwise        -> MODELS_DIR
    """
    cache_key = f"{mode}::{entity}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # pilih folder model berdasarkan mode
    models_dir = MODELS_30 if str(mode).lower() == "real" else MODELS_DIR

    # 1) Cari pack terlebih dulu (packs naming)
    files = sorted(models_dir.glob(f"{entity}*best_pack.joblib"))
    # 2) Fallback: file joblib apa pun yang cocok
    if not files:
        files = sorted(models_dir.glob(f"{entity}*.joblib"))
    if not files:
        raise FileNotFoundError(f"Model file untuk '{entity}' tidak ditemukan di {models_dir}")

    files0 = files[0]
    pack = joblib.load(files0)

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
        print(f">> WARNING: '{files0.name}' bukan pack. Memakai raw estimator.")

    alpha      = float(best_cfg.get("alpha_blend", 1.0))
    mode_cfg   = best_cfg.get("mode", "level")
    transform  = best_cfg.get("transform", "none")

    # === Reconstruct smearing (kalau level+log) ===
    dfe = LONG_DF.loc[LONG_DF["entity"] == entity, ["entity","date","value"]].copy()
    df_feat, _ = make_features_entity(dfe, horizon=1)

    # 🔥 tambahan: kalau mode == real, tentukan train_until yang lebih “pintar”
  
    if str(mode).lower() == "real":

            # for real mode: prefer explicit best_cfg train_until if present,
        # otherwise try conservative fallbacks (file mtime -> overall_max - DEFAULT_TEST_DAYS)
        train_until_raw = best_cfg.get("train_until")
        if train_until_raw:
            try:
                train_until = pd.to_datetime(train_until_raw)
                print(f">> [REAL] train_until from pack.best_config: {train_until.date()}")
            except Exception:
                train_until = None
        else:
            train_until = None

        # fallback 1: try metrics["train_until"]
        if train_until is None and isinstance(metrics, dict) and metrics.get("train_until"):
            try:
                train_until = pd.to_datetime(metrics.get("train_until"))
                print(f">> [REAL] train_until from pack.metrics: {train_until.date()}")
            except Exception:
                train_until = None

        # fallback 2: infer from model file mtime (conservative: mtime - 1 day)
        if train_until is None:
            try:
                mtime = files0.stat().st_mtime
                mdate = pd.to_datetime(time.strftime("%Y-%m-%d", time.localtime(mtime)))
                train_until = mdate - pd.Timedelta(days=60)
                print(f">> [REAL] train_until inferred from file mtime: {train_until.date()}")
            except Exception:
                train_until = None

        # fallback 3: infer by subtracting DEFAULT_TEST_DAYS from df_feat max
        if train_until is None:
            try:
                overall_max = df_feat["date"].max()
                train_until = overall_max - pd.Timedelta(days=DEFAULT_TEST_DAYS)
                print(f">> [REAL] train_until inferred: {train_until.date()} (df_feat.max={overall_max.date()} - {DEFAULT_TEST_DAYS}d)")
            except Exception:
                train_until = df_feat["date"].max()
                print(f">> [REAL] final fallback train_until = df_feat.max: {train_until.date()}")
    else:
        # 🔒 untuk mode test — jangan ubah apa pun
        train_until_raw = best_cfg.get("train_until")
        train_until = pd.to_datetime(train_until_raw) if train_until_raw else df_feat["date"].max()

    df_train = df_feat.loc[df_feat["date"] <= train_until].copy()

    use_log = (mode_cfg == "level" and transform == "log")
    smear = 1.0  # default aman

    if use_log:
        if df_train.empty:
            n = len(df_feat)
            if n >= 20:
                cut = max(10, int(0.8 * n))
                df_train = df_feat.iloc[:cut].copy()
                warnings.warn(f"[{entity}] TRAIN subset kosong (train_until={train_until}); fallback {cut}/{n} baris pertama untuk smearing.")
            else:
                warnings.warn(f"[{entity}] Data terlalu pendek ({n}); set smear=1.0.")

        if not df_train.empty:
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

    _MODEL_CACHE[cache_key] = {
        "model": model,
        "feature_cols": feature_cols,
        "config": best_cfg,
        "metrics": metrics,
        "smear": smear,
        "mode": mode_cfg,
        "transform": transform,
        "alpha": alpha,
        "loaded_from": str(files0)
    }

    print(f">> Loaded model for {entity} (mode={mode}) | smear={smear:.6f} | train_until={train_until.date()} | file={files0.name}")
    return _MODEL_CACHE[cache_key]


def make_features_for_next(dfe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Generator fitur untuk prediksi recursive H+1 yang
    *persis* sama dengan fitur training 'lag30':
      ['lag_1','lag_30','month','dayofweek','time_index', 'is_end_of_year','is_new_year','is_eid_window']
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

import math

def _safe_str(x):
    """
    Return normalized string or empty string if x is None/NaN.
    Avoid calling .strip() on non-string objects.
    """
    if x is None:
        return ""
    # pandas/ numpy NaN check
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
    except Exception:
        pass
    return str(x).strip()

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



_PRECOMP_CACHE = {}
# Prediction caches (if used elsewhere)
_RC_PRED_CACHE = {}
RC_PRED_CACHE_TTL = 60 * 30
RC_PRED_MAX_WORKERS = 6
RC_PRED_HORIZON_LIMIT = 365


def _normalize_city_from_pred(s: str) -> str:
    """
    'kab._banyumas' -> 'kab_banyumas'
    'kota_padang'   -> 'kota_padang'
    """
    return (s or "").lower().replace(".", "").strip()

def load_predicted_city_choropleth(year, month=None, week=None, island=None):
    """
    Load choropleth predicted data at CITY level (no province aggregation)
    """

    base = Path("static/data_01_")
    payloads = []

    def _read_file(fp):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    # =========================
    # LOAD FILE
    # =========================
    if month and week:
        fp = base / f"choropleth_pred_{year:04d}_{int(month):02d}_w{int(week)}.json"
        if fp.exists():
            p = _read_file(fp)
            if p:
                payloads.append(p)
    elif month:
        fp = base / f"choropleth_pred_{year:04d}_{int(month):02d}.json"
        if fp.exists():
            p = _read_file(fp)
            if p:
                payloads.append(p)
    else:
        files = sorted(base.glob(f"choropleth_pred_{year:04d}_*.json"))
        for fp in files:
            p = _read_file(fp)
            if p:
                payloads.append(p)

    if not payloads:
        return None

    # =========================
    # FLATTEN CITY DATA
    # =========================
    rows = []
    for p in payloads:
        for d in p.get("data", []):
            city = d.get("city")
            value = _safe_float_or_none(d.get("value"))
            province = d.get("province")
            isl = d.get("island")

            if not city or value is None:
                continue

            rows.append({
                "city": city.strip(),
                "province": province.strip() if province else None,
                "island": isl.strip() if isinstance(isl, str) else isl,
                "value": value
            })

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # =========================
    # FILTER ISLAND (OPTIONAL)
    # =========================
    if island and island.strip().lower() != "semua pulau":
        df = df[df["island"].str.upper() == island.strip().upper()]

    if df.empty:
        return None

    # =========================
    # AGGREGATE PER CITY
    # =========================
    city_df = (
        df.groupby("city", as_index=False)
          .agg(
              value=("value", "mean"),
              province=("province", "first"),
              island=("island", "first")
          )
    )

    # =========================
    # TERTIL (CITY BASED)
    # =========================
    vals = city_df["value"].dropna().values
    if len(vals):
        q1, q2 = np.quantile(vals, [1/3, 2/3])
    else:
        q1 = q2 = None

    def cat(v):
        if q1 is None:
            return "no-data"
        if v <= q1:
            return "T1"
        if v <= q2:
            return "T2"
        return "T3"

    # =========================
    # MERGE COORDINATES
    # =========================
    data_out = []
    for r in city_df.itertuples(index=False):
        key = _slugify_city(r.city)
        coord = CITY_COORDS.get(key)
        if not coord:
            continue

        data_out.append({
            "city": r.city,
            "province": r.province,
            "island": r.island,
            "value": float(r.value),
            "category": cat(r.value),
            "lat": coord["lat"],
            "lng": coord["lng"]
        })

    return {
        "mode": "predicted",
        "bucket_scope": "city",
        "buckets": {
            "T1": float(q1) if q1 is not None else None,
            "T2": float(q2) if q2 is not None else None
        },
        "data": data_out
    }

## ----------------- API endpoint -----------------
@app.route("/api/choropleth")
def api_choropleth():
    try:
        mode = request.args.get("mode", "actual")

        year  = int(request.args.get("year"))
        month = request.args.get("month", type=int)
        week  = request.args.get("week", type=int)
        island = request.args.get("island")

        # =========================================================
        # =================== PREDICTED (CITY) ====================
        # =========================================================
        if mode == "predicted":
            resp = load_predicted_city_choropleth(
                year=year,
                month=month,
                week=week,
                island=island
            )

            return jsonify({
                "mode": "predicted",
                "bucket_scope": "city",
                "buckets": None,
                "rata_ratanasional": None,
                "data": resp.get("data", []) if resp else []
            })

        # =========================================================
        # ======================= ACTUAL ==========================
        # =========================================================
        df = LONG_DF[["entity", "date", "value"]].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].dt.year == year]

        if month:
            df = df[df["date"].dt.month == month]

        if week and month:
            df["week"] = df["date"].apply(_week_of_month_int)
            df = df[df["week"] == week]

        if df.empty:
            return jsonify({"data": [], "buckets": None})

        # =====================
        # AGREGASI PER KOTA
        # =====================
        city_stats = (
            df.groupby("entity")
              .agg(
                  mean_val=("value", "mean"),
                  cnt=("value", "count"),
              )
              .reset_index()
              .rename(columns={"entity": "city"})
        )

        # =====================
        # TERTIL (CITY BASED)
        # =====================
        city_vals = city_stats["mean_val"].dropna().tolist()
        if city_vals:
            q1, q2 = np.quantile(city_vals, [1/3, 2/3])
        else:
            q1 = q2 = None

        def cat_city(v):
            if v is None or q1 is None:
                return "no-data"
            if q1 == q2:
                return "T2"
            if v <= q1:
                return "T1"
            if v <= q2:
                return "T2"
            return "T3"

        # =====================
        # MERGE LAT-LONG
        # =====================
        data_out = []
        for r in city_stats.itertuples(index=False):
            key = _slugify_city(r.city)
            coord = CITY_COORDS.get(key)
            if not coord:
                continue

            val = float(r.mean_val) if pd.notna(r.mean_val) else None
            data_out.append({
                "city": r.city,
                "value": val,
                "category": cat_city(val),
                "lat": coord["lat"],
                "lng": coord["lng"],
                "count": int(r.cnt),
            })

        return jsonify({
            "mode": "actual",
            "bucket_scope": "city",
            "buckets": {
                "T1": float(q1) if q1 is not None else None,
                "T2": float(q2) if q2 is not None else None,
            },
            "rata_ratanasional": float(np.mean(city_vals)) if city_vals else None,
            "data": data_out,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# static geojson serving helper (optional)
@app.route('/static/geo/<path:filename>')
def static_geo(filename):
    return send_from_directory('static/geo', filename)


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
        if val <= lo: return "T1"
        if val >= hi: return "T3"
        return "T2"

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


#======================= semua yang prediksi
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

def _one_step_predict_series(entity: str, mode: str = "test") -> pd.DataFrame:
    """
    Prediksi one-step-ahead (historis) dengan feature builder yang IDENTIK dgn training.
    Return: DataFrame [date (t+1), pred(level)].
    """
    b = _load_model_for_entity(entity, mode = mode)
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

def _recursive_predict(entity: str, days: int, mode: str = "test"):
    if days <= 0:
        return []

    b = _load_model_for_entity(entity,mode = mode)
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
    mode = (request.args.get("mode") or "test").strip().lower()

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
            one_step = _one_step_predict_series(entity,mode=mode)  # kolom: date, pred
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
            preds = _recursive_predict(entity, days=days_need,mode = mode)  # list {date:'YYYY-MM-DD', pred:float}
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
    last_actual_iso = last_actual_dt.date().isoformat()
    act_for_eval = [a for a in actual_series if a["date"] <= last_actual_iso]
    pred_for_eval = [p for p in predicted_series if p["date"] <= last_actual_iso]

    def _compute_eval_from_series_local(actual_series, predicted_series, min_samples=1):
        try:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            has_sk = True
        except Exception:
            has_sk = False

        if not actual_series or not predicted_series:
            return None
        act_map = {a["date"]: a["value"] for a in actual_series}
        pred_map = {p["date"]: p["value"] for p in predicted_series}
        common = sorted(set(act_map.keys()) & set(pred_map.keys()))
        if len(common) < (min_samples or 1):
            return None
        y_true = np.array([act_map[d] for d in common], dtype=float)
        y_pred = np.array([pred_map[d] for d in common], dtype=float)
        if has_sk:
            mae = float(mean_absolute_error(y_true, y_pred))
            mse = float(mean_squared_error(y_true, y_pred))
        else:
            mae = float(np.mean(np.abs(y_true - y_pred)))
            mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(math.sqrt(mse))
        nz = (y_true != 0)
        mape = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz]))) if nz.sum() > 0 else None
        if has_sk:
            try:
                r2 = float(r2_score(y_true, y_pred))
            except Exception:
                r2 = None
        else:
            try:
                ss_res = float(np.sum((y_true - y_pred) ** 2))
                ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
                r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
            except Exception:
                r2 = None
        return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "r2": r2, "n": len(common)}
    def _zero_metrics():
        return {
            "mae": 0.0,
            "mse": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "r2": 0.0,
            "n": 0
        }

    runtime_eval = None
    
    if mode in ("real", "test"):
        runtime_eval = _compute_eval_from_series_local(
            act_for_eval, pred_for_eval, min_samples=1
        )
        print(f"DEBUG runtime_eval (mode={mode}): {runtime_eval}")

    viz_roll = int((request.args.get("viz_roll") or "0").strip() or 0)
    if viz_roll > 1:
        actual_series    = _apply_rolling(actual_series, window=viz_roll)
        predicted_series = _apply_rolling(predicted_series, window=viz_roll)

    print(f"DEBUG result actual={len(actual_series)} predicted={len(predicted_series)}")
    summary_pred = _series_stats(predicted_series)
    summary_act  = _series_stats(actual_series)

# === Tambahan: ambil evaluasi dari Excel dan gabungkan ke respons ===
    eval_metrics = None
    if start_dt > last_actual_dt:
        # full future → semua metric = 0
        eval_metrics = _zero_metrics()
        if runtime_eval is None:
            runtime_eval = _zero_metrics()
    else:
        # BUKAN full future → cek apakah range ini adalah test window spesial
        TEST_START_MIN = date(2024, 7, 1)
        TEST_START_MAX = date(2024, 7, 7)
        TEST_END_MIN   = date(2025, 7, 1)
        TEST_END_MAX   = date(2025, 7, 7)

        cur_start = start_dt.date()
        cur_end   = end_dt.date()

        is_test_window = (
            TEST_START_MIN <= cur_start <= TEST_START_MAX and
            TEST_END_MIN   <= cur_end   <= TEST_END_MAX
        )

        if is_test_window:
            # HANYA untuk window ini ambil dari Excel
            try:
                eval_metrics = _load_eval_metrics(entity)
                if eval_metrics:
                    metrics = dict(eval_metrics)  # salin biar gak ubah cache

                    for k in ["mae", "rmse", "mse", "mape", "r2"]:
                        if k in metrics and metrics[k] is not None:
                            try:
                                metrics[k] = float(metrics[k])
                            except Exception:
                                metrics[k] = None

                    # MAPE kadang dalam %, ubah ke 0..1
                    if metrics.get("mape") and metrics["mape"] > 1.5:
                        metrics["mape"] = metrics["mape"] / 100.0

                    # Hitung MSE/RMSE jika salah satu kosong
                    if metrics.get("mse") is None and metrics.get("rmse") is not None:
                        metrics["mse"] = float(metrics["rmse"]) ** 2
                    if metrics.get("rmse") is None and metrics.get("mse") is not None:
                        metrics["rmse"] = float(metrics["mse"]) ** 0.5

                    eval_metrics = metrics
                    runtime_eval = dict(metrics)

            except Exception as e:
                print("DEBUG eval attach error:", e)
                eval_metrics = None
        else:
            # Bukan future, tapi juga bukan test window spesial
            # → pakai runtime_eval (manual), atau 0 kalau gak ada
            if runtime_eval is not None:
                eval_metrics = dict(runtime_eval)
            else:
                eval_metrics = _zero_metrics()

    # fallback terakhir kalau masih None (jaga-jaga)
    if eval_metrics is None:
        eval_metrics = _zero_metrics()
    if runtime_eval is None:
        runtime_eval = _zero_metrics()

    return jsonify({
        "city": slug,
        "entity": entity,
        "range": {
            "start": start_dt.date().isoformat(),
            "end":   end_dt.date().isoformat()
        },
        "last_actual": last_actual_dt.date().isoformat(),
        "actual": actual_series,
        "predicted": predicted_series,
        "summary": {
            "actual": summary_act,
            "predicted": summary_pred
        },
        "eval": {
            "excel": eval_metrics,   # now: future → 0, test-window → Excel, lainnya → runtime/0
            "runtime": runtime_eval  # hasil hitung manual utk real+test
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


from flask import request

# Pastikan di file Flask kamu: from flask import request, jsonify
# dan modul lain sudah diimport (re, pandas, etc.)


@app.route("/api/eval_summary")
def api_eval_summary():
    """
    Ambil evaluasi dari Excel untuk satu kota (slug/label longgar):
    /api/eval_summary?city=<slug_or_label>&mode=test|real

    - mode=test (default): kembalikan evaluasi dari Excel (sama seperti sekarang)
    - mode=real: coba hitung runtime eval langsung (one-step preds <= last_actual)
    """
    q = (request.args.get("city", "") or "").strip()
    mode = (request.args.get("mode") or "test").strip().lower()

    if not q:
        return jsonify({"error": "parameter 'city' wajib"}), 400

    # --- helper: runtime eval computation (copy-safe, independent) ---
    import math
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    def _compute_eval_from_series(actual_series, predicted_series, min_samples=5):
        if not actual_series or not predicted_series:
            return None
        act_map = {a["date"]: a["value"] for a in actual_series}
        pred_map = {p["date"]: p["value"] for p in predicted_series}
        common_dates = sorted(set(act_map.keys()) & set(pred_map.keys()))
        if len(common_dates) < min_samples:
            return None
        y_true = np.array([act_map[d] for d in common_dates], dtype=float)
        y_pred = np.array([pred_map[d] for d in common_dates], dtype=float)
        try:
            mae = float(mean_absolute_error(y_true, y_pred))
            mse = float(mean_squared_error(y_true, y_pred))
            rmse = float(math.sqrt(mse))
            nonzero_mask = (y_true != 0)
            if nonzero_mask.sum() > 0:
                mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])))
            else:
                mape = None
            try:
                r2 = float(r2_score(y_true, y_pred))
            except Exception:
                r2 = None
            return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "r2": r2, "n": len(common_dates)}
        except Exception as e:
            print("DEBUG compute_eval error:", e)
            return None

    # --- load Excel metrics (if available) ---
    excel_data = None
    try:
        excel_data = _load_eval_metrics()
    except Exception as e:
        print("DEBUG load excel eval error:", e)
        excel_data = None

    # helper: try several slug candidates like before and return (slug, rec)
    def _find_in_excel(q):
        if not excel_data:
            return None
        for key in _candidates_for_eval_slug(q):
            rec = excel_data.get(key)
            if rec:
                return key, rec
        q_norm = re.sub(r"[.\-]", " ", q.lower()).strip()
        for slug, val in excel_data.items():
            if re.sub(r"[.\-]", " ", val["label"].lower()).strip() == q_norm:
                return slug, val
        for slug, val in excel_data.items():
            if q_norm in val["label"].lower():
                return slug, val
        return None

    # --- If mode=real, compute runtime eval (best-effort) ---
    if mode == "real":
        # try to map incoming q to entity. Prefer Excel-slug mapping if available.
        found = _find_in_excel(q)
        candidate_slugs = []
        if found:
            candidate_slugs.append(found[0])
        # always add the raw q as last resort
        candidate_slugs.append(q)

        runtime_result = None
        runtime_errors = []

        for candidate in candidate_slugs:
            try:
                # try map to internal entity (use your project mapping func)
                try:
                    entity = _slug_to_entity(candidate)
                except Exception:
                    # fallback: assume candidate already matches entity name
                    entity = candidate

                # load actuals
                dfe = LONG_DF.loc[LONG_DF["entity"] == entity, ["date", "value"]].copy()
                if dfe.empty:
                    runtime_errors.append(f"no actuals for entity='{entity}'")
                    continue
                dfe["date"] = dfe["date"].dt.normalize()
                last_actual_dt = LAST_ACTUAL.get(entity, dfe["date"].max().normalize())

                # build actual_series up to cutoff
                mask_act = dfe["date"] <= last_actual_dt
                actual_series = [{"date": d.date().isoformat(), "value": float(v)} for d, v in zip(dfe.loc[mask_act, "date"], dfe.loc[mask_act, "value"])]
                if not actual_series:
                    runtime_errors.append(f"no actuals <= last_actual for entity='{entity}'")
                    continue

                # load one-step preds (use mode="real" to reflect runtime)
                try:
                    one_step = _one_step_predict_series(entity, mode="real")  # expected DataFrame with date,pred
                except FileNotFoundError:
                    runtime_errors.append("one-step model file not found")
                    continue
                except Exception as e:
                    runtime_errors.append(f"failed loading one-step: {e}")
                    continue

                # filter preds up to cutoff
                mask_pred = one_step["date"] <= last_actual_dt
                if mask_pred.sum() == 0:
                    runtime_errors.append("no one-step preds <= last_actual")
                    continue
                predicted_series = [{"date": d.date().isoformat(), "value": float(v)} for d, v in zip(one_step.loc[mask_pred, "date"], one_step.loc[mask_pred, "pred"])]

                # compute eval
                runtime_eval = _compute_eval_from_series(actual_series, predicted_series, min_samples=5)
                if runtime_eval is None:
                    runtime_errors.append("not enough overlap for eval (min_samples=5)")
                    continue

                # success
                runtime_result = {
                    "entity": entity,
                    "last_actual": last_actual_dt.date().isoformat(),
                    "runtime_eval": runtime_eval
                }
                break

            except Exception as e:
                runtime_errors.append(f"unexpected error for candidate '{candidate}': {e}")
                continue

        # assemble response: if Excel available, include it; always include runtime attempt result/errors
        resp = {"ok": True, "query": q, "mode": "real", "runtime": runtime_result, "runtime_errors": runtime_errors}
        if found:
            resp["excel"] = {"slug": found[0], "metrics": found[1]}
        else:
            resp["excel"] = None

        return jsonify(resp)

    # --- else mode == "test": return Excel-only (existing behavior) ---
    # same matching logic as your original function
    if not excel_data:
        return jsonify({"error": "file evaluasi tidak ditemukan atau kosong"}), 404

    for key in _candidates_for_eval_slug(q):
        rec = excel_data.get(key)
        if rec:
            return jsonify({"ok": True, "city": rec["label"], "slug": key, "metrics": rec})

    q_norm = re.sub(r"[.\-]", " ", q.lower()).strip()
    for slug, val in excel_data.items():
        if re.sub(r"[.\-]", " ", val["label"].lower()).strip() == q_norm:
            return jsonify({"ok": True, "city": val["label"], "slug": slug, "metrics": val})

    for slug, val in excel_data.items():
        if q_norm in val["label"].lower():
            return jsonify({"ok": True, "city": val["label"], "slug": slug, "metrics": val})

    return jsonify({"error": f"Evaluasi untuk kota '{q}' tidak ditemukan di Excel"}), 404


def _compute_eval_from_series(actual_series, predicted_series, min_samples=5):
    """
    actual_series / predicted_series: list of {"date":"YYYY-MM-DD","value":float,...}
    Return: dict of metrics or None
    """
    if not actual_series or not predicted_series:
        return None

    # build dict by date
    act_map = {a["date"]: a["value"] for a in actual_series}
    pred_map = {p["date"]: p["value"] for p in predicted_series}

    # intersect dates
    common_dates = sorted(set(act_map.keys()) & set(pred_map.keys()))
    if len(common_dates) < min_samples:
        return None

    y_true = np.array([act_map[d] for d in common_dates], dtype=float)
    y_pred = np.array([pred_map[d] for d in common_dates], dtype=float)

    # safe computations
    try:
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(math.sqrt(mse))
        # MAPE: guard divide-by-zero; skip elements where true == 0
        nonzero_mask = (y_true != 0)
        if nonzero_mask.sum() > 0:
            mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])))
        else:
            mape = None
        # R2
        try:
            r2 = float(r2_score(y_true, y_pred))
        except Exception:
            r2 = None

        # return normalized metrics (MAPE in 0..1 if available)
        return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "r2": r2, "n": len(common_dates)}
    except Exception as e:
        print("DEBUG compute_eval error:", e)
        return None

# helper: compute runtime eval for given eval-slug (best-effort)
def _try_compute_runtime_eval_for_entity(eval_slug):
    """
    eval_slug: key in Excel eval dict (e.g. 'kab_banyumas' etc).
    Returns dict like {"mae":..., "mse":..., "rmse":..., "mape":..., "r2":..., "n":...}
    or {"error": "..."} on failure, or None if not enough overlap.
    """
    try:
        # map eval slug to entity used in LONG_DF; try to re-use your candidate mapping
        # if you have function to do reverse mapping, use it; otherwise try simple heuristics:
        entity = _slug_to_entity(eval_slug) if callable(_slug_to_entity) else eval_slug

        # get actuals until last_actual
        dfe = LONG_DF.loc[LONG_DF["entity"] == entity, ["date", "value"]].copy()
        if dfe.empty:
            return {"error": "data actual tidak ditemukan in LONG_DF for entity"}

        dfe["date"] = dfe["date"].dt.normalize()
        last_actual_dt = LAST_ACTUAL.get(entity, dfe["date"].max().normalize())

        # build actual_series up to cutoff
        mask = dfe["date"] <= last_actual_dt
        actual_series = [{"date": d.date().isoformat(), "value": float(v)} for d, v in zip(dfe.loc[mask, "date"], dfe.loc[mask, "value"])]

        # load one-step predictions (best-effort). Use mode="real" to match runtime behavior.
        try:
            one_step = _one_step_predict_series(entity, mode="real")  # expects DataFrame with date,pred
        except FileNotFoundError:
            return {"error": "one-step model file tidak ditemukan"}
        except Exception as e:
            return {"error": f"gagal load one-step: {e}"}

        # filter preds up to cutoff
        mask_pred = one_step["date"] <= last_actual_dt
        if mask_pred.sum() == 0:
            return {"error": "tidak ada prediksi historis (one-step) sampai cutoff"}

        predicted_series = [{"date": d.date().isoformat(), "value": float(v)} for d, v in zip(one_step.loc[mask_pred, "date"], one_step.loc[mask_pred, "pred"])]

        # compute eval using same util as predict_range (assumes _compute_eval_from_series exists)
        runtime_eval = _compute_eval_from_series(actual_series, predicted_series, min_samples=5)
        if runtime_eval is None:
            return {"error": "tidak cukup overlap untuk menghitung runtime eval (min_samples=5)"}
        return runtime_eval

    except Exception as e:
        return {"error": f"unexpected error: {e}"}


@app.route("/api/provinces")
def api_provinces():
    try:
        df = _load_regional_df()
        provs = sorted({p for p in df["province"].dropna().astype(str).str.strip() if p})
        return jsonify(provs)
    except Exception as e:
        return jsonify({"error": f"Gagal baca provinsi: {e}"}), 500

# --- Normalisasi nama kota agar rapi ---
def normalizes_city_name(name: str) -> str:
    if not name:
        return ""
    name = name.strip().lower()
    name = name.replace("_", " ").replace("-", " ").replace(".", " ")
    name = " ".join(name.split())  # hapus spasi ganda
    # ubah prefiks kab/kota jadi bentuk formal
    if name.startswith("kab "):
        name = "kabupaten " + name[4:]
    elif name.startswith("kabupaten "):
        pass
    elif name.startswith("kota "):
        pass
    # kapitalisasi tiap kata
    return " ".join([w.capitalize() for w in name.split()])

# Terapkan ke semua nama di page_cities

# @app.route("/api/region_summary")
# def api_region_summary():
    
#     mode  = (request.args.get("mode") or "island").strip().lower()
#     value = (request.args.get("value") or "").strip()
#     predict = str(request.args.get("predict") or "0").strip()
#     naive_fallback = str(request.args.get("naive_fallback") or "0").strip()
#     try:
#         horizon = int(request.args.get("horizon") or 30)
#     except:
#         return jsonify({"error":"horizon harus integer (jumlah hari)"}), 400
#     model_mode = (request.args.get("model_mode") or "test").strip()

#     # parse optional date filters
#     year_q  = request.args.get("year", "").strip()
#     month_q = request.args.get("month", "").strip()
#     week_q  = request.args.get("week", "").strip()
#     try:
#         year  = int(year_q)  if year_q  else None
#         month = int(month_q) if month_q else None
#         week  = int(week_q)  if week_q  else None
#     except:
#         return jsonify({"error":"format year/month/week tidak valid"}), 400

#     if mode not in ("island","province"):
#         return jsonify({"error":"mode harus 'island' atau 'province'"}), 400
#     if not value:
#         return jsonify({"error":"parameter 'value' wajib"}), 400

#     do_predict = predict in ("1","true","yes")
#     naive_fallback = naive_fallback in ("1","true","yes")
#     # init containers early
#     results = {}
#     errors = {}
#     if horizon <= 0 or horizon > RC_PRED_HORIZON_LIMIT:
#             return jsonify({"error": f"horizon harus antara 1 dan {RC_PRED_HORIZON_LIMIT} hari"}), 400

#     try:
#         page = int(request.args.get("page") or 1)
#         page_size = int(request.args.get("page_size") or 50)
#     except:
#         return jsonify({"error":"page & page_size harus integer"}), 400
#     if page < 1: page = 1
#     page_size = max(1, min(page_size, 200))

#     # load dataframe
#     df = LONG_DF[["entity","date","value"]].copy()
#     df["date"] = pd.to_datetime(df["date"]).dt.normalize()
#     df = df.dropna(subset=["entity","date","value"])
#     df["province"] = df["entity"].map(ENTITY_TO_PROVINCE)
#     df = df.dropna(subset=["province"])
#     df["island"] = df["province"].str.upper().map(PROVINCE_TO_ISLAND)
#     df["city"] = df["entity"]

#     # --- keep a full copy of the dataframe BEFORE applying year/month/week filters
#     full_df = df.copy()

#     # apply year/month/week filters to df (not full_df)
#     if year is not None:
#         df = df[df["date"].dt.year == year]
#     if month is not None:
#         df = df[df["date"].dt.month == month]
#     if week is not None and month is not None:
#         df["week_in_month"] = df["date"].apply(_week_of_month_int)
#         df = df[df["week_in_month"] == week]

#     # select sub by island/province
#     if mode == "island":
#         if value.lower() == "semua pulau":
#             sub = df.copy()
#         else:
#             sub = df[df["island"].str.casefold() == value.casefold()].copy()
#     else:
#         sub = df[df["province"].str.casefold() == value.casefold()].copy()

#     # if the filtered sub is empty, nothing to show
#     if sub.empty:
#         return jsonify({"mode": mode, "value": value, "count": 0, "rows": []})

#     # --- Normalisasi nama kota: buat city_key (untuk join/agg) dan city_display (untuk output) ---
#     def _make_city_key(s):
#         if not s or pd.isna(s):
#             return ""
#         s = str(s).strip().lower()
#         s = re.sub(r'[_\.\-]+', ' ', s)      # replace _ . - with space
#         s = re.sub(r'\s+', ' ', s).strip()   # collapse spaces
#         s = re.sub(r'^\bkab\s+', 'kabupaten ', s)  # kab -> kabupaten
#         return s

#     def _make_city_display_from_key(key):
#         if not key:
#             return ""
#         disp = " ".join([w.capitalize() for w in key.split()])
#         disp = re.sub(r'^\bKabupaten\s+', 'Kabupaten ', disp)
#         disp = re.sub(r'^\bKota\s+', 'Kota ', disp)
#         return disp

#     # apply to sub and full_df (so future lookups are consistent)
#     sub['city_key'] = sub['city'].apply(_make_city_key)
#     sub['city_display'] = sub['city_key'].apply(_make_city_display_from_key)
#     full_df['city_key'] = full_df['city'].apply(_make_city_key)

#     # fallback: if city_display empty, use original cleaned title of city
#     sub.loc[sub['city_display'].str.strip() == "", 'city_display'] = (
#         sub.loc[sub['city_display'].str.strip() == "", 'city']
#             .astype(str).str.strip().str.title()
#     )

#     # --- build paged list using city_key and display map ---
#     city_key_list = sorted({ str(x).strip() for x in sub['city_key'].unique() if pd.notna(x) and str(x).strip() })
#     city_display_map = sub.drop_duplicates('city_key').set_index('city_key')['city_display'].to_dict()
#     total_cities = len(city_key_list)

#     start = (page - 1) * page_size
#     end = start + page_size
#     page_city_keys = city_key_list[start:end] if start < total_cities else []
#     # page_cities (display names) - used in API output only
#     page_cities = [ city_display_map.get(k, k) for k in page_city_keys ]

#     # prepare time-based columns and monthly aggregation using city_key
#     sub["year"] = sub["date"].dt.year
#     sub["month"] = sub["date"].dt.month
#     monthly = sub.groupby(["city_key","year","month"])["value"].mean().reset_index(name="month_avg")

#     # precompute min/max idx safely (by city_key)
#     try:
#         min_idx = sub.groupby("city_key")["value"].idxmin()
#         max_idx = sub.groupby("city_key")["value"].idxmax()
#     except Exception:
#         min_idx = {}
#         max_idx = {}

#     # city means computed FROM sub (sub already filtered by year/month/week)
#     try:
#         city_mean_df = sub.groupby("city_key", dropna=False).agg(mean_val=("value","mean")).reset_index()
#         # attach display
#         first_display = sub[['city_key','city_display']].drop_duplicates('city_key').set_index('city_key')
#         city_mean_df = city_mean_df.set_index('city_key').join(first_display, how='left').reset_index()
#         # keep column name 'city' for compatibility with older code paths (value is normalized key)
#         city_mean_df = city_mean_df.rename(columns={'city_key':'city'})
#         city_mean_df['city_display'] = city_mean_df['city_display'].fillna(city_mean_df['city'])
#     except Exception:
#         city_mean_df = pd.DataFrame(columns=['city','mean_val','city_display'])

#     # attach province/island for each city using first occurrence in sub (index by city_key)
#     try:
#         city_meta = sub[['city_key','province','island']].drop_duplicates('city_key').set_index('city_key')
#     except Exception:
#         city_meta = pd.DataFrame(columns=['province','island'])

#     # merge meta with means (city_mean_df.city contains the normalized key)
#     if not city_mean_df.empty and not city_meta.empty:
#         city_mean_df = city_mean_df.set_index('city').join(city_meta, how='left').reset_index()
#     else:
#         if "province" not in city_mean_df.columns:
#             city_mean_df["province"] = None
#         if "island" not in city_mean_df.columns:
#             city_mean_df["island"] = None

#     # safe maps keyed by normalized key (which we kept in 'city' column)
#     city_mean_map = { str(row['city']).strip(): _safe_float_or_none(row['mean_val']) for _, row in city_mean_df.iterrows() if pd.notna(row['city']) }
#     city_prov_map = { str(row['city']).strip(): (str(row['province']).strip() if pd.notna(row.get('province')) else None) for _, row in city_mean_df.iterrows() if pd.notna(row['city']) }
#     city_island_map = { str(row['city']).strip(): (str(row['island']).strip() if pd.notna(row.get('island')) else None) for _, row in city_mean_df.iterrows() if pd.notna(row['city']) }

#     # province-level aggregation (mean of city means) — used to compute island/national refs
#     if not city_mean_df.empty:
#         prov_vals = city_mean_df.groupby("province", dropna=False, as_index=False).agg(value=("mean_val","mean"), n_cities=("city","nunique")).reset_index()
#         prov_vals["value"] = prov_vals["value"].apply(_safe_float_or_none)
#         prov_vals["province"] = prov_vals["province"].apply(lambda x: None if pd.isna(x) else str(x).strip())
#         prov_vals["island"] = prov_vals["province"].astype(str).str.upper().map(PROVINCE_TO_ISLAND)
#         # fill missing island from city_island_map if possible
#         for i, row in prov_vals.iterrows():
#             if not row.get("island"):
#                 vals = city_mean_df.loc[city_mean_df["province"].astype(str).str.strip() == (row.get("province") or ""), "island"].dropna().unique()
#                 prov_vals.at[i, "island"] = vals[0] if len(vals) else None
#     else:
#         prov_vals = pd.DataFrame(columns=["province","value","n_cities","island"])

#     # compute island_means and national_mean
#     try:
#         island_means_series = prov_vals.groupby("island", dropna=False)["value"].mean()
#         island_means = { (k if (k is not None and not (isinstance(k,float) and pd.isna(k))) else "unknown"): _safe_float_or_none(v) for k,v in island_means_series.items() }
#     except Exception:
#         island_means = {}

#     try:
#         city_vals = [ _safe_float_or_none(v) for v in city_mean_df["mean_val"].tolist() if v is not None ]
#         national_mean = _safe_float_or_none(np.mean(city_vals) if city_vals else None)
#     except Exception:
#         national_mean = None

#     # helper predict (unchanged)
#     def _get_pred_for_entity(entity, horizon_days, model_mode_local):
#         key = (entity, horizon_days, model_mode_local)
#         now = time.time()
#         cached = _RC_PRED_CACHE.get(key)
#         if cached and now - cached["ts"] < RC_PRED_CACHE_TTL:
#             return cached["preds"]
#         preds = _recursive_predict(entity, days=horizon_days, mode=model_mode_local)
#         normalized = []
#         for p in preds:
#             try:
#                 normalized.append({"date": pd.to_datetime(p["date"]).normalize(), "pred": float(p["pred"])} )
#             except Exception:
#                 pass
#         _RC_PRED_CACHE[key] = {"ts": now, "preds": normalized}
#         return normalized

#     # --- non-predict: build rows for this page slice ---
#     rows = []
#     if not do_predict:
#         for offset_idx, city_key in enumerate(page_city_keys, start=start+1):
#             try:
#                 # filter by normalized key
#                 g = sub[sub["city_key"].astype(str).str.strip() == city_key].sort_values("date")
#                 if g.empty:
#                     app.logger.debug("city not found in sub df (key): %s", city_key)
#                     continue

#                 g_all = full_df[full_df["city_key"].astype(str).str.strip() == city_key].sort_values("date")

#                 # compute min/max and monthly stats based on g
#                 min_value = min_date = max_value = max_date = None
#                 try:
#                     idx_min_local = g["value"].idxmin()
#                     minr = g.loc[idx_min_local]
#                     min_value = float(minr["value"]); min_date = minr["date"].date().isoformat()
#                 except Exception:
#                     min_value = min_date = None

#                 try:
#                     idx_max_local = g["value"].idxmax()
#                     maxr = g.loc[idx_max_local]
#                     max_value = float(maxr["value"]); max_date = maxr["date"].date().isoformat()
#                 except Exception:
#                     max_value = max_date = None

#                 mm_local = pd.DataFrame()
#                 try:
#                     mm_local = g.groupby([g["date"].dt.year.rename("year"), g["date"].dt.month.rename("month")])["value"].mean().reset_index(name="month_avg")
#                 except Exception:
#                     mm_local = pd.DataFrame()

#                 if not mm_local.empty:
#                     try:
#                         idxh = mm_local["month_avg"].idxmax()
#                         idxl = mm_local["month_avg"].idxmin()
#                         high_row = mm_local.loc[idxh]; low_row = mm_local.loc[idxl]
#                         avg_month_high = float(high_row["month_avg"])
#                         avg_month_high_month = int(high_row["month"])
#                         avg_month_high_year = int(high_row["year"])
#                         avg_month_low  = float(low_row["month_avg"])
#                         avg_month_low_month = int(low_row["month"])
#                         avg_month_low_year  = int(low_row["year"])
#                     except Exception:
#                         avg_month_high = avg_month_high_month = avg_month_high_year = None
#                         avg_month_low  = avg_month_low_month  = avg_month_low_year  = None
#                 else:
#                     avg_month_high = avg_month_high_month = avg_month_high_year = None
#                     avg_month_low  = avg_month_low_month  = avg_month_low_year  = None

#                 # display name and maps
#                 city_display = city_display_map.get(city_key, city_key)
#                 mean_val = city_mean_map.get(city_key) if city_key in city_mean_map else (float(g["value"].mean()) if not g["value"].isna().all() else None)
#                 prov_val = g["province"].iloc[0] if "province" in g.columns and not g["province"].isna().all() else city_prov_map.get(city_key)
#                 isl_val = g["island"].iloc[0] if "island" in g.columns and not g["island"].isna().all() else city_island_map.get(city_key)
#                 island_key = (isl_val or "unknown")

#                 rows.append({
#                     "no": offset_idx,
#                     "city": city_display,
#                     "province": prov_val,
#                     "island": island_key,
#                     "min": _safe_float_or_none(min_value),
#                     "min_date": min_date,
#                     "max": _safe_float_or_none(max_value),
#                     "max_date": max_date,
#                     "mean": _safe_float_or_none(mean_val),
#                     "count": int(g["value"].count() if "value" in g.columns else 0),
#                     "avg_month_high": avg_month_high,
#                     "avg_month_high_month": avg_month_high_month,
#                     "avg_month_high_year": avg_month_high_year,
#                     "avg_month_low": avg_month_low,
#                     "avg_month_low_month": avg_month_low_month,
#                     "avg_month_low_year": avg_month_low_year,
#                     "island_mean": island_means.get(isl_val if isl_val else "unknown"),
#                     "national_mean": national_mean
#                 })
#             except Exception as e:
#                 app.logger.exception("failed building non-predict row for city_key %s: %s", city_key, e)
#                 continue

#         safe_rows = [_sanitize_for_json(r) for r in rows]
#         return jsonify({
#             "mode": mode,
#             "value": value,
#             "count": int(total_cities),
#             "page": int(page),
#             "page_size": int(page_size),
#             "rows": safe_rows,
#             "predict": False
#         })



#     if do_predict:
#             try:
#                 # try load precomputed (this function already handles month/week None)
#                 pre = _load_precomputed_choropleth(year=year or full_df["date"].dt.year.max(), month=month, week=week, island=None)
#             except Exception as _e:
#                 pre = None

#             if pre is not None:
#                 # precomputed available: build table rows for the requested province page slice
#                 # pre contains "raw_entries" (per-city rows) or "data" depending on file structure
#                 raw_entries = pre.get("raw_entries") or pre.get("data") or []
#                 # normalize into DataFrame for robust filtering/agg
#                 if isinstance(raw_entries, list) and len(raw_entries):
#                     df_r = pd.DataFrame(raw_entries)
#                     # normalize column names (keep original casing trimmed)
#                     df_r.columns = [str(c).strip() for c in df_r.columns]
#                     colmap = {c: c.lower() for c in df_r.columns}
#                     df_r.rename(columns=colmap, inplace=True)

#                     # Ensure period fields lowercased are preserved (period_year/month/week)
#                     # find likely columns
#                     def find_col(df, candidates):
#                         for cand in candidates:
#                             if cand in df.columns:
#                                 return cand
#                         return None

#                     entity_col = find_col(df_r, ['entity','city','kota','name'])
#                     value_col  = find_col(df_r, ['value','harga','price','val'])
#                     prov_col   = find_col(df_r, ['province','provinsi','prov','province_name'])
#                     date_col   = find_col(df_r, ['date','tanggal','ts','datetime'])

#                     if entity_col:
#                         df_r['entity'] = df_r[entity_col].astype(str)
#                     else:
#                         df_r['entity'] = df_r.index.map(lambda i: f"city_{i}")

#                     if value_col:
#                         df_r['value'] = pd.to_numeric(df_r[value_col], errors='coerce')
#                     else:
#                         df_r['value'] = pd.NA

#                     if prov_col:
#                         df_r['province'] = df_r[prov_col].astype(str)
#                     else:
#                         df_r['province'] = pd.NA

#                     # If date column missing, try to build from period_* fields if available
#                     if date_col:
#                         df_r['date'] = pd.to_datetime(df_r[date_col], errors='coerce')
#                     else:
#                         # try construct a representative date if possible from period_year/period_month/period_week
#                         def _build_date_from_period(row):
#                             try:
#                                 py = int(row.get('period_year') or row.get('year') or year)
#                             except Exception:
#                                 return pd.NaT
#                             pm = row.get('period_month') or row.get('month') or None
#                             pw = row.get('period_week') or row.get('week') or None
#                             try:
#                                 pm = int(pm) if pm not in (None, '') else None
#                             except Exception:
#                                 pm = None
#                             try:
#                                 pw = int(pw) if pw not in (None, '') else None
#                             except Exception:
#                                 pw = None
#                             if pw is not None and pm is not None:
#                                 day = min(1 + (pw - 1) * 7, 28)
#                                 return pd.to_datetime(f"{py:04d}-{pm:02d}-{day:02d}", errors='coerce')
#                             elif pm is not None:
#                                 return pd.to_datetime(f"{py:04d}-{pm:02d}-01", errors='coerce')
#                             else:
#                                 return pd.to_datetime(f"{py:04d}-01-01", errors='coerce')
#                         df_r['date'] = df_r.apply(_build_date_from_period, axis=1)

#                     # optionally apply the same year/month/week filtering as earlier
#                     df_filtered = df_r
#                     if 'date' in df_filtered.columns and not df_filtered['date'].isna().all():
#                         if year is not None:
#                             df_filtered = df_filtered[df_filtered['date'].dt.year == int(year)]
#                         if month is not None:
#                             df_filtered = df_filtered[df_filtered['date'].dt.month == int(month)]
#                         if week is not None and month is not None:
#                             df_filtered['week_in_month'] = df_filtered['date'].apply(_week_of_month_int)
#                             df_filtered = df_filtered[df_filtered['week_in_month'] == int(week)]

#                     # now group by entity and compute per-city summary (min/max/mean/count + dates)
#                     if df_filtered.empty:
#                         table_rows = []
#                     else:
#                         # per-city aggregates (df_filtered already filtered by year/month/week)
#                         agg_df = df_filtered.groupby(df_filtered['entity'].astype(str)).agg(
#                             min_value=('value','min'),
#                             max_value=('value','max'),
#                             mean_value=('value','mean'),
#                             count=('value','count')
#                         ).reset_index().rename(columns={'entity':'city'})

#                         # attach min/max dates if available
#                         if 'date' in df_filtered.columns and not df_filtered['date'].isna().all():
#                             min_dates = df_filtered.groupby('entity')['date'].min().reset_index().rename(columns={'entity':'city','date':'min_date'})
#                             max_dates = df_filtered.groupby('entity')['date'].max().reset_index().rename(columns={'entity':'city','date':'max_date'})
#                             agg_df = agg_df.merge(min_dates, on='city', how='left').merge(max_dates, on='city', how='left')

#                         # attach province if present
#                         if 'province' in df_filtered.columns and df_filtered['province'].notna().any():
#                             prov_map = df_filtered[['entity','province']].drop_duplicates('entity').rename(columns={'entity':'city'})
#                             agg_df = agg_df.merge(prov_map, on='city', how='left')

#                         # --- NEW: compute prov_vals (mean of city means) and island-level stats ---
#                         try:
#                             # ensure 'province' column exists on agg_df (may be missing when source had no province)
#                             if 'province' not in agg_df.columns:
#                                 agg_df['province'] = None

#                             tmp_city_means = agg_df[['city','mean_value','province']].copy()
#                             tmp_city_means['province'] = tmp_city_means['province'].astype(object)
#                             prov_vals = tmp_city_means.groupby('province', dropna=False, as_index=False).agg(
#                                 value=('mean_value','mean'),
#                                 n_cities=('city','nunique')
#                             ).reset_index(drop=True)
#                             prov_vals['province'] = prov_vals['province'].apply(lambda x: None if pd.isna(x) else str(x).strip())
#                             prov_vals['value'] = prov_vals['value'].apply(_safe_float_or_none)
#                             # map island
#                             prov_vals['island'] = prov_vals['province'].astype(str).str.upper().map(PROVINCE_TO_ISLAND)
#                             # fallback island from df_filtered if missing
#                             for i, row in prov_vals.iterrows():
#                                 if not row.get('island'):
#                                     vals = df_filtered.loc[df_filtered.get('province', pd.Series(dtype=object)).astype(str).str.strip() == (row.get('province') or ''), 'island'].dropna().unique()
#                                     prov_vals.at[i, 'island'] = vals[0] if len(vals) else None
#                         except Exception:
#                             prov_vals = pd.DataFrame(columns=['province','value','n_cities','island'])

#                         # compute island-level means (unweighted mean of province values)
#                         try:
#                             island_means_series = prov_vals.groupby('island', dropna=False)['value'].mean()
#                             island_means = { (k if (k is not None and not (isinstance(k,float) and pd.isna(k))) else 'unknown'): _safe_float_or_none(v) for k,v in island_means_series.items() }
#                         except Exception:
#                             island_means = {}

#                         # national mean computed from city-level means (agg_df.mean_value)
#                         try:
#                             city_vals = [ _safe_float_or_none(v) for v in tmp_city_means['mean_value'].tolist() if v is not None ]
#                             national_mean = _safe_float_or_none(np.mean(city_vals) if city_vals else None)
#                         except Exception:
#                             national_mean = None

#                         # now build table_rows and inject island_mean/national_mean
#                         table_rows = []
#                         for r in agg_df.itertuples(index=False):
#                             prov_val = getattr(r, 'province', None) if 'province' in agg_df.columns else None
#                             island_val = None
#                             if prov_val:
#                                 island_val = PROVINCE_TO_ISLAND.get(str(prov_val).strip().upper()) if PROVINCE_TO_ISLAND else None
#                                 # fallback to prov_vals mapping if needed
#                                 if not island_val:
#                                     pv = prov_vals.loc[prov_vals['province'] == (str(prov_val).strip()), 'island']
#                                     if not pv.empty:
#                                         island_val = pv.iloc[0]
#                             isl_key = island_val if island_val else 'unknown'
#                             table_rows.append({
#                                 "city": getattr(r,'city',None),
#                                 "province": prov_val,
#                                 "island": island_val,
#                                 "min": _safe_float_or_none(getattr(r,'min_value',None)),
#                                 "min_date": (getattr(r,'min_date').date().isoformat() if getattr(r,'min_date',None) is not pd.NaT and getattr(r,'min_date',None) is not None else None),
#                                 "max": _safe_float_or_none(getattr(r,'max_value',None)),
#                                 "max_date": (getattr(r,'max_date').date().isoformat() if getattr(r,'max_date',None) is not pd.NaT and getattr(r,'max_date',None) is not None else None),
#                                 "mean": _safe_float_or_none(getattr(r,'mean_value',None)),
#                                 "count": int(getattr(r,'count',0) or 0),
#                                 # supply island_mean computed from prov_vals (mean of province means) and national_mean
#                                 "island_mean": island_means.get(isl_key if isl_key else 'unknown'),
#                                 "national_mean": national_mean
#                             })

#                     # filter table_rows for the requested province (since this endpoint is province-scoped)
#                     if page_cities:
#                         # we only need rows that belong to the page slice; but raw precomputed likely contains many provinces/cities
#                         # normalize match by province name casefold
#                         prov_l = value.strip().casefold()
#                         filtered_rows = [tr for tr in table_rows if (str(tr.get('province') or '').strip().casefold() == prov_l)]
#                     else:
#                         filtered_rows = table_rows

#                     # now prepare the response for predict mode using precomputed
#                     safe_rows = [_sanitize_for_json(r) for r in filtered_rows]
#                     return jsonify({
#                         "mode": "province",
#                         "value": value,
#                         "count": len(safe_rows),
#                         "page": int(page),
#                         "page_size": int(page_size),
#                         "rows": safe_rows,
#                         "predict": True,
#                         "horizon_days": int(horizon),
#                         "precomputed": True,
#                         "generated_at": pre.get("generated_at"),
#                         "model_version": pre.get("model_version")
#                     })
#                 # if no usable raw_entries -> continue to fallback to runtime predict
#             # else pre is None -> fallback to runtime per-entity prediction

#         # --- PREDICT MODE: build rows with predictions (runtime fallback) ---
#         # (if you have runtime predict code below, it should return a response as well)
#         # SAFE FALLBACK: jika tidak ada response di jalur predict, kembalikan pesan jelas agar Flask tidak return None
#     if do_predict:
#         app.logger.error("api_region_summary: predict requested but no valid response produced (precomputed missing/empty and runtime fallback did not return). "
#                             "mode=%s value=%s year=%s month=%s week=%s", mode, value, year, month, week)
#         return jsonify({
#             "error": "predict requested but no data available",
#             "mode": mode,
#             "value": value,
#             "count": 0,
#             "rows": [],
#             "predict": True,
#             "horizon_days": int(horizon),
#             "precomputed": pre is not None
#         }), 500

def _test_bounds_for_entity(entity):
    """
    Return dict with last_train, test_end, last_actual
    Adjust last_train if you store it per-model; here contoh fixed.
    """
    last_train = pd.Timestamp("2024-07-31").normalize()   # ganti kalau ada per-entity metadata
    test_end   = last_train + pd.DateOffset(years=1)      # -> 2025-07-31
    # last_actual = LAST_ACTUAL.get(entity, dfe["date"].max().normalize())  # caller must pass dfe if needed
    return dict(last_train=last_train, test_end=test_end)

def _predict_meta_for_date(entity, pred_dt, last_actual_dt):
    """
    pred_dt: pandas.Timestamp normalized
    last_train/test_end computed elsewhere
    """
    bounds = _test_bounds_for_entity(entity)
    test_end = bounds["test_end"]
    within_test = pred_dt <= test_end
    days_beyond_test = max(0, (pred_dt - test_end).days)
    # is extrapolation relative to last_actual
    beyond_actual = pred_dt > last_actual_dt
    days_beyond_actual = max(0, (pred_dt - last_actual_dt).days)
    # crude confidence rule (tweak to taste)
    if not within_test:
        conf = "low" if days_beyond_test > 30 else "medium"
    else:
        conf = "high"
    return {
        "within_test": bool(within_test),
        "days_beyond_test": int(days_beyond_test),
        "beyond_actual": bool(beyond_actual),
        "days_beyond_actual": int(days_beyond_actual),
        "confidence": conf,
        "test_end": test_end.date().isoformat()
    }


@app.route("/api/quick_predict")
def api_quick_predict():
    """
    Quick predict ringan untuk UI Beranda.
    Query:
      - city (slug/entity) required
      - mode (test|real) optional, default 'test'
    Returns keys: ok, city, entity, last_actual, last_value, predictions, history, naive
    """
    slug = (request.args.get("city") or "").strip().lower()
    mode = (request.args.get("mode") or "real").strip().lower()
    if not slug:
        return jsonify({"error": "param ?city= wajib"}), 400

    try:
        entity = _slug_to_entity(slug)
    except Exception as e:
        return jsonify({"error": f"slug->entity mapping failed: {e}"}), 400

    # historical series for entity
    dfe = LONG_DF.loc[LONG_DF["entity"] == entity, ["date", "value"]].copy()
    if dfe.empty:
        return jsonify({"error": f"data '{entity}' kosong"}), 404
    dfe["date"] = pd.to_datetime(dfe["date"]).dt.normalize()

    # last actual cutoff and last value - use the same source as predict_range
    last_actual_dt = LAST_ACTUAL.get(entity, dfe["date"].max().normalize())
    last_row = dfe.loc[dfe["date"] == last_actual_dt]
    if not last_row.empty:
        last_val = float(last_row["value"].iloc[0])
    else:
        # fallback take last non-null
        last_val = float(pd.to_numeric(dfe["value"], errors="coerce").dropna().iloc[-1])

    # build ~30-day history (most recent)
    hist_df = dfe.sort_values("date").tail(30)
    history = [{"date": d.date().isoformat(), "value": float(v)} for d, v in zip(hist_df["date"], hist_df["value"])]

    # try to predict next N days using the SAME recursive pipeline as predict_range
    horizons = [1, 2,7, 10]
    predictions = {str(h): None for h in horizons}
    naive = False

    max_h = max(horizons)
    try:
        # IMPORTANT: _recursive_predict simulates from last_actual_dt forward using observed data up to last_actual
        preds = _recursive_predict(entity, days=max_h, mode=mode)  # [{'date','pred'},...]
        # map to requested horizons (preds list indexing: 0 -> H+1)
        for h in horizons:
            idx = h - 1
            if idx < len(preds):
                p = preds[idx]
                # include meta per-prediction so UI can show within_test/confidence
                p_dt = pd.to_datetime(p["date"]).normalize()
                p_meta = _predict_meta_for_date(entity, p_dt, last_actual_dt)  # see helper below
                predictions[str(h)] = {"date": p["date"], "value": float(p["pred"]), "meta": p_meta}
            else:
                predictions[str(h)] = None
    except FileNotFoundError:
        # no model file — naive fallback: repeat last_val
        naive = True
        for h in horizons:
            d = (last_actual_dt + pd.Timedelta(days=h)).date().isoformat()
            # meta for fallback (all beyond actual)
            p_meta = _predict_meta_for_date(entity, pd.to_datetime(d).normalize(), last_actual_dt)
            predictions[str(h)] = {"date": d, "value": float(last_val), "meta": p_meta}
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": f"prediction failed: {e}", "trace": tb}), 500

    return jsonify({
        "ok": True,
        "city": slug,
        "entity": entity,
        "last_actual": last_actual_dt.date().isoformat(),
        "last_value": float(last_val),
        "predictions": predictions,
        "history": history,
        "naive": bool(naive)
    })




#  UPLOAD


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

def upload_iterative_forecast_from_pack(pack: dict, series: pd.Series, horizons=[1,2,7,10]):
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

                # === Safe TimeSeriesSplit creation ===
        def make_safe_tscv(n_splits_wanted, n_samples, test_size, gap):
            from sklearn.model_selection import TimeSeriesSplit
            wanted = max(2, int(n_splits_wanted))
            available = max(0, n_samples - gap)
            if available <= test_size:
                safe_splits = 2
            else:
                safe_splits = min(wanted, max(2, available // test_size))
            safe_splits = max(2, min(safe_splits, wanted))
            try:
                tscv = TimeSeriesSplit(n_splits=safe_splits, gap=gap)
                _ = list(tscv.split(range(n_samples)))  # validate
                return tscv, safe_splits
            except Exception:
                try:
                    tscv = TimeSeriesSplit(n_splits=2, gap=0)
                    _ = list(tscv.split(range(n_samples)))
                    return tscv, 2
                except Exception:
                    raise

        try:
            tscv, used_splits = make_safe_tscv(UPLOAD_N_SPLITS, len(X_train), test_days, max_lookback)
        except Exception as e:
            print(f"[TSCV FALLBACK] failed to create tscv: {e}. Using simple train/test (no CV).")
            tscv, used_splits = None, 0

        base = HGBR(loss="squared_error", random_state=UPLOAD_SEED, **UPLOAD_EARLY_STOP)

        if tscv is not None:
            gs = GridSearchCV(
                estimator=base,
                param_grid=UPLOAD_PARAM_GRID,
                scoring="r2",
                refit=True,
                cv=tscv,
                n_jobs=1,
                verbose=0
            )
            print(f"[GRIDSEARCH] {city_name} • {exp_name}: training {len(X_train)} / test={len(X_test)} • splits={used_splits} • gap={max_lookback}")
            t0 = time.perf_counter()
            gs.fit(X_train, y_train)
            train_secs = time.perf_counter() - t0
            best_model = gs.best_estimator_
        else:
            print(f"[NO-CV] {city_name} • {exp_name}: fallback simple training (no CV)")
            t0 = time.perf_counter()
            base.fit(X_train, y_train)
            train_secs = time.perf_counter() - t0
            best_model = base


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

    preds = upload_iterative_forecast_from_pack(pack, series_full, horizons=list(range(1, 31)))

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

    # server_upload.py (Flask)


# Config: sesuaikan
UPLOAD_BASE = Path(r"C:\tmp\uploads")
ALLOWED_EXT = {'.xlsx', '.xls', '.csv'}
MAX_FILE_MB = 20

# import helper functions you showed earlier:
# from your_helpers import upload_iterative_forecast_from_pack, upload_train_one_city, upload_make_flags, upload_build_features_level_target
# and ensure UPLOAD_EID_DATES etc are defined in that module or in this file

def allowed_file(fname):
    return Path(fname).suffix.lower() in ALLOWED_EXT

def save_uploaded_file(storage, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = secure_filename(storage.filename)
    path = dest_dir / fname
    storage.save(str(path))
    return path

# quick parsing helpers
def read_tabular_file(path: Path):
    suf = path.suffix.lower()
    if suf in ('.xlsx', '.xls'):
        # fallback: read first sheet
        return pd.read_excel(path, engine='openpyxl')
    elif suf == '.csv':
        return pd.read_csv(path)
    else:
        raise ValueError("unsupported file format")




def _fmt_rp(v):
    try:
        return "Rp " + f"{int(round(float(v))):,}".replace(",", ".")
    except Exception:
        return str(v)

def generate_pdf_evaluation(city_name: str, result: dict, output_dir: Path):
    """
    PDF: EVALUASI — <city>
    - Menampilkan metrik dan prediksi horizon 1..30
    - Format prediksi: TGL — Rp X.XXX
    - Dua kolom, auto-paginasi
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        city_safe = str(city_name).replace('/', '_').replace('\\', '_').strip()
        pdf_path = output_dir / f"{city_safe}__evaluasi.pdf"

        # Start PDF
        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        w, h = A4
        top_y = h - 2 * cm
        y = top_y

        # Title
        c.setFont("Helvetica-Bold", 18)
        c.drawString(2 * cm, y, f"EVALUASI — {city_safe}")
        y -= 1.2 * cm
        c.setFont("Helvetica", 9)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        c.drawString(2 * cm, y, f"Generated: {timestamp}")
        y -= 0.8 * cm

        # Basic info
        n_total = result.get("n_total", "-")
        test_days = result.get("test_days", "-")
        best_r2 = result.get("best_r2", "-")
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2 * cm, y, "Informasi Data")
        y -= 0.6 * cm
        c.setFont("Helvetica", 10)
        c.drawString(2.2 * cm, y, f"Total Data (n_total): {n_total}")
        y -= 0.4 * cm
        c.drawString(2.2 * cm, y, f"Test Days: {test_days}")
        y -= 0.4 * cm
        c.drawString(2.2 * cm, y, f"Best R²: {best_r2}")
        y -= 0.8 * cm

        # Metrics
        metrics = result.get("metrics", {})
        if metrics:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(2 * cm, y, "Metrik Evaluasi")
            y -= 0.6 * cm
            c.setFont("Helvetica", 10)
            for k in ("r2", "mae", "cv_best_r2", "train_time_seconds"):
                if k in metrics:
                    c.drawString(2.2 * cm, y, f"{k}: {metrics.get(k)}")
                    y -= 0.4 * cm
            y -= 0.3 * cm

        # Predictions: prefer result['predictions'], otherwise try others
        preds = result.get("predictions") or result.get("predictions_full") or {}
        # Normalize keys to integers
        normalized = {}
        for k, v in preds.items():
            try:
                ik = int(str(k).strip())
            except Exception:
                # try extract digits
                digits = ''.join(filter(str.isdigit, str(k)))
                ik = int(digits) if digits else k
            normalized[int(ik) if isinstance(ik, int) else ik] = v

        # Build horizon list 1..30
        horizons = list(range(1, 31))

        # Header for predictions
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2 * cm, y, "Perkiraan Harga Minyak Goreng (30 hari terakhir)")
        y -= 0.6 * cm

        # Layout two columns
        left_x = 2.2 * cm
        right_x = w / 2 + 0.5 * cm
        line_h = 0.5 * cm
        # Start row y
        start_y = y

        # Prepare rows as list of strings like "2025-07-02 — Rp 21.352"
        rows = []
        for h in horizons:
            v = normalized.get(h)
            if isinstance(v, dict):
                date = v.get("date", "-")
                val = v.get("value", "-")
            else:
                # If preds keyed differently (e.g. 'pred_1d'), try fallback access
                date = "-"
                val = v if v is not None else "-"
            # format
            date_str = str(date) if date is not None else "-"
            val_str = _fmt_rp(val)
            rows.append((h, f"{date_str} — {val_str}"))

        # Draw rows in two columns, ascending order
        col = 0
        row_index = 0
        for idx, (horizon, text) in enumerate(rows):
            col = idx % 2  # 0 left, 1 right
            row_index = idx // 2
            cur_y = start_y - row_index * line_h
            # if not enough space on page, add new page and reset
            if cur_y < 3 * cm:
                c.showPage()
                # redraw header & title on new page for clarity
                c.setFont("Helvetica-Bold", 12)
                c.drawString(2 * cm, h - 2 * cm, f"EVALUASI — {city_safe}")
                c.setFont("Helvetica-Bold", 12)
                c.drawString(2 * cm, h - 3 * cm, "Prediksi (lanjutan)")
                start_y = h - 3.8 * cm
                cur_y = start_y
                # reset row_index relative to new page
                row_index = 0
                # recompute positions for this item after page break
                # we place current item at start_y (left column)
            x = left_x if col == 0 else right_x
            # prepend horizon label e.g. "1d: " optionally — user wanted date first, so we write date — value
            # but include small horizon label in parentheses to keep clarity
            display = f"{text}    ({horizon}d)"
            c.setFont("Helvetica", 10)
            c.drawString(x, cur_y, display)
        # after drawing predictions, set y to below the drawn block
        # compute how many physical rows were used (pairs)
        used_rows = ((len(rows) + 1) // 2)
        y = start_y - used_rows * line_h - 0.6 * cm

        # Footer
        c.setFont("Helvetica-Oblique", 8)
        c.drawString(2 * cm, 1.5 * cm, f"Generated by system — {timestamp}")

        c.save()
        print(f"[PDF] generated: {pdf_path}")
        return pdf_path

    except Exception as e:
        print("[ERROR] generate_pdf_evaluation:", e, traceback.format_exc())
        return None
@app.route('/api/upload_file', methods=['POST'])
def api_upload_file():
    """
    Expects multipart form with:
      - file: uploaded file (.xlsx/.xls/.csv)
      - mode: 'quick' or 'full' (optional, default 'quick')
    Response JSON shape (examples below)
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error":"no file provided"}), 400
        f = request.files['file']
        if f.filename == '':
            return jsonify({"error":"empty filename"}), 400
        if not allowed_file(f.filename):
            return jsonify({"error":"file type not allowed"}), 400

        mode = (request.form.get('mode') or 'quick').strip().lower()
        if mode not in ('quick','full'):
            mode = 'quick'

        # basic size check (werkzeug keeps file in memory/file)
        f.stream.seek(0, 2)
        size = f.stream.tell()
        f.stream.seek(0)
        if size > (MAX_FILE_MB * 1024 * 1024):
            return jsonify({"error":f"file too large (> {MAX_FILE_MB} MB)"}), 413

        # save to temp folder with timestamp
        ts = int(time.time())
        dest = UPLOAD_BASE / str(ts)
        uploaded_path = save_uploaded_file(f, dest)

        # parse file into DataFrame (simple)
        try:
            df = read_tabular_file(uploaded_path)
        except Exception as e:
            return jsonify({"error":"failed parse uploaded file", "detail": str(e)}), 400

        # ---- QUICK MODE: compute lightweight summary & quick preds using packs/naive ----
        if mode == 'quick':
            # Quick summary stats — adjust column names expectation accordingly
            # assume uploaded file has 'date' column + one or multiple city columns OR two columns: date,value (single city)
            df_cols = list(df.columns)
            result = {"mode":"quick", "cols": df_cols}

            # if the file is single-city (date + value), compute 1/7/30 day preds using naive last-value
            if 'date' in (c.lower() for c in df_cols) and df.shape[1] <= 2:
                # normalize: find date col name
                date_col = next(c for c in df_cols if c.lower()=='date')
                value_cols = [c for c in df_cols if c!=date_col]
                val_col = value_cols[0] if value_cols else df_cols[1] if len(df_cols)>1 else None
                if val_col:
                    df[date_col] = pd.to_datetime(df[date_col])
                    s = pd.Series(pd.to_numeric(df[val_col], errors='coerce').values,
                                  index=pd.DatetimeIndex(df[date_col])).asfreq('D').ffill().bfill()
                    last_val = float(s.dropna().iloc[-1])
                    # quick preds: 1,7,30
                    out_pred = {}
                    for h in (1,7,30):
                        out_pred[f'pred_{h}d'] = {"horizon":h, "value": last_val}
                    result['stats'] = {
                        "n_rows": int(len(df)),
                        "last_value": last_val,
                        "predictions": out_pred
                    }
                return jsonify({"ok":True, "mode":"quick", "data": result}), 200

            # fallback quick: give basic stats per numeric column
            stats = {}
            for c in df.select_dtypes(include='number').columns:
                col = df[c].dropna()
                stats[c] = {"count":int(col.count()), "mean": float(col.mean()) if len(col) else None,
                            "min": float(col.min()) if len(col) else None, "max": float(col.max()) if len(col) else None}
            return jsonify({"ok":True, "mode":"quick", "stats":stats, "cols":df_cols}), 200

        # ---- FULL MODE: train per-city (heavy) ----
        # In your helpers you have upload_train_one_city(series_full, city_name, outdir,...)
        # We'll iterate through columns (except date) and call that function.
        if mode == 'full':
            # expect first column is date
            cols = list(df.columns)
            date_col = None
            for c in cols:
                if c.lower() == 'date':
                    date_col = c; break
            if date_col is None:
                return jsonify({"error":"full mode expects a 'date' column in uploaded file"}), 400

            # prepare time series per city
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).reset_index(drop=True)
            results = []
            # output dir for packs specific to this upload
            outdir = UPLOAD_BASE / 'packs' / str(ts)
            upload_ensure_dir(outdir)

            # iterate columns excluding date
            city_cols = [c for c in df.columns if c != date_col]
                        # jumlah hari yang kita kirim sebagai trend (atur sesuai kebutuhan UI)
            TREND_DAYS = 180
            for city in city_cols:
                try:
                    series = pd.Series(
                        pd.to_numeric(df[city], errors='coerce').values,
                        index=pd.DatetimeIndex(df[date_col])
                    ).asfreq('D').ffill().bfill()

                    # train / proses seperti biasa
                    res = upload_train_one_city(series, city, outdir)
                    # === Generate PDF evaluasi otomatis ===
                    try:
                        pdf_path = generate_pdf_evaluation(city, res, outdir)
                        print("[DEBUG] pdf_path:", pdf_path)
                        print("[DEBUG] pdf exists?:", pdf_path.exists() if pdf_path is not None else "None")
                        print("[DEBUG] UPLOAD_BASE:", UPLOAD_BASE.resolve())
                        for p in UPLOAD_BASE.rglob('*evaluasi.pdf'):
                            print("[FOUND UNDER UPLOAD_BASE]", p)
                        if pdf_path and pdf_path.exists():
                            # Buat URL download-nya biar bisa diakses dari frontend
                            rel_path = pdf_path.relative_to(UPLOAD_BASE)
                            res["pdf_url"] = f"/uploads/{rel_path.as_posix()}"
                    except Exception as e:
                        print(f"[WARN] gagal buat PDF evaluasi untuk {city}: {e}")

                    # --- tambahkan trend dan stats agar frontend bisa langsung render chart ---
                    try:
                        s = series.dropna()
                        # potong ke TREND_DAYS terakhir supaya response tidak terlalu besar
                        if len(s) > TREND_DAYS:
                            s_sub = s.iloc[-TREND_DAYS:]
                        else:
                            s_sub = s

                        trend = {
                            "dates": [d.strftime('%Y-%m-%d') for d in s_sub.index],
                            "values": [float(x) if not pd.isna(x) else None for x in s_sub.values]
                        }
                        stats = {
                            "n_points": int(s.shape[0]) if s is not None else 0,
                            "avg": float(s.mean()) if len(s) else None,
                            "min": float(s.min()) if len(s) else None,
                            "max": float(s.max()) if len(s) else None
                        }

                        if isinstance(res, dict):
                            # hanya set kalau belum ada, supaya tidak menimpa info penting
                            res.setdefault("trend", trend)
                            res.setdefault("stats", stats)
                    except Exception as _e:
                        # jangan biarkan kegagalan pembuatan trend memutus seluruh proses
                        print(f"[WARN] gagal menyiapkan trend untuk {city}: {_e}")

                    results.append(res)
                except Exception as e:
                    results.append({"city": city, "ok": False, "reason": str(e), "trace": traceback.format_exc()})
            return jsonify({"ok":True, "mode":"full", "results": results}), 200

        # default fallback
        return jsonify({"error":"unknown mode"}), 400

    except Exception as exc:
        current_app.logger.exception("upload_file failed")
        return jsonify({"error":"internal server error", "detail": str(exc), "trace": traceback.format_exc()}), 500
from flask import send_file, abort, current_app

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    try:
        base = UPLOAD_BASE.resolve()
        target = (base / filename).resolve()
        # security: pastikan target di bawah base
        if not str(target).startswith(str(base)):
            abort(403)
        if not target.exists():
            abort(404)
        return send_file(str(target), as_attachment=True)
    except Exception:
        current_app.logger.exception("serve_uploads error")
        abort(500)


# ---------- route ----------

# =======================
# MAIN
# =======================
if __name__ == "__main__":
    print(">> Starting server on http://localhost:5000")
# --- robust normalize df_c & compute city_stats (replace existing block) ---
    app.run(host="0.0.0.0", port=5000, debug=True)

