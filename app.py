# app.py
import os, json, warnings, re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, abort, render_template
from flask_cors import CORS
import joblib
import math


# =======================
# KONFIGURASI DASAR
# =======================
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH   = BASE_DIR / "data" / "dataset_filled_ffill_bfill.xlsx"
MODELS_DIR  = BASE_DIR / "models"

ENTITY_PROV_PATH = BASE_DIR / "static" / "entity_to_province.json"
CITY_COORDS_PATH = BASE_DIR / "static" / "city_coords.json"
EVAL_XLSX = BASE_DIR / "models" / "training_summary_all_cities.xlsx"

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
    "banjarmasin": "kota_banjarmasin",
    "jakarta-pusat": "kota_administrasi_jakarta_pusat",
    "jakarta-selatan": "kota_administrasi_jakarta_selatan",
    "jakarta-barat": "kota_administrasi_jakarta_barat",
    "jakarta-timur": "kota_administrasi_jakarta_timur",
    "jakarta-utara": "kota_administrasi_jakarta_utara",
    "bandung": "kota_bandung",
    "surabaya": "kota_surabaya",
    "medan": "kota_medan",
    "semarang": "kota_semarang",
    "makassar": "kota_makassar",
    "palembang": "kota_palembang",
    "pontianak": "kota_pontianak",
    "manado": "kota_manado",
    "denpasar": "kota_denpasar",
    "mataram": "kota_mataram",
    "kupang": "kota_kupang",
    "ambon": "kota_ambon",
    "jayapura": "kota_jayapura",
    "biak": "kab._biak_numfor",
    "bogor": "kota_bogor",
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

def to_numeric_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.replace({"-": np.nan, "": np.nan})
    s = s.str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def build_wide_long(path_xlsx: str, fill_method: str):
    raw = pd.read_excel(path_xlsx)
    date_col   = raw.columns[0]
    value_cols = raw.columns[1:]

    raw[date_col] = raw[date_col].astype(str).str.replace(r"\s+", "", regex=True)
    raw[date_col] = pd.to_datetime(raw[date_col], dayfirst=True, errors="coerce")
    raw = raw.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    for c in value_cols:
        raw[c] = to_numeric_series(raw[c])

    full_idx = pd.date_range(raw[date_col].min(), raw[date_col].max(), freq="D")
    wide = (raw.set_index(date_col).reindex(full_idx).rename_axis("date").sort_index())

    if fill_method == "ffill_bfill":
        wide[value_cols] = wide[value_cols].ffill().bfill()
    elif fill_method == "interpolate":
        wide[value_cols] = wide[value_cols].interpolate(method="time", limit_direction="both")
        wide[value_cols] = wide[value_cols].ffill().bfill()
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
    """Cari last tanggal NON-NaN per kolom di file mentah (bukan hasil ffill)."""
    raw = pd.read_excel(path_xlsx)
    date_col = raw.columns[0]
    raw[date_col] = pd.to_datetime(raw[date_col], dayfirst=True, errors="coerce")
    raw = raw.dropna(subset=[date_col]).sort_values(date_col)
    value_cols = raw.columns[1:]

    last = {}
    for c in value_cols:
        ent = re.sub(r"\s+", "_", str(c).strip().lower())
        s = to_numeric_series(raw[c])
        valid_dates = raw.loc[s.notna(), date_col]
        if not valid_dates.empty:
            last[ent] = valid_dates.max().normalize()
    return last

def add_fourier(df: pd.DataFrame, date_col: str, periods, K=2, prefix="fyr"):
    t = (df[date_col] - df[date_col].min()).dt.days.astype(float)
    for P in periods:
        for k in range(1, K+1):
            df[f"{prefix}_sin_P{P}_k{k}"] = np.sin(2*np.pi*k*t/P)
            df[f"{prefix}_cos_P{P}_k{k}"] = np.cos(2*np.pi*k*t/P)

def make_features_entity(dfe: pd.DataFrame, horizon=1):
    df = dfe.sort_values("date").copy()

    lags_short  = [1, 2, 3, 7, 14, 21, 30]
    lags_long   = [60, 90]
    for L in lags_short + lags_long:
        df[f"lag_{L}"] = df["value"].shift(L)

    for W in [7, 14, 30]:
        s = df["value"].shift(1).rolling(W, min_periods=3)
        df[f"roll{W}_mean"]   = s.mean()
        df[f"roll{W}_std"]    = s.std()
        df[f"roll{W}_median"] = s.median()
    for W in [30, 60, 90]:
        s = df["value"].shift(1).rolling(W, min_periods=5)
        df[f"roll{W}_min"] = s.min()
        df[f"roll{W}_max"] = s.max()

    df["diff1"] = df["value"].diff(1)
    df["diff7"] = df["value"].diff(7)

    df["dayofweek"]  = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"]      = df["date"].dt.month

    add_fourier(df, "date", periods=[7, 365.25], K=2, prefix="fyr")

    df["y_next"] = df["value"].shift(-1)
    df["y_diff"] = df["y_next"] - df["value"]

    df = df.dropna().reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ["entity","date","value","y_next","y_diff"]]
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
    feat_in = getattr(model, "feature_names_in_", None)
    if feat_in is not None:
        cols = [c for c in feat_in if c in X_df.columns]
        return model.predict(X_df[cols])
    return model.predict(X_df.to_numpy())

def _load_model_for_entity(entity: str):
    if entity in _MODEL_CACHE:
        return _MODEL_CACHE[entity]

    files = sorted(MODELS_DIR.glob(f"{entity}*.joblib"))
    if not files:
        raise FileNotFoundError(f"Model file untuk '{entity}' tidak ditemukan di {MODELS_DIR}")

    pack = joblib.load(files[0])
    model = pack["model"]
    feature_cols = pack["feature_cols"]
    best_cfg = pack.get("best_config", {"mode": "level", "transform": "none", "train_until": None})
    alpha = float(best_cfg.get("alpha_blend", 1.0))
    metrics = pack.get("metrics", {})

    # reconstruct smearing factor (kalau level+log)
    dfe = LONG_DF.loc[LONG_DF["entity"] == entity, ["entity","date","value"]].copy()
    df_feat, _ = make_features_entity(dfe, horizon=1)

    train_until = pd.to_datetime(best_cfg.get("train_until")) if best_cfg.get("train_until") else df_feat["date"].max()
    tr_mask = df_feat["date"] <= train_until
    df_train = df_feat.loc[tr_mask].copy()
    if df_train.empty:
        raise RuntimeError("TRAIN subset kosong saat reconstruct smearing.")

    mode = best_cfg.get("mode", "level")
    transform = best_cfg.get("transform", "none")
    use_log = (mode == "level" and transform == "log")

    if mode == "level":
        y_train = df_train["y_next"].values
        y_train_tr = np.log(y_train) if use_log else y_train
    else:
        y_train = df_train["y_diff"].values
        y_train_tr = y_train

    if use_log:
        yhat_tr = _predict_with_safe_names(model, df_train[feature_cols].copy())
        resid_log = y_train_tr - yhat_tr
        smear = float(np.mean(np.exp(resid_log)))
    else:
        smear = 1.0

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
    print(f">> Loaded model for {entity} | smear={smear:.6f} | mode={mode}/{transform}")
    return _MODEL_CACHE[entity]

def _one_step_predict_series(entity: str) -> pd.DataFrame:
    """Prediksi one-step-ahead (historis). Return: [date (t+1), pred(level)]."""
    b = _load_model_for_entity(entity)
    model = b["model"]; feature_cols = b["feature_cols"]
    mode = b["mode"]; transform = b["transform"]; smear = b["smear"]; alpha = b.get("alpha", 1.0)

    dfe = LONG_DF.loc[LONG_DF["entity"] == entity, ["entity","date","value"]].copy()
    df_feat, _ = make_features_entity(dfe, horizon=1)

    yhat_tr = _predict_with_safe_names(model, df_feat[feature_cols].copy())
    if mode == "level":
        yhat_level = np.exp(yhat_tr) * smear if transform == "log" else yhat_tr
    else:
        yhat_level = df_feat["value"].values + (alpha * yhat_tr)

    dates = (df_feat["date"] + pd.Timedelta(days=1)).dt.normalize()
    out = pd.DataFrame({"date": dates, "pred": yhat_level})
    return out.sort_values("date").reset_index(drop=True)

def _recursive_predict(entity: str, days: int):
    """Prediksi multi-step (ke depan). Return list dict {date, pred}."""
    if days <= 0:
        return []
    b = _load_model_for_entity(entity)
    model = b["model"]; feature_cols = b["feature_cols"]
    mode = b["mode"]; transform = b["transform"]; smear = b["smear"]; alpha = b.get("alpha", 1.0)

    dfe = LONG_DF.loc[LONG_DF["entity"] == entity, ["entity","date","value"]].copy()
    preds = []

    for _ in range(days):
        df_feat, _ = make_features_entity(dfe, horizon=1)
        if df_feat.empty:
            raise RuntimeError("Fitur kosong saat inference.")
        x_last_df = df_feat[feature_cols].iloc[[-1]].copy()

        yhat_tr = float(_predict_with_safe_names(model, x_last_df)[0])
        last_date  = df_feat.iloc[-1]["date"]
        last_value = dfe["value"].iloc[-1]

        if mode == "level":
            if transform == "log":
                y_next_level = float(np.exp(yhat_tr) * smear)
            else:
                y_next_level = float(yhat_tr)
        else:
            delta_hat = float(alpha * yhat_tr)
            delta_hat = float(np.clip(delta_hat, -500, 500))
            y_next_level = float(last_value + delta_hat)

        next_date = last_date + pd.Timedelta(days=1)
        preds.append({"date": next_date.date().isoformat(), "pred": round(y_next_level, 4)})

        dfe = pd.concat(
            [dfe, pd.DataFrame([{"entity": entity, "date": next_date, "value": y_next_level}])],
            ignore_index=True
        )
    return preds

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

@app.route("/api/predict_range")
def api_predict_range():
    """
    Query:
      - city  : slug (wajib)
      - start : 'YYYY-MM-DD' (wajib)
      - end   : 'YYYY-MM-DD' (wajib)
      - hide_actual=1|0       (opsional)
      - future_only=1|0       (opsional, jika 1 maka one-step historis tidak dikirim)
      - naive_fallback=1|0    (opsional, jika model tak ada → isi flat = last actual)
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

    dfe = LONG_DF.loc[LONG_DF["entity"] == entity, ["date","value"]].copy()
    if dfe.empty:
        return jsonify({"error": f"data '{entity}' kosong"}), 404
    dfe["date"] = dfe["date"].dt.normalize()

    last_actual_dt = LAST_ACTUAL.get(entity, dfe["date"].max().normalize())

    print(f"DEBUG /api/predict_range entity={entity} start={start_dt.date()} end={end_dt.date()} last_actual={last_actual_dt.date()} future_only={future_only} naive_fallback={naive_fallback}")

    # ===== ACTUAL (hanya sampai last_actual_dt)
    hist_end_for_actual = min(end_dt, last_actual_dt)
    mask_hist = (dfe["date"] >= start_dt) & (dfe["date"] <= hist_end_for_actual)
    actual_series = [{"date": d.date().isoformat(), "value": float(v)}
                     for d, v in zip(dfe.loc[mask_hist, "date"], dfe.loc[mask_hist, "value"])]
    if hide_actual:
        actual_series = []

    predicted_series = []

    # ===== ONE-STEP HISTORIS (skip jika future_only)
    if not future_only:
        try:
            one_step = _one_step_predict_series(entity)
            hist_end = min(end_dt, last_actual_dt)
            mask_hist_pred = (one_step["date"] >= start_dt) & (one_step["date"] <= hist_end)
            for d, v in zip(one_step.loc[mask_hist_pred, "date"], one_step.loc[mask_hist_pred, "pred"]):
                predicted_series.append({"date": d.date().isoformat(), "value": float(v), "pred": float(v)})
            print(f"DEBUG one-step added: {len(predicted_series)}")
        except FileNotFoundError:
            print("DEBUG one-step: no model file")
        except Exception as e:
            print(f"DEBUG one-step error: {e}")
            return jsonify({"error": f"gagal memprediksi (historis): {e}"}), 500

    # ===== FUTURE (recursive) =====
    if end_dt > last_actual_dt:
        anchor = max(last_actual_dt, start_dt - pd.Timedelta(days=1))
        days_need = int((end_dt - anchor).days)
        print(f"DEBUG future anchor={anchor.date()} days_need={days_need}")
        try:
            preds = _recursive_predict(entity, days=days_need)  # [{date, pred}]
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

    predicted_series.sort(key=lambda x: x["date"])
    print(f"DEBUG result actual={len(actual_series)} predicted={len(predicted_series)}")
    return jsonify({
        "city": slug, "entity": entity,
        "range": {"start": start_dt.date().isoformat(), "end": end_dt.date().isoformat()},
        "last_actual": last_actual_dt.date().isoformat(),
        "actual": actual_series,
        "predicted": predicted_series
    })

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

# =======================
# MAIN
# =======================
if __name__ == "__main__":
    print(">> Starting server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
