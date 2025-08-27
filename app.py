import os, json, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
import joblib

# =======================
# KONFIGURASI DASAR
# =======================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset_filled_ffill_bfill.xlsx"     # <-- pastikan file ada
MODELS_DIR = BASE_DIR / "models"                                       # <-- taruh .joblib di sini
FILL_METHOD = "ffill_bfill"                                            # harus sama dg saat training
RELOAD_TOKEN = os.getenv("RELOAD_TOKEN", "secret123")                  # guard untuk /api/reload

# Mapping dropdown <select> -> nama kolom entity di dataset (silakan sesuaikan)
# app.py (di atas, dekat CITY_SLUG_TO_ENTITY)
PROVINCE_TO_ISLAND = {
    "DKI Jakarta": "Jawa",
    "Jawa Barat": "Jawa",
    "Jawa Tengah": "Jawa",
    "DI Yogyakarta": "Jawa",
    "Jawa Timur": "Jawa",
    "Banten": "Jawa",
    "Sumatera Utara": "Sumatra",
    "Sumatera Barat": "Sumatra",
    "Riau": "Sumatra",
    "Kepulauan Riau": "Sumatra",
    "Aceh": "Sumatra",
    "Sumatera Selatan": "Sumatra",
    "Bangka Belitung": "Sumatra",
    "Jambi": "Sumatra",
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
    "Papua Tengah": "Papua",
    "Papua Pegunungan": "Papua",
    "Papua Selatan": "Papua",
    "Papua Barat Daya": "Papua"
}
# ENTITY_TO_PROVINCE: simpan di file JSON 'static/entity_to_province.json'
# Contoh beberapa baris:
# {
#   "kota_banjarmasin": "Kalimantan Selatan",
#   "kota_surabaya": "Jawa Timur",
#   "kab._biak_numfor": "Papua"
# }

CITY_SLUG_TO_ENTITY = {
    "banjarmasin": "kota_banjarmasin",
    "jakarta-pusat": "kota_administrasi_jakarta_pusat",
    "jakarta-selatan": "kota_administrasi_jakarta_selatan",
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
    "biak": "kab._biak_numfor",  # sesuaikan jika beda di dataset
}

# =======================
# FLASK APP
# =======================
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

ENTITY_PROV_PATH = BASE_DIR / "static" / "entity_to_province.json"

with open(ENTITY_PROV_PATH, "r", encoding="utf-8") as f:
    ENTITY_TO_PROVINCE = json.load(f)

# =======================
# UTIL DATA (sesuai training)
# =======================
def to_numeric_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.replace({"-": np.nan, "": np.nan})
    s = s.str.replace(",", "", regex=False)  # "14,250" -> "14250"
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
    wide = (raw.set_index(date_col)
               .reindex(full_idx)
               .rename_axis("date")
               .sort_index())

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

def add_fourier(df: pd.DataFrame, date_col: str, periods, K=2, prefix="fyr"):
    t = (df[date_col] - df[date_col].min()).dt.days.astype(float)
    for P in periods:
        for k in range(1, K+1):
            df[f"{prefix}_sin_P{P}_k{k}"] = np.sin(2*np.pi*k*t/P)
            df[f"{prefix}_cos_P{P}_k{k}"] = np.cos(2*np.pi*k*t/P)

def make_features_entity(dfe: pd.DataFrame, horizon=1):
    df = dfe.sort_values("date").copy()

    # Lags (ringan)
    lags_short  = [1, 2, 3, 7, 14, 21, 30]
    lags_long   = [60, 90]
    for L in lags_short + lags_long:
        df[f"lag_{L}"] = df["value"].shift(L)

    # Rolling (hindari leakage)
    for W in [7, 14, 30]:
        s = df["value"].shift(1).rolling(W, min_periods=3)
        df[f"roll{W}_mean"]   = s.mean()
        df[f"roll{W}_std"]    = s.std()
        df[f"roll{W}_median"] = s.median()
    for W in [30, 60, 90]:
        s = df["value"].shift(1).rolling(W, min_periods=5)
        df[f"roll{W}_min"] = s.min()
        df[f"roll{W}_max"] = s.max()

    # Diff features
    df["diff1"] = df["value"].diff(1)
    df["diff7"] = df["value"].diff(7)

    # Kalender
    df["dayofweek"]  = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"]      = df["date"].dt.month

    # Fourier
    add_fourier(df, "date", periods=[7, 365.25], K=2, prefix="fyr")

    # Targets
    df["y_next"] = df["value"].shift(-1)
    df["y_diff"] = df["y_next"] - df["value"]

    df = df.dropna().reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ["entity","date","value","y_next","y_diff"]]
    return df, feature_cols

# =======================
# LOAD DATA SEKALI
# =======================
print(">> Starting app.py")
print(">> Expecting Excel at:", DATA_PATH)
WIDE, LONG_DF = build_wide_long(str(DATA_PATH), FILL_METHOD)
print(">> Dataset loaded:", len(LONG_DF), "rows")
ENTITIES = set(LONG_DF["entity"].unique())

# =======================
# MODEL CACHE
# =======================
_MODEL_CACHE = {}  # entity -> { model, feature_cols, config, metrics, smear, mode, transform }

def _slug_to_entity(slug: str) -> str:
    slug = slug.strip().lower()
    if slug in CITY_SLUG_TO_ENTITY:
        return CITY_SLUG_TO_ENTITY[slug]
    # fallback heuristik
    cand = f"kota_{slug.replace('-', '_')}"
    if cand in ENTITIES:
        return cand
    cand = f"kab._{slug.replace('-', '_')}"
    if cand in ENTITIES:
        return cand
    raise ValueError(f"Mapping slug '{slug}' ke entity tidak ditemukan. Tambahkan di CITY_SLUG_TO_ENTITY.")

def _load_model_for_entity(entity: str):
    """Load artefak model .joblib dan hitung smearing (jika log-level) dari TRAIN subset."""
    if entity in _MODEL_CACHE:
        return _MODEL_CACHE[entity]

    files = sorted(MODELS_DIR.glob(f"{entity}*.joblib"))
    if not files:
        raise FileNotFoundError(f"Model file untuk '{entity}' tidak ditemukan di {MODELS_DIR}")
    pack = joblib.load(files[0])

    model = pack["model"]
    feature_cols = pack["feature_cols"]
    best_cfg = pack["best_config"]     # berisi: mode, transform, train_until, dst
    metrics = pack["metrics"]

    dfe = LONG_DF.loc[LONG_DF["entity"] == entity, ["entity","date","value"]].copy()
    df_feat, _ = make_features_entity(dfe, horizon=1)

    train_until = pd.to_datetime(best_cfg["train_until"])
    tr_mask = df_feat["date"] <= train_until
    df_train = df_feat.loc[tr_mask].copy()
    if df_train.empty:
        raise RuntimeError("TRAIN subset kosong saat reconstruct untuk smearing.")

    mode = best_cfg["mode"]
    transform = best_cfg["transform"]
    use_log = (mode == "level" and transform == "log")

    if mode == "level":
        y_train = df_train["y_next"].values
        y_train_tr = np.log(y_train) if use_log else y_train
    else:
        y_train = df_train["y_diff"].values
        y_train_tr = y_train

    X_train = df_train[feature_cols].values

    if use_log:
        yhat_tr = model.predict(X_train)
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
    }
    print(f">> Loaded model for {entity} | smear={smear:.6f} | mode={mode}/{transform}")
    return _MODEL_CACHE[entity]
# ========= Helper agregasi tren =========
def _week_of_month(d: pd.Timestamp) -> int:
    # minggu ke-1: tgl 1-7, ke-2: 8-14, dst (sederhana & stabil)
    return int(((d.day - 1) // 7) + 1)

def _trend_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if len(values) == 0: 
        return {"min": None, "max": None, "mean": None, "std": None, "vol_pct": None, "n": 0}
    mean = float(values.mean())
    std  = float(values.std(ddof=0))
    vol  = float((std / mean) * 100) if mean != 0 else None
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": mean,
        "std": std,
        "vol_pct": vol,
        "n": int(len(values))
    }
# --- tambahkan util ini (di dekat _recursive_predict) ---
def _one_step_predict_series(entity: str) -> pd.DataFrame:
    """
    Prediksi one-step-ahead untuk SELURUH sejarah (tanpa kebocoran).
    Output: DataFrame [date (target t+1, normalized), pred (level)] untuk tanggal <= last_actual.
    """
    bundle = _load_model_for_entity(entity)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    mode = bundle["mode"]
    transform = bundle["transform"]
    smear = bundle["smear"]

    dfe = LONG_DF.loc[LONG_DF["entity"] == entity, ["entity","date","value"]].copy()
    df_feat, _ = make_features_entity(dfe, horizon=1)

    yhat_tr = model.predict(df_feat[feature_cols].values)
    if mode == "level":
        yhat_level = np.exp(yhat_tr) * smear if transform == "log" else yhat_tr
    else:
        yhat_level = df_feat["value"].values + yhat_tr

    # target date = t+1
    dates = (df_feat["date"] + pd.Timedelta(days=1)).dt.normalize()
    out = pd.DataFrame({"date": dates, "pred": yhat_level})
    return out.sort_values("date").reset_index(drop=True)

def _recursive_predict(entity: str, days: int):
    """Prediksi multi-step (horizon=1 beruntun) -> skala LEVEL."""
    bundle = _load_model_for_entity(entity)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    mode = bundle["mode"]
    transform = bundle["transform"]
    smear = bundle["smear"]

    dfe = LONG_DF.loc[LONG_DF["entity"] == entity, ["entity","date","value"]].copy()
    preds = []

    for _ in range(days):
        df_feat, _ = make_features_entity(dfe, horizon=1)
        if df_feat.empty:
            raise RuntimeError("Fitur kosong saat inference.")

        x_last = df_feat.iloc[-1][feature_cols].values.reshape(1, -1)
        last_date = df_feat.iloc[-1]["date"]
        last_value = dfe["value"].iloc[-1]

        yhat_tr = model.predict(x_last)[0]
        if mode == "level":
            if transform == "log":
                y_next_level = float(np.exp(yhat_tr) * smear)
            else:
                y_next_level = float(yhat_tr)
        else:
            y_next_level = float(last_value + yhat_tr)

        next_date = last_date + pd.Timedelta(days=1)
        preds.append({"date": next_date.date().isoformat(), "pred": round(y_next_level, 4)})

        # append untuk langkah berikutnya
        dfe = pd.concat([
            dfe,
            pd.DataFrame([{"entity": entity, "date": next_date, "value": y_next_level}])
        ], ignore_index=True)

    return preds

# =======================
# ROUTES
# =======================
from datetime import datetime  # sudah ada di atas
@app.route("/api/islands")
def api_islands():
    # Nama pulau yang dipakai di UI
    islands = ["Semua Pulau","Jawa","Sumatra","Kalimantan","Sulawesi","Bali–NT","Maluku","Papua"]
    return jsonify(islands)

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

    # ambil data historis kota tsb
    dfe = LONG_DF.loc[LONG_DF["entity"] == entity, ["date","value"]].copy()
    if dfe.empty:
        return jsonify({"error": "data kota kosong"}), 400

    last_dt = dfe["date"].max().normalize()

    # Jika tanggal yang diminta masih di historis -> kembalikan nilai aktual
    row = dfe.loc[dfe["date"] == target_date]
    if not row.empty:
        val = float(row["value"].iloc[0])
        return jsonify({
            "city": slug,
            "entity": entity,
            "status": "actual",
            "date": target_date.date().isoformat(),
            "value": val
        })

    # Jika future -> prediksi sampai tanggal tsb
    if target_date > last_dt:
        days_ahead = int((target_date - last_dt).days)
        preds = _recursive_predict(entity, days=days_ahead)
        if not preds:
            return jsonify({"error": "gagal memprediksi"}), 500
        val = preds[-1]["pred"]
        return jsonify({
            "city": slug,
            "entity": entity,
            "status": "predicted",
            "from_last_actual": last_dt.date().isoformat(),
            "date": target_date.date().isoformat(),
            "steps": days_ahead,
            "value": val
        })

    # tanggal di masa lalu tapi tidak tepat pada index (mis. lubang kalender)
    return jsonify({
        "city": slug,
        "entity": entity,
        "status": "no_data",
        "date": target_date.date().isoformat(),
        "message": "tanggal tidak ada di dataset (bukan titik observasi)"
    })

@app.route("/api/choropleth")
def api_choropleth():
    """
    Query:
      - island : "Semua Pulau" | "Jawa" | ...
      - year   : wajib (int)
      - month  : opsional (1..12)
      - week   : opsional (1..5), aktif jika month ada
    Return:
      { last_actual, buckets:{low, mid, high}, data:[{province, value, category}] }
    """
    island = request.args.get("island","Semua Pulau").strip()
    year   = request.args.get("year","").strip()
    month  = request.args.get("month","").strip()
    week   = request.args.get("week","").strip()

    if not year: return jsonify({"error":"param ?year= wajib"}), 400
    try:
        year = int(year)
        month = int(month) if month else None
        week  = int(week)  if week  else None
    except:
        return jsonify({"error":"format year/month/week tidak valid"}), 400

    # siapkan data harian
    df = LONG_DF[["entity","date","value"]].copy()
    df["date"] = df["date"].dt.normalize()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week_in_month"] = df["date"].apply(lambda d: int(((d.day - 1)//7)+1))

    # filter waktu
    df = df[df["year"]==year]
    if month: df = df[df["month"]==month]
    if week and month: df = df[df["week_in_month"]==week]

    if df.empty:
        return jsonify({"data":[],"buckets":None,"last_actual":None})

    # map entity -> province -> island
    df["province"] = df["entity"].map(ENTITY_TO_PROVINCE)
    df = df.dropna(subset=["province"]).copy()
    df["island"]   = df["province"].map(PROVINCE_TO_ISLAND)

    if island and island!="Semua Pulau":
        df = df[df["island"]==island]

    if df.empty:
        return jsonify({"data":[],"buckets":None,"last_actual":None})

    # agregasi ke provinsi (mean dari kab/kota pada provinsi tsb)
    grp = df.groupby("province")["value"].mean().reset_index()
    vals = grp["value"].values
    # bikin 3 bucket: low / mid / high (tertiles)
    q1, q2 = np.quantile(vals, [1/3, 2/3])
    def cat(v):
        if v <= q1: return "low"
        if v <= q2: return "mid"
        return "high"

    data = [{"province": r.province, "value": float(r.value), "category": cat(r.value)} for r in grp.itertuples(index=False)]

    last_actual = str(LONG_DF["date"].max().date())
    return jsonify({
        "last_actual": last_actual,
        "buckets": {"low": float(q1), "mid": float(q2)},
        "data": data
    })

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/cities")
def api_cities():
    return jsonify(sorted(list(CITY_SLUG_TO_ENTITY.keys())))

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
MONTH_NAMES = {
    1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"Mei", 6:"Jun",
    7:"Jul", 8:"Agu", 9:"Sep", 10:"Okt", 11:"Nov", 12:"Des"
}

def _week_of_month_int(dt: pd.Timestamp) -> int:
    # Minggu ke-n dalam bulan versi sederhana: 1..5
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
        # Volatilitas sederhana: std/mean*100 dari data harian (bukan agregat)
        "vol_pct": float(np.std(arr, ddof=0) / np.mean(arr) * 100.0) if np.mean(arr) != 0 else None
    }

@app.route("/api/trend")
def api_trend():
    """
    Query:
      - city   (slug UI, wajib)  -> contoh: banjarmasin
      - year   (wajib)           -> contoh: 2023
      - month  (opsional)        -> 1..12 atau kosong
      - week   (opsional)        -> 1..5 atau kosong (aktif jika month terisi)
    Output:
      { city, entity, year, month, week, granularity, series: [{label, value}], stats:{...} }
    """
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

    # map slug -> entity kolom data
    try:
        entity = _slug_to_entity(slug)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # ambil data harian si kota
    df = LONG_DF.loc[LONG_DF["entity"] == entity, ["date", "value"]].copy()
    if df.empty:
        return jsonify({"error": f"data untuk '{entity}' kosong"}), 404

    # filter tahun
    df = df[df["date"].dt.year == year]
    if df.empty:
        return jsonify({"city": slug, "entity": entity, "year": year,
                        "granularity": "yearly", "series": [], "stats": _basic_stats([])})

    # parse month & week (opsional)
    month = None
    if month_str:
        try:
            m = int(month_str)
            if 1 <= m <= 12:
                month = m
            else:
                return jsonify({"error": "param ?month= harus 1..12"}), 400
        except:
            return jsonify({"error": "param ?month= tidak valid"}), 400

    week = None
    if week_str:
        try:
            w = int(week_str)
            if 1 <= w <= 5:
                week = w
            else:
                return jsonify({"error": "param ?week= harus 1..5"}), 400
        except:
            return jsonify({"error": "param ?week= tidak valid"}), 400

    # ====== CASE 1: YEARLY (bulanan, jika month tidak dipilih) ======
    if month is None:
        # Seri = rata-rata per bulan, pakai data harian
        df["mon"] = df["date"].dt.month
        grp = df.groupby("mon")["value"].mean().reset_index()
        grp = grp.sort_values("mon")
        series = [{"label": MONTH_NAMES.get(int(r.mon), str(int(r.mon))), "value": float(r.value)} for r in grp.itertuples(index=False)]

        stats = _basic_stats(df["value"].values)  # statistik dari data harian sepanjang tahun
        return jsonify({
            "city": slug, "entity": entity, "year": year,
            "month": None, "week": None, "granularity": "yearly",
            "series": series, "stats": stats
        })

    # ====== CASE 2: MONTHLY (mingguan, jika month dipilih tapi week tidak) ======
    dfm = df[df["date"].dt.month == month].copy()
    if dfm.empty:
        return jsonify({"city": slug, "entity": entity, "year": year,
                        "month": month, "granularity": "monthly", "series": [], "stats": _basic_stats([])})

    if week is None:
        # Seri = rata-rata per 'minggu-ke' (1..5) dalam bulan tsb
        dfm["week_in_month"] = dfm["date"].apply(_week_of_month_int)
        grp = dfm.groupby("week_in_month")["value"].mean().reset_index()
        grp = grp.sort_values("week_in_month")
        series = [{"label": f"Minggu {int(r.week_in_month)}", "value": float(r.value)} for r in grp.itertuples(index=False)]
        stats = _basic_stats(dfm["value"].values)  # stats dari data harian dalam bulan
        return jsonify({
            "city": slug, "entity": entity, "year": year,
            "month": month, "week": None, "granularity": "monthly",
            "series": series, "stats": stats
        })

    # ====== CASE 3: DAILY (jika month & week dipilih) ======
    dfm["week_in_month"] = dfm["date"].apply(_week_of_month_int)
    dfd = dfm[dfm["week_in_month"] == week].copy().sort_values("date")
    series = [{"label": d.date().isoformat(), "value": float(v)} for d, v in zip(dfd["date"], dfd["value"])]
    stats = _basic_stats(dfd["value"].values)
    return jsonify({
        "city": slug, "entity": entity, "year": year,
        "month": month, "week": week, "granularity": "daily",
        "series": series, "stats": stats
    })
@app.route("/api/metrics")
def api_metrics():
    slug = request.args.get("city", "").strip().lower()
    if not slug:
        return jsonify({"error": "param ?city= wajib"}), 400
    try:
        entity = _slug_to_entity(slug)
        pack = _load_model_for_entity(entity)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({
        "city": slug,
        "entity": entity,
        "best_config": pack["config"],
        "metrics": pack["metrics"]
    })

@app.route("/api/predict_range")
def api_predict_range():
    """
    Query:
      - city  : slug, contoh 'banjarmasin' (wajib)
      - start : 'YYYY-MM-DD' (wajib)
      - end   : 'YYYY-MM-DD' (wajib)
    Output:
      {
        city, entity, range:{start,end}, last_actual,
        actual:    [{date,value}],
        predicted: [{date,value}]   # one-step utk historis + recursive utk masa depan
      }
    """
    slug = request.args.get("city", "").strip().lower()
    start_str = request.args.get("start", "").strip()
    end_str   = request.args.get("end", "").strip()

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

    # Data historis kota
    dfe = LONG_DF.loc[LONG_DF["entity"] == entity, ["date","value"]].copy()
    if dfe.empty:
        return jsonify({"error": f"data '{entity}' kosong"}), 404

    dfe["date"] = dfe["date"].dt.normalize()
    last_actual_dt = dfe["date"].max()

    # --- seri aktual dalam rentang ---
    mask_hist = (dfe["date"] >= start_dt) & (dfe["date"] <= end_dt)
    actual_series = [
        {"date": d.date().isoformat(), "value": float(v)}
        for d, v in zip(dfe.loc[mask_hist, "date"], dfe.loc[mask_hist, "value"])
    ]

    # --- seri prediksi (historis one-step + future recursive) ---
    predicted_series = []

    # 1) One-step untuk tanggal ≤ last_actual (backtest tanpa kebocoran)
    try:
        one_step = _one_step_predict_series(entity)  # kolom: date (t+1), pred
        hist_end = min(end_dt, last_actual_dt)
        mask_hist_pred = (one_step["date"] >= start_dt) & (one_step["date"] <= hist_end)
        for d, v in zip(one_step.loc[mask_hist_pred, "date"], one_step.loc[mask_hist_pred, "pred"]):
            predicted_series.append({"date": d.date().isoformat(), "value": float(v)})
    except FileNotFoundError:
        # tidak ada model untuk kota ini -> biarkan predicted kosong
        pass
    except Exception as e:
        return jsonify({"error": f"gagal memprediksi (historis): {e}"}), 500

    # 2) Recursive untuk tanggal > last_actual (forecast ke depan)
    if end_dt > last_actual_dt:
        try:
            days_need = int((end_dt - last_actual_dt).days)  # dari last_actual+1 s/d end_dt
            preds = _recursive_predict(entity, days=days_need)  # [{date, pred}]
            for p in preds:
                p_dt = pd.to_datetime(p["date"]).normalize()
                if (p_dt >= start_dt) and (p_dt <= end_dt):
                    predicted_series.append({"date": p_dt.date().isoformat(), "value": float(p["pred"])})
        except FileNotFoundError:
            pass
        except Exception as e:
            return jsonify({"error": f"gagal memprediksi (future): {e}"}), 500

    # Kembalikan hasil
    predicted_series.sort(key=lambda x: x["date"])
    return jsonify({
        "city": slug,
        "entity": entity,
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
    global WIDE, LONG_DF, ENTITIES, _MODEL_CACHE
    WIDE, LONG_DF = build_wide_long(str(DATA_PATH), FILL_METHOD)
    ENTITIES = set(LONG_DF["entity"].unique())
    _MODEL_CACHE.clear()
    return jsonify({
        "ok": True,
        "entities": sorted(list(ENTITIES)),
        "rows": int(len(LONG_DF))
    })

# =======================
# MAIN
# =======================
if __name__ == "__main__":
    print(">> Starting server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
