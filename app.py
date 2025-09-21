# app.py
import os, json, warnings, re
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

# Pastikan file ini ada
DATA_PATH   = BASE_DIR / "data" / "dataset_filled_ffill_bfill.xlsx"
# Folder model .joblib per-kota (opsional; endpoint prediksi bisa jalan tanpa model untuk bagian historis)
MODELS_DIR  = BASE_DIR / "models"

# File JSON pendukung (mapping entity→provinsi dan koordinat kota)
ENTITY_PROV_PATH = BASE_DIR / "static" / "entity_to_province.json"
CITY_COORDS_PATH = BASE_DIR / "static" / "city_coords.json"

# Halaman depan
INDEX_HTML = "index.html"  # letakkan di folder ./static/index.html

# Isi dengan metode yang dipakai saat bikin dataset "filled"
FILL_METHOD = "ffill_bfill"  # atau "interpolate" kalau kamu pakai interpolasi saat persiapan data

# Token untuk reload data (opsional)
RELOAD_TOKEN = os.getenv("RELOAD_TOKEN", "secret123")

# =======================
# MAPPING PROVINSI → PULAU (KUNCI WAJIB KECOCOKAN)
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
    # Wilayah Papua (baru & lama)
    "Papua": "Papua",
    "Papua Barat": "Papua",
    "Papua Barat Daya": "Papua",
    "Papua Tengah": "Papua",
    "Papua Pegunungan": "Papua",
    "Papua Selatan": "Papua",
}

# =======================
# SLUG → NAMA ENTITY (opsional; sisanya pakai heuristik)
# Pastikan nilainya SAMA PERSIS dengan nama kolom yang dislug di dataset (lower + underscore)
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
    "bogor" : "kota_bogor",
    # contoh kabupaten
}

# =======================
# FLASK APP
# =======================
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# =======================
# UTIL DATA
# =======================
def to_numeric_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.replace({"-": np.nan, "": np.nan})
    s = s.str.replace(",", "", regex=False)  # "14,250" -> "14250"
    return pd.to_numeric(s, errors="coerce")

def build_wide_long(path_xlsx: str, fill_method: str):
    """
    Menghasilkan:
      - wide: index = tanggal harian, kolom = entity (dislug), sudah diisi (ffill/bfill)
      - long_df: kolom [entity, date, value] dengan entity = lower+underscore
    """
    raw = pd.read_excel(path_xlsx)
    date_col   = raw.columns[0]
    value_cols = raw.columns[1:]

    # normalisasi tanggal
    raw[date_col] = raw[date_col].astype(str).str.replace(r"\s+", "", regex=True)
    raw[date_col] = pd.to_datetime(raw[date_col], dayfirst=True, errors="coerce")
    raw = raw.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    # angka
    for c in value_cols:
        raw[c] = to_numeric_series(raw[c])

    # grid harian full
    full_idx = pd.date_range(raw[date_col].min(), raw[date_col].max(), freq="D")
    wide = (raw.set_index(date_col)
               .reindex(full_idx)
               .rename_axis("date")
               .sort_index())

    # isi kekosongan
    if fill_method == "ffill_bfill":
        wide[value_cols] = wide[value_cols].ffill().bfill()
    elif fill_method == "interpolate":
        wide[value_cols] = wide[value_cols].interpolate(method="time", limit_direction="both")
        wide[value_cols] = wide[value_cols].ffill().bfill()
    else:
        raise ValueError("FILL_METHOD harus 'ffill_bfill' atau 'interpolate'.")

    # long format + slug kolom
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

    # lags
    lags_short  = [1, 2, 3, 7, 14, 21, 30]
    lags_long   = [60, 90]
    for L in lags_short + lags_long:
        df[f"lag_{L}"] = df["value"].shift(L)

    # rolling (hindari leakage -> shift(1))
    for W in [7, 14, 30]:
        s = df["value"].shift(1).rolling(W, min_periods=3)
        df[f"roll{W}_mean"]   = s.mean()
        df[f"roll{W}_std"]    = s.std()
        df[f"roll{W}_median"] = s.median()
    for W in [30, 60, 90]:
        s = df["value"].shift(1).rolling(W, min_periods=5)
        df[f"roll{W}_min"] = s.min()
        df[f"roll{W}_max"] = s.max()

    # diff
    df["diff1"] = df["value"].diff(1)
    df["diff7"] = df["value"].diff(7)

    # kalender
    df["dayofweek"]  = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"]      = df["date"].dt.month

    # fourier
    add_fourier(df, "date", periods=[7, 365.25], K=2, prefix="fyr")

    # target
    df["y_next"] = df["value"].shift(-1)
    df["y_diff"] = df["y_next"] - df["value"]

    df = df.dropna().reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ["entity","date","value","y_next","y_diff"]]
    return df, feature_cols

def _week_of_month_int(dt: pd.Timestamp) -> int:
    # minggu 1..5 per bulan (sederhana)
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

# =======================
# LOAD DATA SEKALI
# =======================
print(">> Starting app.py")
print(">> Expecting Excel at:", DATA_PATH)
WIDE, LONG_DF = build_wide_long(str(DATA_PATH), FILL_METHOD)
print(">> Dataset loaded:", len(LONG_DF), "rows")
ENTITIES = set(LONG_DF["entity"].unique())

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

# Uppercase-kan kunci PROVINCE_TO_ISLAND agar map tahan ejaan
PROVINCE_TO_ISLAND = {k.upper(): v for k, v in PROVINCE_TO_ISLAND.items()}

# LOG CEK: entitas yang belum termap provinsi atau koordinat
missing_etp = sorted(e for e in ENTITIES if e not in ENTITY_TO_PROVINCE)
if missing_etp:
    print(f"⚠️ Belum terpetakan ke provinsi ({len(missing_etp)}):", ", ".join(missing_etp[:15]), "...")
missing_coords = sorted(e for e in ENTITIES if e not in CITY_COORDS)
if missing_coords:
    print(f"⚠️ Belum ada koordinat kota ({len(missing_coords)}):", ", ".join(missing_coords[:15]), "...")

# =======================
# MODEL CACHE (opsional)
# =======================
_MODEL_CACHE = {}  # entity -> { model, feature_cols, config, metrics, smear, mode, transform }


def _normalize_to_slug(text: str) -> str:
    text = (text or "").strip().lower()
    text =  re.sub(r'^(kota administrasi|kota|kab\.?|kabupaten)\s+', '', text)
    # ganti non-alnum jadi '-'
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')

# def _slug_to_entity(slug: str) -> str:
#     slug = _normalize_to_slug(slug)   # <-- baris penting
#     if slug in CITY_SLUG_TO_ENTITY:
#         return CITY_SLUG_TO_ENTITY[slug]
#     cand = f"kota_{slug.replace('-', '_')}"
#     if cand in ENTITIES: return cand
#     cand = f"kab._{slug.replace('-', '_')}"
#     if cand in ENTITIES: return cand
#     raise ValueError(
#         f"Mapping slug '{slug}' ke entity tidak ditemukan. "
#         f"Tambahkan di CITY_SLUG_TO_ENTITY atau samakan nama kolom Excel-nya."
#     )

def _slug_to_entity(s: str) -> str:
    """
    Terima: entity (kota_*/kab._*), label ('Kota Banjarmasin'), atau slug ('banjarmasin').
    Balikkan: entity yang ada di ENTITIES.
    """
    s0 = (s or "").strip()
    s_lower = s0.lower()

    # 0) kalau sudah entity persis di dataset
    if s_lower in ENTITIES:
        return s_lower

    # 1) normalisasi ke slug
    slug = _normalize_to_slug(s0)

    # 2) cek override manual (boleh kosong / sebagian)
    if slug in CITY_SLUG_TO_ENTITY:
        ent = CITY_SLUG_TO_ENTITY[slug]
        if ent in ENTITIES:
            return ent

    # 3) heuristik cocok nama kolom dataset
    cand = f"kota_{slug.replace('-', '_')}"
    if cand in ENTITIES:
        return cand
    cand = f"kab._{slug.replace('-', '_')}"
    if cand in ENTITIES:
        return cand

    # 4) cocokkan dengan label di CITY_COORDS
    for ent, meta in CITY_COORDS.items():
        label = (meta.get("label") or ent)
        if _normalize_to_slug(label) == slug and ent in ENTITIES:
            return ent

    raise ValueError(
        f"Mapping untuk '{s}' tidak ditemukan. "
        f"Pastikan label ada di city_coords.json atau sesuaikan nama kolom Excel."
    )
def _load_model_for_entity(entity: str):
    """Load artefak model .joblib dan hitung smearing (jika level+log)."""
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
        "alpha": alpha,  # <--- tambahkan ini

    }
    print(f">> Loaded model for {entity} | smear={smear:.6f} | mode={mode}/{transform}")
    return _MODEL_CACHE[entity]

def _one_step_predict_series(entity: str) -> pd.DataFrame:
    """
    Prediksi one-step-ahead untuk keseluruhan histori (tanpa kebocoran).
    Output: DataFrame [date (target t+1, normalized), pred (level)].
    """
    bundle = _load_model_for_entity(entity)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    mode = bundle["mode"]
    transform = bundle["transform"]
    smear = bundle["smear"]
    alpha = bundle.get("alpha", 1.0)  # <--- baru


    dfe = LONG_DF.loc[LONG_DF["entity"] == entity, ["entity","date","value"]].copy()
    df_feat, _ = make_features_entity(dfe, horizon=1)

    yhat_tr = model.predict(df_feat[feature_cols].values)
    if mode == "level":
        yhat_level = np.exp(yhat_tr) * smear if transform == "log" else yhat_tr
    else:
        yhat_level = df_feat["value"].values + (alpha * yhat_tr)  # <--- pakai alpha

    dates = (df_feat["date"] + pd.Timedelta(days=1)).dt.normalize()
    out = pd.DataFrame({"date": dates, "pred": yhat_level})
    return out.sort_values("date").reset_index(drop=True)
def _recursive_predict(entity: str, days: int):
    """Prediksi multi-step (berantai) -> skala LEVEL."""
    bundle = _load_model_for_entity(entity)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    mode = bundle["mode"]
    transform = bundle["transform"]
    smear = bundle["smear"]
    alpha = bundle.get("alpha", 1.0)  # ← ambil alpha_blend (default 1.0)

    dfe = LONG_DF.loc[LONG_DF["entity"] == entity, ["entity","date","value"]].copy()
    preds = []

    for _ in range(days):
        df_feat, _ = make_features_entity(dfe, horizon=1)
        if df_feat.empty:
            raise RuntimeError("Fitur kosong saat inference.")

        x_last = df_feat.iloc[-1][feature_cols].values.reshape(1, -1)
        last_date  = df_feat.iloc[-1]["date"]
        last_value = dfe["value"].iloc[-1]

        yhat_tr = float(model.predict(x_last)[0])
        if mode == "level":
            if transform == "log":
                y_next_level = float(np.exp(yhat_tr) * smear)
            else:
                y_next_level = float(yhat_tr)
        else:
            # MODE 'diff' → y_t = y_{t-1} + alpha * Δ̂_t
            delta_hat = alpha * yhat_tr
            # optional clamp biar tidak ekstrem (sesuaikan kalau mau)
            delta_hat = float(np.clip(delta_hat, -500, 500))
            y_next_level = float(last_value + delta_hat)

        next_date = last_date + pd.Timedelta(days=1)
        preds.append({"date": next_date.date().isoformat(), "pred": round(y_next_level, 4)})

        # append untuk langkah berikutnya
        dfe = pd.concat(
            [dfe, pd.DataFrame([{"entity": entity, "date": next_date, "value": y_next_level}])],
            ignore_index=True
        )

    return preds

# =======================
# ROUTES
# =======================
MONTH_NAMES = {
    1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"Mei", 6:"Jun",
    7:"Jul", 8:"Agu", 9:"Sep", 10:"Okt", 11:"Nov", 12:"Des"
}

@app.route("/")
def index():
    return send_from_directory(app.static_folder, INDEX_HTML)

@app.route("/api/islands")
def api_islands():
    islands = ["Semua Pulau","Jawa","Sumatra","Kalimantan","Sulawesi","Bali–NT","Maluku","Papua"]
    return jsonify(islands)

@app.route("/api/cities_full")
def api_cities_full():
    """
    Daftar kota dari CITY_COORDS / ENTITIES:
    [{entity, slug, label}]
    """
    out = []
    source = CITY_COORDS if CITY_COORDS else {e: {} for e in ENTITIES}
    for ent in sorted(source.keys()):
        meta  = CITY_COORDS.get(ent, {})
        label = meta.get("label") or ent.replace("_", " ").title().replace("Kab. ", "Kabupaten ")
        slug  = _normalize_to_slug(label)
        out.append({"entity": ent, "slug": slug, "label": label})
    # urutkan berdasarkan label
    out.sort(key=lambda x: x["label"].lower())
    return jsonify(out)

@app.route("/api/cities")
def api_cities():
    """
    Untuk kompatibilitas lama: kembalikan list label saja (diambil dari CITY_COORDS).
    Saran: gunakan /api/cities_full di FE.
    """
    if CITY_COORDS:
        labels = []
        for ent, meta in CITY_COORDS.items():
            label = meta.get("label") or ent.replace("_", " ").title()
            labels.append(label)
        labels = sorted(set(labels), key=lambda x: x.lower())
        return jsonify(labels)
    # fallback: generate dari ENTITIES
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
    """
    Ambil harga untuk kota & tanggal tertentu.
    - Jika tanggal ada di historis -> status=actual
    - Jika tanggal di depan historis -> status=predicted (butuh model)
    """
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

    last_dt = dfe["date"].max().normalize()
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

    if target_date > last_dt:
        try:
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
        except FileNotFoundError:
            return jsonify({"error": "model untuk kota ini belum tersedia"}), 404
        except Exception as e:
            return jsonify({"error": f"gagal memprediksi: {e}"}), 500

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
    Aggregasi rata-rata per provinsi untuk periode filter.
    Query:
      - island : "Semua Pulau" | "Jawa" | ...
      - year   : wajib (int)
      - month  : opsional (1..12)
      - week   : opsional (1..5), aktif jika month ada
    Return:
      { last_actual, buckets:{low, mid}, data:[{province, value, category}] }
    """
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

    # map entity -> province -> island
    df["province"] = df["entity"].map(ENTITY_TO_PROVINCE)
    df = df.dropna(subset=["province"]).copy()
    df["island"]   = df["province"].str.upper().map(PROVINCE_TO_ISLAND)

    if island and island != "Semua Pulau":
        df = df[df["island"] == island]
    if df.empty:
        return jsonify({"data": [], "buckets": None, "last_actual": None})

    # agregasi ke provinsi (mean)
    grp = df.groupby("province")["value"].mean().reset_index()
    vals = grp["value"].values
    q1, q2 = np.quantile(vals, [1/3, 2/3])  # bucket terciles

    def cat(v):
        if v <= q1: return "low"
        if v <= q2: return "mid"
        return "high"

    data = [{"province": r.province, "value": float(r.value), "category": cat(r.value)}
            for r in grp.itertuples(index=False)]
    last_actual = str(LONG_DF["date"].max().date())
    return jsonify({
        "last_actual": last_actual,
        "buckets": {"low": float(q1), "mid": float(q2)},
        "data": data
    })

@app.route("/api/city_points")
def api_city_points():
    """
    Titik kota di peta beranda, warna: low/mid/high dibanding mean periode.
    Query:
      - island : "Semua Pulau" | "Jawa" | ...
      - year   : wajib (int)
      - month  : opsional 1..12
      - week   : opsional 1..5 (aktif kalau month ada)
      - band   : opsional float (default 0.05 -> ±5%)
    """
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
        if band < 0 or band > 0.5:
            band = 0.05
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

    # Tambah province & island
    df["province"] = df["entity"].map(ENTITY_TO_PROVINCE)
    df = df.dropna(subset=["province"]).copy()
    df["island"] = df["province"].str.upper().map(PROVINCE_TO_ISLAND)

    if island and island != "Semua Pulau":
        df = df[df["island"] == island]
    if df.empty:
        return jsonify({"last_actual": None, "mean_ref": None, "band_pct": band, "points": []})

    # Rata-rata per kota untuk periode
    agg = (
        df.groupby(["entity", "province", "island"])["value"]
          .mean()
          .reset_index()
          .rename(columns={"value": "avg_value"})
    )

    # Mean referensi + ambang
    mean_ref = float(agg["avg_value"].mean())
    lo = mean_ref * (1 - band)
    hi = mean_ref * (1 + band)

    def cat(val: float) -> str:
        if val <= lo: return "low"   # hijau
        if val >= hi: return "high"  # merah
        return "mid"                 # kuning

    # Gabungkan koordinat dari CITY_COORDS
    points = []
    for r in agg.itertuples(index=False):
        coords = CITY_COORDS.get(r.entity)
        if not coords:
            continue
        lat = coords.get("lat"); lng = coords.get("lng")
        if lat is None or lng is None:
            continue
        points.append({
            "entity": r.entity,
            "label": coords.get("label") or r.entity.replace("_", " ").title(),
            "province": r.province,
            "island": r.island,
            "value": float(r.avg_value),
            "lat": float(lat),
            "lng": float(lng),
            "category": cat(float(r.avg_value)),
        })

    last_actual = str(LONG_DF["date"].max().date())
    return jsonify({
        "last_actual": last_actual,
        "mean_ref": mean_ref,
        "band_pct": band,
        "points": points
    })

@app.route("/api/trend")
def api_trend():
    """
    Query:
      - city   (slug UI, wajib)  -> contoh: banjarmasin
      - year   (wajib)           -> contoh: 2023
      - month  (opsional)        -> 1..12
      - week   (opsional)        -> 1..5 (aktif jika month terisi)
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

    # parse month & week
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

    # CASE 1: YEARLY -> seri bulanan
    if month is None:
        df["mon"] = df["date"].dt.month
        grp = df.groupby("mon")["value"].mean().reset_index().sort_values("mon")
        series = [{"label": MONTH_NAMES.get(int(r.mon), str(int(r.mon))), "value": float(r.value)}
                  for r in grp.itertuples(index=False)]
        stats = _basic_stats(df["value"].values)
        return jsonify({
            "city": slug, "entity": entity, "year": year,
            "month": None, "week": None, "granularity": "yearly",
            "series": series, "stats": stats
        })

    # CASE 2: MONTHLY -> seri mingguan (1..5)
    dfm = df[df["date"].dt.month == month].copy()
    if dfm.empty:
        return jsonify({"city": slug, "entity": entity, "year": year,
                        "month": month, "granularity": "monthly", "series": [], "stats": _basic_stats([])})

    if week is None:
        dfm["week_in_month"] = dfm["date"].apply(_week_of_month_int)
        grp = dfm.groupby("week_in_month")["value"].mean().reset_index().sort_values("week_in_month")
        series = [{"label": f"Minggu {int(r.week_in_month)}", "value": float(r.value)}
                  for r in grp.itertuples(index=False)]
        stats = _basic_stats(dfm["value"].values)
        return jsonify({
            "city": slug, "entity": entity, "year": year,
            "month": month, "week": None, "granularity": "monthly",
            "series": series, "stats": stats
        })

    # CASE 3: DAILY -> harian dalam minggu tsb
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
    except FileNotFoundError:
        return jsonify({"error": "model untuk kota ini belum tersedia"}), 404
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
      - city  : slug (wajib)
      - start : 'YYYY-MM-DD' (wajib)
      - end   : 'YYYY-MM-DD' (wajib)
    Output:
      { actual:[], predicted:[] }  # predicted = one-step historis + recursive future (jika ada model)
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

    dfe = LONG_DF.loc[LONG_DF["entity"] == entity, ["date","value"]].copy()
    if dfe.empty:
        return jsonify({"error": f"data '{entity}' kosong"}), 404

    dfe["date"] = dfe["date"].dt.normalize()
    last_actual_dt = dfe["date"].max()

    # Seri aktual
    mask_hist = (dfe["date"] >= start_dt) & (dfe["date"] <= end_dt)
    actual_series = [
        {"date": d.date().isoformat(), "value": float(v)}
        for d, v in zip(dfe.loc[mask_hist, "date"], dfe.loc[mask_hist, "value"])
    ]

    predicted_series = []

    # One-step historis
    try:
        one_step = _one_step_predict_series(entity)  # butuh model
        hist_end = min(end_dt, last_actual_dt)
        mask_hist_pred = (one_step["date"] >= start_dt) & (one_step["date"] <= hist_end)
        for d, v in zip(one_step.loc[mask_hist_pred, "date"], one_step.loc[mask_hist_pred, "pred"]):
            predicted_series.append({"date": d.date().isoformat(), "value": float(v)})
    except FileNotFoundError:
        # tidak ada model: bagian historis predicted dikosongkan (tetap ok)
        pass
    except Exception as e:
        return jsonify({"error": f"gagal memprediksi (historis): {e}"}), 500

    # Recursive ke depan
    if end_dt > last_actual_dt:
        try:
            days_need = int((end_dt - last_actual_dt).days)
            preds = _recursive_predict(entity, days=days_need)  # [{date, pred}]
            for p in preds:
                p_dt = pd.to_datetime(p["date"]).normalize()
                if (p_dt >= start_dt) and (p_dt <= end_dt):
                    predicted_series.append({"date": p_dt.date().isoformat(), "value": float(p["pred"])})
        except FileNotFoundError:
            pass
        except Exception as e:
            return jsonify({"error": f"gagal memprediksi (future): {e}"}), 500

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
    global WIDE, LONG_DF, ENTITIES, _MODEL_CACHE, ENTITY_TO_PROVINCE, CITY_COORDS
    WIDE, LONG_DF = build_wide_long(str(DATA_PATH), FILL_METHOD)
    ENTITIES = set(LONG_DF["entity"].unique())
    _MODEL_CACHE.clear()
    ENTITY_TO_PROVINCE = _safe_load_json(ENTITY_PROV_PATH, {})
    CITY_COORDS        = _safe_load_json(CITY_COORDS_PATH, {})
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
