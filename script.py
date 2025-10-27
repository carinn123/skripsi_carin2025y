
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import logging
import pandas as pd
import calendar
import time
import hashlib
import sys
from app import LONG_DF, LAST_ACTUAL, ENTITY_TO_PROVINCE, PROVINCE_TO_ISLAND, _recursive_predict, _week_of_month_int

import os

LOG = logging.getLogger("precompute_pack")
logging.basicConfig(level=logging.INFO)

# --------- CONFIG ---------
# Windows path to your models folder (you gave this)
MODELS_DIR = Path(r"C:\Users\ASUS\skripsi_carin\models")
OUT_DIR = Path("static/data_01_")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = 4            # mulai konservatif (naikkan jika mesin punya banyak CPU/RAM)
TARGET_END = pd.Timestamp(year=2026, month=12, day=31).normalize()
YEARS = list(range(2020, 2026+1))
HAVE_WEEK_FILES = True
# --------------------------

# --- Import project objects (adjust module name if needed) ---
# The script expects to run from project root where `app` module exists.
# If your app module is named differently, change the import below.
try:
    # try import app-level symbols
    from app import LONG_DF, LAST_ACTUAL, ENTITY_TO_PROVINCE, _recursive_predict, _week_of_month_int
except Exception as e:
    LOG.error("Failed to import required objects from app module: %s", e)
    LOG.error("Make sure you run this script from project root and that 'app' exposes LONG_DF, LAST_ACTUAL, ENTITY_TO_PROVINCE, _recursive_predict, _week_of_month_int")
    raise

# --- helpers ---
def month_date_range(year, month):
    start = pd.Timestamp(year=year, month=month, day=1).normalize()
    last = calendar.monthrange(year, month)[1]
    end = pd.Timestamp(year=year, month=month, day=last).normalize()
    return pd.date_range(start, end, freq='D')

def week_in_month_date_range(year, month, week):
    last_day = calendar.monthrange(year, month)[1]
    if week <= 0: week = 1
    if week >= 5:
        start_day = 29
        end_day = last_day
    else:
        start_day = (week - 1) * 7 + 1
        end_day = min(week * 7, last_day)
    start = pd.Timestamp(year=year, month=month, day=start_day).normalize()
    end = pd.Timestamp(year=year, month=month, day=end_day).normalize()
    return pd.date_range(start, end, freq='D')

def get_last_actual(entity):
    try:
        if isinstance(LAST_ACTUAL, dict) and LAST_ACTUAL.get(entity) is not None:
            return pd.to_datetime(LAST_ACTUAL[entity]).normalize()
        s = LONG_DF.loc[LONG_DF["entity"] == entity, "date"]
        if s.empty:
            return None
        return pd.to_datetime(s.max()).normalize()
    except Exception:
        return None

def available_model_files():
    if not MODELS_DIR.exists():
        LOG.error("MODELS_DIR not found: %s", MODELS_DIR)
        return []
    return list(MODELS_DIR.glob("**/*.joblib")) + list(MODELS_DIR.glob("**/*.pkl"))

def model_hash_for_index():
    """Compute combined hash of all model filenames & mtimes (lightweight model_version)."""
    parts = []
    for fp in sorted(available_model_files()):
        st = fp.stat()
        parts.append(f"{fp.name}:{st.st_mtime_ns}")
    h = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:10]
    return h

def compute_entity_predictions_until(entity):
    """
    Predict for entity from last_actual+1 up to TARGET_END.
    Returns list of {"date": Timestamp, "pred": float}
    """
    last_dt = get_last_actual(entity)
    if last_dt is None:
        return []
    if last_dt >= TARGET_END:
        return []
    days_needed = int((TARGET_END - last_dt).days)
    if days_needed <= 0:
        return []
    try:
        preds = _recursive_predict(entity, days=days_needed, mode="real")
        out = []
        for p in preds:
            p_dt = pd.to_datetime(p["date"]).normalize()
            out.append({"date": p_dt, "pred": float(p["pred"])})
        return out
    except FileNotFoundError:
        LOG.warning("Model not found for entity %s", entity)
        return []
    except Exception as e:
        LOG.exception("Error predicting for entity %s: %s", entity, e)
        return []

def build_series_for_entity(entity, preds):
    # actuals
    s_act = LONG_DF.loc[LONG_DF["entity"] == entity, ["date","value"]].copy()
    if not s_act.empty:
        s_act["date"] = pd.to_datetime(s_act["date"]).dt.normalize()
        s_act = s_act.drop_duplicates(subset=["date"]).set_index("date")["value"].sort_index()
    else:
        s_act = pd.Series(dtype=float)
    # preds series
    s_pred = pd.Series({p["date"]: p["pred"] for p in preds})
    # only include preds beyond last actual
    last = get_last_actual(entity)
    if last is not None and not s_pred.empty:
        s_pred = s_pred[s_pred.index > last]
    combined = pd.concat([s_act, s_pred[~s_pred.index.isin(s_act.index)]])
    combined = combined.sort_index()
    return combined

def agg_series_to_maps(series_by_entity):
    """
    Menghasilkan mapping:
      month_map[(year,month)] -> dict(entity -> {"province":..., "island":..., "value":...})
      week_map[(year,month,week)] -> dict(entity -> {"province":..., "island":..., "value":...})
    """
    rows = []
    for ent, ser in series_by_entity.items():
        if ser is None or ser.empty:
            continue
        for d, v in ser.items():
            rows.append({"entity": ent, "date": d, "value": float(v)})
    if not rows:
        return {}, {}

    dfall = pd.DataFrame(rows)
    dfall["year"] = dfall["date"].dt.year
    dfall["month"] = dfall["date"].dt.month
    dfall["week_in_month"] = dfall["date"].apply(_week_of_month_int)

    # mapping entity -> province (menggunakan ENTITY_TO_PROVINCE)
    mapping = pd.DataFrame({"entity": list(ENTITY_TO_PROVINCE.keys()), "province": list(ENTITY_TO_PROVINCE.values())})
    dfall = dfall.merge(mapping, on="entity", how="left")
    dfall = dfall.dropna(subset=["province"])

    # tambahkan kolom island berdasarkan province (jaga case keys)
    def prov_to_island(p):
        if p is None: return None
        # PROVINCE_TO_ISLAND keys mungkin uppercase; coba beberapa bentuk
        return PROVINCE_TO_ISLAND.get(p) or PROVINCE_TO_ISLAND.get(p.upper()) or PROVINCE_TO_ISLAND.get(p.title())

    dfall["island"] = dfall["province"].apply(prov_to_island)

    # group per entity untuk tiap year/month (mengambil rata2)
    month_map = {}
    grp_m = dfall.groupby(["year","month","entity","province","island"])["value"].mean().reset_index()
    for (y,m), g in grp_m.groupby(["year","month"]):
        key = (int(y), int(m))
        # buat dict entity -> {province,island,value}
        month_map[key] = {row.entity: {"province": row.province, "island": row.island, "value": float(row.value)} for row in g.itertuples(index=False)}

    week_map = {}
    if HAVE_WEEK_FILES:
        grp_w = dfall.groupby(["year","month","week_in_month","entity","province","island"])["value"].mean().reset_index()
        for (y,m,w), g in grp_w.groupby(["year","month","week_in_month"]):
            key = (int(y), int(m), int(w))
            week_map[key] = {row.entity: {"province": row.province, "island": row.island, "value": float(row.value)} for row in g.itertuples(index=False)}

    return month_map, week_map
def main():
    LOG.info("Models dir: %s", MODELS_DIR)
    LOG.info("Scanning model files...")
    files = available_model_files()
    LOG.info("Found %d model files", len(files))
    model_version = model_hash_for_index()
    LOG.info("Computed model_version: %s", model_version)

    entities = sorted(LONG_DF["entity"].unique())
    LOG.info("Entities count from LONG_DF: %d", len(entities))

    # compute predictions in parallel
    series_by_entity = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(compute_entity_predictions_until, ent): ent for ent in entities}
        for fut in as_completed(futures):
            ent = futures[fut]
            try:
                preds = fut.result()
            except Exception as e:
                LOG.exception("Error for %s: %s", ent, e)
                preds = []
            series_by_entity[ent] = build_series_for_entity(ent, preds)

    # aggregate
    month_map, week_map = agg_series_to_maps(series_by_entity)

    # write files
    index = {"generated_at": datetime.datetime.utcnow().isoformat()+"Z", "model_version": model_version, "files": []}
    
        # write files (bagian month_map)
    for (y,m), city_map in month_map.items():
        if y not in YEARS: continue
        fname = f"choropleth_pred_{y:04d}_{m:02d}.json"
        fp = OUT_DIR / fname
        payload = {
            "year": y,
            "month": m,
            "generated_at": index["generated_at"],
            "model_version": model_version,
            "data": [
                {"city": ent, "province": info["province"], "island": info["island"], "value": info["value"]}
                for ent, info in city_map.items()
            ]
        }
        fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        index["files"].append({"type":"month","year":y,"month":m,"path":str(fp),"count":len(city_map)})

    # week files
    if HAVE_WEEK_FILES:
        for (y,m,w), city_map in week_map.items():
            if y not in YEARS: continue
            fname = f"choropleth_pred_{y:04d}_{m:02d}_w{w}.json"
            fp = OUT_DIR / fname
            payload = {
                "year": y,
                "month": m,
                "week": w,
                "generated_at": index["generated_at"],
                "model_version": model_version,
                "data": [
                    {"city": ent, "province": info["province"], "island": info["island"], "value": info["value"]}
                    for ent, info in city_map.items()
                ]
            }
            fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            index["files"].append({"type":"week","year":y,"month":m,"week":w,"path":str(fp),"count":len(city_map)})


    idx_fp = OUT_DIR / "choropleth_pred_index.json"
    idx_fp.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("Wrote index %s", idx_fp)
    LOG.info("Done.")

if __name__ == "__main__":
    main()
