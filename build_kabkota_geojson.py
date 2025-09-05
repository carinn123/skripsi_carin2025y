import json, re, unicodedata
from pathlib import Path
from typing import Optional
import pandas as pd
from difflib import get_close_matches

DATA_XLSX = Path("data/dataset_filled_ffill_bfill.xlsx")
CITY_COORDS = Path("static/city_coords.json")          # { "Kota Bandung": [lon,lat], ... }
E2P_PATH    = Path("static/entity_to_province.json")   # { "Kota Bandung": "Jawa Barat", ... }
OUT_POINTS  = Path("static/kabkota_points.geojson")
OUT_PROV    = Path("static/province_agg.geojson")      # opsional, kalau provinces geojson tersedia
PROV_GEO    = Path("static/indonesia_provinces.geojson")

# === 1) util normalisasi ===
def norm_basic(s: str) -> str:
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def strip_prefixes(s: str) -> str:
    s = re.sub(r"^(KAB\.?|KOTA)\s*", "", s, flags=re.I)
    return s

def simplify(s: str) -> str:
    s = norm_basic(s)
    s = re.sub(r"\(.*?\)", "", s)              # buang (Solo), dll
    s = s.replace("-", " ")                    # samakan strip
    s = re.sub(r"\s+", " ", s).strip()
    return s.upper()

# === 2) alias/typo overrides -> kanonik nama di city_coords.json ===
ALIAS = {
    "Kota Tanggerang": "Kota Tangerang",
    "Kota Surakarta (Solo)": "Kota Surakarta",
    "Kota Pare -Pare": "Kota Parepare",
    "Kota Bau - Bau": "Kota Baubau",
    "Kotamobagu": "Kota Kotamobagu",
    "Kab Sragen": "Kab. Sragen",
    "Kab.Klaten": "Kab. Klaten",
    "Kab Kotabaru": "Kab. Kotabaru",
    " Kota Lubuk Linggau": "Kota Lubuklinggau",
    "Kota Gunung Sitoli": "Kota Gunungsitoli",
    "Kota Sumenep": "Kab. Sumenep",
    "Kota Sampit": "Kab. Kotawaringin Timur",
    "Kota Tembilahan": "Kab. Indragiri Hilir",
    "Kota Watampone": "Kab. Bone",
    "Kota Maumere": "Kab. Sikka",
    "Kota Tanjung Pandan": "Kab. Belitung",
    "Kota Tanjung Pinang": "Kota Tanjungpinang",
    "Kota Tanjung": "Kab. Tabalong",  # asumsi: Tanjung (Tabalong)
    "Kota Maluku": "Kota Ambon",      # asumsi: yang dimaksud Ambon
}

# === 3) load master koordinat & province map ===
coords = json.loads(CITY_COORDS.read_text(encoding="utf-8"))
e2p    = json.loads(E2P_PATH.read_text(encoding="utf-8"))

# index utk fuzzy match
canon_names = list(coords.keys())
canon_index_simple = {simplify(k): k for k in canon_names}

def to_canonical(raw: str) -> Optional[str]:
    raw = norm_basic(raw)
    if raw in ALIAS:
        return ALIAS[raw]
    # coba exact
    if raw in coords:
        return raw
    # coba pakai bentuk sederhana
    simp = simplify(raw)
    if simp in canon_index_simple:
        return canon_index_simple[simp]
    # fuzzy last resort
    cand = get_close_matches(simp, list(canon_index_simple.keys()), n=1, cutoff=0.88)
    if cand:
        return canon_index_simple[cand[0]]
    return None

# === 4) baca excel wide & melt ===
df = pd.read_excel(DATA_XLSX, header=0)
if "date" not in df.columns:
    raise RuntimeError("Kolom 'date' tidak ditemukan.")
value_cols = [c for c in df.columns if c != "date"]

long = df.melt(id_vars=["date"], value_vars=value_cols,
               var_name="raw_entity", value_name="value")

# keep hanya angka (drop NaN)
long = long.dropna(subset=["value"])

# map ke nama kanonik
long["entity"] = long["raw_entity"].apply(to_canonical)

unmatched = sorted(set(long.loc[long["entity"].isna(), "raw_entity"]))
if unmatched:
    print("⚠️ Belum ketemu di city_coords.json (cek ALIAS / city_coords):")
    for x in unmatched[:25]: print("  -", x)
    print(f"... total {len(unmatched)} nama. Lanjut bikin yang sudah match dulu.")

matched = long.dropna(subset=["entity"]).copy()

# === 5) ambil nilai tanggal paling baru utk peta point cepat ===
latest_date = matched["date"].max()
latest = matched.loc[matched["date"] == latest_date].copy()

# gabung provinsi
def map_province(name):
    return e2p.get(name) or e2p.get(strip_prefixes(name)) or e2p.get(simplify(name))
latest["province"] = latest["entity"].map(map_province)

# build FeatureCollection (Point)
features = []
for _, r in latest.iterrows():
    name = r["entity"]
    lon, lat = coords[name]
    features.append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
        "properties": {
            "kabkota": name,
            "province": r["province"],
            "date": pd.to_datetime(r["date"]).date().isoformat(),
            "value": float(r["value"]),
        }
    })
geo_points = {"type": "FeatureCollection", "features": features}
OUT_POINTS.write_text(json.dumps(geo_points, ensure_ascii=False), encoding="utf-8")
print(f"✅ Saved {OUT_POINTS} ({len(features)} fitur) untuk tanggal {latest_date.date()}.")

# === 6) (opsional) agregasi ke provinsi (mean terbaru), masukkan ke prov geojson ===
if PROV_GEO.exists():
    prov_geo = json.loads(PROV_GEO.read_text(encoding="utf-8"))
    # hitung mean per prov
    prov_mean = (latest.dropna(subset=["province"])
                       .groupby("province")["value"].mean().to_dict())
    # tambahkan properti 'value'
    for f in prov_geo["features"]:
        p = f.get("properties", {})
        # coba baca nama prov dari properti umum
        prov_name = p.get("PROVINSI") or p.get("Propinsi") or p.get("provinsi") or p.get("prov_name")
        if prov_name in prov_mean:
            p["value"] = float(prov_mean[prov_name])
            f["properties"] = p
    OUT_PROV.write_text(json.dumps(prov_geo, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Saved {OUT_PROV} (choropleth provinsi, mean nilai terbaru).")
else:
    print("ℹ️ Lewati agregat provinsi (file provinces geojson tidak ditemukan).")
