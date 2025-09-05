# make_city_coords.py
import json, time, pathlib, re
from typing import Optional
import pandas as pd

try:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
except Exception as e:
    raise SystemExit("Install dulu: pip install geopy") from e

# ==== KONFIG ====
XLSX = "data/dataset_filled_ffill_bfill.xlsx"          # ganti sesuai file kamu
ENTITY_TO_PROV = "static/entity_to_province.json"
OUT_JSON = "static/city_coords.json"                   # hasil: { "Kota Bandung": [lon, lat], ... }

# ==== util normalisasi (samakan dengan yang tadi) ====
def norm_entity(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^Kab\.?\s*", "Kab. ", s, flags=re.I)
    s = re.sub(r"^Kota\s*", "Kota ", s, flags=re.I)
    # koreksi umum sesuai header Excel kamu
    s = s.replace("Kab Kotabaru", "Kab. Kotabaru")
    s = s.replace("Kab.Klaten", "Kab. Klaten")
    s = s.replace("Kab Sragen", "Kab. Sragen")
    s = s.replace("Kota Tanggerang", "Kota Tangerang")
    s = s.replace("Kota mobagu", "Kotamobagu")  # typo di list kamu
    s = s.replace("Kotamobagu", "Kotamobagu")   # biarkan nama lajur apa adanya
    s = s.strip()
    return s

def base_name(ent: str) -> str:
    # buang prefix kab/kota untuk query geocode
    return ent.replace("Kab. ", "").replace("Kota ", "").strip()

# ==== load data ====
etp = json.loads(pathlib.Path(ENTITY_TO_PROV).read_text(encoding="utf-8"))
df = pd.read_excel(XLSX)
cols = [c for c in df.columns if c != "date"] if "date" in df.columns else list(df.columns)
entities = [norm_entity(c) for c in cols]

# pakai yang benar-benar ada di mapping
entities = [e for e in entities if e in etp]

# ==== geocoder ====
geolocator = Nominatim(user_agent="kabkota-geocoder")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)  # rate-limit 1 detik

coords = {}
missing = []

for e in entities:
    prov = etp[e]
    q = f"{base_name(e)}, {prov}, Indonesia"
    try:
        loc = geocode(q, addressdetails=False, country_codes="id")
    except Exception:
        loc = None
    if loc:
        coords[e] = [loc.longitude, loc.latitude]
        print(f"OK  {e:<30} -> {coords[e]}  ({q})")
    else:
        missing.append((e, q))
        print(f"??  {e:<30} -> GAGAL: {q}")

# tulis file (yang dapat dulu)
pathlib.Path(OUT_JSON).write_text(json.dumps(coords, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n✅ Tulis: {OUT_JSON} (got {len(coords)}/{len(entities)})")

if missing:
    print("\n⚠️ Belum dapat koordinat, isi manual di city_coords.json (format [lon, lat]):")
    for e, q in missing:
        print(f"  - {e}  | saran query: {q}")
