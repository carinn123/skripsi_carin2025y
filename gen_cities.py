# gen_cities_json.py
from pathlib import Path
import json
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset_filled_ffill_bfill.xlsx"
OUT_PATH  = BASE_DIR / "static" / "cities.json"

df = pd.read_excel(DATA_PATH, nrows=0)
cols = list(df.columns)

# buang kolom tanggal jika ada
drop_keys = {"date", "tanggal", "time", "waktu"}
cities = [c for c in cols if str(c).strip().lower() not in drop_keys]

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUT_PATH.open("w", encoding="utf-8") as f:
  json.dump({"cities": sorted(cities)}, f, ensure_ascii=False, indent=2)

print(f"OK -> {OUT_PATH}")
d