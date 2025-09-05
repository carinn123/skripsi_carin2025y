# build_province_values.py
# ------------------------------------------------------------
# - Baca Excel harian per kota/kab
# - Mapping ke provinsi (ENTITY_TO_PROV)
# - Agregasi rata-rata per provinsi per hari
# - Tulis:
#     static/province_daily.json   -> { "2020-01-01": { "JAWA BARAT": 13150, ... }, ... }
#     static/province_latest.json  -> { "JAWA BARAT": 13200, ... } (tanggal terakhir)
# ------------------------------------------------------------

import json
import pathlib
import re
from typing import Optional

import numpy as np
import pandas as pd

# ====== KONFIG ======
XLSX = "data/dataset_filled_ffill_bfill.xlsx"
ENTITY_TO_PROV = "static/entity_to_province.json"
OUT_DAILY = "static/province_daily.json"
OUT_LATEST = "static/province_latest.json"

# Pastikan folder output ada
pathlib.Path("static").mkdir(parents=True, exist_ok=True)

# ====== NORMALISASI NAMA ENTITAS ======
def norm_entity(s: str) -> str:
    """Samakan pola penulisan supaya cocok dengan key di entity_to_province.json."""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^Kab\.?\s*", "Kab. ", s, flags=re.I)
    s = re.sub(r"^Kota\s*", "Kota ", s, flags=re.I)

    # correction untuk variasi yang muncul di Excel contohmu
    s = s.replace("Kab Kotabaru", "Kab. Kotabaru")
    s = s.replace("Kab.Klaten", "Kab. Klaten")
    s = s.replace("Kab Sragen", "Kab. Sragen")
    s = s.replace("Kota Tanggerang", "Kota Tangerang")
    s = s.replace("Kota mobagu", "Kotamobagu")
    s = s.replace(" Kota Lubuk Linggau", "Kota Lubuk Linggau")  # ada spasi depan kadang

    return s.strip()

# ====== BACA DATA ======
# mapping entity -> provinsi (kamu sudah isi file ini sebelumnya)
etp_path = pathlib.Path(ENTITY_TO_PROV)
if not etp_path.exists():
    raise SystemExit(f"❌ File mapping tidak ditemukan: {ENTITY_TO_PROV}")

etp = json.loads(etp_path.read_text(encoding="utf-8"))

xlsx_path = pathlib.Path(XLSX)
if not xlsx_path.exists():
    raise SystemExit(f"❌ File Excel tidak ditemukan: {XLSX}")

df = pd.read_excel(xlsx_path)

if "date" not in df.columns:
    raise SystemExit("❌ Kolom 'date' wajib ada di Excel.")

# ====== LONG FORMAT ======
value_cols = [c for c in df.columns if c != "date"]
if not value_cols:
    raise SystemExit("❌ Tidak ada kolom nilai selain 'date' di Excel.")

long = df.melt(
    id_vars=["date"],
    value_vars=value_cols,
    var_name="entity",
    value_name="value",
)

# normalisasi
long["entity"] = long["entity"].map(norm_entity)
long["value"] = pd.to_numeric(long["value"], errors="coerce")
long["province"] = long["entity"].map(etp)

# list entitas yang belum kepetakan (sekadar warning)
unknown = sorted(set(long.loc[long["province"].isna(), "entity"]))
if unknown:
    print("⚠️  Masih ada entitas belum termapping ke provinsi (cek ENTITY_TO_PROV):")
    for name in unknown[:30]:
        print("   -", name)
    if len(unknown) > 30:
        print(f"   ... dan {len(unknown)-30} lagi.")
# hanya pakai yang sudah terpetakan
long = long.dropna(subset=["province"])

# pastikan date jadi datetime lalu group pakai .dt.date
long["date"] = pd.to_datetime(long["date"], errors="coerce")
long = long.dropna(subset=["date"])  # buang baris tanggal invalid

# ====== AGREGASI PROVINSI / HARI ======
# index = datetime.date (BUKAN Timestamp)
daily = (
    long.groupby([long["date"].dt.date, "province"])["value"]
        .mean()
        .unstack()               # kolom = provinsi
        .sort_index()
)

# ====== TULIS province_daily.json ======
# simpan semua prov (termasuk yang NaN -> tulis None)
daily_dict = {}
for idx, row in daily.iterrows():
    row_dict = {}
    for prov, val in row.items():
        if pd.isna(val):
            row_dict[prov] = None
        else:
            row_dict[prov] = float(val)
    daily_dict[idx.isoformat()] = row_dict  # kunci tanggal jadi string ISO

pathlib.Path(OUT_DAILY).write_text(
    json.dumps(daily_dict, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(f"✅ Tulis: {OUT_DAILY}  (hari: {len(daily_dict)})")

# ====== TULIS province_latest.json (berdasarkan tanggal terakhir) ======
if len(daily.index) == 0:
    # tidak ada data setelah filter, tulis kosong
    pathlib.Path(OUT_LATEST).write_text("{}", encoding="utf-8")
    print("⚠️  Tidak ada baris harian setelah agregasi. province_latest.json dikosongkan.")
else:
    last_day = daily.index.max()           # <-- datetime.date
    latest_series = daily.loc[last_day]    # Series prov -> nilai (bisa NaN)
    latest = {prov: float(val) for prov, val in latest_series.items() if pd.notna(val)}

    pathlib.Path(OUT_LATEST).write_text(
        json.dumps(latest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"✅ Tulis: {OUT_LATEST} (tanggal: {last_day.isoformat()}, provinsi: {len(latest)})")
