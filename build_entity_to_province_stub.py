import json, pathlib, re
import pandas as pd

# === KONFIG ===
XLSX = "data/dataset_filled_ffill_bfill.xlsx"          # ganti sesuai lokasi file Excel kamu
PROV_GEO = "static/indonesia_provinces.geojson"        # ganti kalau beda nama
OUT_JSON = "static/entity_to_province.json"

# === normalisasi nama entitas (kolom Excel) ===
def norm_entity(s):
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)

    # seragamkan penulisan Kab/Kota
    s = re.sub(r"^Kab\.?\s*", "Kab. ", s, flags=re.I)
    s = re.sub(r"^Kota\s*", "Kota ", s, flags=re.I)

    # perbaikan ejaan umum
    s = s.replace("Kab Kotabaru", "Kab. Kotabaru")
    s = s.replace("Kab.Klaten", "Kab. Klaten")
    s = s.replace("Kab Sragen", "Kab. Sragen")
    s = s.replace("Kota Tanggerang", "Kota Tangerang")
    s = s.replace("Kota Pare -Pare", "Kota Pare -Pare")  # biarkan seperti di Excel
    s = s.replace("Kota Bau - Bau", "Kota Bau - Bau")
    s = s.replace("Kota Palangkaraya", "Kota Palangkaraya")  # biarkan style Excel
    s = s.replace("Kota Sumenep", "Kota Sumenep")  # ada kolom ini & "Kab. Sumenep"
    s = s.replace("Kota Maluku", "Kota Maluku")    # heading unik, kita map ke Prov. MALUKU
    s = s.strip()
    return s

# === normalisasi nama provinsi untuk perbandingan longgar ===
def norm_prov(s):
    s = str(s).strip().upper()
    s = re.sub(r"\s+", " ", s)
    # hilangkan kata umum dan tanda
    s = s.replace("PROVINSI", "").replace("DAERAH KHUSUS IBUKOTA", "DKI").replace("DAERAH ISTIMEWA", "DI")
    s = s.replace("KEPULAUAN", "").replace("Kepulauan", "")
    s = s.replace("KOTA", "").replace("KABUPATEN", "")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s):
    return set(norm_prov(s).split())

def best_match_prov(target, prov_list):
    """cocokkan target ke daftar prov dari GeoJSON berdasarkan irisan token terbanyak"""
    tt = tokens(target)
    best, score = None, -1.0
    for p in prov_list:
        pp = tokens(p)
        if not tt or not pp:
            continue
        inter = len(tt & pp)
        uni = len(tt | pp)
        sc = inter / uni
        if sc > score:
            best, score = p, sc
    return best, score

# === kamus tebakan awal: kolom Excel -> provinsi (standar) ===
SUGGEST = {
    # ACEH
    "Kota Banda Aceh": "ACEH",
    "Kota Lhokseumawe": "ACEH",
    "Kota Meulaboh": "ACEH",
    # SUMATERA UTARA
    "Kota Medan": "SUMATERA UTARA",
    "Kota Pematang Siantar": "SUMATERA UTARA",
    "Kota Gunung Sitoli": "SUMATERA UTARA",
    "Kota Padang Sidempuan": "SUMATERA UTARA",
    "Kota Sibolga": "SUMATERA UTARA",
    # SUMATERA BARAT
    "Kota Padang": "SUMATERA BARAT",
    # RIAU
    "Kota Pekanbaru": "RIAU",
    "Kota Tembilahan": "RIAU",
    "Kota Dumai": "RIAU",
    # KEP. RIAU
    "Kota Batam": "KEPULAUAN RIAU",
    "Kota Tanjung Pinang": "KEPULAUAN RIAU",
    # JAMBI
    "Kota Jambi": "JAMBI",
    "Kab. Bungo": "JAMBI",
    # SUMSEL
    "Kota Palembang": "SUMATERA SELATAN",
    " Kota Lubuk Linggau": "SUMATERA SELATAN",
    # BENGKULU
    "Kota Bengkulu": "BENGKULU",
    # LAMPUNG
    "Kota Bandar Lampung": "LAMPUNG",
    "Kota Metro": "LAMPUNG",
    # BABEL
    "Kota Pangkalpinang": "KEPULAUAN BANGKA BELITUNG",
    "Kota Tanjung Pandan": "KEPULAUAN BANGKA BELITUNG",
    # DKI
    "Kota Jakarta Pusat": "DKI JAKARTA",
    # BANTEN
    "Kota Cilegon": "BANTEN",
    "Kota Serang": "BANTEN",
    "Kota Tangerang": "BANTEN",
    "Kota Tanggerang": "BANTEN",  # kalau belum ternormalisasi
    # JABAR
    "Kota Bandung": "JAWA BARAT",
    "Kota Bekasi": "JAWA BARAT",
    "Kota Bogor": "JAWA BARAT",
    "Kota Sukabumi": "JAWA BARAT",
    "Kota Tasikmalaya": "JAWA BARAT",
    "Kota Cirebon": "JAWA BARAT",
    "Kab. Cirebon": "JAWA BARAT",
    # DIY
    "Kota Yogyakarta": "DI YOGYAKARTA",
    # JATENG
    "Kota Semarang": "JAWA TENGAH",
    "Kota Surakarta (Solo)": "JAWA TENGAH",
    "Kab. Banyumas": "JAWA TENGAH",
    "Kab. Boyolali": "JAWA TENGAH",
    "Kab Sragen": "JAWA TENGAH",
    "Kab. Sragen": "JAWA TENGAH",
    "Kab. Sukoharjo": "JAWA TENGAH",
    "Kab. Karanganyar": "JAWA TENGAH",
    "Kab. Kudus": "JAWA TENGAH",
    "Kab. Cilacap": "JAWA TENGAH",
    # JATIM
    "Kota Surabaya": "JAWA TIMUR",
    "Kota Malang": "JAWA TIMUR",
    "Kota Kediri": "JAWA TIMUR",
    "Kota Probolinggo": "JAWA TIMUR",
    "Kota Blitar": "JAWA TIMUR",
    "Kab. Banyuwangi": "JAWA TIMUR",
    "Kab. Jember": "JAWA TIMUR",
    "Kab. Sumenep": "JAWA TIMUR",
    "Kota Sumenep": "JAWA TIMUR",
    "Kota Madiun": "JAWA TIMUR",
    # BALI & NUSRA
    "Kota Denpasar": "BALI",
    "Kota Mataram": "NUSA TENGGARA BARAT",
    "Kab. Lombok Timur": "NUSA TENGGARA BARAT",
    "Kab. Sumbawa": "NUSA TENGGARA BARAT",
    "Kota Bima": "NUSA TENGGARA BARAT",
    "Kota Kupang": "NUSA TENGGARA TIMUR",
    "Kota Maumere": "NUSA TENGGARA TIMUR",
    "Kab. Sumba Timur": "NUSA TENGGARA TIMUR",
    # KALIMANTAN
    "Kota Pontianak": "KALIMANTAN BARAT",
    "Kota Singkawang": "KALIMANTAN BARAT",
    "Kab. Sintang": "KALIMANTAN BARAT",
    "Kota Palangkaraya": "KALIMANTAN TENGAH",
    "Kota Sampit": "KALIMANTAN TENGAH",
    "Kota Banjarmasin": "KALIMANTAN SELATAN",
    "Kota Tanjung": "KALIMANTAN SELATAN",
    "Kab. Kotabaru": "KALIMANTAN SELATAN",
    "Kab Kotabaru": "KALIMANTAN SELATAN",
    "Kota Balikpapan": "KALIMANTAN TIMUR",
    "Kota Samarinda": "KALIMANTAN TIMUR",
    "Kota Bontang": "KALIMANTAN TIMUR",
    "Kota Tarakan": "KALIMANTAN UTARA",
    "Kab. Bulungan": "KALIMANTAN UTARA",
    # SULAWESI
    "Kota Makassar": "SULAWESI SELATAN",
    "Kota Pare -Pare": "SULAWESI SELATAN",
    "Kota Palopo": "SULAWESI SELATAN",
    "Kab. Bulukomba": "SULAWESI SELATAN",
    "Kota Watampone": "SULAWESI SELATAN",
    "Kab. Polewali Mandar": "SULAWESI BARAT",
    "Kota Mamuju": "SULAWESI BARAT",
    "Kota Manado": "SULAWESI UTARA",
    "Kotamobagu": "SULAWESI UTARA",
    "Kota Gorontalo": "GORONTALO",
    "Kab. Gorontalo": "GORONTALO",
    "Kota Palu": "SULAWESI TENGAH",
    "Kota Banggai": "SULAWESI TENGAH",
    "Kota Kendari": "SULAWESI TENGGARA",
    "Kota Bau - Bau": "SULAWESI TENGGARA",
    # MALUKU & PAPUA
    "Kota Ambon": "MALUKU",
    "Kota Maluku": "MALUKU",
    "Kota Tual": "MALUKU",
    "Kota Ternate": "MALUKU UTARA",
    "Kota Sorong": "PAPUA BARAT",     # akan otomatis dialihkan ke PAPUA BARAT DAYA kalau ada di geojson
    "Kab. Manokwari": "PAPUA BARAT",
    "Kota Jayapura": "PAPUA",
    "Kab. Jayawijaya": "PAPUA",
    "Kab. Mimika": "PAPUA",
    "Kab. Nabire": "PAPUA",
    "Kab. Merauke": "PAPUA",
}

# === baca Excel & ambil daftar entitas ===
df = pd.read_excel(XLSX)
if "date" in df.columns:
    cols = [c for c in df.columns if c != "date"]
else:
    cols = list(df.columns)

entities = [norm_entity(c) for c in cols]

# === baca daftar provinsi dari GeoJSON ===
gj = json.loads(pathlib.Path(PROV_GEO).read_text(encoding="utf-8"))
features = gj.get("features", [])
# cari key nama provinsi
prov_key_candidates = ["PROVINSI","Provinsi","Propinsi","province","provinsi","NAME_1","NAME","prov_name"]
prov_names = []
for f in features:
    props = f.get("properties", {})
    key = next((k for k in prov_key_candidates if k in props), None)
    if key:
        prov_names.append(str(props[key]).strip())
prov_names = sorted(set(prov_names))
prov_tokens = {p: tokens(p) for p in prov_names}

has_pbd = any(norm_prov(p) == "PAPUA BARAT DAYA" for p in prov_names)

# === bentuk mapping: entitas -> provinsi (dicocokkan ke ejaan di GeoJSON) ===
result = {}
not_suggested = []

for e in entities:
    sug = SUGGEST.get(e)
    # aturan khusus Sorong jika geojson sudah punya Papua Barat Daya
    if e.startswith("Kota Sorong") and has_pbd:
        sug = "PAPUA BARAT DAYA"

    if not sug:
        not_suggested.append(e)
        continue

    # cari ejaan provinsi yang ada di geojson (best token overlap)
    best, score = best_match_prov(sug, prov_names)
    # ambang seadanya; kalau nggak ketemu, pakai sug apa adanya
    result[e] = best if best else sug

# === tulis JSON ===
pathlib.Path(OUT_JSON).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"✅ Tulis: {OUT_JSON}")
print(f"  - total entitas: {len(entities)}")
print(f"  - sudah disarankan: {len(result)}")
if not_suggested:
    print("⚠️ Belum ada saran (isi manual di JSON setelah file dibuat):")
    for x in not_suggested:
        print("  -", x)
