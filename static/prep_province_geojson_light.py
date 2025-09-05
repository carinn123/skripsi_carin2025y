# prep_province_geojson_light.py
import json, argparse, sys, unicodedata, re
from pathlib import Path

# Nama provinsi "kanonikal" yang dipakai backend (harus match dgn ENTITY_TO_PROVINCE)
CANON = {
    "aceh": "Aceh",
    "sumatera utara": "Sumatera Utara",
    "sumatera barat": "Sumatera Barat",
    "riau": "Riau",
    "kepulauan riau": "Kepulauan Riau",
    "jambi": "Jambi",
    "sumatera selatan": "Sumatera Selatan",
    "bengkulu": "Bengkulu",
    "bangka belitung": "Bangka Belitung",  # catatan: banyak dataset tulis "Kepulauan Bangka Belitung"
    "lampung": "Lampung",

    "dki jakarta": "DKI Jakarta",
    "banten": "Banten",
    "jawa barat": "Jawa Barat",
    "jawa tengah": "Jawa Tengah",
    "di yogyakarta": "DI Yogyakarta",      # sering muncul "Daerah Istimewa Yogyakarta"
    "jawa timur": "Jawa Timur",

    "bali": "Bali",
    "nusa tenggara barat": "Nusa Tenggara Barat",
    "nusa tenggara timur": "Nusa Tenggara Timur",

    "kalimantan barat": "Kalimantan Barat",
    "kalimantan tengah": "Kalimantan Tengah",
    "kalimantan selatan": "Kalimantan Selatan",
    "kalimantan timur": "Kalimantan Timur",
    "kalimantan utara": "Kalimantan Utara",

    "sulawesi utara": "Sulawesi Utara",
    "gorontalo": "Gorontalo",
    "sulawesi tengah": "Sulawesi Tengah",
    "sulawesi barat": "Sulawesi Barat",
    "sulawesi selatan": "Sulawesi Selatan",
    "sulawesi tenggara": "Sulawesi Tenggara",

    "maluku": "Maluku",
    "maluku utara": "Maluku Utara",

    # Catatan: banyak GeoJSON belum pecah Papua (tengah/selatan/pegunungan/barat daya).
    # Minimal pastikan dua ini ada. Kalau datasetmu masih model lama, "Papua Barat" & "Papua".
    "papua": "Papua",
    "papua barat": "Papua Barat",

    # Jika dataset baru:
    "papua barat daya": "Papua Barat Daya",
    "papua pegunungan": "Papua Pegunungan",
    "papua selatan": "Papua Selatan",
    "papua tengah": "Papua Tengah",
}

# Alias umum → kanonikal
ALIASES = {
    "daerah istimewa yogyakarta": "di yogyakarta",
    "d.i. yogyakarta": "di yogyakarta",
    "yogyakarta": "di yogyakarta",
    "daerah khusus ibukota jakarta": "dki jakarta",
    "dki": "dki jakarta",
    "kepulauan bangka belitung": "bangka belitung",
    "kep. bangka belitung": "bangka belitung",
}

CANDIDATE_NAME_KEYS = [
    "provinsi", "name", "NAME_1", "shapeName", "Propinsi", "Provinsi", "PROVINSI", "province", "prov_name"
]

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.lower()
    return s

def to_canon(raw: str) -> str:
    key = norm(raw)
    key = ALIASES.get(key, key)
    return CANON.get(key)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path GeoJSON sumber (ADM1 Indonesia)")
    ap.add_argument("--out", required=True, help="Path output GeoJSON: static/indonesia_provinces.geojson")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    with src.open("r", encoding="utf-8") as f:
        gj = json.load(f)

    if gj.get("type") != "FeatureCollection":
        print("ERROR: GeoJSON bukan FeatureCollection", file=sys.stderr)
        sys.exit(1)

    fixed_features = []
    missing = []
    seen_names = set()

    for feat in gj.get("features", []):
        props = feat.get("properties") or {}
        # cari field nama provinsi yang tersedia
        raw_name = None
        for k in CANDIDATE_NAME_KEYS:
            if k in props and props[k]:
                raw_name = str(props[k])
                break
        if not raw_name:
            # fallback: coba gabungkan beberapa field umum
            raw_name = props.get("shapeName") or props.get("NAME_1") or props.get("name")

        if not raw_name:
            missing.append("(tanpa nama)")
            continue

        canon = to_canon(raw_name)
        if not canon:
            missing.append(raw_name)
            # tetap lanjutkan, tapi isi provinsi = raw_name biar bisa dilihat di popup
            prov_name = raw_name
        else:
            prov_name = canon

        # tulis 'provinsi' saja + pertahankan geometry
        new_props = {"provinsi": prov_name}
        fixed_features.append({
            "type": "Feature",
            "properties": new_props,
            "geometry": feat.get("geometry")
        })
        seen_names.add(prov_name)

    fixed = {"type": "FeatureCollection", "features": fixed_features}

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False)

    print(f"OK: tulis {out} dengan {len(fixed_features)} feature.")
    if missing:
        uniq = sorted(set(missing))
        print("\nCatatan: ada nama yang tidak terpetakan ke kanonikal:")
        for n in uniq:
            print(" -", n)
        print("\nKalau perlu, tambahkan ke ALIASES atau CANON di script ini.")

    # Bantuan verifikasi cepat:
    expected = sorted(set(CANON.values()))
    print("\nNama provinsi yang terdeteksi di output:")
    for n in sorted(seen_names):
        print(" •", n)

    # Optional: warning kalau ada kunci di ENTITY_TO_PROVINCE (kanonikal) tapi tidak ada di GeoJSON
    missing_expected = [n for n in expected if n not in seen_names]
    if missing_expected:
        print("\nPERINGATAN: berikut provinsi kanonikal tidak muncul di GeoJSON output:")
        for n in missing_expected:
            print(" !", n)

if __name__ == "__main__":
    main()
