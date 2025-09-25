import json, re, pathlib

INP = "static/city_coords.json"
OUT = "static/city_coords_fixed.json"

def norm(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("Kab Kotabaru", "Kab. Kotabaru")
    s = s.replace("Kab.Klaten", "Kab. Klaten")
    s = s.replace("Kab Sragen", "Kab. Sragen")
    s = s.replace("Kota mobagu", "Kotamobagu")
    return re.sub(r"\s+", "_", s.lower())

raw = json.loads(pathlib.Path(INP).read_text(encoding="utf-8"))
out = {}
for pretty, val in raw.items():
    key = norm(pretty)
    if isinstance(val, (list, tuple)) and len(val)==2:
        lng, lat = float(val[0]), float(val[1])
        out[key] = {"lat": lat, "lng": lng, "label": pretty}
    elif isinstance(val, dict) and "lat" in val and "lng" in val:
        out[key] = {"lat": float(val["lat"]), "lng": float(val["lng"]), "label": val.get("label", pretty)}

pathlib.Path(OUT).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"✅ Tulis: {OUT} | total {len(out)} entri")
