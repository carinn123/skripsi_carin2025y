# scripts/fix_entity_to_province_keys.py
import json, re
from pathlib import Path

IN = Path("static/entity_to_province.json")
etp = json.loads(IN.read_text(encoding="utf-8"))

def canon(k: str) -> str:
    k = str(k).strip().lower()
    k = re.sub(r"\s+", "_", k)   # spasi -> _
    return k

fixed = {}
for k, v in etp.items():
    if not v: 
        continue
    fixed[canon(k)] = v  # contoh "Kab. Banyumas" -> "kab._banyumas"

# opsional: tampilkan berapa yang berubah
changed = [k for k in etp if canon(k) != k]
print(f"Total entri: {len(etp)} | Berubah kunci: {len(changed)}")

IN.write_text(json.dumps(fixed, ensure_ascii=False, indent=2), encoding="utf-8")
print("✅ entity_to_province.json diseragamkan.")
