import json, re

raw = open("raw.geojson.txt","r",encoding="utf-8").read()

# find all feature objects by balancing braces starting at each {"type":"Feature"
features = []
for m in re.finditer(r'\{"type"\s*:\s*"Feature"', raw):
    i = m.start()
    depth = 0
    in_str = False
    esc = False
    end = None
    for j,ch in enumerate(raw[i:], start=i):
        if in_str:
            esc = (ch == '\\') and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = j+1
                break
    if end:
        try:
            feat = json.loads(raw[i:end])
            features.append(feat)
        except Exception:
            pass  # skip partial/broken ones

clean = {"type":"FeatureCollection","features":features}
open("cleaned.geojson","w",encoding="utf-8").write(json.dumps(clean, ensure_ascii=False))
print(f"Extracted {len(features)} complete feature(s) -> cleaned.geojson")
