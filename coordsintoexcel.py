import json
import pandas as pd

# Path file JSON
json_path = r"C:\Users\ASUS\skripsi_carin\coords.json"

# Baca file JSON
with open(json_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)

# Ambil data yang dibutuhkan
rows = []
for item in json_data["data"]:
    rows.append({
        "kab_kota": item["city"],
        "value": round(item["value"], 2),
        "tertil": item["category"]
    })

# Buat DataFrame
df = pd.DataFrame(rows)

# Simpan ke Excel
output_path = r"C:\Users\ASUS\skripsi_carin\harga_minyak_tertil_per_kota.xlsx"
df.to_excel(output_path, index=False)

print(f"Excel berhasil dibuat di: {output_path}")
print(df.head())
