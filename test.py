import pandas as pd
import numpy as np
from datetime import datetime

# Path file
fp = r"C:\Users\ASUS\skripsi_carin\data\dataset.xlsx"

# Baca excel, parse date column
df = pd.read_excel(fp, sheet_name="Sheet1", parse_dates=['date'])

# Pastikan kolom date tersedia
if 'date' not in df.columns:
    raise KeyError("Kolom 'date' tidak ditemukan di sheet. Periksa nama kolom di Excel.")

# Fungsi pembersihan angka (menghandle format '1.234,56' atau '1234,56' atau '1234.56')
def clean_numeric_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    # Kalau ada titik dan koma, anggap format ribuan.titik dan desimal koma (E.g. 1.234,56)
    mask_both = s.str.contains(r'[.,]') & s.str.contains(',')
    if mask_both.any():
        # Hapus titik ribuan, ganti koma jadi titik desimal
        s = s.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    else:
        # Kalau cuma ada koma (mis. '13120,96') ganti koma ke titik
        if s.str.contains(',').any():
            s = s.str.replace(',', '.', regex=False)
        # if only dots exist we leave it (likely already '1234.56' or '1.234' -> may be ambiguous)
        # remove any stray spaces
        s = s.str.replace(r'\s+', '', regex=True)
    # Convert to numeric, coercing errors to NaN
    return pd.to_numeric(s, errors='coerce')

# Bersihkan nama kolom whitespace (opsional)
df.columns = df.columns.str.strip()

# Konversi kolom kecuali 'date' menjadi numeric setelah pembersihan
city_cols = [c for c in df.columns if c.lower() != 'date']
for c in city_cols:
    df[c] = clean_numeric_series(df[c])

# Parse & set index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Tambah kolom year (opsional, akan berguna)
df['year'] = df.index.year

# Fungsi volatility (CV %)
def calculate_volatility(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0 or np.mean(arr) == 0:
        return 0.0
    return (np.std(arr, ddof=0) / np.mean(arr)) * 100.0

# Siapkan struktur hasil
vol_per_year = {}
full_period_vol = {}
test_period_vol = {}

# Definisikan periode
full_start = datetime(2020, 1, 1)
full_end   = datetime(2024, 7, 1)
test_start = datetime(2024, 7, 1)
test_end   = datetime(2025, 7, 1)

for city in city_cols:
    vol_per_year[city] = {}
    city_series = df[city].dropna()

    for year in range(2020, 2026):
        year_data = city_series[city_series.index.year == year]
        vol_per_year[city][year] = calculate_volatility(year_data.values)

    full_data = city_series[(city_series.index >= full_start) & (city_series.index < full_end)]
    full_period_vol[city] = calculate_volatility(full_data.values)

    test_data = city_series[(city_series.index >= test_start) & (city_series.index < test_end)]
    test_period_vol[city] = calculate_volatility(test_data.values)

# Buat DataFrame hasil
years = list(range(2020, 2026))
vol_df = pd.DataFrame(vol_per_year).T.reindex(columns=years)
vol_df['Full Period (2020-01-01 to 2024-07-01)'] = pd.Series(full_period_vol)
vol_df['Test Period (2024-07-01 to 2025-07-01)'] = pd.Series(test_period_vol)
vol_df = vol_df.round(2)

# Simpan
output_file = "volatilities_per_city_per_yeasr.xlsx"
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    vol_df.to_excel(writer, sheet_name='Volatility by City and Year')

print(f"Excel file generated: {output_file}")
print(vol_df.head(10))
