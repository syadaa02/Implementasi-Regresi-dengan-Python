import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

# Path to the CSV file
file_path = 'D:\Smt 4\Metnum\student_performance.csv'

# Membaca file CSV dengan delimiter ';'
data = pd.read_csv(file_path, delimiter=';')

# Normalisasi data: mengganti nilai '91.00.00' dengan '91' di kolom NT
def normalize_NT(x):
    if isinstance(x, str):
        return float(x.split('.')[0])
    elif isinstance(x, float) and not np.isnan(x):
        return x
    else:
        return np.nan

data['NT'] = data['NT'].apply(normalize_NT)

# Konversi kolom NL ke tipe data numerik tanpa penanganan kesalahan
data['NL'] = pd.to_numeric(data['NL'], errors='coerce')

# Menghapus baris dengan nilai NaN
data.dropna(subset=['NL', 'NT'], inplace=True)

# Mengekstrak kolom dari CSV
NL = data['NL'].tolist()
NT = data['NT'].tolist()

# Mendefinisikan fungsi pangkat sederhana
def fungsi_pangkat(x, a, b):
    return a * np.power(x, b)

# Melakukan fitting fungsi pangkat sederhana ke data
popt, pcov = curve_fit(fungsi_pangkat, NL, NT, maxfev=10000)

# Mengekstrak koefisien
a, b = popt

# Menghitung nilai prediksi
y_pred_pangkat = [fungsi_pangkat(x, a, b) for x in NL]

# Menghitung galat RMS
error_sum = sum((nt - yp) ** 2 for nt, yp in zip(NT, y_pred_pangkat))
rms_error_pangkat = (error_sum / len(NT)) ** 0.5
print("Galat RMS (Regresi Pangkat):", rms_error_pangkat)

# Memplot titik data dan kurva pangkat dengan ukuran titik yang lebih kecil
plt.scatter(NL, NT, label='Titik Data', s=1)  # Mengurangi ukuran titik data
x_fit = np.linspace(min(NL), max(NL), 100)
y_fit = fungsi_pangkat(x_fit, a, b)
plt.plot(x_fit, y_fit, color='red', label='Fit Pangkat: y = {:.2f}x^{:.2f}'.format(a, b))
plt.xlabel('Sample Question Papers Practiced (NL)')
plt.ylabel('Performance Index (NT)')
plt.legend()
plt.title('Regresi Pangkat dari Performance Index vs. Sample Question Papers Practiced')
plt.show()

# Menguji kode
def uji_regresi():
    assert isinstance(rms_error_pangkat, float), "Galat RMS pangkat tidak valid"
    assert len(NL) == len(NT), "Panjang data tidak sesuai"
    assert len(NL) > 0, "Array data kosong"

uji_regresi()
