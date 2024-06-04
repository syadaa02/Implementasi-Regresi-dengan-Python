import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

# Path to the CSV file
file_path = 'D:\Smt 4\Metnum\student_performance.csv'

# Membaca file CSV dengan delimiter ';'
data = pd.read_csv(file_path, delimiter=';')

# Normalisasi data: mengganti '91.00.00' dengan '91' di kolom NT
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

# Mendefinisikan fungsi linear
def fungsi_linear(x, m, c):
    return m * x + c

# Melakukan fitting fungsi linear ke data
popt, pcov = curve_fit(fungsi_linear, NL, NT)

# Mengekstrak koefisien
m, c = popt

# Menghitung nilai prediksi
y_pred_linear = [fungsi_linear(x, m, c) for x in NL]

# Menghitung galat RMS
error_sum = sum((nt - yp) ** 2 for nt, yp in zip(NT, y_pred_linear))
rms_error_linear = (error_sum / len(NT)) ** 0.5
print("Galat RMS (Regresi Linear):", rms_error_linear)

# Memplot titik data dan garis regresi linear dengan ukuran titik yang lebih kecil
plt.scatter(NL, NT, label='Titik Data', s=1)  # Mengurangi ukuran titik data
plt.plot(NL, y_pred_linear, color='red', label='Fit Linear: y = {:.2f}x + {:.2f}'.format(m, c))
plt.xlabel('Sample Question Papers Practiced (NL)')
plt.ylabel('Performance Index (NT)')
plt.legend()
plt.title('Regresi Linear dari Performance Index vs. Sample Question Papers Practiced')
plt.show()

# Menguji kode
def uji_regresi():
    assert isinstance(rms_error_linear, float), "Galat RMS linear tidak valid"
    assert len(NL) == len(NT), "Panjang data tidak sesuai"
    assert len(NL) > 0, "Array data kosong"

uji_regresi()
