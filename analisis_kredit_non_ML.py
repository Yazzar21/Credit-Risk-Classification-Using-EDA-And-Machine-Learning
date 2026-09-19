import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def inspeksi_dataset(lokasi_file):
    df = pd.read_csv(lokasi_file)

    print("--- DIMENSI DATA ---")
    jumlah_baris = df.shape[0]
    jumlah_kolom = df.shape[1]
    print(f"Dataset ini memiliki {jumlah_baris} baris dan {jumlah_kolom} kolom.\n")
    
    print("--- DATA KOSONG (MISSING VALUES) ---")
    data_kosong = df.isnull().sum()
    
    print(data_kosong[data_kosong > 0]) 
    print("\n")
    
    print("--- TIPE DATA KOLOM ---")
    print(df.dtypes)

def bersihkan_data(df):
    print("\n--- FASE 2: PEMBERSIHAN DATA ---")
    
    median_emp_length = df['person_emp_length'].median()
    df['person_emp_length'] = df['person_emp_length'].fillna(median_emp_length)
    
    median_int_rate = df['loan_int_rate'].median()
    df['loan_int_rate'] = df['loan_int_rate'].fillna(median_int_rate)
    
    data_kosong_sisa = df.isnull().sum()
    print("Jumlah data kosong setelah dibersihkan:")
    sisa = data_kosong_sisa[data_kosong_sisa > 0]
    if sisa.empty:
        print("Status: BERSIH (0 missing values)")
    else:
        print(sisa)
        
    return df

def deteksi_outlier(df):
    print("\n--- FASE 3: DETEKSI PENCILAN (OUTLIER) ---")
    
    kolom_fokus = ['person_age', 'person_emp_length']
    ringkasan_statistik = df[kolom_fokus].describe()
    
    print("Ringkasan Statistik (Umur & Lama Bekerja):")
    print(ringkasan_statistik)
    
    print("\n--- PENGECEKAN DATA MUSTAHIL ---")
    umur_mustahil = df[df['person_age'] > 80]
    kerja_mustahil = df[df['person_emp_length'] > 60]
    
    print(f"Jumlah baris dengan umur > 80 tahun: {len(umur_mustahil)}")
    print(f"Jumlah baris dengan lama kerja > 60 tahun: {len(kerja_mustahil)}")

def bersihkan_outlier_dan_visualisasi(df):
    print("\n--- FASE 4: HAPUS OUTLIER & VISUALISASI ---")
    
    df_final = df[(df['person_age'] <= 80) & (df['person_emp_length'] <= 60)]
    
    baris_dibuang = len(df) - len(df_final)
    print(f"Berhasil membuang {baris_dibuang} baris data aneh (outlier).")
    print(f"Sisa data bersih siap analisis: {len(df_final)} baris.")
    
    print("\nMenyiapkan grafik... (Silakan cek jendela popup yang muncul)")
    
    plt.figure(figsize=(10, 5))
    
    sns.countplot(data=df_final, x='loan_intent', hue= 'loan_intent', palette='Set2', legend=False)
    
    plt.title('Distribusi Tujuan Pinjaman Nasabah', fontsize=14, fontweight='bold')
    plt.xlabel('Tujuan Pinjaman (Loan Intent)', fontsize=12)
    plt.ylabel('Jumlah Nasabah', fontsize=12)
    
    plt.show()
    
    return df_final

def analisis_bivariat(df):
    print("\n--- FASE 5: ANALISIS BIVARIAT (KORELASI) ---")
    print("Menyiapkan grafik korelasi status gagal bayar... (Silakan cek jendela popup)")
    
    plt.figure(figsize=(12, 6))
    
    sns.countplot(data=df, x='loan_intent', hue='loan_status', palette='Set1')
    
    plt.title('Status Pinjaman Berdasarkan Tujuan (0 = Lancar, 1 = Macet)', fontsize=14, fontweight='bold')
    plt.xlabel('Tujuan Pinjaman (Loan Intent)', fontsize=12)
    plt.ylabel('Jumlah Nasabah', fontsize=12)
    
    plt.legend(title='Status', labels=['Lancar (0)', 'Gagal Bayar (1)'])
    
    plt.show()

def simpan_data_bersih(df, nama_file_baru):
    print("\n--- FASE 6: MENYIMPAN DATA BERSIH ---")
    
    df.to_csv(nama_file_baru, index=False)
    
    print(f"SUKSES! Data bersih berhasil disimpan dengan nama: {nama_file_baru}")

# ==========================================
# --- AREA EKSEKUSI UTAMA ---
# ==========================================

# 1. Menentukan Lokasi File Dataset Mentahnya
path_dataset = 'Credit Risk Dataset/credit_risk_dataset.csv'

# 2. Memuat Dataset Mentah ke Variabel Utama
df_utama = pd.read_csv(path_dataset)

# 3. Menginspeksi Dataset Mentah
inspeksi_dataset(path_dataset)

# 4. Membersihkan Data Mentah Tadi
df_bersih = bersihkan_data(df_utama)

# 5. Mendeteksi Outlier Pada Dataset yang Sudah Bersih
deteksi_outlier(df_bersih)

# 6. Menghapus Outlier dan Memvisualisasikan Distribusi Dari Dataset yang Sudah Bersih
df_final = bersihkan_outlier_dan_visualisasi(df_bersih)

# 7. Menganalisis Korelasi Bivariat Dari Dataset yang Sudah Bersih
analisis_bivariat(df_final)

# 8. Mengekspor Dataset Bersih ke File CSV yang Baru
path_simpan = 'Credit Risk Dataset/credit_risk_dataset_bersih.csv'
simpan_data_bersih(df_final, path_simpan)