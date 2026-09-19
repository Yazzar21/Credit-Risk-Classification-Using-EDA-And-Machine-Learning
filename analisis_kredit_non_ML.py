import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def inspeksi_dataset(lokasi_file):
    # Membaca data CSV ke dalam DataFrame (tabel virtual)
    df = pd.read_csv(lokasi_file)
    
    # 1. Menghitung total baris dan kolom
    print("--- DIMENSI DATA ---")
    jumlah_baris = df.shape[0]
    jumlah_kolom = df.shape[1]
    print(f"Dataset ini memiliki {jumlah_baris} baris dan {jumlah_kolom} kolom.\n")
    
    # 2. Mendeteksi data yang hilang/kosong
    print("--- DATA KOSONG (MISSING VALUES) ---")
    data_kosong = df.isnull().sum()
    # Memfilter agar terminal hanya mencetak kolom yang benar-benar ada data kosongnya
    print(data_kosong[data_kosong > 0]) 
    print("\n")
    
    # 3. Memeriksa tipe data masing-masing kolom
    print("--- TIPE DATA KOLOM ---")
    print(df.dtypes)

def bersihkan_data(df):
    print("\n--- FASE 2: PEMBERSIHAN DATA ---")
    
    # 1. Mengisi data kosong dengan nilai median (nilai tengah)
    median_emp_length = df['person_emp_length'].median()
    df['person_emp_length'] = df['person_emp_length'].fillna(median_emp_length)
    
    median_int_rate = df['loan_int_rate'].median()
    df['loan_int_rate'] = df['loan_int_rate'].fillna(median_int_rate)
    
    # 2. Verifikasi ulang apakah masih ada data kosong
    data_kosong_sisa = df.isnull().sum()
    print("Jumlah data kosong setelah dibersihkan:")
    # Jika Series kosong (tidak ada > 0), cetak keterangan aman
    sisa = data_kosong_sisa[data_kosong_sisa > 0]
    if sisa.empty:
        print("Status: BERSIH (0 missing values)")
    else:
        print(sisa)
        
    return df

def deteksi_outlier(df):
    print("\n--- FASE 3: DETEKSI PENCILAN (OUTLIER) ---")
    
    # 1. Menampilkan ringkasan statistik khusus untuk kolom umur dan lama bekerja
    # Kita menggunakan describe() untuk melihat nilai rata-rata, min, dan max
    kolom_fokus = ['person_age', 'person_emp_length']
    ringkasan_statistik = df[kolom_fokus].describe()
    
    print("Ringkasan Statistik (Umur & Lama Bekerja):")
    print(ringkasan_statistik)
    
    # 2. Mari kita buat filter logika untuk mengecek apakah ada umur yang mustahil (misal > 80 tahun)
    print("\n--- PENGECEKAN DATA MUSTAHIL ---")
    umur_mustahil = df[df['person_age'] > 80]
    kerja_mustahil = df[df['person_emp_length'] > 60]
    
    print(f"Jumlah baris dengan umur > 80 tahun: {len(umur_mustahil)}")
    print(f"Jumlah baris dengan lama kerja > 60 tahun: {len(kerja_mustahil)}")

def bersihkan_outlier_dan_visualisasi(df):
    print("\n--- FASE 4: HAPUS OUTLIER & VISUALISASI ---")
    
    # 1. Membuang data yang tidak masuk akal
    # Logika: "Pertahankan data yang umurnya <= 80 DAN kerjanya <= 60"
    df_final = df[(df['person_age'] <= 80) & (df['person_emp_length'] <= 60)]
    
    baris_dibuang = len(df) - len(df_final)
    print(f"Berhasil membuang {baris_dibuang} baris data aneh (outlier).")
    print(f"Sisa data bersih siap analisis: {len(df_final)} baris.")
    
    # 2. Visualisasi Data: Alasan Nasabah Meminjam Uang
    print("\nMenyiapkan grafik... (Silakan cek jendela popup yang muncul)")
    
    # Mengatur ukuran kanvas
    plt.figure(figsize=(10, 5))
    
    # Membuat diagram batang (Bar Chart)
    
    sns.countplot(data=df_final, x='loan_intent', hue= 'loan_intent', palette='Set2', legend=False)
    
    #Memberi judul dan label
    
    plt.title('Distribusi Tujuan Pinjaman Nasabah', fontsize=14, fontweight='bold')
    plt.xlabel('Tujuan Pinjaman (Loan Intent)', fontsize=12)
    plt.ylabel('Jumlah Nasabah', fontsize=12)
    
    # Menampilkan grafik ke layar
    plt.show()
    
    return df_final

def analisis_bivariat(df):
    print("\n--- FASE 5: ANALISIS BIVARIAT (KORELASI) ---")
    print("Menyiapkan grafik korelasi status gagal bayar... (Silakan cek jendela popup)")
    
    # Mengatur ukuran kanvas sedikit lebih lebar
    plt.figure(figsize=(12, 6))
    
    # Menggunakan countplot, tapi kali ini kita tambahkan argumen 'hue'
    # hue='loan_status' akan membelah setiap batang berdasarkan status pinjaman
    sns.countplot(data=df, x='loan_intent', hue='loan_status', palette='Set1')
    
    # Memberi judul dan label
    plt.title('Status Pinjaman Berdasarkan Tujuan (0 = Lancar, 1 = Macet)', fontsize=14, fontweight='bold')
    plt.xlabel('Tujuan Pinjaman (Loan Intent)', fontsize=12)
    plt.ylabel('Jumlah Nasabah', fontsize=12)
    
    # Mengganti teks pada legenda agar lebih manusiawi dibaca
    plt.legend(title='Status', labels=['Lancar (0)', 'Gagal Bayar (1)'])
    
    # Menampilkan grafik ke layar
    plt.show()

def simpan_data_bersih(df, nama_file_baru):
    print("\n--- FASE 6: MENYIMPAN DATA BERSIH ---")
    
    # Mengekspor tabel virtual (DataFrame) kembali menjadi wujud file fisik CSV
    # index=False mencegah Pandas menambahkan kolom nomor urut baru yang tidak berguna
    df.to_csv(nama_file_baru, index=False)
    
    print(f"SUKSES! Data bersih berhasil disimpan dengan nama: {nama_file_baru}")

# ==========================================
# --- AREA EKSEKUSI UTAMA ---
# ==========================================

# 1. Tentukan lokasi file
path_dataset = 'Credit Risk Dataset/credit_risk_dataset.csv'

# 2. Muat data awal ke variabel utama
df_utama = pd.read_csv(path_dataset)

# 3. Eksekusi Fase 1: Inspeksi Data
inspeksi_dataset(path_dataset)

# 4. Eksekusi Fase 2: Pembersihan Data
df_bersih = bersihkan_data(df_utama)

# 5. Eksekusi Fase 3: Deteksi Outlier
deteksi_outlier(df_bersih)

# 6. Eksekusi Fase 4: Hapus Outlier & Visualisasi
df_final = bersihkan_outlier_dan_visualisasi(df_bersih)

# 7. Eksekusi Fase 5: Analisis Bivariat (TAMBAHKAN BARIS INI)
analisis_bivariat(df_final)

# 8. Eksekusi Fase 6: Ekspor Data Bersih
path_simpan = 'Credit Risk Dataset/credit_risk_dataset_bersih.csv'
simpan_data_bersih(df_final, path_simpan)