import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# ==========================================
# --- AREA FUNGSI (RESEP) ---
# ==========================================

def siapkan_data_ml(lokasi_file):
    print("--- FASE 1: PERSIAPAN DATA (ENCODING) ---")
    df = pd.read_csv(lokasi_file)
    print(f"Dimensi data awal: {df.shape}")
    
    df_encoded = pd.get_dummies(df, drop_first=True)
    print(f"Dimensi data setelah Encoding: {df_encoded.shape}")
    
    y = df_encoded['loan_status']
    X = df_encoded.drop('loan_status', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Jumlah data untuk dilatih (Train): {X_train.shape[0]} baris")
    print(f"Jumlah data untuk diuji (Test): {X_test.shape[0]} baris\n")
    
    return X_train, X_test, y_train, y_test

def lihat_fitur_penting(model, X_train):
    print("\n--- FASE 4: MENGUNGKAP FITUR TERPENTING ---")
    print("Menyiapkan grafik feature importance... (Silakan cek popup)")
    
    # Mengambil persentase kepentingan masing-masing kolom dari model
    bobot_fitur = model.feature_importances_
    
    # Menyatukan nama kolom dan bobotnya ke dalam satu tabel, lalu diurutkan dari yang terbesar
    df_bobot = pd.DataFrame({
        'Fitur': X_train.columns,
        'Bobot': bobot_fitur
    }).sort_values(by='Bobot', ascending=False)
    
    # Menggambar grafik batang untuk 10 faktor paling atas
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_bobot.head(10), x='Bobot', y='Fitur', hue='Fitur', palette='viridis', legend=False)
    plt.title('10 Faktor Paling Menentukan Risiko Kredit Macet', fontsize=14, fontweight='bold')
    plt.xlabel('Tingkat Kepentingan (Bobot)', fontsize=12)
    plt.ylabel('Faktor / Variabel', fontsize=12)
    plt.tight_layout()
    plt.show()

def visualisasi_metrik(model, X_test, y_test):
    print("\n--- FASE 5: VISUALISASI METRIK PERFORMA ---")
    print("Menyiapkan grafik diagram batang metrik... (Silakan cek popup)")
    
    # 1. Mesin melakukan prediksi ulang untuk mendapatkan data tebakan
    y_prediksi = model.predict(X_test)
    
    # 2. Menghitung nilai masing-masing metrik secara spesifik
    akurasi = accuracy_score(y_test, y_prediksi)
    presisi = precision_score(y_test, y_prediksi)
    recall = recall_score(y_test, y_prediksi)
    f1 = f1_score(y_test, y_prediksi)
    
    # 3. Menyiapkan label dan nilai untuk sumbu X dan Y
    nama_metrik = ['Akurasi', 'Presisi', 'Recall', 'F1-Score']
    nilai_metrik = [akurasi, presisi, recall, f1]
    
    # 4. Menggambar diagram batang
    plt.figure(figsize=(8, 5))
    sns.barplot(x=nama_metrik, y=nilai_metrik, palette='Blues_r')
    
    # Menambahkan teks angka persentase tepat di atas masing-masing batang
    for i, nilai in enumerate(nilai_metrik):
        plt.text(i, nilai + 0.02, f'{nilai*100:.1f}%', ha='center', fontsize=11, fontweight='bold')
        
    plt.ylim(0, 1.1) # Memperluas atap grafik agar angka tidak terpotong
    plt.title('Perbandingan Metrik Evaluasi Model', fontsize=14, fontweight='bold')
    plt.ylabel('Skor (0.0 - 1.0)', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def latih_model_ml(X_train, y_train):
    print("--- FASE 2: MELATIH MODEL (TRAINING) ---")
    print("Mesin sedang mempelajari pola data, mohon tunggu sebentar...")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("SUKSES! Mesin telah selesai belajar dan membuat aturan klasifikasi.")
    return model

def evaluasi_model_ml(model, X_test, y_test):
    print("\n--- FASE 3: EVALUASI KECERDASAN MODEL ---")
    print("Mesin sedang mengerjakan soal ujian (memprediksi 6.515 data)...")
    
    # 1. Tebakan mesin
    y_prediksi = model.predict(X_test)
    
    # 2. Rapor teks (Classification Report)
    print("\nRapor Nilai Mesin (Classification Report):")
    print(classification_report(y_test, y_prediksi))
    
    # 3. Visualisasi Confusion Matrix
    print("Menyiapkan grafik Confusion Matrix... (Silakan cek popup)")
    
    # Menghitung angka persimpangan antara tebakan dan kenyataan
    cm = confusion_matrix(y_test, y_prediksi)
    
    # Menggambar matriks ke dalam kanvas
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Lancar (0)', 'Macet (1)'])
    
    # values_format='d' memastikan angka ditampilkan utuh, bukan format ilmiah (e.g., 1e4)
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    
    plt.title('Confusion Matrix - Prediksi Risiko Kredit', fontsize=14, fontweight='bold')
    plt.xlabel('Tebakan Mesin (Predicted Label)', fontsize=12)
    plt.ylabel('Kenyataan Asli (True Label)', fontsize=12)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# --- AREA EKSEKUSI UTAMA ---
# ==========================================

# 1. Tentukan lokasi data
path_dataset_bersih = 'Credit Risk Dataset/credit_risk_dataset_bersih.csv'

# 2. Eksekusi Fase 1 (Persiapan)
X_train, X_test, y_train, y_test = siapkan_data_ml(path_dataset_bersih)

# 3. Eksekusi Fase 2 (Pelatihan)
model_klasifikasi = latih_model_ml(X_train, y_train)

# (Tambahkan baris ini)
# 4. Eksekusi Fase 3 (Evaluasi)
evaluasi_model_ml(model_klasifikasi, X_test, y_test)

# 5. Eksekusi Fase 4 (Melihat Faktor Penentu)
lihat_fitur_penting(model_klasifikasi, X_train)

# 6. Eksekusi Fase 5 (Visualisasi Metrik)
visualisasi_metrik(model_klasifikasi, X_test, y_test)