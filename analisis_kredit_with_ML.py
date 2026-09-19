import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score

def siapkan_data_ml(lokasi_file):
    print("--- FASE 1: PERSIAPAN DATA (ENCODING) ---")
    df = pd.read_csv(lokasi_file)
    print(f"Dimensi data awal: {df.shape}")
    
    df_encoded = pd.get_dummies(df, drop_first=True)
    print(f"Dimensi data setelah Encoding: {df_encoded.shape}")
    
    y = df_encoded['loan_status']
    X = df_encoded.drop('loan_status', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Train: {X_train.shape[0]} baris")
    print(f"Test: {X_test.shape[0]} baris\n")
    
    return X_train, X_test, y_train, y_test

def lihat_fitur_penting(model, X_train):
    print("\n--- FASE 4: FITUR TERPENTING (FEATURE IMPORTANCE) ---")
    print("Menyiapkan grafik feature importance")
    
    bobot_fitur = model.feature_importances_
    
    df_bobot = pd.DataFrame({
        'Fitur': X_train.columns,
        'Bobot': bobot_fitur
    }).sort_values(by='Bobot', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_bobot.head(10), x='Bobot', y='Fitur', hue='Fitur', palette='viridis', legend=False)
    plt.title('10 Faktor Paling Menentukan Risiko Kredit Macet', fontsize=14, fontweight='bold')
    plt.xlabel('Tingkat Kepentingan (Bobot)', fontsize=12)
    plt.ylabel('Faktor / Variabel', fontsize=12)
    plt.tight_layout()
    plt.show()

def visualisasi_metrik(model, X_test, y_test):
    print("\n--- FASE 5: VISUALISASI METRIK PERFORMA ---")
    print("Menyiapkan grafik diagram batang metrik")
    
    y_prediksi = model.predict(X_test)
    
    akurasi = accuracy_score(y_test, y_prediksi)
    presisi = precision_score(y_test, y_prediksi)
    recall = recall_score(y_test, y_prediksi)
    f1 = f1_score(y_test, y_prediksi)
    
    nama_metrik = ['Akurasi', 'Presisi', 'Recall', 'F1-Score']
    nilai_metrik = [akurasi, presisi, recall, f1]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=nama_metrik, y=nilai_metrik, palette='Blues_r')
    
    for i, nilai in enumerate(nilai_metrik):
        plt.text(i, nilai + 0.02, f'{nilai*100:.1f}%', ha='center', fontsize=11, fontweight='bold')
        
    plt.ylim(0, 1.1)
    plt.title('Perbandingan Metrik Evaluasi Model', fontsize=14, fontweight='bold')
    plt.ylabel('Skor (0.0 - 1.0)', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def latih_model_ml(X_train, y_train):
    print("--- FASE 2: MELATIH MODEL (TRAINING) ---")
    print("Sedang Proses Belajar Pola Dari Datasetnya")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("SUKSES!")
    return model

def evaluasi_model_ml(model, X_test, y_test):
    print("\n--- FASE 3: EVALUASI KECERDASAN MODEL ---")
    print("Sedang Melakukan Proses")
    
    y_prediksi = model.predict(X_test)
    
    print("\nHasil Klasifikasi (Classification Report):")
    print(classification_report(y_test, y_prediksi))
    
    print("Menyiapkan grafik Confusion Matrix")
    
    cm = confusion_matrix(y_test, y_prediksi)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Lancar (0)', 'Macet (1)'])
    
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    
    plt.title('Confusion Matrix - Prediksi Risiko Kredit', fontsize=14, fontweight='bold')
    plt.xlabel('Tebakan Mesin (Predicted Label)', fontsize=12)
    plt.ylabel('Kenyataan Asli (True Label)', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def bandingkan_berbagai_model(daftar_model, X_train, X_test, y_train, y_test):
    print("\n--- FASE TAMBAHAN: MODEL BENCHMARKING ---")
    print("Melatih berbagai model Machine Learning")
    
    hasil_evaluasi = []
    
    for nama_model, model in daftar_model.items():
        print(f"-> Sedang melatih model: {nama_model}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)
        
        hasil_evaluasi.append({
            "Model": nama_model,
            "Akurasi": round(acc * 100, 2),
            "Presisi": round(prec * 100, 2),
            "Recall": round(rec * 100, 2),
            "F1-Score": round(f1 * 100, 2),
            "ROC-AUC": round(roc * 100, 2)
        })
    
    df_hasil = pd.DataFrame(hasil_evaluasi)
    df_hasil = df_hasil.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
    
    print("\n========================================================")
    print("TABEL PERBANDINGAN PERFORMA MODEL (BENCHMARKING RESULT)")
    print("========================================================")
    print(df_hasil.to_string(index=False))
    print("========================================================")
    
    return df_hasil

def visualisasi_benchmark(df_hasil):
    print("\n--- FASE TAMBAHAN: VISUALISASI GRAFIK BENCHMARKING ---")
    print("Menyiapkan Grafik Perbandingan Model Machine Learning Lain")
    
    plt.figure(figsize=(12, 6))
    
    df_melted = df_hasil.melt(id_vars="Model", value_vars=["F1-Score", "ROC-AUC", "Akurasi"], 
                              var_name="Metrik", value_name="Skor (%)")
    
    sns.barplot(data=df_melted, x="Model", y="Skor (%)", hue="Metrik", palette="viridis")
    
    plt.title("Perbandingan Performa Multi-Model Machine Learning (Credit Risk)", fontsize=14, fontweight='bold')
    plt.xlabel("Algoritma Model", fontsize=12)
    plt.ylabel("Skor Persentase (%)", fontsize=12)
    plt.ylim(0, 110)
    plt.xticks(rotation=15, fontsize=10)
    plt.legend(title="Metrik Evaluasi", loc='lower right')
    
    plt.tight_layout()
    plt.show()

def analisis_detail_tiap_model(daftar_model, X_train, X_test, y_train, y_test):
    print("\n--- FASE ANALISIS MENDALAM: GRAFIK INDIVIDU TIAP MODEL ---")
    
    for nama_model, model in daftar_model.items():
        print(f"\nMemproses visualisasi Untuk Model Machine Learning: {nama_model}...")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(7, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Lancar (0)', 'Macet (1)'])
        disp.plot(cmap='Blues', ax=ax, values_format='d')
        plt.title(f'Confusion Matrix - {nama_model}', fontsize=14, fontweight='bold')
        plt.xlabel('Tebakan Mesin (Predicted Label)', fontsize=12)
        plt.ylabel('Kenyataan Asli (True Label)', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        if hasattr(model, "feature_importances_"):
            bobot = model.feature_importances_
            df_bobot = pd.DataFrame({
                'Fitur': X_train.columns,
                'Bobot': bobot
            }).sort_values(by='Bobot', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_bobot.head(10), x='Bobot', y='Fitur', hue='Fitur', palette='viridis', legend=False)
            plt.title(f'10 Faktor Penentu - {nama_model}', fontsize=14, fontweight='bold')
            plt.xlabel('Tingkat Kepentingan (Bobot)', fontsize=12)
            plt.ylabel('Faktor / Variabel', fontsize=12)
            plt.tight_layout()
            plt.show()
            
        elif hasattr(model, "coef_"):
            bobot = abs(model.coef_[0])
            df_bobot = pd.DataFrame({
                'Fitur': X_train.columns,
                'Bobot': bobot
            }).sort_values(by='Bobot', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_bobot.head(10), x='Bobot', y='Fitur', hue='Fitur', palette='magma', legend=False)
            plt.title(f'10 Faktor Penentu (Koefisien Absolut) - {nama_model}', fontsize=14, fontweight='bold')
            plt.xlabel('Besaran Pengaruh (Koefisien)', fontsize=12)
            plt.ylabel('Faktor / Variabel', fontsize=12)
            plt.tight_layout()
            plt.show()

# ==========================================
# --- AREA EKSEKUSI UTAMA ---
# ==========================================

# 1. Menentukan Lokasi Dataset Bersih
path_dataset_bersih = 'Credit Risk Dataset/credit_risk_dataset_bersih.csv'

# 2. Persiapan Data untuk Machine Learning
X_train, X_test, y_train, y_test = siapkan_data_ml(path_dataset_bersih)

# 3. Pelatihan Baseline Untuk Model Random Forest
model_klasifikasi = latih_model_ml(X_train, y_train)

# 4. Evaluasi Baseline Pada Model Random Forest
evaluasi_model_ml(model_klasifikasi, X_test, y_test)

# 5. Faktor Penting (Feature Importance) Dari Model Random Forest
lihat_fitur_penting(model_klasifikasi, X_train)

# 6. Memvisualiasikan Metrik Evaluasi Model Random Forest
visualisasi_metrik(model_klasifikasi, X_test, y_test)

daftar_model = {
    "Logistic Regression": LogisticRegression(max_iter=5000, solver='liblinear', random_state=42),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
}

# 7. Mengeksekusi Perbandingan Model Machine Learning Lainnya
tabel_perbandingan = bandingkan_berbagai_model(daftar_model, X_train, X_test, y_train, y_test)

# 8. Memvisualisasikan Perbandingan Model Machine Learning Lainnya
visualisasi_benchmark(tabel_perbandingan)

# 9. Menganalisis Detail Tiap Model Machine Learning Lainnya
analisis_detail_tiap_model(daftar_model, X_train, X_test, y_train, y_test)