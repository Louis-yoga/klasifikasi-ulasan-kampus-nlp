import sys
import os
import pandas as pd
sys.path.insert(0, '.')

from src.classifier import build_and_train, predict_single, LABEL_SENTIMEN, LABEL_KATEGORI, save_model
from src.feature_extractor import save_vectorizer
from src.evaluator import evaluate_both_models

print("=== MELATIH ULANG MODEL DENGAN DATASET BOOK12.XLSX ===")
df = pd.read_csv('data/dataset.csv')
print("Dataset size:", df.shape)

train_df, test_df, vectorizer, model_s, model_k = build_and_train(df, random_state=42)

os.makedirs('models', exist_ok=True)
save_vectorizer(vectorizer, 'models/vectorizer.pkl')
save_model(model_s, 'models/model_sentimen.pkl')
save_model(model_k, 'models/model_kategori.pkl')
print("Model baru tersimpan di models/: vectorizer.pkl, model_sentimen.pkl, model_kategori.pkl")

eval_r = evaluate_both_models(model_s, model_k, test_df, vectorizer)
es = eval_r['sentimen']
ek = eval_r['kategori']

print("\n--- MODEL SENTIMEN (3 KELAS: Positif, Negatif, Netral) ---")
print(f"Accuracy : {es['accuracy']*100:.1f}%")
print(f"Precision: {es['precision_macro']*100:.1f}% (Macro)")
print(f"Recall   : {es['recall_macro']*100:.1f}% (Macro)")
print(f"F1-Score : {es['f1_macro']*100:.1f}% (Macro)")
print("Confusion Matrix:\n", es['conf_matrix'])

print("\n--- MODEL KATEGORI (4 KELAS: Administrasi, Akademik, Fasilitas, Kemahasiswaan) ---")
print(f"Accuracy : {ek['accuracy']*100:.1f}%")
print(f"Precision: {ek['precision_macro']*100:.1f}% (Macro)")
print(f"Recall   : {ek['recall_macro']*100:.1f}% (Macro)")
print(f"F1-Score : {ek['f1_macro']*100:.1f}% (Macro)")
print("Confusion Matrix:\n", ek['conf_matrix'])

print("\n=== UJI PREDIKSI BATCH DENGAN DATA/BOOK2.CSV ===")
from src.classifier import predict_batch
df_b2 = pd.read_csv('data/Book2.csv', sep=';')
df_hasil_b2 = predict_batch(df_b2, vectorizer, model_s, model_k)
print("Hasil klasifikasi Book2.csv (100 ulasan):")
print("\nDistribusi Sentimen Terprediksi:")
print(df_hasil_b2['sentimen'].value_counts())
print("\nDistribusi Kategori Terprediksi:")
print(df_hasil_b2['kategori'].value_counts())
print("\nDistribusi Low Confidence:")
print(df_hasil_b2['low_confidence'].value_counts())

# Simpan hasil klasifikasi Book2.csv ke data/hasil_klasifikasi.csv
df_hasil_b2.to_csv('data/hasil_klasifikasi.csv', index=False, encoding='utf-8-sig')
print("\nHasil klasifikasi Book2.csv berhasil diperbarui di data/hasil_klasifikasi.csv!")
