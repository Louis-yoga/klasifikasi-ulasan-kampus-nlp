"""
scripts/evaluate.py
-------------------
Skrip evaluasi model independen terhadap dataset eksternal (data/external_test.csv)
serta verifikasi keselarasan metrik metadata.json (MOD-01, MOD-02).
"""

import os
import sys

# Tambahkan root workspace ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

from src.artifacts import load_artifacts
from src.classifier import LABEL_KATEGORI, LABEL_SENTIMEN
from src.evaluator import evaluate_model
from src.feature_extractor import preprocess_corpus, transform_texts

EXTERNAL_TEST_PATH = os.path.join("data", "external_test.csv")
MODELS_DIR = "models"


def main():
    print("=" * 65)
    print("EVALUASI INDEPENDEN MODEL & UJI GENERALISASI EKSTERNAL (MOD-01)")
    print("=" * 65)

    # 1. Muat artefak model dan metadata
    print(f"\n1. Memuat artefak model dari direktori '{MODELS_DIR}'...")
    try:
        objs, metadata = load_artifacts(MODELS_DIR)
        print("   [OK] Verifikasi SHA-256 dan versi scikit-learn berhasil.")
    except Exception as e:  # noqa: BLE001
        print(f"   [ERROR] Gagal memuat artefak: {e}")
        sys.exit(1)

    # Tampilkan ringkasan dari metadata.json
    print("\n2. Ringkasan Pelatihan (dari models/metadata.json):")
    print(f"   - Algoritma         : {metadata.get('model_algorithm')}")
    print(f"   - Versi Scikit-Learn: {metadata.get('sklearn_version')}")
    print(f"   - Tanggal Pelatihan : {metadata.get('created_utc')}")

    cv_s = metadata['metrics']['sentimen']['cv_repeated_kfold']
    cv_k = metadata['metrics']['kategori']['cv_repeated_kfold']
    print(f"   - CV Sentimen (5x5) : Akurasi {cv_s['accuracy_mean']:.4f} (+/- {cv_s['accuracy_std']:.4f}), F1-Macro {cv_s['f1_macro_mean']:.4f} (+/- {cv_s['f1_macro_std']:.4f})")
    print(f"   - CV Kategori (5x5) : Akurasi {cv_k['accuracy_mean']:.4f} (+/- {cv_k['accuracy_std']:.4f}), F1-Macro {cv_k['f1_macro_mean']:.4f} (+/- {cv_k['f1_macro_std']:.4f})")

    hold_s = metadata['metrics']['sentimen']['holdout']
    hold_k = metadata['metrics']['kategori']['holdout']
    print(f"   - Holdout Test (n=20) Sentimen: Akurasi {hold_s['accuracy']:.4f}, F1-Macro {hold_s['f1_macro']:.4f}")
    print(f"   - Holdout Test (n=20) Kategori: Akurasi {hold_k['accuracy']:.4f}, F1-Macro {hold_k['f1_macro']:.4f}")

    # 2. Evaluasi pada External Test Set
    if not os.path.exists(EXTERNAL_TEST_PATH):
        print(f"\n[Warning] External test set '{EXTERNAL_TEST_PATH}' tidak ditemukan.")
        return

    print(f"\n3. Mengevaluasi pada External Held-Out Test Set ({EXTERNAL_TEST_PATH})...")
    df_ext = pd.read_csv(EXTERNAL_TEST_PATH)
    print(f"   Total data eksternal: {len(df_ext)} baris (100% unseen)")

    ext_corpus = preprocess_corpus(df_ext['ulasan'].tolist())
    X_ext = transform_texts(ext_corpus, objs['vectorizer'])

    res_sentimen = evaluate_model(
        objs['sentimen'], X_ext, df_ext['sentimen'].tolist(), LABEL_SENTIMEN
    )
    res_kategori = evaluate_model(
        objs['kategori'], X_ext, df_ext['kategori'].tolist(), LABEL_KATEGORI
    )

    print("\n--- HASIL EVALUASI EKSTERNAL (SENTIMEN) ---")
    print(f"Akurasi        : {res_sentimen['accuracy']:.4f}")
    print(f"Macro Precision: {res_sentimen['precision_macro']:.4f}")
    print(f"Macro Recall   : {res_sentimen['recall_macro']:.4f}")
    print(f"Macro F1-Score : {res_sentimen['f1_macro']:.4f}")
    print("\nPer-class Metrics (Sentimen):")
    print(res_sentimen['per_class'].to_string(index=False))
    print("\nConfusion Matrix (Sentimen):\n", res_sentimen['conf_matrix'])

    print("\n--- HASIL EVALUASI EKSTERNAL (KATEGORI) ---")
    print(f"Akurasi        : {res_kategori['accuracy']:.4f}")
    print(f"Macro Precision: {res_kategori['precision_macro']:.4f}")
    print(f"Macro Recall   : {res_kategori['recall_macro']:.4f}")
    print(f"Macro F1-Score : {res_kategori['f1_macro']:.4f}")
    print("\nPer-class Metrics (Kategori):")
    print(res_kategori['per_class'].to_string(index=False))
    print("\nConfusion Matrix (Kategori):\n", res_kategori['conf_matrix'])


if __name__ == '__main__':
    main()
