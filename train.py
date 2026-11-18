"""
train.py
--------
Skrip pelatihan terpisah untuk model Multinomial Naive Bayes (SEC-04, MOD-01).
Melakukan:
1. Validasi integritas dan hashing dataset (data/dataset.csv).
2. Stratified Split 80/20 bebas data leakage (random_state=42).
3. Evaluasi Repeated Stratified K-Fold (5x5).
4. Evaluasi holdout test set (20 sampel).
5. Penyimpanan artefak model atomik dan metadata.json bersertifikasi SHA-256.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.artifacts import save_artifacts, sha256
from src.classifier import LABEL_KATEGORI, LABEL_SENTIMEN
from src.evaluator import evaluate_model
from src.feature_extractor import build_vectorizer, preprocess_corpus, transform_texts

DATA_PATH = os.path.join("data", "dataset.csv")
MODELS_DIR = "models"


def run_training():
    print("=" * 60)
    print("PROSES PELATIHAN MODEL KLASIFIKASI ULASAN KAMPUS (SEC-04, MOD-01)")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset tidak ditemukan di {DATA_PATH}")
        sys.exit(1)

    # 1. Hitung SHA-256 dataset
    data_hash = sha256(DATA_PATH)
    print(f"Dataset SHA-256: {data_hash}")

    df = pd.read_csv(DATA_PATH)
    print(f"Total baris dataset: {len(df)}")

    # Validasi kolom wajib
    required_cols = {'id', 'ulasan', 'sentimen', 'kategori'}
    if not required_cols.issubset(df.columns):
        print(f"Error: Dataset harus memiliki kolom: {required_cols}")
        sys.exit(1)

    # 2. Pra-pemrosesan korpus ulasan
    print("\nMelakukan pra-pemrosesan teks...")
    clean_corpus = preprocess_corpus(df['ulasan'].tolist())
    df['clean_ulasan'] = clean_corpus

    # 3. Kunci stratifikasi gabungan untuk 80/20 split
    strat_key = df['kategori'].astype(str) + "_" + df['sentimen'].astype(str)
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=strat_key
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Ukuran Data Latih: {len(train_df)} baris")
    print(f"Ukuran Data Uji   : {len(test_df)} baris")

    # 4. Fit vectorizer hanya pada data latih (Zero Leakage)
    print("\nMembangun CountVectorizer (Unigram + Bigram) pada data latih...")
    vectorizer = build_vectorizer(train_df['clean_ulasan'].tolist(), ngram_range=(1, 2))
    X_train = transform_texts(train_df['clean_ulasan'].tolist(), vectorizer)
    X_test = transform_texts(test_df['clean_ulasan'].tolist(), vectorizer)

    # 5. Latih model MultinomialNB
    print("Melatih MultinomialNB (alpha=0.2) untuk Sentimen...")
    model_sentimen = MultinomialNB(alpha=0.2)
    model_sentimen.fit(X_train, train_df['sentimen'])

    print("Melatih MultinomialNB (alpha=0.2) untuk Kategori...")
    model_kategori = MultinomialNB(alpha=0.2)
    model_kategori.fit(X_train, train_df['kategori'])

    # 6. Evaluasi Holdout Test Set (20 sampel)
    print("\nEvaluasi Holdout Test Set (n=20):")
    eval_sentimen = evaluate_model(
        model_sentimen, X_test, test_df['sentimen'].tolist(), LABEL_SENTIMEN
    )
    eval_kategori = evaluate_model(
        model_kategori, X_test, test_df['kategori'].tolist(), LABEL_KATEGORI
    )

    print(f"  Sentimen -> Akurasi: {eval_sentimen['accuracy']:.4f}, Macro F1: {eval_sentimen['f1_macro']:.4f}")
    print(f"  Kategori -> Akurasi: {eval_kategori['accuracy']:.4f}, Macro F1: {eval_kategori['f1_macro']:.4f}")

    # 7. Repeated Stratified K-Fold CV (5 fold x 5 repeat = 25 evaluasi)
    print("\nMenghitung Repeated Stratified K-Fold Cross-Validation (5x5)...")
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    cv_sent_acc, cv_sent_f1 = [], []
    cv_kat_acc, cv_kat_f1 = [], []

    for fold_train_idx, fold_val_idx in rskf.split(df, strat_key):
        f_train = df.iloc[fold_train_idx]
        f_val = df.iloc[fold_val_idx]

        f_vec = build_vectorizer(f_train['clean_ulasan'].tolist(), ngram_range=(1, 2))
        f_X_train = transform_texts(f_train['clean_ulasan'].tolist(), f_vec)
        f_X_val = transform_texts(f_val['clean_ulasan'].tolist(), f_vec)

        # Sentimen
        m_s = MultinomialNB(alpha=0.2)
        m_s.fit(f_X_train, f_train['sentimen'])
        preds_s = m_s.predict(f_X_val)
        cv_sent_acc.append(accuracy_score(f_val['sentimen'], preds_s))
        cv_sent_f1.append(f1_score(f_val['sentimen'], preds_s, average='macro', zero_division=0))

        # Kategori
        m_k = MultinomialNB(alpha=0.2)
        m_k.fit(f_X_train, f_train['kategori'])
        preds_k = m_k.predict(f_X_val)
        cv_kat_acc.append(accuracy_score(f_val['kategori'], preds_k))
        cv_kat_f1.append(f1_score(f_val['kategori'], preds_k, average='macro', zero_division=0))

    cv_summary = {
        'sentimen': {
            'n_splits': 5,
            'n_repeats': 5,
            'accuracy_mean': round(float(np.mean(cv_sent_acc)), 4),
            'accuracy_std': round(float(np.std(cv_sent_acc)), 4),
            'f1_macro_mean': round(float(np.mean(cv_sent_f1)), 4),
            'f1_macro_std': round(float(np.std(cv_sent_f1)), 4)
        },
        'kategori': {
            'n_splits': 5,
            'n_repeats': 5,
            'accuracy_mean': round(float(np.mean(cv_kat_acc)), 4),
            'accuracy_std': round(float(np.std(cv_kat_acc)), 4),
            'f1_macro_mean': round(float(np.mean(cv_kat_f1)), 4),
            'f1_macro_std': round(float(np.std(cv_kat_f1)), 4)
        }
    }
    print(f"  CV Sentimen -> Akurasi: {cv_summary['sentimen']['accuracy_mean']:.4f} (+/- {cv_summary['sentimen']['accuracy_std']:.4f}), Macro F1: {cv_summary['sentimen']['f1_macro_mean']:.4f} (+/- {cv_summary['sentimen']['f1_macro_std']:.4f})")
    print(f"  CV Kategori -> Akurasi: {cv_summary['kategori']['accuracy_mean']:.4f} (+/- {cv_summary['kategori']['accuracy_std']:.4f}), Macro F1: {cv_summary['kategori']['f1_macro_mean']:.4f} (+/- {cv_summary['kategori']['f1_macro_std']:.4f})")

    # 8. Susun metadata lengkap
    metadata = {
        'model_algorithm': 'Multinomial Naive Bayes (Laplace alpha=1.0) with Bag-of-Words (Unigram + Bigram)',
        'dataset': {
            'path': DATA_PATH,
            'sha256': data_hash,
            'total_rows': len(df),
            'train_rows': len(train_df),
            'test_rows': len(test_df),
            'stratification': 'stratified_80_20_compound',
            'random_state': 42
        },
        'classes': {
            'sentimen': LABEL_SENTIMEN,
            'kategori': LABEL_KATEGORI
        },
        'distribution': {
            'sentimen': df['sentimen'].value_counts().to_dict(),
            'kategori': df['kategori'].value_counts().to_dict()
        },
        'metrics': {
            'sentimen': {
                'holdout': {
                    'n': len(test_df),
                    'accuracy': eval_sentimen['accuracy'],
                    'precision_macro': eval_sentimen['precision_macro'],
                    'recall_macro': eval_sentimen['recall_macro'],
                    'f1_macro': eval_sentimen['f1_macro'],
                    'conf_matrix': eval_sentimen['conf_matrix'].tolist(),
                    'per_class': eval_sentimen['per_class'].to_dict(orient='records')
                },
                'cv_repeated_kfold': cv_summary['sentimen']
            },
            'kategori': {
                'holdout': {
                    'n': len(test_df),
                    'accuracy': eval_kategori['accuracy'],
                    'precision_macro': eval_kategori['precision_macro'],
                    'recall_macro': eval_kategori['recall_macro'],
                    'f1_macro': eval_kategori['f1_macro'],
                    'conf_matrix': eval_kategori['conf_matrix'].tolist(),
                    'per_class': eval_kategori['per_class'].to_dict(orient='records')
                },
                'cv_repeated_kfold': cv_summary['kategori']
            }
        }
    }

    # 9. Simpan artefak secara atomik
    print("\nMenyimpan artefak model dan metadata.json ke direktori models/...")
    objs = {
        'vectorizer': vectorizer,
        'sentimen': model_sentimen,
        'kategori': model_kategori
    }
    saved_meta = save_artifacts(MODELS_DIR, objs, metadata)
    print("Pelatihan selesai dengan sukses!")
    print(f"File hashes tersimpan: {saved_meta.get('files')}")


if __name__ == '__main__':
    run_training()
