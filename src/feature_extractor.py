"""
feature_extractor.py
--------------------
Wrapper Feature Extraction (Bag-of-Words & N-Gram TF-IDF) untuk Sistem Klasifikasi Ulasan Kampus.
Satu vectorizer digunakan bersama oleh kedua model MNB (Sentimen & Kategori).
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocessor import tokens_to_string, preprocess_pipeline


def build_vocabulary(corpus: list) -> TfidfVectorizer:
    """
    Fit TfidfVectorizer pada training corpus.
    Menggunakan unigram + bigram dan sublinear_tf=True untuk meningkatkan
    akurasi diskriminasi sentimen & kategori pada ulasan kampus.

    Parameters
    ----------
    corpus : list[str]  — Daftar teks ulasan (string token hasil preprocessing)

    Returns
    -------
    TfidfVectorizer  — Vectorizer yang sudah di-fit
    """
    vectorizer = TfidfVectorizer(
        analyzer='word',
        min_df=1,           # Seluruh fitur kosakata disertakan
        max_df=0.95,        # Abaikan kata yang terlalu umum di >95% dokumen
        ngram_range=(1, 2), # Unigram + Bigram (menangkap frasa penting)
        sublinear_tf=True,  # Reduksi dominasi frekuensi kata ekstrem (1 + log(tf))
    )
    vectorizer.fit(corpus)
    return vectorizer


def preprocess_corpus(texts: list) -> list:
    """
    Jalankan pipeline preprocessing pada daftar teks dan
    kembalikan sebagai daftar string siap-vectorize.

    Parameters
    ----------
    texts : list[str]  — Daftar teks mentah

    Returns
    -------
    list[str]  — Daftar string token yang sudah digabung
    """
    result = []
    for teks in texts:
        hasil = preprocess_pipeline(teks)
        result.append(tokens_to_string(hasil['tokens']))
    return result


def transform_texts(texts: list, vectorizer):
    """
    Transform daftar teks menjadi sparse matrix fitur.

    Parameters
    ----------
    texts      : list[str]  — Teks yang sudah dipreprocess
    vectorizer : TfidfVectorizer

    Returns
    -------
    scipy.sparse matrix
    """
    return vectorizer.transform(texts)


def transform_single(teks_token_str: str, vectorizer):
    """
    Transform satu teks (token string) menjadi vektor fitur.

    Parameters
    ----------
    teks_token_str : str
    vectorizer     : TfidfVectorizer

    Returns
    -------
    scipy.sparse matrix (1 baris)
    """
    return vectorizer.transform([teks_token_str])


def get_feature_names(vectorizer) -> list:
    """
    Kembalikan daftar seluruh fitur dalam vocabulary.
    """
    return list(vectorizer.get_feature_names_out())


def save_vectorizer(vectorizer, path: str) -> None:
    """Simpan vectorizer ke file menggunakan joblib."""
    joblib.dump(vectorizer, path)


def load_vectorizer(path: str):
    """Muat vectorizer dari file."""
    return joblib.load(path)
