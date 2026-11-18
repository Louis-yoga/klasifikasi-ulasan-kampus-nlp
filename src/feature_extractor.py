"""
feature_extractor.py
--------------------
Ekstraksi Fitur Teks (Sublinear TF-IDF N-Gram) untuk Sistem Klasifikasi Ulasan Kampus.
Satu vectorizer digunakan bersama oleh model Sentimen dan Kategori.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessor import preprocess_pipeline, tokens_to_string


def build_vocabulary(corpus: list, ngram_range: tuple = (1, 2)) -> TfidfVectorizer:
    """
    Fit TfidfVectorizer pada training corpus.
    Menggunakan unigram + bigram dan sublinear_tf=True untuk meningkatkan
    akurasi diskriminasi sentimen & kategori pada ulasan kampus.

    Parameters
    ----------
    corpus      : list[str]  — Daftar teks ulasan (string token hasil preprocessing)
    ngram_range : tuple      — Range n-gram (default: (1, 2))

    Returns
    -------
    TfidfVectorizer  — Vectorizer yang sudah di-fit
    """
    vectorizer = TfidfVectorizer(
        analyzer='word',
        min_df=1,           # Seluruh fitur kosakata disertakan
        max_df=0.95,        # Abaikan kata yang terlalu umum di >95% dokumen
        ngram_range=ngram_range, # Unigram + Bigram (menangkap frasa penting)
        sublinear_tf=True,  # Reduksi dominasi frekuensi kata ekstrem (1 + log(tf))
    )
    vectorizer.fit(corpus)
    return vectorizer


# Alias untuk fleksibilitas pemanggilan
build_vectorizer = build_vocabulary


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


def transform_texts(texts: list, vectorizer: TfidfVectorizer):
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


def transform_single(teks_token_str: str, vectorizer: TfidfVectorizer):
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


def get_feature_names(vectorizer: TfidfVectorizer) -> list:
    """
    Kembalikan daftar seluruh fitur dalam vocabulary.
    """
    return list(vectorizer.get_feature_names_out())
