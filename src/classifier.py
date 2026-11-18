"""
classifier.py
-------------
Wrapper dua model Multinomial Naive Bayes independen:
  - model_sentimen : Positif / Negatif
  - model_kategori : Akademik / Fasilitas/WiFi / Perpustakaan / Sistem KRS

Keduanya menggunakan vectorizer yang sama dari feature_extractor.py.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from src.preprocessor import preprocess_pipeline, tokens_to_string
from src.feature_extractor import (
    build_vocabulary, preprocess_corpus,
    transform_texts, transform_single,
    save_vectorizer, load_vectorizer
)

# ── Konstanta ────────────────────────────────────────────────────────────────
LABEL_SENTIMEN  = ['Positif', 'Negatif', 'Netral']
LABEL_KATEGORI  = ['Administrasi', 'Akademik', 'Fasilitas', 'Kemahasiswaan']
LOW_CONF_THRESHOLD = 0.60  # (Q10) Flag jika confidence < 60%

EMOJI_SENTIMEN = {
    'Positif': '😊 Positif',
    'Negatif': '😟 Negatif',
    'Netral' : '😐 Netral',
}
EMOJI_KATEGORI = {
    'Administrasi' : '📑 Administrasi',
    'Akademik'     : '🎓 Akademik',
    'Fasilitas'    : '🏢 Fasilitas',
    'Kemahasiswaan': '👥 Kemahasiswaan',
}


# ── Training ─────────────────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame, random_state: int = 42):
    """
    Split dataset menjadi train dan test menggunakan stratified split
    berdasarkan kombinasi sentimen × kategori.

    Parameters
    ----------
    df           : DataFrame  — Dataset lengkap dengan kolom ulasan/sentimen/kategori
    random_state : int

    Returns
    -------
    train_df, test_df : DataFrame, DataFrame
    """
    # Buat stratify key: gabungan sentimen + kategori (8 kelas kombinasi)
    df = df.copy()
    df['stratify_key'] = df['sentimen'] + '_' + df['kategori']

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=random_state,
        stratify=df['stratify_key']
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def train_all_models(train_df: pd.DataFrame, vectorizer):
    """
    Melatih dua model MNB independen pada training set yang sama.

    Parameters
    ----------
    train_df   : DataFrame  — Training set (sudah di-split)
    vectorizer : CountVectorizer  — Sudah di-fit pada training corpus

    Returns
    -------
    model_sentimen, model_kategori : MultinomialNB, MultinomialNB
    """
    train_corpus = preprocess_corpus(train_df['ulasan'].tolist())
    X_train = transform_texts(train_corpus, vectorizer)

    model_sentimen = MultinomialNB(alpha=0.2)  # Smoothing teroptimasi
    model_sentimen.fit(X_train, train_df['sentimen'])

    model_kategori = MultinomialNB(alpha=0.2)  # Smoothing teroptimasi
    model_kategori.fit(X_train, train_df['kategori'])

    return model_sentimen, model_kategori


def build_and_train(df: pd.DataFrame, random_state: int = 42):
    """
    Pipeline lengkap: split → preprocess → fit vectorizer → train 2 model.
    Digunakan oleh app.py pada saat startup pertama.

    Returns
    -------
    train_df, test_df, vectorizer, model_sentimen, model_kategori
    """
    train_df, test_df = prepare_data(df, random_state)

    train_corpus = preprocess_corpus(train_df['ulasan'].tolist())
    vectorizer = build_vocabulary(train_corpus)

    model_sentimen, model_kategori = train_all_models(train_df, vectorizer)

    return train_df, test_df, vectorizer, model_sentimen, model_kategori


# ── Prediksi ─────────────────────────────────────────────────────────────────

def predict_single(
    teks: str,
    vectorizer,
    model_sentimen: MultinomialNB,
    model_kategori: MultinomialNB,
) -> dict:
    """
    Prediksi sentimen dan kategori untuk satu teks ulasan.

    Parameters
    ----------
    teks           : str  — Teks ulasan mentah
    vectorizer     : CountVectorizer
    model_sentimen : MultinomialNB
    model_kategori : MultinomialNB

    Returns
    -------
    dict:
        prep_result      : dict   — Hasil preprocess_pipeline (untuk Panel 2)
        sentimen         : str    — Kelas sentimen prediksi
        conf_sentimen    : float  — Keyakinan sentimen (0–1)
        kategori         : str    — Kelas kategori prediksi
        conf_kategori    : float  — Keyakinan kategori (0–1)
        proba_sentimen   : dict   — Probabilitas per kelas sentimen
        proba_kategori   : dict   — Probabilitas per kelas kategori
        low_confidence   : bool   — True jika salah satu < LOW_CONF_THRESHOLD
        low_conf_message : str    — Pesan human oversight (jika low_confidence)
    """
    # Preprocessing
    prep_result = preprocess_pipeline(teks)
    token_str = tokens_to_string(prep_result['tokens'])

    # Vectorize
    X = transform_single(token_str, vectorizer)

    # Prediksi sentimen
    proba_s_arr = model_sentimen.predict_proba(X)[0]
    kelas_s     = model_sentimen.classes_
    proba_s     = {k: float(round(v, 4)) for k, v in zip(kelas_s, proba_s_arr)}
    pred_s      = kelas_s[np.argmax(proba_s_arr)]
    conf_s      = float(np.max(proba_s_arr))

    # Prediksi kategori
    proba_k_arr = model_kategori.predict_proba(X)[0]
    kelas_k     = model_kategori.classes_
    proba_k     = {k: float(round(v, 4)) for k, v in zip(kelas_k, proba_k_arr)}
    pred_k      = kelas_k[np.argmax(proba_k_arr)]
    conf_k      = float(np.max(proba_k_arr))

    # Low Confidence Flag (Q10)
    low_conf = (conf_s < LOW_CONF_THRESHOLD) or (conf_k < LOW_CONF_THRESHOLD)
    low_conf_msg = (
        "⚠️ Keyakinan model rendah (< 60%). "
        "Disarankan tinjauan manual oleh staf sebelum mengambil keputusan."
        if low_conf else ""
    )

    return {
        "prep_result"     : prep_result,
        "sentimen"        : pred_s,
        "conf_sentimen"   : conf_s,
        "kategori"        : pred_k,
        "conf_kategori"   : conf_k,
        "proba_sentimen"  : proba_s,
        "proba_kategori"  : proba_k,
        "low_confidence"  : low_conf,
        "low_conf_message": low_conf_msg,
    }


def predict_batch(
    df_input: pd.DataFrame,
    vectorizer,
    model_sentimen: MultinomialNB,
    model_kategori: MultinomialNB,
) -> pd.DataFrame:
    """
    Prediksi batch untuk DataFrame yang memiliki kolom 'ulasan'.

    Returns
    -------
    DataFrame dengan kolom tambahan: sentimen, conf_sentimen,
                                     kategori, conf_kategori, low_confidence
    """
    hasil_list = []
    for idx, row in df_input.iterrows():
        teks_ulasan = str(row['ulasan']) if pd.notna(row.get('ulasan')) else ""
        res = predict_single(
            teks_ulasan, vectorizer, model_sentimen, model_kategori
        )
        row_id = row['id'] if ('id' in row and pd.notna(row['id']) and str(row['id']).strip() != '') else (idx + 1)
        hasil_list.append({
            'id'            : row_id,
            'ulasan'        : teks_ulasan,
            'sentimen'      : res['sentimen'],
            'conf_sentimen' : f"{res['conf_sentimen']*100:.1f}%",
            'kategori'      : res['kategori'],
            'conf_kategori' : f"{res['conf_kategori']*100:.1f}%",
            'low_confidence': '⚠️ Ya' if res['low_confidence'] else '✅ Tidak',
        })
    return pd.DataFrame(hasil_list)


# ── Simpan / Muat Model ───────────────────────────────────────────────────────

def save_model(model, path: str) -> None:
    """Simpan model ke file menggunakan joblib."""
    joblib.dump(model, path)


def load_model(path: str):
    """Muat model dari file."""
    return joblib.load(path)
