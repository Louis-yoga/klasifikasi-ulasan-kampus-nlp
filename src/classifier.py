"""
classifier.py
-------------
Pengklasifikasi teks berbasis dua model Multinomial Naive Bayes independen:
  - model_sentimen : Positif / Negatif / Netral
  - model_kategori : Administrasi / Akademik / Fasilitas / Kemahasiswaan
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.config import MAX_CHARS
from src.feature_extractor import (
    build_vocabulary,
    preprocess_corpus,
    transform_single,
    transform_texts,
)
from src.preprocessor import preprocess_pipeline, tokens_to_string

# ── Konstanta Label & Konfigurasi ────────────────────────────────────────────
LABEL_SENTIMEN = ['Positif', 'Negatif', 'Netral']
LABEL_KATEGORI = ['Administrasi', 'Akademik', 'Fasilitas', 'Kemahasiswaan']
LOW_CONF_THRESHOLD = 0.60  # Flag jika confidence < 60%

EMOJI_SENTIMEN = {
    'Positif': '😊 Positif',
    'Negatif': '😟 Negatif',
    'Netral' : '😐 Netral',
    'Tidak dapat diklasifikasikan': '❓ Tidak Diketahui',
}
EMOJI_KATEGORI = {
    'Administrasi' : '📑 Administrasi',
    'Akademik'     : '🎓 Akademik',
    'Fasilitas'    : '🏢 Fasilitas',
    'Kemahasiswaan': '👥 Kemahasiswaan',
    'Tidak dapat diklasifikasikan': '❓ Tidak Diketahui',
}


# ── Persiapan Data & Pelatihan ───────────────────────────────────────────────

def prepare_data(df: pd.DataFrame, random_state: int = 42) -> tuple:
    """
    Split dataset menjadi data latih dan data uji menggunakan stratified split
    berdasarkan kombinasi sentimen × kategori (12 kombinasi kelas).

    Parameters
    ----------
    df           : DataFrame
    random_state : int

    Returns
    -------
    train_df, test_df : DataFrame, DataFrame
    """
    df = df.copy()
    df['stratify_key'] = df['sentimen'].astype(str) + '_' + df['kategori'].astype(str)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=random_state,
        stratify=df['stratify_key']
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def train_all_models(train_df: pd.DataFrame, vectorizer, alpha: float = 0.2) -> tuple:
    """
    Melatih dua model Multinomial Naive Bayes independen pada training set.

    Parameters
    ----------
    train_df   : DataFrame
    vectorizer : TfidfVectorizer (sudah di-fit pada train_df)
    alpha      : float (smoothing parameter)

    Returns
    -------
    model_sentimen, model_kategori : MultinomialNB, MultinomialNB
    """
    train_corpus = preprocess_corpus(train_df['ulasan'].tolist())
    X_train = transform_texts(train_corpus, vectorizer)

    model_sentimen = MultinomialNB(alpha=alpha)
    model_sentimen.fit(X_train, train_df['sentimen'])

    model_kategori = MultinomialNB(alpha=alpha)
    model_kategori.fit(X_train, train_df['kategori'])

    return model_sentimen, model_kategori


def build_and_train(df: pd.DataFrame, random_state: int = 42, alpha: float = 0.2) -> tuple:
    """
    Pipeline lengkap pelatihan model dari dataset mentah.

    Returns
    -------
    train_df, test_df, vectorizer, model_sentimen, model_kategori
    """
    train_df, test_df = prepare_data(df, random_state)

    train_corpus = preprocess_corpus(train_df['ulasan'].tolist())
    vectorizer = build_vocabulary(train_corpus)

    model_sentimen, model_kategori = train_all_models(train_df, vectorizer, alpha=alpha)

    return train_df, test_df, vectorizer, model_sentimen, model_kategori


# ── Inferensi / Prediksi ─────────────────────────────────────────────────────

def predict_single(
    teks: str,
    vectorizer,
    model_sentimen: MultinomialNB,
    model_kategori: MultinomialNB,
) -> dict:
    """
    Prediksi sentimen dan kategori untuk satu teks ulasan secara aman.

    Parameters
    ----------
    teks           : str
    vectorizer     : TfidfVectorizer
    model_sentimen : MultinomialNB
    model_kategori : MultinomialNB

    Returns
    -------
    dict:
        prep_result      : dict (hasil preprocessing dan masking PII)
        sentimen         : str
        conf_sentimen    : float
        kategori         : str
        conf_kategori    : float
        proba_sentimen   : dict
        proba_kategori   : dict
        low_confidence   : bool
        low_conf_message : str
    """
    # Preprocessing & PII sanitization
    prep_result = preprocess_pipeline(teks)

    # Tangani input kosong / tanpa token bermakna (FUN-04 / M-07)
    if prep_result['jumlah_token'] == 0:
        return {
            "prep_result"     : prep_result,
            "sentimen"        : "Tidak dapat diklasifikasikan",
            "conf_sentimen"   : 0.0,
            "kategori"        : "Tidak dapat diklasifikasikan",
            "conf_kategori"   : 0.0,
            "proba_sentimen"  : {k: 0.0 for k in model_sentimen.classes_},
            "proba_kategori"  : {k: 0.0 for k in model_kategori.classes_},
            "low_confidence"  : True,
            "low_conf_message": "⚠️ Teks tidak mengandung kata bermakna yang dapat dianalisis model.",
        }

    token_str = tokens_to_string(prep_result['tokens'])
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

    # Low Confidence Flag
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
) -> tuple:
    """
    Prediksi batch untuk DataFrame.
    Secara ketat memastikan:
    1. Teks dipotong pada MAX_CHARS (SEC-01).
    2. Kolom 'ulasan' pada hasil berisi TEKS TERSAMAR (masked PII), BUKAN teks mentah (SEC-02).
    3. Mengembalikan (hasil_df, n_truncated).

    Parameters
    ----------
    df_input : DataFrame (memiliki kolom 'ulasan')

    Returns
    -------
    tuple : (DataFrame hasil prediksi, int jumlah sel yang dipotong)
    """
    hasil_list = []
    n_truncated = 0

    for idx, row in df_input.iterrows():
        raw_text = str(row['ulasan']) if pd.notna(row.get('ulasan')) else ""
        if len(raw_text) > MAX_CHARS:
            teks_ulasan = raw_text[:MAX_CHARS]
            n_truncated += 1
        else:
            teks_ulasan = raw_text

        res = predict_single(teks_ulasan, vectorizer, model_sentimen, model_kategori)
        row_id = row['id'] if ('id' in row and pd.notna(row['id']) and str(row['id']).strip() != '') else (idx + 1)

        # Simpan teks_masked_pii, BUKAN teks mentah (SEC-02)
        hasil_list.append({
            'id'            : row_id,
            'ulasan'        : res['prep_result']['teks_masked_pii'],
            'sentimen'      : res['sentimen'],
            'conf_sentimen' : f"{res['conf_sentimen']*100:.1f}%",
            'kategori'      : res['kategori'],
            'conf_kategori' : f"{res['conf_kategori']*100:.1f}%",
            'low_confidence': '⚠️ Ya' if res['low_confidence'] else '✅ Tidak',
        })

    return pd.DataFrame(hasil_list), n_truncated
