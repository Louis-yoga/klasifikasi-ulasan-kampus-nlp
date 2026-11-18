"""
evaluator.py
------------
Evaluasi performa model MNB dan perhitungan Top-N fitur untuk explainability.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.feature_extractor import preprocess_corpus, transform_texts


def evaluate_model(
    model,
    X_test,
    y_test: list,
    label_names: list,
) -> dict:
    """
    Hitung metrik evaluasi lengkap untuk satu model.

    Parameters
    ----------
    model       : MultinomialNB  — Model yang sudah dilatih
    X_test      : sparse matrix  — Fitur test set
    y_test      : list           — Label ground-truth
    label_names : list[str]      — Urutan nama kelas

    Returns
    -------
    dict:
        accuracy        : float
        precision_macro : float
        recall_macro    : float
        f1_macro        : float
        per_class       : DataFrame  (Precision/Recall/F1 per label)
        conf_matrix     : np.ndarray
        y_pred          : list
        y_true          : list
    """
    y_pred = model.predict(X_test)
    y_true = list(y_test)

    accuracy        = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro    = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro        = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Laporan per kelas
    report = classification_report(
        y_true, y_pred,
        labels=label_names,
        output_dict=True,
        zero_division=0
    )
    per_class_data = []
    for label in label_names:
        if label in report:
            per_class_data.append({
                'Kelas'    : label,
                'Precision': round(report[label]['precision'], 4),
                'Recall'   : round(report[label]['recall'], 4),
                'F1-Score' : round(report[label]['f1-score'], 4),
                'Support'  : int(report[label]['support']),
            })
    per_class_df = pd.DataFrame(per_class_data)

    cm = confusion_matrix(y_true, y_pred, labels=label_names)

    return {
        'accuracy'       : round(accuracy, 4),
        'precision_macro': round(precision_macro, 4),
        'recall_macro'   : round(recall_macro, 4),
        'f1_macro'       : round(f1_macro, 4),
        'per_class'      : per_class_df,
        'conf_matrix'    : cm,
        'y_pred'         : list(y_pred),
        'y_true'         : y_true,
        'label_names'    : label_names,
    }


def evaluate_both_models(
    model_sentimen,
    model_kategori,
    test_df: pd.DataFrame,
    vectorizer,
) -> dict:
    """
    Evaluasi kedua model sekaligus pada test set.

    Parameters
    ----------
    model_sentimen : MultinomialNB
    model_kategori : MultinomialNB
    test_df        : DataFrame  — Test set dengan kolom ulasan/sentimen/kategori
    vectorizer     : CountVectorizer

    Returns
    -------
    dict:
        sentimen : dict  — Hasil evaluate_model untuk sentimen
        kategori : dict  — Hasil evaluate_model untuk kategori
    """
    from src.classifier import LABEL_KATEGORI, LABEL_SENTIMEN

    test_corpus = preprocess_corpus(test_df['ulasan'].tolist())
    X_test = transform_texts(test_corpus, vectorizer)

    hasil_sentimen = evaluate_model(
        model_sentimen, X_test,
        test_df['sentimen'].tolist(),
        LABEL_SENTIMEN
    )
    hasil_kategori = evaluate_model(
        model_kategori, X_test,
        test_df['kategori'].tolist(),
        LABEL_KATEGORI
    )
    return {'sentimen': hasil_sentimen, 'kategori': hasil_kategori}


def get_top_features(
    model,
    vectorizer,
    class_name: str,
    input_tokens: list | None = None,
    top_n: int = 5,
) -> list:
    """
    Ambil Top-N kata paling berpengaruh untuk kelas tertentu berdasarkan log-odds.

    Log-odds membandingkan log-probabilitas fitur pada kelas target dengan rata-rata
    log-probabilitas pada kelas lainnya:
        log_odds = log P(w|C_k) - log mean_{j != k} P(w|C_j)

    Jika input_tokens diberikan, hanya filter token (unigram & bigram) dari input
    yang ada di vocabulary. Jika tidak ada yang cocok, kembalikan list kosong []
    (tanpa fallback global yang menyesatkan).

    Parameters
    ----------
    model        : MultinomialNB
    vectorizer   : CountVectorizer
    class_name   : str        — Nama kelas yang ingin dilihat top fiturnya
    input_tokens : list[str]  — Token dari teks yang sedang diprediksi (opsional)
    top_n        : int        — Jumlah kata teratas

    Returns
    -------
    list[tuple]  — [(kata, log_odds), ...] diurutkan dari skor tertinggi
    """
    classes    = list(model.classes_)
    feat_names = list(vectorizer.get_feature_names_out())
    flp        = model.feature_log_prob_  # shape: (n_classes, n_features)

    if class_name not in classes:
        return []

    ci = classes.index(class_name)
    oth = [i for i in range(len(classes)) if i != ci]

    if oth:
        # log-sum-exp over other classes minus log(number of other classes)
        bg = np.logaddexp.reduce(flp[oth], axis=0) - np.log(len(oth))
        scores = flp[ci] - bg
    else:
        scores = flp[ci]

    if input_tokens is not None:
        vocab = vectorizer.vocabulary_
        idx_list = []
        for token in input_tokens:
            if token in vocab:
                idx_list.append(vocab[token])
        for i in range(len(input_tokens) - 1):
            bg_gram = f"{input_tokens[i]} {input_tokens[i+1]}"
            if bg_gram in vocab:
                idx_list.append(vocab[bg_gram])

        unique_idx = list(set(idx_list))
        if not unique_idx:
            # Tidak ada token yang cocok di vocabulary -> jangan fallback ke global
            return []

        idx_sorted = sorted(unique_idx, key=lambda i: scores[i], reverse=True)[:top_n]
    else:
        idx_sorted = np.argsort(scores)[::-1][:top_n]

    return [(feat_names[i], round(float(scores[i]), 4)) for i in idx_sorted]


def get_misclassified(y_true: list, y_pred: list, texts: list) -> pd.DataFrame:
    """
    Kembalikan DataFrame berisi sampel yang salah diklasifikasikan.

    Parameters
    ----------
    y_true : list  — Label sebenarnya
    y_pred : list  — Label prediksi
    texts  : list  — Teks ulasan asli

    Returns
    -------
    DataFrame dengan kolom: ulasan, label_benar, label_prediksi
    """
    salah = [
        {'ulasan': t, 'label_benar': yt, 'label_prediksi': yp}
        for t, yt, yp in zip(texts, y_true, y_pred)
        if yt != yp
    ]
    return pd.DataFrame(salah)
