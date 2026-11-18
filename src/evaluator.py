"""
evaluator.py
------------
Evaluasi performa model MNB dan perhitungan Top-N fitur untuk explainability.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, classification_report
)

from src.preprocessor import preprocess_pipeline, tokens_to_string
from src.feature_extractor import transform_texts, preprocess_corpus


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
    from src.classifier import LABEL_SENTIMEN, LABEL_KATEGORI

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
    input_tokens: list = None,
    top_n: int = 5,
) -> list:
    """
    Ambil Top-N kata paling berpengaruh untuk kelas tertentu.

    Jika input_tokens diberikan, filter hanya kata yang ada dalam input
    (kontekstual). Jika tidak, tampilkan top-N global dari vocabulary.

    Parameters
    ----------
    model        : MultinomialNB
    vectorizer   : CountVectorizer
    class_name   : str        — Nama kelas yang ingin dilihat top fiturnya
    input_tokens : list[str]  — Token dari teks yang sedang diprediksi (opsional)
    top_n        : int        — Jumlah kata teratas

    Returns
    -------
    list[tuple]  — [(kata, log_prob), ...] diurutkan dari tertinggi
    """
    classes     = list(model.classes_)
    feat_names  = list(vectorizer.get_feature_names_out())
    log_probs   = model.feature_log_prob_

    if class_name not in classes:
        return []

    class_idx = classes.index(class_name)
    class_log_prob = log_probs[class_idx]  # shape: (n_features,)

    if input_tokens:
        # Filter token (unigram dan bigram) dari input yang ada di vocabulary
        vocab = vectorizer.vocabulary_
        idx_list = []
        for token in input_tokens:
            if token in vocab:
                idx_list.append(vocab[token])
        for i in range(len(input_tokens) - 1):
            bg = f"{input_tokens[i]} {input_tokens[i+1]}"
            if bg in vocab:
                idx_list.append(vocab[bg])

        if not idx_list:
            # Fallback: global top-N jika tidak ada token yang dikenal
            idx_sorted = np.argsort(class_log_prob)[::-1][:top_n]
        else:
            # Urutkan idx yang ada di input berdasarkan log-prob tertinggi
            idx_sorted = sorted(list(set(idx_list)), key=lambda i: class_log_prob[i], reverse=True)[:top_n]
    else:
        idx_sorted = np.argsort(class_log_prob)[::-1][:top_n]

    result = [(feat_names[i], float(class_log_prob[i])) for i in idx_sorted]
    return result


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
