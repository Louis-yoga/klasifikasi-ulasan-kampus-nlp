"""
tests/test_models.py
--------------------
Uji integritas model dan evaluasi: non-sirkularitas metrik, penanganan negasi,
log-odds explainability, dan validasi metadata (MOD-01 s.d. MOD-07, U-01).
"""

import json
import os

import pytest

from src.artifacts import load_artifacts
from src.evaluator import evaluate_model, get_top_features
from src.preprocessor import mark_negation, preprocess_pipeline


def test_metrik_tidak_sirkular():
    """Uji verifikasi kebenaran perhitungan metrik evaluasi independen (M-04)."""
    class MockModel:
        def predict(self, X):
            return ['A', 'B', 'B', 'B']

    r = evaluate_model(MockModel(), None, ['A', 'A', 'B', 'B'], ['A', 'B'])
    assert r['accuracy'] == 0.75
    assert r['f1_macro'] == pytest.approx(0.7333, abs=1e-4)
    assert r['conf_matrix'].tolist() == [[1, 1], [0, 2]]


def test_negasi_tidak_hilang():
    """Kata negasi (belum, tidak, bukan) harus menghasilkan token yang berbeda dari kalimat positif (M-03)."""
    tok_belum = preprocess_pipeline("WiFi belum stabil")['tokens']
    tok_stabil = preprocess_pipeline("WiFi stabil")['tokens']
    assert tok_belum != tok_stabil, f"Negasi hilang! Token sama: {tok_belum} vs {tok_stabil}"


def test_mark_negation_prefix():
    """Fungsi mark_negation harus memberikan prefix NOT_ pada token setelah negasi."""
    tokens = ["pelayanan", "tidak", "ramah", "dan", "lambat"]
    marked = mark_negation(tokens)
    assert "NOT_ramah" in marked


def test_explainability_log_odds_no_fallback():
    """Jika tidak ada token input yang cocok di vocab, get_top_features harus return [] (U-01)."""
    objs, _meta = load_artifacts("models")
    vec = objs['vectorizer']
    model_s = objs['sentimen']

    # Input kata asing / OOV yang tidak ada dalam vocabulary
    oov_tokens = ["asdfghjk", "qwertyuiop", "zxcvbnm"]
    feats = get_top_features(model_s, vec, "Positif", input_tokens=oov_tokens)
    assert feats == [], f"Harusnya [] tetapi menghasilkan: {feats}"

    # Input dengan kata yang dikenal
    known_tokens = ["dosen", "sangat", "ramah", "baik"]
    feats_known = get_top_features(model_s, vec, "Positif", input_tokens=known_tokens)
    assert len(feats_known) > 0
    words = [w for w, score in feats_known]
    # Semua kata yang dikembalikan harus berasal dari token input (unigram atau bigram)
    for w in words:
        parts = w.split()
        assert any(p in known_tokens for p in parts)


def test_metadata_json_integrity():
    """Metadata model harus ada, valid JSON, dan memuat metrik CV serta holdout."""
    meta_path = os.path.join("models", "metadata.json")
    assert os.path.exists(meta_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "files" in meta
    assert "sklearn_version" in meta
    assert "dataset" in meta
    assert "metrics" in meta
    assert "cv_repeated_kfold" in meta["metrics"]["sentimen"]
    assert "holdout" in meta["metrics"]["sentimen"]
