"""
scripts/bench.py
----------------
Benchmark performa dan anggaran waktu sistem (PERF-01 s.d. PERF-05).
Memvalidasi batas latensi inferensi tunggal (<50ms), batch 100 ulasan (<500ms),
dan parsing CSV ketat 500 baris (<100ms).
"""

import os
import sys
import time

# Pastikan root workspace ada di sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

from src.artifacts import load_artifacts
from src.classifier import predict_batch, predict_single
from src.parsers import parse_csv_strict
from src.preprocessor import preprocess_pipeline

MODELS_DIR = "models"
DEMO_PATH = os.path.join("data", "demo_unseen.csv")


def run_benchmark():
    print("=" * 60)
    print("BENCHMARK PERFORMA SISTEM (PERF-01 s.d. PERF-05)")
    print("=" * 60)

    # 1. Benchmark pemuatan artefak model
    t0 = time.perf_counter()
    objs, _metadata = load_artifacts(MODELS_DIR)
    t_load = (time.perf_counter() - t0) * 1000
    print(f"1. Model Artifacts Loading: {t_load:.2f} ms")
    assert t_load < 500, f"Loading terlalu lambat: {t_load} ms"

    # 2. Benchmark Preprocessing Teks Tunggal
    sample_text = (
        "Pelayanan administrasi di loket sangat lambat dan staf kurang ramah. "
        "Hubungi saya di 081234567890 atau email mahasiswa@kampus.ac.id untuk bukti."
    )
    t0 = time.perf_counter()
    preprocess_pipeline(sample_text)
    t_prep = (time.perf_counter() - t0) * 1000
    print(f"2. Preprocessing & PII Masking (Single Text): {t_prep:.2f} ms")
    assert t_prep < 20, f"Preprocessing terlalu lambat: {t_prep} ms"

    # 3. Benchmark Inferensi Tunggal End-to-End
    t0 = time.perf_counter()
    pred_res = predict_single(
        sample_text,
        objs['vectorizer'],
        objs['sentimen'],
        objs['kategori']
    )
    t_single = (time.perf_counter() - t0) * 1000
    print(f"3. Single Prediction Latency: {t_single:.2f} ms")
    print(f"   Prediksi -> Sentimen: {pred_res['sentimen']}, Kategori: {pred_res['kategori']}")
    assert t_single < 50, f"Single prediction melebihi anggaran waktu: {t_single} ms"

    # 4. Benchmark Batch Prediction (100 ulasan dari data/demo_unseen.csv)
    if os.path.exists(DEMO_PATH):
        df_demo = pd.read_csv(DEMO_PATH)
        t0 = time.perf_counter()
        _df_out, _stats = predict_batch(
            df_demo,
            objs['vectorizer'],
            objs['sentimen'],
            objs['kategori']
        )
        t_batch = (time.perf_counter() - t0) * 1000
        per_item = t_batch / len(df_demo)
        print(f"4. Batch Prediction (100 ulasan): {t_batch:.2f} ms ({per_item:.2f} ms/item)")
        assert t_batch < 500, f"Batch prediction melebihi anggaran waktu: {t_batch} ms"

    # 5. Benchmark Strict CSV Parsing (500 baris sintetis)
    dummy_csv_lines = ["id,ulasan"] + [f'{i},"Pelayanan loket gedung B sangat memuaskan dan cepat."' for i in range(1, 501)]
    csv_bytes = "\n".join(dummy_csv_lines).encode('utf-8')
    t0 = time.perf_counter()
    df_parsed = parse_csv_strict(csv_bytes)
    t_parse = (time.perf_counter() - t0) * 1000
    print(f"5. Strict CSV Parsing (500 baris): {t_parse:.2f} ms")
    assert len(df_parsed) == 500
    assert t_parse < 100, f"CSV parsing terlalu lambat: {t_parse} ms"

    print("\n[SUKSES] Seluruh pengujian performa memenuhi anggaran waktu!")


if __name__ == '__main__':
    run_benchmark()
