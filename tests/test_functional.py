"""
tests/test_functional.py
------------------------
Uji fungsionalitas sistem: strict CSV parsing, error handling, batch prediction
tanpa kebocoran raw PII (FUN-01 s.d. FUN-04, F-01 s.d. F-04).
"""

import pandas as pd
import pytest

from src.artifacts import load_artifacts
from src.classifier import predict_batch, predict_single
from src.parsers import CsvError, parse_csv_strict


@pytest.fixture(scope="module")
def loaded_models():
    objs, _meta = load_artifacts("models")
    return objs


def test_parse_csv_strict_valid():
    """Parser harus berhasil membaca CSV berkoma dan bertitik-koma."""
    csv_comma = b"id,ulasan\n1,Dosen menjelaskan materi dengan sangat baik.\n2,Fasilitas lab memadai."
    df = parse_csv_strict(csv_comma)
    assert len(df) == 2
    assert list(df.columns) == ['id', 'ulasan']

    csv_semi = b"id;ulasan\n1;WiFi di perpus cepat sekali;Tambahan info\n"
    # Berbeda jumlah kolom harus gagal
    with pytest.raises(CsvError):
        parse_csv_strict(csv_semi)

    csv_semi_valid = b"id;ulasan\n1;WiFi di perpus cepat sekali\n2;Gedung bersih"
    df2 = parse_csv_strict(csv_semi_valid)
    assert len(df2) == 2


def test_parse_csv_strict_rejects_binary():
    """Parser wajib menolak berkas biner berkarakter null."""
    binary_content = b"id,ulasan\n1,\x00\x01\x02binarydata"
    with pytest.raises(CsvError, match="bukan CSV teks yang valid"):
        parse_csv_strict(binary_content)


def test_parse_csv_strict_rejects_missing_column():
    """Parser wajib menolak jika kolom ulasan tidak ada."""
    invalid_cols = b"id,komentar_mhs\n1,Kuliah asik"
    with pytest.raises(CsvError, match="Kolom 'ulasan' tidak ditemukan"):
        parse_csv_strict(invalid_cols)


def test_parse_csv_strict_rejects_column_mismatch():
    """Parser wajib menolak baris dengan koma tak berkutip yang memecah kolom."""
    broken = b"id,ulasan\n1,Dosen santai, ramah, dan baik\n"  # 1 baris punya 3 koma tanpa kutip
    with pytest.raises(CsvError, match="Gunakan tanda kutip ganda"):
        parse_csv_strict(broken)


def test_parse_csv_strict_rejects_empty():
    """Parser wajib menolak file kosong atau hanya header."""
    with pytest.raises(CsvError, match="File tidak berisi data"):
        parse_csv_strict(b"id,ulasan\n")


def test_empty_review_handling(loaded_models):
    """Ulasan kosong atau whitespace harus menghasilkan 'Tidak dapat diklasifikasikan'."""
    res = predict_single(
        "   ",
        loaded_models['vectorizer'],
        loaded_models['sentimen'],
        loaded_models['kategori']
    )
    assert res['sentimen'] == "Tidak dapat diklasifikasikan"
    assert res['kategori'] == "Tidak dapat diklasifikasikan"
    assert res['low_confidence'] is True


def test_batch_prediction_no_raw_pii_leak(loaded_models):
    """Output dataframe hasil batch prediction hanya boleh memuat teks bertopeng PII."""
    raw_df = pd.DataFrame({
        'id': [1],
        'ulasan': ["Dosen Pak Bambang ramah sekali, hubungi beliau di 081234567890 atau email test@kampus.ac.id"]
    })
    df_out, _stats = predict_batch(
        raw_df,
        loaded_models['vectorizer'],
        loaded_models['sentimen'],
        loaded_models['kategori']
    )
    output_text = df_out.iloc[0]['ulasan']
    # Memastikan tidak ada nomor telepon atau email mentah di output
    assert "081234567890" not in output_text
    assert "test@kampus.ac.id" not in output_text
    assert "[TELEPON]" in output_text
    assert "[EMAIL]" in output_text
