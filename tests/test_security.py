"""
tests/test_security.py
----------------------
Uji keamanan: ReDoS, batas input, penyamaran PII, netralisasi formula CSV,
integritas artefak model (SEC-01 s.d. SEC-06).
"""

import time

import pandas as pd
import pytest

from src.artifacts import ArtifactError, load_artifacts, save_artifacts
from src.config import MAX_CHARS
from src.exporters import sanitize_cell, to_safe_csv
from src.guard import throttle
from src.preprocessor import InputTooLong, mask_pii, preprocess_pipeline


def test_email_waktu_linear():
    """Uji ketahanan ReDoS: pola email repetitif panjang harus selesai <1 detik (SEC-01)."""
    # String berbahaya tipe ReDoS pada regex kuantifier bertumpuk
    malicious = "a" * 1000 + "@" + "b" * 1000 + "." + "c" * 50
    t0 = time.perf_counter()
    mask_pii(malicious)
    durasi = time.perf_counter() - t0
    assert durasi < 0.5, f"ReDoS terdeteksi! Waktu eksekusi: {durasi:.3f}s"


def test_input_terlalu_panjang():
    """Input melebihi MAX_CHARS wajib melempar InputTooLong (SEC-01)."""
    with pytest.raises(InputTooLong):
        preprocess_pipeline("a" * (MAX_CHARS + 1))


@pytest.mark.parametrize("input_text, expected_tag", [
    ("hubungi +6281234567890 sekarang", "[TELEPON]"),
    ("hubungi +62 812-3456-7890 sekarang", "[TELEPON]"),
    ("hubungi 0812-3456-7890 segera", "[TELEPON]"),
    ("hubungi 081234567890 segera", "[TELEPON]"),  # Harus TELEPON, bukan NIM
    ("NIM 24.06.05.110.034 tercatat", "[NIM]"),
    ("NIM 240605110034 tercatat", "[NIM]"),
    ("kirim ke budi.s@student.uin-malang.ac.id ya", "[EMAIL]"),
    ("disampaikan oleh Pak Budi Santoso kemarin", "[NAMA]"),
])
def test_pii_tersamar(input_text, expected_tag):
    """Seluruh variasi format PII harus tersamar dengan benar tanpa sisa raw (SEC-02)."""
    masked = mask_pii(input_text)
    assert expected_tag in masked, f"Tag {expected_tag} tidak ditemukan pada hasil: {masked}"
    # Pastikan nomor/identitas mentah tidak tertinggal
    if "081234567890" in input_text:
        assert "081234567890" not in masked
    if "240605110034" in input_text:
        assert "240605110034" not in masked
    if "student.uin-malang.ac.id" in input_text:
        assert "student.uin-malang.ac.id" not in masked


def test_csv_injection_neutralization():
    """Sel diawali formula (=, +, -, @) harus dinetralkan dengan kutip tunggal ' (SEC-03)."""
    assert sanitize_cell("=1+1") == "'=1+1"
    assert sanitize_cell("+cmd|' /C calc'!A0") == "'+cmd|' /C calc'!A0"
    assert sanitize_cell("-5+2") == "'-5+2"
    assert sanitize_cell("@SUM(1,2)") == "'@SUM(1,2)"
    assert sanitize_cell("Teks normal") == "Teks normal"


def test_safe_csv_bom_prefix():
    """Ekspor CSV aman harus diawali UTF-8 BOM (\ufeff) (SEC-03)."""
    df = pd.DataFrame({'ulasan': ['Bagus', '=cmd|x', '+628123']})
    csv_bytes = to_safe_csv(df)
    assert csv_bytes.startswith(b'\xef\xbb\xbf'), "CSV tidak memuat UTF-8 BOM"
    decoded = csv_bytes.decode('utf-8')
    assert "'=cmd|x" in decoded
    assert "'+628123" in decoded


def test_throttle_rate_limiting():
    """Throttle session state membatasi lonjakan request (SEC-06)."""
    state = {}
    # Request pertama diperbolehkan
    assert throttle(state, min_interval=0.1) is True
    # Request langsung berikutnya ditolak (terlalu cepat)
    assert throttle(state, min_interval=0.1) is False
    # Setelah jeda cukup waktu, diperbolehkan kembali
    time.sleep(0.12)
    assert throttle(state, min_interval=0.1) is True


def test_artifact_tampering_detection(tmp_path):
    """Modifikasi berkas model .pkl wajib ditolak oleh verifikasi SHA-256 (SEC-04)."""
    model_dir = str(tmp_path / "models")
    dummy_obj = {'a': 1}
    objs = {'vectorizer': dummy_obj, 'sentimen': dummy_obj, 'kategori': dummy_obj}
    save_artifacts(model_dir, objs, {'test': True})

    # Verifikasi normal sukses
    loaded, _meta = load_artifacts(model_dir)
    assert 'vectorizer' in loaded

    # Rusak salah satu file
    tampered_file = tmp_path / "models" / "model_sentimen.pkl"
    with open(tampered_file, "ab") as f:
        f.write(b"\x00TAMPERED")

    with pytest.raises(ArtifactError, match="Integritas .* gagal diverifikasi"):
        load_artifacts(model_dir)
