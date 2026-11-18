"""
parsers.py
----------
Parser CSV ketat untuk batch prediction (F-01, F-02, F-03, SEC-01).
Menolak file corrupt, non-teks, melebihi batas ukuran, atau tanpa kolom target.
"""

import csv
import io

import pandas as pd

from src.config import MAX_BYTES, MAX_CHARS, MAX_ROWS


class CsvError(ValueError):
    """Exception khusus untuk kegagalan validasi CSV."""


def parse_csv_strict(b: bytes) -> pd.DataFrame:
    """
    Parse file CSV biner dengan validasi ketat.

    Aturan:
    1. Ukuran maksimal MAX_BYTES (1 MB).
    2. Mendukung encoding UTF-8 / UTF-8-BOM dan UTF-16.
    3. Menolak file biner (mengandung null-byte '\\x00').
    4. Mendeteksi pemisah (',', ';', '\\t').
    5. Wajib memiliki baris header dan minimal 1 baris data.
    6. Wajib memiliki kolom 'ulasan' (atau 'review', 'text', 'komentar').
    7. Baris data maksimal MAX_ROWS (500 baris).
    8. Setiap baris harus konsisten dengan jumlah kolom header.
    9. Teks ulasan dipotong maksimal MAX_CHARS (2000 karakter).

    Parameters
    ----------
    b : bytes — Raw bytes file yang diunggah

    Returns
    -------
    pd.DataFrame dengan kolom ['id', 'ulasan']
    """
    if len(b) > MAX_BYTES:
        raise CsvError(f"Ukuran file melebihi batas maksimum {MAX_BYTES // 1_000_000} MB.")

    enc = 'utf-16' if b.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
    try:
        text = b.decode(enc)
    except UnicodeDecodeError:
        raise CsvError("Encoding tidak dikenali. Simpan ulang sebagai CSV UTF-8.")

    if '\x00' in text:
        raise CsvError("File bukan CSV teks yang valid (mengandung karakter null biner).")

    first_line = text.split('\n', 1)[0] + '\n'
    try:
        sep = csv.Sniffer().sniff(first_line, delimiters=',;\t').delimiter
    except csv.Error:
        sep = ','

    rows = [r for r in csv.reader(io.StringIO(text), delimiter=sep) if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise CsvError("File tidak berisi data atau hanya berisi header.")

    header = [h.strip().lower() for h in rows[0]]
    key = next((header.index(k) for k in ('ulasan', 'review', 'text', 'komentar') if k in header), None)
    if key is None:
        raise CsvError(f"Kolom 'ulasan' tidak ditemukan. Header terdeteksi: {rows[0]}")

    body = rows[1:]
    if len(body) > MAX_ROWS:
        raise CsvError(f"File melebihi batas maksimum {MAX_ROWS} baris (ditemukan {len(body)} baris).")

    for n, r in enumerate(body, start=2):
        if len(r) != len(header):
            raise CsvError(
                f"Baris {n} memiliki {len(r)} kolom, seharusnya {len(header)}. "
                f"Gunakan tanda kutip ganda (\") pada teks yang mengandung '{sep}'."
            )

    idc = header.index('id') if 'id' in header else None
    return pd.DataFrame({
        'id': [r[idc] if idc is not None else i for i, r in enumerate(body, 1)],
        'ulasan': [r[key].strip()[:MAX_CHARS] for r in body]
    })
