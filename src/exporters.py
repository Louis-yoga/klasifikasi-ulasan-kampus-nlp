"""
exporters.py
------------
Modul ekspor data yang aman dari kerentanan CSV Formula Injection (SEC-03)
dan memastikan kompatibilitas encoding Excel (UTF-8 dengan BOM).
"""

import pandas as pd

FORMULA_PREFIX = ('=', '+', '-', '@', '\t', '\r')


def sanitize_cell(v):
    """
    Menetralkan sel string yang berpotensi mengeksekusi rumus Excel (SEC-03).
    Diawali tanda petik tunggal (') jika dimulai dengan =, +, -, @, dll.
    """
    if isinstance(v, str) and v.startswith(FORMULA_PREFIX):
        return "'" + v
    return v


def to_safe_csv(df: pd.DataFrame, as_bytes: bool = True):
    """
    Mengonversi DataFrame ke CSV dengan netralisasi rumus Excel.
    Setiap sel string yang diawali karakter eksekusi rumus (=, +, -, @, dll)
    diberi awalan tanda petik tunggal (') dan string diawali UTF-8 BOM (\ufeff).

    Parameters
    ----------
    df       : pd.DataFrame
    as_bytes : bool — Jika True, kembalikan bytes ber-BOM UTF-8 b'\xef\xbb\xbf' (default True)

    Returns
    -------
    bytes atau str  — Konten CSV aman siap unduh
    """
    safe = df.copy()
    for col in safe.select_dtypes(include=['object', 'string']).columns:
        safe[col] = safe[col].map(sanitize_cell)

    csv_str = '\ufeff' + safe.to_csv(index=False)
    if as_bytes:
        return csv_str.encode('utf-8')
    return csv_str
