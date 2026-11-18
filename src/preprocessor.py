"""
preprocessor.py
---------------
Pipeline preprocessing teks untuk Sistem Klasifikasi Ulasan Kampus.
Langkah: Input Validation → PII Masking → Normalisasi & Lowercase → Cleaning → Stopword Removal → Tokenisasi → Negation Marking
"""

import re

from src.config import MAX_CHARS


class InputTooLong(ValueError):
    """Exception jika teks ulasan melebihi batas panjang maksimum."""


# ── Daftar Stopword Bahasa Indonesia ─────────────────────────────────────────
# Catatan Keamanan & Semantik (MOD-03):
# Kata-kata negasi, intensifier, dan kata domain kunci sengaja TIDAK dimasukkan
# (misal: "tidak", "belum", "bukan", "sekali", "lebih", "sering", "waktu", "bisa", "hal")
STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "ini", "itu",
    "atau", "juga", "pada", "sudah", "saya", "kami", "kita", "adalah",
    "ada", "akan", "nya", "pun", "lah",
    "ya", "si", "para", "sang", "agar", "atas", "bagi", "bahwa",
    "demi", "dalam", "hingga", "jika", "kalau", "karena", "lagi",
    "maka", "menjadi", "meski", "oleh", "sama", "sebelum", "sebuah",
    "sedang", "sehingga", "setelah", "sini", "suatu", "supaya",
    "telah", "tentang", "tiap", "walaupun", "seperti", "setiap", "antara"
}

# ── Pola Negasi untuk Contextual Marking (MOD-03) ─────────────────────────────
NEGATORS = {
    "tidak", "belum", "bukan", "kurang", "tanpa",
    "jangan", "tak", "gak", "nggak", "tdk"
}

# ── Pola Regex PII dengan Waktu Eksekusi Linear (SEC-01 & SEC-02) ────────────
# 1. Email: kuantifier terbatas mencegah catastrophic backtracking (ReDoS)
_EMAIL = re.compile(r'[\w.+-]{1,64}@[\w-]{1,63}(?:\.[\w-]{1,63}){1,5}')

# 2. Nomor Telepon (+62, 08xx, spasi/strip/titik) - dieksekusi sebelum NIM agar tidak salah tangkap
_TELEPON = re.compile(r'(?<![\w+])(?:\+62|62|0)[\s.-]?8\d{1,3}(?:[\s.-]?\d{3,4}){2,3}(?!\d)')

# 3. NIM: format bertitik (24.06.05.110.034) atau 9-12 digit berurutan
_NIM = re.compile(r'(?<!\d)\d{2}[.\s-]?\d{2}[.\s-]?\d{2}[.\s-]?\d{3}[.\s-]?\d{3}(?!\d)|(?<!\d)\d{9,12}(?!\d)')

# 4. Heuristik Nama Orang: Peka huruf besar-kecil (JANGAN gunakan re.IGNORECASE)
_NAMA = re.compile(r'\b(?:Pak|Bu|Bapak|Ibu|Mas|Mbak|Dr|Prof)\.?\s+[A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){0,2}\b')

_PII_PATTERNS = [
    (_EMAIL, '[EMAIL]'),
    (_TELEPON, '[TELEPON]'),
    (_NIM, '[NIM]'),
    (_NAMA, '[NAMA]'),
]


def mask_pii(teks: str) -> str:
    """
    Mendeteksi dan menyamarkan data pribadi (PII): Email, Nomor Telepon, NIM, dan Nama.
    Menghasilkan teks tersanitasi yang aman dari kebocoran privasi.

    Parameters
    ----------
    teks : str  — Teks mentah masukan

    Returns
    -------
    str  — Teks dengan data pribadi yang sudah disamarkan
    """
    if not isinstance(teks, str):
        teks = '' if teks is None else str(teks)

    for pattern, placeholder in _PII_PATTERNS:
        teks = pattern.sub(placeholder, teks)
    return teks


def clean_text(teks: str) -> str:
    """
    Normalisasi kata kunci umum (MOD-05), konversi ke huruf kecil,
    serta menghapus karakter non-alfabet kecuali tanda kurung placeholder PII.

    Parameters
    ----------
    teks : str

    Returns
    -------
    str  — Teks bersih ternormalisasi
    """
    if not isinstance(teks, str):
        teks = '' if teks is None else str(teks)

    # Normalisasi ejaan/frasa domain umum
    teks = re.sub(r'\bwi[\s-]?fi\b', 'wifi', teks, flags=re.IGNORECASE)
    teks = re.sub(r'\be[\s-]?mail\b', 'email', teks, flags=re.IGNORECASE)

    teks = teks.lower()
    # Hapus semua karakter kecuali huruf alfabet, spasi, dan bracket placeholder PII
    teks = re.sub(r'[^a-z\s\[\]]', ' ', teks)
    # Rampingkan spasi ganda
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks


def remove_stopwords(teks: str, stopwords: set = STOPWORDS) -> str:
    """
    Membuang kata stopword non-semantik dari teks yang telah bersih.

    Parameters
    ----------
    teks      : str  — Teks bersih
    stopwords : set  — Set kata henti

    Returns
    -------
    str  — Teks tanpa stopword
    """
    kata_list = teks.split()
    hasil = [kata for kata in kata_list if kata not in stopwords]
    return ' '.join(hasil)


def tokenize(teks: str) -> list:
    """
    Memecah teks menjadi daftar token dan memfilter token pendek (< 2 karakter).

    Parameters
    ----------
    teks : str

    Returns
    -------
    list[str]  — Daftar token
    """
    tokens = teks.split()
    return [token for token in tokens if len(token) >= 2]


def mark_negation(tokens: list) -> list:
    """
    Menandai token yang berada tepat setelah kata negasi dengan awalan 'NOT_'
    untuk mempertahankan konteks polaritas (MOD-03).
    Contoh: ["wifi", "belum", "stabil"] -> ["wifi", "belum", "NOT_stabil"]

    Parameters
    ----------
    tokens : list[str]

    Returns
    -------
    list[str]  — Daftar token dengan negasi kontekstual
    """
    out = []
    flag = False
    for w in tokens:
        if flag:
            out.append(f"NOT_{w}")
            flag = False
        else:
            out.append(w)
        if w in NEGATORS:
            flag = True
    return out


def preprocess_pipeline(teks: str) -> dict:
    """
    Menjalankan seluruh pipeline preprocessing secara terurut dan defensif.

    Alur:
        1. Validasi Batas Panjang Input (SEC-01)
        2. PII Sanitization (SEC-02)
        3. Normalisasi & Lowercasing (MOD-05)
        4. Stopword Removal Terkalibrasi (MOD-03)
        5. Tokenisasi
        6. Negation Context Marking (MOD-03)

    Parameters
    ----------
    teks : str  — Teks ulasan masukan

    Returns
    -------
    dict dengan kunci:
        teks_asli       : str
        teks_masked_pii : str
        teks_bersih     : str
        teks_no_stop    : str
        tokens          : list[str]
        jumlah_token    : int
        pii_ditemukan   : bool

    Raises
    ------
    InputTooLong : jika panjang karakter melebihi MAX_CHARS
    """
    if not isinstance(teks, str):
        teks = '' if teks is None else str(teks)

    if len(teks) > MAX_CHARS:
        raise InputTooLong(f"Ulasan melebihi batas maksimum {MAX_CHARS} karakter (panjang: {len(teks)}).")

    teks_asli = teks.strip()
    teks_masked = mask_pii(teks_asli)
    pii_ditemukan = teks_masked != teks_asli

    teks_clean = clean_text(teks_masked)
    teks_no_stop = remove_stopwords(teks_clean)
    raw_tokens = tokenize(teks_no_stop)
    tokens = mark_negation(raw_tokens)

    return {
        "teks_asli"       : teks_asli,
        "teks_masked_pii" : teks_masked,
        "teks_bersih"     : teks_clean,
        "teks_no_stop"    : teks_no_stop,
        "tokens"          : tokens,
        "jumlah_token"    : len(tokens),
        "pii_ditemukan"   : pii_ditemukan,
    }


def tokens_to_string(tokens: list) -> str:
    """Helper: Gabungkan token list menjadi string untuk ekstraksi fitur."""
    return ' '.join(tokens)
