"""
preprocessor.py
---------------
Pipeline preprocessing teks untuk Sistem Klasifikasi Ulasan Kampus.
Langkah: PII Masking → Lowercase → Hapus Tanda Baca → Stopword Removal → Tokenisasi
"""

import re

# ── Daftar Stopword Ringan Bahasa Indonesia ─────────────────────────────────
# "tidak" sengaja TIDAK dimasukkan karena mengandung informasi sentimen penting
STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "ini", "itu",
    "atau", "juga", "pada", "sudah", "saya", "kami", "kita", "adalah",
    "ada", "bisa", "akan", "lebih", "sekali", "nya", "pun", "lah",
    "ya", "si", "para", "sang", "agar", "atas", "bagi", "bahwa",
    "belum", "bila", "bukan", "demi", "dalam", "hal", "hingga",
    "jika", "kalau", "karena", "lagi", "maka", "menjadi", "meski",
    "oleh", "pun", "sama", "sebelum", "sebuah", "sedang", "sehingga",
    "sering", "setelah", "sini", "suatu", "supaya", "telah", "tentang",
    "tiap", "waktu", "walaupun", "seperti", "setiap", "antara"
}

# ── Pola Regex PII ───────────────────────────────────────────────────────────
_PII_PATTERNS = [
    # NIM: 9–12 digit angka berurutan (tanpa spasi)
    (re.compile(r'\b\d{9,12}\b'), '[NIM]'),
    # Email
    (re.compile(r'\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b', re.IGNORECASE), '[EMAIL]'),
    # Nomor telepon: 08xx atau +62xx (8–13 digit)
    (re.compile(r'\b(?:\+62|08)\d{7,11}\b'), '[TELEPON]'),
]


def mask_pii(teks: str) -> str:
    """
    Mendeteksi dan menyamarkan pola PII (NIM, Email, Nomor Telepon)
    dengan placeholder yang aman.

    Parameters
    ----------
    teks : str  — Teks mentah masukan

    Returns
    -------
    str  — Teks dengan PII yang sudah di-mask
    """
    for pattern, placeholder in _PII_PATTERNS:
        teks = pattern.sub(placeholder, teks)
    return teks


def clean_text(teks: str) -> str:
    """
    Lowercase + hapus tanda baca, angka, dan karakter non-alfabet.
    Karakter '[' dan ']' dipertahankan agar placeholder PII tidak rusak.

    Parameters
    ----------
    teks : str

    Returns
    -------
    str  — Teks bersih huruf kecil
    """
    teks = teks.lower()
    # Hapus semua karakter kecuali huruf, spasi, dan placeholder bracket
    teks = re.sub(r'[^a-z\s\[\]]', ' ', teks)
    # Kolaps spasi ganda
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks


def remove_stopwords(teks: str, stopwords: set = STOPWORDS) -> str:
    """
    Membuang kata-kata stopword dari teks yang sudah bersih.

    Parameters
    ----------
    teks      : str   — Teks bersih (sudah lowercase)
    stopwords : set   — Set kata stopword

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


def preprocess_pipeline(teks: str) -> dict:
    """
    Menjalankan seluruh pipeline preprocessing secara berurutan dan
    mengembalikan hasil setiap tahap untuk keperluan explainability.

    Alur:
        Teks Asli
        → PII Masking
        → Lowercase + Hapus Tanda Baca
        → Stopword Removal
        → Tokenisasi

    Parameters
    ----------
    teks : str  — Teks ulasan mentah

    Returns
    -------
    dict dengan kunci:
        teks_asli       : str
        teks_masked_pii : str
        teks_bersih     : str       (setelah cleaning)
        teks_no_stop    : str       (setelah stopword removal)
        tokens          : list[str]
        jumlah_token    : int
        pii_ditemukan   : bool
    """
    teks_asli = teks.strip()
    teks_masked = mask_pii(teks_asli)
    pii_ditemukan = teks_masked != teks_asli

    teks_clean = clean_text(teks_masked)
    teks_no_stop = remove_stopwords(teks_clean)
    tokens = tokenize(teks_no_stop)

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
    """Helper: Gabungkan token list menjadi string untuk vectorizer."""
    return ' '.join(tokens)


# ── Self-test sederhana ──────────────────────────────────────────────────────
if __name__ == '__main__':
    test_cases = [
        "WiFi di kelas sangat lambat!",
        "NIM: 240605110034 susah login KRS",
        "Hubungi 081234567890 atau email test@uin.ac.id untuk info",
        "Bagus sekali!!!",
        "",
    ]
    for tc in test_cases:
        hasil = preprocess_pipeline(tc)
        print(f"Input  : {repr(tc)}")
        print(f"Tokens : {hasil['tokens']}")
        print(f"PII    : {hasil['pii_ditemukan']}")
        print("-" * 50)
