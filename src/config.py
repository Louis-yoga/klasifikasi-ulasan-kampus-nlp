"""
config.py
---------
Konfigurasi batasan input dan batas sistem untuk Sistem Klasifikasi Ulasan Kampus.
Mencegah ReDoS, memory exhaustion, dan unconstrained inputs.
"""

MAX_CHARS = 2000        # Panjang maksimum karakter untuk satu ulasan teks
MAX_ROWS = 500          # Jumlah baris maksimum untuk satu batch CSV
MAX_BYTES = 1_000_000   # Ukuran berkas maksimum unggahan (1 MB)
