"""
log.py
------
Modul structured audit logging tanpa PII (SEC-06).
Hanya mencatat metadata operasional: durasi, jumlah baris/karakter, status, dan kode error.
DILARANG mencatat teks ulasan, nama file asli, atau identitas pengguna.
"""

import json
import logging

logger = logging.getLogger('ulasan_audit')
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def log_event(event_name: str, **attributes) -> None:
    """
    Mencatat event audit operasional dalam format JSON terstruktur.
    Memastikan tidak ada isi teks mentah pengguna yang tersimpan.

    Parameters
    ----------
    event_name : str
    attributes : kwargs (misal: n_chars, n_rows, durasi_ms, status)
    """
    payload = {
        'event': event_name,
        **attributes
    }
    logger.info(json.dumps(payload, ensure_ascii=False))
