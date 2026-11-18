"""
guard.py
--------
Modul pembatasan laju permintaan tingkat sesi (Session Throttling) untuk
mencegah spamming dan brute-force inferensi (SEC-06).
"""

import time


def throttle(session_state_dict: dict, key: str = 'req_timestamps', min_interval: float = 0.5, max_per_minute: int = 30) -> bool:
    """
    Memeriksa apakah permintaan saat ini diizinkan berdasarkan batas interval
    dan frekuensi maksimum per menit pada sesi pengguna.

    Parameters
    ----------
    session_state_dict : dict or streamlit.session_state
    key                : str
    min_interval       : float (detik)
    max_per_minute     : int

    Returns
    -------
    bool : True jika diizinkan, False jika terkena pembatasan laju
    """
    now = time.time()
    hist = [t for t in session_state_dict.get(key, []) if now - t < 60]

    if (hist and now - hist[-1] < min_interval) or len(hist) >= max_per_minute:
        return False

    hist.append(now)
    session_state_dict[key] = hist
    return True
