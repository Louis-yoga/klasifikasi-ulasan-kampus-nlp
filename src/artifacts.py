"""
artifacts.py
------------
Modul pengelolaan persistensi artefak model machine learning dengan
verifikasi integritas kriptografis SHA-256 dan pemeriksaan versi scikit-learn (SEC-04).
"""

import datetime
import hashlib
import json
import os
import tempfile

import joblib
import sklearn

NAMES = {
    'vectorizer': 'vectorizer.pkl',
    'sentimen'  : 'model_sentimen.pkl',
    'kategori'  : 'model_kategori.pkl'
}


class ArtifactError(RuntimeError):
    """Exception jika berkas model rusak, hash tidak cocok, atau metadata tidak ditemukan."""


def sha256(path: str) -> str:
    """Menghitung nilai SHA-256 hash dari sebuah berkas."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _atomic_write(path: str, writer_func) -> None:
    """Menulis berkas secara atomik untuk mencegah berkas terkorupsi saat terputus."""
    target_dir = os.path.dirname(path) or '.'
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=target_dir)
    os.close(fd)
    try:
        writer_func(tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def save_artifacts(model_dir: str, objs: dict, metadata: dict) -> dict:
    """
    Menyimpan artefak model (.pkl) dan berkas metadata.json secara atomik
    disertai checksum SHA-256 untuk tiap berkas.

    Parameters
    ----------
    model_dir : str
    objs      : dict dengan kunci 'vectorizer', 'sentimen', 'kategori'
    metadata  : dict memuat metrik, parameter, dan info dataset

    Returns
    -------
    dict  — Metadata lengkap yang tersimpan
    """
    os.makedirs(model_dir, exist_ok=True)
    hashes = {}
    for key, filename in NAMES.items():
        filepath = os.path.join(model_dir, filename)
        _atomic_write(filepath, lambda p, o=objs[key]: joblib.dump(o, p))
        hashes[key] = sha256(filepath)

    meta_payload = {
        **metadata,
        'files': hashes,
        'sklearn_version': sklearn.__version__,
        'created_utc': datetime.datetime.now(datetime.UTC).isoformat()
    }

    def write_json(p):
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(meta_payload, f, indent=2, ensure_ascii=False)

    _atomic_write(os.path.join(model_dir, 'metadata.json'), write_json)
    return meta_payload


def load_artifacts(model_dir: str) -> tuple:
    """
    Memuat artefak model setelah memverifikasi keberadaan dan keutuhan hash SHA-256
    serta kompatibilitas versi scikit-learn.

    Parameters
    ----------
    model_dir : str

    Returns
    -------
    tuple : (objs: dict, metadata: dict)

    Raises
    ------
    ArtifactError : jika metadata tidak ada, hash tidak cocok, atau versi tidak kompatibel
    """
    metadata_path = os.path.join(model_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise ArtifactError("metadata.json tidak ditemukan. Jalankan: python train.py")

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise ArtifactError(f"Gagal membaca metadata.json: {e}") from e

    # Periksa kompatibilitas versi scikit-learn (major.minor)
    meta_sklearn = metadata.get('sklearn_version', '')
    cur_sklearn = sklearn.__version__
    if meta_sklearn.split('.')[:2] != cur_sklearn.split('.')[:2]:
        raise ArtifactError(
            f"Versi scikit-learn ({cur_sklearn}) berbeda dari saat pelatihan ({meta_sklearn}). Jalankan: python train.py"
        )

    objs = {}
    for key, filename in NAMES.items():
        filepath = os.path.join(model_dir, filename)
        if not os.path.exists(filepath):
            raise ArtifactError(f"Berkas model {filename} tidak ditemukan. Jalankan: python train.py")

        expected_hash = metadata.get('files', {}).get(key)
        actual_hash = sha256(filepath)
        if actual_hash != expected_hash:
            raise ArtifactError(
                f"Integritas {filename} gagal diverifikasi (hash tidak cocok). Jalankan: python train.py"
            )
        # nosec: Verifikasi integritas kriptografis SHA-256 telah divalidasi sebelum deserialisasi
        objs[key] = joblib.load(filepath)

    return objs, metadata
