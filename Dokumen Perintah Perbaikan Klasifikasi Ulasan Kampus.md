# Dokumen Perintah Perbaikan: Klasifikasi Ulasan Kampus

*Dibuat dari audit commit 85cb982 · Urutan prioritas: Keamanan → Performa → Integritas Model → UX → Nilai Jual · 18 November 2026*

## 0. Tujuan dan cara memakai

Dokumen ini adalah **perintah kerja** untuk engineer atau AI coding agent. Tujuannya memperbaiki seluruh temuan audit sebelum pengujian mendalam (penetration test, uji beban, uji pengguna). Setiap tugas memuat ID (merujuk ke audit), file target, langkah, dan **Kriteria Selesai** yang dapat dibuktikan dengan perintah.

Catatan jujur: kata “sempurna” tidak dapat dijamin oleh dokumen mana pun. Yang dapat dijamin adalah gerbang yang terukur (§10). Nilai jual produk pada akhirnya ditentukan oleh akurasi pada **data nyata** (§8), bukan oleh kerapian kode.

## 1. Aturan eksekusi (wajib dipatuhi)

1. Buat branch `fix/pra-uji-mendalam`. Satu commit per tugas, pesan `[ID] ringkasan`.
2. Tes lebih dulu: tulis tes yang gagal (merah), perbaiki, pastikan hijau. Semua tes berada di `tests/` (§9).
3. Jangan mengubah label pada file data. Jangan melatih pada set uji eksternal untuk angka yang dilaporkan (MOD-01).
4. Jangan menulis angka metrik, jumlah data, atau nama algoritma secara manual di UI atau README. Semuanya dibaca dari `models/metadata.json` (SEC-04).
5. Jangan menambah dependensi tanpa menyematkan versinya, lalu jalankan `pip-audit`.
6. Setiap perubahan yang terlihat pengguna wajib memperbarui teks UI dan README pada commit yang sama.
7. Bila instruksi di sini bertentangan dengan kode yang ada, berhenti dan laporkan. Jangan menebak.
8. Laporan akhir per tugas: `ID | status | bukti (output perintah) | catatan`.

## 2. Peta fase dan gerbang keluar

| Fase | Fokus | Tugas | Gerbang keluar |
| --- | --- | --- | --- |
| 1 | Keamanan | SEC-01 … SEC-09 | `pytest tests/test_security.py` hijau; `bandit` dan `pip-audit` bersih |
| 2 | Performa | PERF-01 … PERF-05 | `python scripts/bench.py` memenuhi anggaran waktu §4 |
| 3 | Integritas model dan data | MOD-01 … MOD-07 | `python scripts/evaluate.py` menghasilkan laporan CV dan uji eksternal; tes model hijau |
| 4 | Fungsi dan kontrak data | FUN-01 … FUN-04 | `pytest tests/test_functional.py` hijau |
| 5 | UX | UX-01 … UX-14 | Daftar periksa UX §7 lulus; uji kegunaan 5 orang |
| 6 | Kemasan produk | PRD-01 … PRD-07 | Docker build dan health check OK; dokumen lengkap |

Jangan memulai fase berikutnya sebelum gerbang fase sebelumnya lulus.

## 3. FASE 1: KEAMANAN

### SEC-01 · Hilangkan ReDoS dan batasi seluruh input (audit S-01, Critical)

**File:** `src/config.py` (baru), `src/preprocessor.py`, `src/classifier.py`, `app.py`.

**Langkah:**

1. Buat `src/config.py`:

```python
MAX_CHARS = 2000        # panjang maksimum satu ulasan
MAX_ROWS = 500          # baris maksimum satu batch
MAX_BYTES = 1_000_000   # ukuran maksimum file unggahan
```

2. Di `preprocessor.py`, ganti pola email (`preprocessor.py:29`) dengan kuantifier terbatas agar waktu eksekusi linear, dan tolak input terlalu panjang:

```python
from src.config import MAX_CHARS

class InputTooLong(ValueError):
    pass

# pola lama r'\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b' bersifat kuadratik (200 KB = 112 detik)
_EMAIL = re.compile(r'[\w.+-]{1,64}@[\w-]{1,63}(?:\.[\w-]{1,63}){1,5}')

def preprocess_pipeline(teks):
    if not isinstance(teks, str):
        teks = '' if teks is None else str(teks)
    if len(teks) > MAX_CHARS:
        raise InputTooLong(f'Ulasan melebihi {MAX_CHARS} karakter.')
    # ... lanjutan pipeline
```

3. Di `app.py` (`st.text_area`, baris 319), tambahkan `max_chars=MAX_CHARS`.
4. Di `predict_batch`, potong tiap sel dengan `teks[:MAX_CHARS]` dan hitung berapa sel yang dipotong (dipakai UX-06).

**Kriteria selesai:**

- `mask_pii('a.' * 20000 + '@')` selesai di bawah 1 detik (sebelum perbaikan: 40 KB = 4,4 detik; 200 KB = 112 detik).
- `preprocess_pipeline('a.' * 100000)` melempar `InputTooLong` seketika.
- Tes `test_email_waktu_linear` dan `test_input_terlalu_panjang` hijau.

### SEC-02 · Perbaiki PII dan jangan pernah mengeluarkan teks mentah (S-02, High)

**File:** `src/preprocessor.py`, `src/classifier.py`, `app.py`, `scripts/check_pii_fp.py` (baru).

**Langkah:**

1. Ganti `_PII_PATTERNS` (`preprocessor.py:25-32`). Urutan penting: telepon sebelum NIM. Pola ini sudah diuji pada audit untuk email, telepon, dan NIM (0 false positive pada 100 ulasan). Pola nama adalah heuristik baru dan **wajib** lolos pemeriksaan di bawah:

```python
_PII_PATTERNS = [
    (_EMAIL, '[EMAIL]'),
    (re.compile(r'(?<![\w+])(?:\+62|62|0)[\s.-]?8\d{1,3}(?:[\s.-]?\d{3,4}){2,3}(?!\d)'), '[TELEPON]'),
    (re.compile(r'(?<!\d)\d{2}[.\s-]?\d{2}[.\s-]?\d{2}[.\s-]?\d{3}[.\s-]?\d{3}(?!\d)|(?<!\d)\d{9,12}(?!\d)'), '[NIM]'),
    # heuristik nama: peka huruf besar-kecil, JANGAN pakai IGNORECASE
    (re.compile(r'\b(?:Pak|Bu|Bapak|Ibu|Mas|Mbak|Dr|Prof)\.?\s+[A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){0,2}\b'), '[NAMA]'),
]
```

2. `predict_batch` (`classifier.py:204-211`): kolom `ulasan` harus berisi `prep_result['teks_masked_pii']`, bukan teks mentah.
3. `app.py:344-345` dan `app.py:538`: simpan dan tampilkan hanya teks tersamar. Ubah judul “Teks Asli” menjadi “Teks setelah penyamaran data pribadi”.
4. Buat `scripts/check_pii_fp.py` yang menjalankan `mask_pii` pada semua ulasan di `data/dataset.csv` dan set uji eksternal, lalu mencetak jumlah ulasan yang berubah. Jumlahnya harus **0**. Bila pola nama menghasilkan false positive, perketat pola sampai 0.
5. Nama tidak mungkin terdeteksi sempurna oleh regex. Tampilkan peringatan di UI (UX-12): “Nama orang mungkin tidak terdeteksi. Jangan menulis data pribadi.”

**Kriteria selesai:** tes parametrik mencakup `+6281234567890`, `+62 812-3456-7890`, `0812-3456-7890`, `081234567890` (harus `[TELEPON]`, bukan `[NIM]`), `24.06.05.110.034`, `240605110034`, email berhyphen `budi.s@student.uin-malang.ac.id` (tanpa sisa domain), dan `Pak Budi Santoso`. `scripts/check_pii_fp.py` = 0. Unduhan CSV dan tabel batch tidak memuat digit/email dari input uji.

### SEC-03 · Ekspor CSV aman (S-03, Medium)

**File:** `app.py:514`, `src/exporters.py` (baru).

Penyebab: sel yang diawali `=`, `+`, `-`, `@` dieksekusi sebagai rumus oleh Excel, dan `to_csv(encoding='utf-8-sig')` tanpa path **mengabaikan** encoding (tanpa BOM, emoji rusak di Excel).

```python
FORMULA_PREFIX = ('=', '+', '-', '@', '\t', '\r')

def to_safe_csv(df):
    safe = df.copy()
    for c in safe.select_dtypes(include=['object', 'string']).columns:
        safe[c] = safe[c].map(lambda v: chr(39) + v if isinstance(v, str) and v.startswith(FORMULA_PREFIX) else v)
    return '\ufeff' + safe.to_csv(index=False)
```

Gunakan `to_safe_csv` di semua tombol unduh (termasuk XLSX pada UX-06: terapkan awalan yang sama pada sel string).

**Kriteria selesai:** tes memasukkan ulasan `=HYPERLINK("http://evil.example","x")` dan `@SUM(1+1)`; hasil unduhan diawali `'`; string hasil diawali BOM (`\ufeff`).

### SEC-04 · Integritas artefak model dan metadata (S-04, F-05)

**File:** `src/artifacts.py` (baru), `train.py` (baru), `src/classifier.py:223-230`, `src/feature_extractor.py:97-104`.

**Langkah:**

1. Hapus `save_model`/`load_model` dan `save_vectorizer`/`load_vectorizer`. Semua `joblib.dump/load` hanya boleh ada di `src/artifacts.py`.
2. Tulis `src/artifacts.py`:

```python
import datetime, hashlib, json, os, tempfile
import joblib, sklearn

NAMES = {'vectorizer': 'vectorizer.pkl', 'sentimen': 'model_sentimen.pkl', 'kategori': 'model_kategori.pkl'}

class ArtifactError(RuntimeError):
    pass

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blok in iter(lambda: f.read(1 << 20), b''):
            h.update(blok)
    return h.hexdigest()

def _atomic_write(path, writer):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or '.')
    os.close(fd)
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

def save_artifacts(model_dir, objs, metadata):
    os.makedirs(model_dir, exist_ok=True)
    hashes = {}
    for key, nama in NAMES.items():
        path = os.path.join(model_dir, nama)
        _atomic_write(path, lambda p, o=objs[key]: joblib.dump(o, p))
        hashes[key] = sha256(path)
    meta = {**metadata, 'files': hashes, 'sklearn_version': sklearn.__version__,
            'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    def tulis(p):
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    _atomic_write(os.path.join(model_dir, 'metadata.json'), tulis)

def load_artifacts(model_dir):
    mp = os.path.join(model_dir, 'metadata.json')
    if not os.path.exists(mp):
        raise ArtifactError('metadata.json tidak ditemukan. Jalankan: python train.py')
    with open(mp, encoding='utf-8') as f:
        meta = json.load(f)
    if meta['sklearn_version'].split('.')[:2] != sklearn.__version__.split('.')[:2]:
        raise ArtifactError('Versi scikit-learn berbeda dari saat pelatihan. Jalankan: python train.py')
    objs = {}
    for key, nama in NAMES.items():
        path = os.path.join(model_dir, nama)
        if not os.path.exists(path) or sha256(path) != meta['files'][key]:
            raise ArtifactError(f'Integritas {nama} gagal diverifikasi (berkas hilang atau hash tidak cocok).')
        objs[key] = joblib.load(path)
    return objs, meta
```

3. `train.py` (satu-satunya tempat pelatihan): baca `data/dataset.csv`, split stratifikasi, tuning (MOD-05), fit, evaluasi (MOD-01), lalu `save_artifacts`. `metadata` memuat: `data_sha256`, jumlah per kelas, `params`, `metrics`, `labels`, `thresholds` (MOD-06).
4. `app.py:190-237` tidak boleh melatih model. Aplikasi hanya memuat artefak (FUN-02).

Catatan keamanan: hash di dalam repositori yang sama melindungi dari berkas rusak atau tertukar, **bukan** dari penyerang yang bisa menulis ke repositori. Untuk produksi, simpan hash rujukan di variabel lingkungan atau tanda tangani di CI.

**Kriteria selesai:** (a) ubah satu byte `model_sentimen.pkl` → aplikasi menampilkan pesan integritas gagal, bukan traceback; (b) hapus `metadata.json` → pesan “Jalankan: python train.py”; (c) `grep -rn "joblib.load" --include=*.py .` hanya menunjuk `src/artifacts.py`; (d) tidak ada file sementara tersisa setelah pelatihan yang diinterupsi.

### SEC-05 · Dependensi (S-05)

1. Buat `requirements.in` dengan batas bawah aman: `streamlit>=1.54.0`, `scikit-learn>=1.5.0`, `pandas`, `numpy`, `joblib`, `openpyxl` (dipakai ekspor XLSX di UX-06). **Hapus** `matplotlib` dan `seaborn`: semua grafik diganti komponen bawaan/Altair (PERF-03).
2. `pip-compile --generate-hashes -o requirements.lock requirements.in`. Instalasi produksi: `pip install --require-hashes -r requirements.lock`.
3. Versi yang terbukti berjalan pada audit (Python 3.12.3): streamlit 1.67.0, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.4. Tetapkan versi Python minimum dari hasil `pip-compile` dan CI, bukan menebak. Perbarui README (saat ini tertulis “3.9 atau lebih baru”).
4. Ganti 9 pemakaian `use_container_width=True` dengan `width='stretch'` (periksa nama argumen via `help(st.dataframe)`). Versi Streamlit terbaru sudah mengumumkan penghapusan argumen lama.
5. Latih ulang model (`python train.py`) setelah versi dikunci, karena artefak terikat versi scikit-learn.

**Kriteria selesai:** `pip-audit -r requirements.lock` melaporkan 0 kerentanan; `grep -c use_container_width app.py` = 0; log aplikasi tanpa peringatan deprecation; batas bawah lama (streamlit 1.32.0, scikit-learn 1.4.0) tidak lagi diizinkan (PYSEC-2024-153, PYSEC-2026-212, PYSEC-2026-2285, PYSEC-2024-110).

### SEC-06 · Pengerasan runtime, pembatasan laju, kontrol akses, logging (S-06)

1. Buat `.streamlit/config.toml` dan **hapus** baris `.streamlit/` dari `.gitignore` (tambahkan `.streamlit/secrets.toml` sebagai gantinya):

```toml
[server]
maxUploadSize = 1
enableXsrfProtection = true
headless = true

[browser]
gatherUsageStats = false

[client]
toolbarMode = "viewer"
showErrorDetails = "none"
```

Verifikasi tiap kunci dengan `streamlit config show`; hapus kunci yang tidak dikenal oleh versi terpasang.

2. Pembatas sesi `src/guard.py` (dipanggil sebelum analisis tunggal dan batch):

```python
import time
import streamlit as st

def throttle(key='req', min_interval=1.0, max_per_minute=20):
    now = time.time()
    hist = [t for t in st.session_state.get(key, []) if now - t < 60]
    if (hist and now - hist[-1] < min_interval) or len(hist) >= max_per_minute:
        return False
    hist.append(now)
    st.session_state[key] = hist
    return True
```

Pembatas ini dapat dilewati dengan membuka sesi baru. Pembatas **yang sebenarnya** berada di reverse proxy:

```nginx
limit_req_zone $binary_remote_addr zone=ulasan:10m rate=5r/s;
server {
  client_max_body_size 1m;
  location / {
    limit_req zone=ulasan burst=20 nodelay;
    proxy_pass http://127.0.0.1:8501;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection 'upgrade';
    proxy_set_header Host $host;
  }
}
```

3. Kontrol akses: bila di-deploy ke jaringan yang dapat diakses publik, letakkan di belakang proxy dengan autentikasi atau gunakan `st.login`. Rahasia hanya lewat `st.secrets` atau variabel lingkungan, tidak pernah di repositori.
4. Logging `src/log.py` (tanpa PII):

```python
import json, logging
log = logging.getLogger('ulasan')

def event(nama, **kolom):
    log.info(json.dumps({'event': nama, **kolom}, ensure_ascii=False))
```

Aturan: boleh mencatat `n_chars`, `n_rows`, `durasi_ms`, `error_id`, `status`. **Dilarang** mencatat teks ulasan, nama file, atau isi sel.

**Kriteria selesai:** unggahan 2 MB ditolak server; `tests/test_security.py::test_throttle` hijau; uji manual 30 permintaan cepat ke proxy menghasilkan sebagian 429/503; `grep -rn "log\." src app.py` tidak memuat variabel teks ulasan; tidak ada telemetri keluar.

### SEC-07 · Hentikan HTML/markdown dari teks pengguna (S-07, sekaligus U-03)

**File:** `app.py:53-87, 363-379, 400-404, 538-545, 560-566, 593-595, 709-716`.

1. Hapus seluruh `unsafe_allow_html=True`. Ganti kartu HTML dengan `st.container(border=True)`, `st.metric`, `st.badge`, `st.warning`/`st.info`, dan `st.dataframe`. Komponen bawaan mengikuti tema terang/gelap, sehingga kontras dan kotak peringatan tak terbaca di mode gelap (`.low-conf-box`) ikut teratasi.
2. Teks pengguna hanya ditampilkan lewat `st.text`, `st.code`, atau `st.dataframe`. Jangan pernah memasukkannya ke f-string untuk `st.markdown` (`app.py:538,543,545`).
3. Tabel Responsible AI (`app.py:709-716`) menjadi `st.dataframe` yang dibangun dari data (MOD-07).

**Kriteria selesai:** `grep -c unsafe_allow_html app.py` = 0. Tes `AppTest` mengisi input `![x](http://evil.example/p.png) [klik](javascript:alert(1)) <script>x</script>`; tidak ada elemen markdown yang memuat `evil.example`.

### SEC-08 · Gerbang analisis statis

Tambahkan ke CI (§9): `ruff check .`, `bandit -r src app.py train.py -ll`, `pip-audit -r requirements.lock`. Temuan bandit pada `joblib.load` di `src/artifacts.py` boleh diberi `# nosec` hanya dengan komentar yang menyebut verifikasi SHA-256.

**Kriteria selesai:** ketiga perintah keluar dengan kode 0 di CI.

### SEC-09 · Kebersihan repositori dan kebocoran identitas (S-08, F-07)

1. Hapus gambar hotlink Wikimedia (`app.py:254`); pakai `assets/logo.png` lokal. Pastikan hak pakai logo institusi; bila tidak ada izin, gunakan logo produk sendiri.
2. Hapus nama dan NIM pengembang dari footer UI (`app.py:732-737`). Ganti dengan nama produk, versi, dan tautan Panduan. Kredit akademik cukup di README.
3. README: buang path lokal `d:/KULIAH/...` (baris 35) dan tautan `file:///` (baris 82, 156, 159); gunakan path relatif.
4. Hapus `data/data_latih.csv` (duplikat `dataset.csv`), `data/hasil_klasifikasi.csv` (keluaran sirkular), `data/Book2.csv` (diganti pada MOD-02), dan `data/Book12.xlsx` bila tak terpakai. Pindahkan `Data.csv` menjadi `data/external_test.csv`, dan PRD `.docx` ke `docs/`.
5. Hapus `test_pipeline.py` (diganti pada FUN-04).

**Kriteria selesai:** `git ls-files` tidak memuat file yang disebut di atas; `grep -rn "240605110034" .` hanya ditemukan di README bagian kredit (bila dipertahankan).
