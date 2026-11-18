# 🎓 Sistem Klasifikasi Ulasan Layanan Kampus Berbasis NLP & Multinomial Naive Bayes

> **Mini Project Mata Kuliah Artificial Intelligence — Pertemuan 15**  
> **Program Studi S1 Teknik Informatika, Fakultas Sains dan Teknologi**  
> **UIN Maulana Malik Ibrahim Malang**  
> **Pengembang:** Louis Ainur Rofiq Prayoga (NIM: 240605110034 / Kelas H)  
> **Branch Repositori:** `fix/pra-uji-mendalam` | **Versi:** 2.0.0 (Audit-Hardened)

---

## 📌 1. Ringkasan Eksekutif & Latar Belakang

Sistem ini dirancang untuk mengotomatiskan analisis ulasan mahasiswa terkait layanan kampus dari teks bebas (*unstructured text*) menjadi rekapitulasi terstruktur yang dapat diaudit dan dipercaya. Sistem bekerja secara simultan menghasilkan dua dimensi prediksi:

1. **Sentimen (3 Kelas):** `Positif`, `Negatif`, dan `Netral`
2. **Kategori Layanan (4 Unit):** `Administrasi`, `Akademik`, `Fasilitas`, dan `Kemahasiswaan`

Sistem dirancang dengan arsitektur **Augmented Intelligence** (sebagai *decision support* pendukung staf kampus, bukan pengganti kebijakan manusia) dan memenuhi **5 Pilar Trustworthy AI** (*Privacy, Fairness, Explainability, Transparency, dan Human Oversight*).

---

## 🔒 2. Fitur Keamanan & Integritas Sistem (Hasil Audit Teknis)

Berdasarkan hasil audit mendalam terhadap kerentanan dan integritas data, sistem telah diperkuat dengan perlindungan berikut:

1. **Anti-ReDoS & Bounded Input (SEC-01):**
   * Pola regex email dan entitas teks menggunakan kuantifier terbatas (`{1,64}`) dengan kompleksitas waktu linear $\mathcal{O}(n)$.
   * Pembatasan panjang input maksimum: teks tunggal $\le 2.000$ karakter (`MAX_CHARS`), berkas batch $\le 500$ baris (`MAX_ROWS`), dan ukuran unggahan $\le 1\text{ MB}$ (`MAX_BYTES`).
2. **Sanitasi PII Tanpa Kebocoran (SEC-02):**
   * Deteksi nomor telepon (`+62`, `08xx`), NIM (9–12 digit bertitik maupun undotted), email, dan nama staf/dosen disamarkan menjadi `[TELEPON]`, `[NIM]`, `[EMAIL]`, dan `[NAMA]`.
   * Teks mentah ber-PII tidak pernah diekspos ke antarmuka, `session_state`, maupun berkas unduhan hasil klasifikasi.
3. **Proteksi CSV Formula Injection & UTF-8 BOM (SEC-03):**
   * Sel yang diawali karakter eksekusi rumus (`=`, `+`, `-`, `@`, `\t`, `\r`) dinetralkan dengan awalan tanda petik tunggal (`'`).
   * Ekspor berkas CSV menyertakan UTF-8 BOM (`\ufeff`) sehingga karakter aksen dan format terbaca rapi di Microsoft Excel.
4. **Verifikasi Kriptografis Model SHA-256 (SEC-04):**
   * Seluruh artefak biner (`vectorizer.pkl`, `model_sentimen.pkl`, `model_kategori.pkl`) diverifikasi menggunakan nilai hash SHA-256 yang dicatat di `models/metadata.json`.
   * Pemuatan model gagal secara aman (*fail-closed*) jika berkas diubah atau tidak sesuai versi `scikit-learn`.
5. **Strict CSV Parser (F-01, F-02, F-03):**
   * Menolak berkas biner/korup, berkas tanpa kolom target `ulasan`, atau baris dengan jumlah kolom tidak konsisten.
   * Menggunakan *file content signature* (SHA-256) untuk mereset hasil batch secara otomatis saat berkas baru diunggah.
6. **XSS & HTML Injection Prevention (SEC-07):**
   * 100% menggunakan komponen bawaan Streamlit (`st.container(border=True)`, `st.metric`, `st.dataframe`) tanpa `unsafe_allow_html=True`.
7. **Rate Limiting & Structured Logging (SEC-06):**
   * Mekanisme *throttle* berbasis waktu untuk mencegah spamming inferensi.
   * Logging terstruktur berformat JSON tanpa merekam teks ulasan atau identitas pengguna.

---

## 🚀 3. Panduan Menjalankan Proyek

### A. Prasyarat Sistem
* **Sistem Operasi:** Windows 10/11, macOS, atau Linux
* **Python:** Versi 3.10 atau lebih baru (direkomendasikan 3.12+)
* **Package Manager:** `pip`

---

### B. Langkah Instalasi

1. **Clone Repositori & Masuk ke Folder Proyek:**
   ```bash
   git clone https://github.com/Louis-yoga/klasifikasi-ulasan-kampus-nlp.git
   cd klasifikasi-ulasan-kampus-nlp
   ```

2. **(Direkomendasikan) Buat Virtual Environment:**
   ```bash
   # Windows PowerShell
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Pasang Dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

---

### C. Melatih Model & Verifikasi Artefak

Jalankan skrip pelatihan terpisah untuk memvalidasi dataset, melakukan pembagian *stratified split 80/20*, menghitung validasi silang berulang, dan menandatangani artefak model:
```bash
python train.py
```
*Output: Memperbarui berkas model di folder `models/` beserta `models/metadata.json` bersertifikat hash SHA-256.*

---

### D. Menjalankan Suite Pengujian & Benchmark

* **Menjalankan Pengujian Otomatis (Pytest):**
  ```bash
  python -m pytest -v tests/
  ```
  *(Memvalidasi 26 pengujian unit: keamanan ReDoS, PII masking, CSV injection, integritas model, dan parser CSV).*

* **Menjalankan Evaluasi Dataset Eksternal (MOD-01):**
  ```bash
  python scripts/evaluate.py
  ```
  *(Mengevaluasi generalisasi model terhadap 100 ulasan held-out independen pada `data/external_test.csv`).*

* **Menjalankan Benchmark Performa (PERF-01 s.d. PERF-05):**
  ```bash
  python scripts/bench.py
  ```
  *(Memvalidasi latensi inferensi tunggal < 50ms, inferensi batch 100 ulasan < 500ms, dan pemuatan model < 500ms).*

* **Memeriksa Linter Kode (Ruff):**
  ```bash
  python -m ruff check .
  ```

---

### E. Menjalankan Aplikasi Web

Jalankan server aplikasi Streamlit:
```bash
python -m streamlit run app.py
```
Akses aplikasi melalui peramban di: **[http://localhost:8501](http://localhost:8501)**

---

## 💻 4. Panduan Penggunaan Antarmuka (3 Tab)

### Tab 1 — 🔍 Uji Ulasan Langsung
* **Mode Teks Tunggal:**
  1. Masukkan ulasan kampus pada area teks.
  2. Klik tombol **"🔍 Analisis Ulasan"**.
  3. Sistem menyajikan kartu prediksi sentimen dan kategori layanan beserta skor keyakinan (*confidence level*). Jika skor keyakinan di bawah 60%, sistem memunculkan peringatan *Low Confidence* yang menyarankan ulasan ditinjau manual oleh staf kampus.
* **Mode Upload CSV Batch:**
  1. Klik sub-mode **"📂 Upload CSV Batch"**.
  2. Anda dapat mengklik **"🚀 Gunakan Data Demo (100 Ulasan Independen)"** untuk demonstrasi langsung menggunakan `data/demo_unseen.csv`, atau mengunggah berkas CSV mandiri.
  3. Klik **"🔍 Proses Klasifikasi Batch"**.
  4. Sistem menyajikan tabel hasil prediksi (dengan teks ulasan yang telah disamarkan PII), visualisasi grafik distribusi sentimen dan unit layanan, serta tombol unduh CSV yang aman dari formula injection.

---

### Tab 2 — 🔬 Transparansi Hitungan & Explainability (XAI)
Membedah langkah demi langkah pemrosesan teks dari awal hingga keputusan akhir:
1. **Teks Masukan Asli:** Ditampilkan apa adanya tanpa interpretasi HTML.
2. **Penyamaran PII:** Hasil masking otomatis data sensitif (NIM, nomor HP, email, nama).
3. **Pembersihan & Negasi:** Teks setelah *case folding*, penghapusan karakter khusus, serta penandaan negasi (frasa seperti *"tidak ramah"* ditandai menjadi token `NOT_ramah`).
4. **Token Aktif:** Token kata yang lolos seleksi kosakata.
5. **Kata / Frasa Pemicu (Log-Odds):** Menampilkan Top-5 kata yang paling mendorong keputusan model terhadap kelas terpilih dibandingkan kelas lainnya tanpa *fallback* ke kata global yang tidak relevan.

---

### Tab 3 — 📊 Evaluasi Model & Audit Responsible AI
Menampilkan metrik evaluasi ilmiah yang bersumber langsung dari `models/metadata.json`:
* **Pembagian Dataset:** 80 ulasan data latih dan 20 ulasan data uji (*holdout*) dari `data/dataset.csv`.
* **Validasi Silang Berulang (Repeated Stratified 5-Fold, 25 Iterasi):**
  * **Sentimen:** Akurasi rata-rata $80.0\% \pm 9.4\%$, Macro F1 rata-rata $79.8\% \pm 9.5\%$.
  * **Kategori Layanan:** Akurasi rata-rata $75.6\% \pm 10.2\%$, Macro F1 rata-rata $75.6\% \pm 10.2\%$.
* **Holdout Test Set ($n=20$):**
  * **Sentimen:** Akurasi $70.0\%$, Macro F1 $69.4\%$.
  * **Kategori:** Akurasi $75.0\%$, Macro F1 $74.2\%$.
  * Disertai matriks kebingungan (*Confusion Matrix*) $3 \times 3$ untuk sentimen dan $4 \times 4$ untuk kategori.
* **Kepatuhan 5 Pilar Responsible AI:** Matriks status privasi, keadilan (*fairness* distribusi data), transparansi, dan pengawasan manusia (*human-in-the-loop*).

---

## 📂 5. Struktur Direktori Proyek

```
Mini Project/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI (pytest, ruff, bandit)
├── .streamlit/
│   └── config.toml              # Pengerasan keamanan Streamlit (XSRF, upload limit 1MB)
├── assets/
│   └── logo.png                 # Logo lokal antarmuka
├── data/
│   ├── dataset.csv              # Dataset pelatihan utama (100 baris seimbang)
│   ├── demo_unseen.csv          # 100 ulasan tanpa label khusus demonstrasi batch
│   └── external_test.csv        # 100 ulasan independen untuk uji generalisasi eksternal
├── docs/
│   ├── Audit Teknis.docx        # Laporan audit teknis lengkap
│   └── Klasifikasi Ulasan Kampus.docx # Dokumen PRD sistem
├── models/
│   ├── metadata.json            # Metadata pelatihan, parameter, metrik, & hash SHA-256
│   ├── model_kategori.pkl       # Model MNB klasifikasi kategori layanan
│   ├── model_sentimen.pkl       # Model MNB klasifikasi sentimen
│   └── vectorizer.pkl           # Sublinear TF-IDF vectorizer (Unigram + Bigram)
├── scripts/
│   ├── bench.py                 # Skrip pengujian benchmark performa dan latensi
│   ├── check_pii_fp.py          # Skrip verifikasi false positive PII masking
│   ├── cleanup_data.py          # Skrip sanitasi dan pembersihan data redundan
│   └── evaluate.py              # Skrip evaluasi independen dataset eksternal
├── src/
│   ├── __init__.py
│   ├── artifacts.py             # Pengelolaan artefak atomik & verifikasi SHA-256
│   ├── classifier.py            # Inferensi tunggal & batch prediksi MNB
│   ├── config.py                # Batasan batas beban sistem (MAX_CHARS, MAX_ROWS, MAX_BYTES)
│   ├── evaluator.py             # Evaluasi metrik makro & XAI Log-Odds
│   ├── exporters.py             # Ekspor CSV aman (anti-formula injection & UTF-8 BOM)
│   ├── feature_extractor.py     # Sublinear TF-IDF vectorizer & ekstraksi fitur teks
│   ├── guard.py                 # Throttling & rate limiting session state
│   ├── log.py                   # Structured audit logging (tanpa kebocoran PII)
│   ├── parsers.py               # Strict CSV parser
│   └── preprocessor.py          # Regex linear PII masking, cleaning, & negasi NOT_
├── tests/
│   ├── test_functional.py       # Pengujian unit fungsional dan parser
│   ├── test_models.py           # Pengujian integritas model, negasi, & XAI
│   └── test_security.py         # Pengujian ReDoS, batas input, PII, & CSV injection
├── app.py                       # Aplikasi web Streamlit (3 tab)
├── pyproject.toml               # Konfigurasi linter Ruff & pytest
├── requirements.in              # Spesifikasi dependensi aman
├── requirements.txt             # Versi dependensi terkunci (pinned)
├── train.py                     # Skrip terpisah pelatihan model & evaluasi CV
└── README.md                    # Dokumentasi teknis proyek
```

---

## 🛠️ 6. Spesifikasi Lingkungan Teknis

* **Bahasa:** Python 3.12+ (Kompatibel dengan Python 3.10+)
* **Pustaka Utama:**
  * `streamlit == 1.67.0` (Antarmuka web)
  * `scikit-learn == 1.9.1` (Multinomial Naive Bayes & TF-IDF)
  * `pandas == 3.1.1` (Manipulasi struktur data tabular)
  * `numpy == 2.5.5` (Operasi matriks dan numerik)
  * `joblib == 1.6.0` (Serialisasi artefak model)
  * `pytest == 9.2.0` (Pengujian unit otomatis)
  * `ruff == 0.17.5` (Linter dan pemformat kode)
