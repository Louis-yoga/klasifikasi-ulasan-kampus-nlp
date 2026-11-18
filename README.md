# 🎓 Sistem Klasifikasi Ulasan Layanan Kampus Berbasis NLP & Multinomial Naive Bayes

> **Mini Project Mata Kuliah Artificial Intelligence — Pertemuan 15**  
> **Program Studi S1 Teknik Informatika, Fakultas Sains dan Teknologi**  
> **UIN Maulana Malik Ibrahim Malang**  
> **Pengembang:** Louis Ainur Rofiq Prayoga (NIM: 240605110034 / Kelas H)

---

## 📌 1. Ringkasan Eksekutif & Latar Belakang

Sistem ini dirancang untuk mengotomatiskan analisis ulasan mahasiswa terkait layanan kampus dari teks bebas (*unstructured text*) menjadi rekapitulasi terstruktur yang dapat diaudit dan dipercaya. Sistem bekerja secara simultan menghasilkan dua dimensi prediksi:

1. **Sentimen (3 Kelas):** `Positif`, `Negatif`, dan `Netral`
2. **Kategori Layanan (4 Kelas):** `Administrasi`, `Akademik`, `Fasilitas`, dan `Kemahasiswaan`

Dibangun dengan prinsip filosofis **Augmented Intelligence** (mendukung pengambilan keputusan manusia, bukan menggantikannya) dan memenuhi **5 Pilar Trustworthy AI** (*Explainability, Fairness, Robustness, Transparency, dan Privacy*).

---

## 🚀 2. Panduan Langkah-Langkah Menjalankan Proyek

### A. Prasyarat Sistem
* **Sistem Operasi:** Windows, macOS, atau Linux
* **Python:** Versi 3.9 atau lebih baru
* **Package Manager:** `pip`

---

### B. Langkah Instalasi & Eksekusi

#### Langkah 1: Buka Terminal / PowerShell
Arahkan direktori terminal ke folder proyek:
```bash
cd "d:/KULIAH/SEMESTER 5/ARTIFICIAL INTELEGENT/Mini Project"
```

#### Langkah 2: (Opsional) Buat & Aktifkan Virtual Environment
```bash
# Membuat virtual environment
python -m venv .venv

# Aktivasi di Windows PowerShell:
.venv\Scripts\Activate.ps1

# Aktivasi di Linux/macOS:
source .venv/bin/activate
```

#### Langkah 3: Pasang Dependensi
Pasang seluruh pustaka yang dibutuhkan menggunakan `requirements.txt`:
```bash
pip install -r requirements.txt
```

#### Langkah 4: Jalankan Aplikasi Web Streamlit
Jalankan server aplikasi lokal:
```bash
python -m streamlit run app.py
```
*(Atau gunakan perintah: `streamlit run app.py`)*

#### Langkah 5: Buka Aplikasi di Browser
Setelah server aktif, buka peramban (*web browser*) Anda pada alamat:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 💻 3. Panduan Penggunaan Antarmuka (3 Panel Interaktif)

Aplikasi terbagi menjadi **3 Tab Utama** sesuai mandat PRD:

### Tab 1 — 🔍 Uji Ulasan Langsung
* **Mode Teks Tunggal:**
  1. Masukkan kalimat ulasan bebas pada kotak input (contoh: *"Proses pengurusan surat aktif kuliah sangat cepat dan staf loket ramah"*).
  2. Klik tombol **"🔍 Analisis Ulasan"**.
  3. Sistem menampilkan kartu sentimen, kartu kategori unit layanan, persentase probabilitas keyakinan (*confidence score*), serta *flag* **Low Confidence (< 60%)** jika terdeteksi ambiguitas.
* **Mode Upload CSV Batch (Demonstrasi Staf):**
  1. Klik sub-mode **"📂 Upload CSV Batch"**.
  2. Untuk pengujian instan, klik tombol **"🚀 Gunakan File Demo Book2.csv (100 Ulasan)"** atau unggah file CSV mandiri.
  3. Klik tombol **"🔍 Proses Semua Ulasan"**.
  4. Sistem menyajikan tabel prediksi massal, diagram pai (*pie chart*) distribusi sentimen, grafik batang kategori layanan, serta tombol **"⬇️ Download Hasil CSV"** ([hasil_klasifikasi.csv](file:///d:/KULIAH/SEMESTER%205/ARTIFICIAL%20INTELEGENT/Mini%20Project/data/hasil_klasifikasi.csv)).

---

### Tab 2 — 🔬 Transparansi Hitungan (*Explainable AI*)
Setelah melakukan prediksi ulasan di Tab 1, buka tab ini untuk membedah proses penalaran model:
1. **Teks Asli:** Kalimat masukan pengguna sebelum diubah.
2. **Anonimisasi PII:** Status deteksi data pribadi (NIM, email, nomor telepon) yang otomatis disamarkan menjadi `[NIM]`, `[EMAIL]`, atau `[TELEPON]`.
3. **Pembersihan & Stopword:** Teks setelah *lowercasing*, penghapusan karakter khusus, dan pembuangan kata henti (*stopwords*).
4. **Daftar Token Aktif:** Potongan kata yang lolos filter tokenisasi.
5. **Top-5 Kata Pemicu Prediksi:** Visualisasi bobot kontribusi kata kunci (*log-likelihood*) yang mendorong keputusan model pada kelas sentimen dan unit layanan terkait.

---

### Tab 3 — 📊 Evaluasi Model & Responsible AI Audit
Menampilkan metrik performa model pada *unseen test set* (20% data uji):
1. **Matriks Evaluasi Lengkap:** *Accuracy, Macro-Precision, Macro-Recall, dan Macro F1-Score*.
2. **Confusion Matrix Interaktif:**
   * **Model Sentimen:** Heatmap $3 \times 3$ (`Positif`, `Negatif`, `Netral`).
   * **Model Kategori:** Heatmap $4 \times 4$ (`Administrasi`, `Akademik`, `Fasilitas`, `Kemahasiswaan`).
3. **Tabel Kinerja Per Kelas:** Rincian performa presisi dan penarikan untuk tiap kategori ulasan.
4. **Audit 5 Pilar Trustworthy AI & Catatan Human Oversight:** Transparansi batasan algoritma Bag-of-Words terhadap sarkasme dan kata informal baru.

---

## 🔬 4. Arsitektur Teknis & Pipeline AI

```
[ Input Teks Ulasan Mahasiswa ]
              │
              ▼
  ┌─────────────────────────────────────────────────────────┐
  │  1. Privacy Guardrail: Masking PII (Regex Engine)       │
  │     - NIM (9–12 digit)            → "[NIM]"             │
  │     - Email (RFC 5322 compliant)  → "[EMAIL]"           │
  │     - No. Telepon (08xx / +62xx)  → "[TELEPON]"         │
  └───────────────────────────┬─────────────────────────────┘
                              ▼
  ┌─────────────────────────────────────────────────────────┐
  │  2. Text Cleaning & Tokenization                        │
  │     - Case Folding: Konversi ke huruf kecil (lowercase) │
  │     - Regex Filter: Pembersihan tanda baca & simbol     │
  │     - Stopword Removal: Kamus bahasa Indonesia terarah  │
  │     - Tokenizer: Ekstraksi unigram & bigram berbobot    │
  └───────────────────────────┬─────────────────────────────┘
                              ▼
  ┌─────────────────────────────────────────────────────────┐
  │  3. Feature Extraction (Sublinear TF-IDF N-Gram)        │
  │     - Vektorisasi Unigram + Bigram (1, 2)               │
  │     - Sublinear scaling (1 + log(tf)) untuk redam noise │
  └───────────────────────────┬─────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
  ┌───────────────────────┐       ┌───────────────────────┐
  │ Model MNB: Sentimen   │       │ Model MNB: Kategori   │
  │ (3 Kelas Target)      │       │ (4 Kelas Target)      │
  │ Smoothing α = 0.2     │       │ Smoothing α = 0.2     │
  └───────────┬───────────┘       └───────────┬───────────┘
              └───────────────┬───────────────┘
                              ▼
  ┌─────────────────────────────────────────────────────────┐
  │  4. Post-Processing & Trustworthy AI Verification       │
  │     - Normalisasi Peluang Posterior (Softmax/Prob)      │
  │     - Evaluasi Ambang Batas Keyakinan (Threshold: 60%)  │
  │     - Flag "Low Confidence — Perlu Tinjauan Manusia"   │
  └─────────────────────────────────────────────────────────┘
```

---

## 📊 5. Spesifikasi Dataset & Hasil Evaluasi Model

### Pembagian Data (*Stratified 80/20*)
* **Dataset Utama:** [data/dataset.csv](file:///d:/KULIAH/SEMESTER%205/ARTIFICIAL%20INTELEGENT/Mini%20Project/data/dataset.csv) (100 baris ulasan teranotasi, seimbang 25 per kategori, 34 Positif / 33 Negatif / 33 Netral).
* **Data Latih (*Train Set*):** 80 ulasan.
* **Data Uji (*Test Set*):** 20 ulasan (*unseen ground truth*).
* **File Uji Demo Massal:** [data/Book2.csv](file:///d:/KULIAH/SEMESTER%205/ARTIFICIAL%20INTELEGENT/Mini%20Project/data/Book2.csv) (100 ulasan tanpa label khusus uji coba batch).

### Performa Model pada Data Uji (20% Unseen Set):

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
|---|:---:|:---:|:---:|:---:|
| **Sentimen (3 Kelas)** | **80.0%** | **80.2%** | **80.2%** | **79.8%** |
| **Kategori Layanan (4 Kelas)** | **70.0%** | **73.3%** | **69.6%** | **69.9%** |

*Catatan: Fluktuasi metrik pada dataset berskala 100 baris wajar terjadi karena setiap 1 sampel data uji mewakili 5% nilai akurasi.*

---

## 🛡️ 6. Matriks Kepatuhan 5 Pilar Trustworthy AI

| Pilar | Status | Bentuk Implementasi Nyata pada Sistem |
|---|:---:|---|
| 🔒 **Privacy** | ✅ Terpenuhi | Deteksi dan sensor otomatis terhadap PII (NIM, nomor telepon, alamat email) sebelum masuk tahap vektorisasi maupun logging data ulasan. |
| ⚖️ **Fairness** | ✅ Terpenuhi | Dataset pelatihan diaudit dengan distribusi seimbang 25 sampel di setiap 4 unit layanan kampus guna mencegah bias representasi. |
| 🔍 **Explainability** | ✅ Terpenuhi | Panel 2 secara transparan menyajikan dekonstruksi preprocessing teks dan visualisasi Top-5 token dengan kontribusi probabilitas tertinggi. |
| 📢 **Transparency** | ✅ Terpenuhi | Seluruh probabilitas posterior per kelas (persentase keyakinan) ditampilkan secara terbuka kepada pengguna. |
| 👤 **Human Oversight** | ✅ Terpenuhi | Mekanisme *Low Confidence Flag* (< 60%) otomatis menyarankan ulasan yang ambigu untuk ditinjau oleh staf kampus secara manual. |

---

## 📂 7. Struktur Direktori Proyek

```
Mini Project/
├── app.py                     # Antarmuka web utama Streamlit (3 tab interaktif)
├── requirements.txt           # Daftar pustaka Python yang diperlukan
├── README.md                  # Dokumentasi teknis proyek ini
├── implementation_plan.md     # Rencana arsitektur dan spesifikasi teknis awal
│
├── data/
│   ├── dataset.csv            # Dataset utama (100 baris berlabel: sentimen & kategori)
│   ├── Book12.xlsx            # Dataset referensi awal format Excel
│   ├── Book2.csv              # File ulasan tanpa label untuk demo batch upload
│   └── hasil_klasifikasi.csv  # Output hasil ekspor klasifikasi batch terakhir
│
├── src/
│   ├── __init__.py
│   ├── preprocessor.py        # Modul regex PII masking, case-folding, cleaning, stopword
│   ├── feature_extractor.py   # Modul ekstraksi fitur (TF-IDF N-gram & Vocabulary)
│   ├── classifier.py          # Modul pelatihan 2 model MNB, prediksi, & batch processor
│   └── evaluator.py           # Modul Confusion Matrix, metrik per kelas, & feature importance
│
└── models/                    # Folder persistensi artefak model biner (.pkl)
    ├── vectorizer.pkl         # Vektor representasi kosakata teks
    ├── model_sentimen.pkl     # Model MNB klasifikasi sentimen
    └── model_kategori.pkl     # Model MNB klasifikasi unit layanan
```

---

## 🛠️ 8. Tech Stack & Dependensi

* **Bahasa Pemrograman:** Python 3.9+
* **Algoritma Klasifikasi:** Multinomial Naive Bayes (`scikit-learn`)
* **Representasi Teks:** N-Gram Term Frequency–Inverse Document Frequency (TF-IDF)
* **Web Framework:** Streamlit
* **Analisis & Manipulasi Data:** Pandas, NumPy, openpyxl
* **Visualisasi Data:** Matplotlib, Seaborn
* **Model Serialization:** Joblib