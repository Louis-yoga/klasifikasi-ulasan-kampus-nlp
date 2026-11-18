"""
app.py
------
Antarmuka Web Streamlit Sistem Klasifikasi Ulasan Layanan Kampus.
Mengimplementasikan pengerasan keamanan (SEC-01 s.d. SEC-07), transparansi XAI log-odds (U-01),
validasi input ketat (F-01 s.d. F-04), dan integritas evaluasi berbasis metadata (MOD-01 s.d. MOD-07).
"""

import hashlib
import os
import sys

import pandas as pd
import streamlit as st

# Pastikan modul src dapat diimpor
sys.path.insert(0, os.path.dirname(__file__))

from src.artifacts import ArtifactError, load_artifacts
from src.classifier import (
    EMOJI_KATEGORI,
    EMOJI_SENTIMEN,
    LOW_CONF_THRESHOLD,
    predict_batch,
    predict_single,
)
from src.config import MAX_CHARS, MAX_ROWS
from src.evaluator import get_top_features
from src.exporters import to_safe_csv
from src.guard import throttle
from src.parsers import CsvError, parse_csv_strict

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DEMO_PATH = os.path.join(BASE_DIR, 'data', 'demo_unseen.csv')
LOGO_PATH = os.path.join(BASE_DIR, 'assets', 'logo.png')

# ── Konfigurasi Halaman Streamlit ─────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Ulasan Kampus",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Pemuatan Artefak Model & Metadata (SEC-04, F-05) ──────────────────────────
@st.cache_resource(show_spinner="⚙️ Memverifikasi dan memuat model terenkripsi SHA-256...")
def get_model_resources():
    try:
        objs, metadata = load_artifacts(MODEL_DIR)
        return objs, metadata, None
    except ArtifactError as err:
        return None, None, str(err)
    except Exception as err:  # noqa: BLE001
        return None, None, f"Kesalahan saat memuat model: {err}"


objs, metadata, load_err = get_model_resources()

if load_err or objs is None:
    st.error(
        f"⛔ **Gagal Memuat Model:** {load_err}\n\n"
        "Silakan jalankan skrip pelatihan terlebih dahulu melalui terminal:\n"
        "```bash\npython train.py\n```"
    )
    st.stop()

vectorizer = objs['vectorizer']
model_sentimen = objs['sentimen']
model_kategori = objs['kategori']


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR INFORMASI & STATUS SISTEM
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=80)
    else:
        st.title("🎓")

    st.title("Klasifikasi Ulasan")
    st.caption("Campus Review NLP Analytics")

    st.divider()
    st.subheader("📊 Info Dataset & Model")

    ds_info = metadata.get('dataset', {})
    col_sb1, col_sb2 = st.columns(2)
    col_sb1.metric("Data Latih", ds_info.get('train_rows', 80))
    col_sb2.metric("Data Uji", ds_info.get('test_rows', 20))
    col_sb1.metric("Total Data", ds_info.get('total_rows', 100))
    col_sb2.metric("Vocabulary", len(vectorizer.vocabulary_))

    st.divider()
    st.subheader("🤖 Integritas Model")
    st.success(f"✅ SHA-256 Verified (Scikit-Learn v{metadata.get('sklearn_version', '1.9')})")

    cv_s = metadata['metrics']['sentimen']['cv_repeated_kfold']
    cv_k = metadata['metrics']['kategori']['cv_repeated_kfold']
    st.caption(
        f"**CV F1 Sentimen (5x5):** {cv_s['f1_macro_mean']:.3f} (±{cv_s['f1_macro_std']:.3f})\n\n"
        f"**CV F1 Kategori (5x5):** {cv_k['f1_macro_mean']:.3f} (±{cv_k['f1_macro_std']:.3f})"
    )

    st.divider()
    st.caption(
        "**Metode:** Multinomial Naive Bayes\n\n"
        "**Fitur:** Sublinear TF-IDF (Unigram + Bigram)\n\n"
        f"**Low Confidence Threshold:** < {int(LOW_CONF_THRESHOLD * 100)}%"
    )


# ══════════════════════════════════════════════════════════════════════════════
# HEADER UTAMA
# ══════════════════════════════════════════════════════════════════════════════
st.title("🎓 Sistem Klasifikasi Ulasan Layanan Kampus")
st.markdown(
    "Klasifikasi otomatis **sentimen** (3 kelas) dan **kategori layanan** (4 unit) "
    "menggunakan model NLP **Multinomial Naive Bayes** yang terlindungi dengan verifikasi integritas."
)

tab1, tab2, tab3 = st.tabs([
    "🔍 Uji Ulasan Langsung",
    "🔬 Transparansi Hitungan",
    "📊 Evaluasi & Responsible AI",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Uji Ulasan Langsung
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🔍 Uji Ulasan Langsung")

    input_mode = st.radio(
        "Mode Input:",
        ["📝 Teks Tunggal", "📂 Upload CSV Batch"],
        horizontal=True,
    )
    st.divider()

    # ── Sub-mode A: Teks Tunggal ─────────────────────────────────────────────
    if input_mode == "📝 Teks Tunggal":
        teks_input = st.text_area(
            "Masukkan ulasan layanan kampus:",
            placeholder="Contoh: WiFi di ruang perpustakaan sangat cepat dan staf melayani dengan ramah...",
            height=120,
            max_chars=MAX_CHARS,
            help=f"Maksimal {MAX_CHARS} karakter. PII (NIM, Telepon, Email, Nama) akan disamarkan secara otomatis."
        )

        if st.button("🔍 Analisis Ulasan", type="primary", width="stretch"):
            # Rate limiting / Throttling (SEC-06)
            if not throttle(st.session_state, min_interval=0.2):
                st.warning("⚠️ Permintaan terlalu cepat. Mohon tunggu sesaat.")
                st.stop()

            # Validasi panjang kata mentah (U-02)
            kata_mentah = teks_input.strip().split()
            if len(kata_mentah) < 3:
                st.error("⛔ Ulasan terlalu pendek. Masukkan minimal 3 kata agar model dapat menganalisis konteks.")
            else:
                result = predict_single(
                    teks_input, vectorizer, model_sentimen, model_kategori
                )

                st.session_state['last_result'] = result
                st.session_state['last_teks'] = result['prep_result']['teks_masked_pii']

                st.subheader("📋 Hasil Prediksi")
                c1, c2 = st.columns(2)

                with c1, st.container(border=True):
                    st.caption(f"{EMOJI_SENTIMEN.get(result['sentimen'], '🏷️')} SENTIMEN")
                    st.metric(label="Prediksi Sentimen", value=result['sentimen'])
                    st.caption(f"Tingkat Keyakinan: **{result['conf_sentimen']*100:.1f}%**")

                with c2, st.container(border=True):
                    st.caption(f"{EMOJI_KATEGORI.get(result['kategori'], '🏢')} KATEGORI LAYANAN")
                    st.metric(label="Prediksi Kategori", value=result['kategori'])
                    st.caption(f"Tingkat Keyakinan: **{result['conf_kategori']*100:.1f}%**")

                # Distribusi Probabilitas
                st.markdown("**Distribusi Probabilitas Kelas:**")
                col_s, col_k = st.columns(2)

                with col_s:
                    st.caption("Probabilitas Sentimen")
                    for lbl, prob in sorted(result['proba_sentimen'].items(), key=lambda x: x[1], reverse=True):
                        st.progress(prob, text=f"{lbl}: {prob*100:.1f}%")

                with col_k:
                    st.caption("Probabilitas Kategori Layanan")
                    for lbl, prob in sorted(result['proba_kategori'].items(), key=lambda x: x[1], reverse=True):
                        st.progress(prob, text=f"{lbl}: {prob*100:.1f}%")

                # Peringatan Low Confidence
                if result['low_confidence']:
                    st.warning(result['low_conf_message'])

                st.info("💡 Buka tab **🔬 Transparansi Hitungan** untuk melihat langkah preprocessing, PII masking, dan kata pemicu log-odds.")

    # ── Sub-mode B: Upload CSV Batch ─────────────────────────────────────────
    else:
        st.markdown(
            f"**Ketentuan Batch:** Format CSV dengan kolom `ulasan` (opsional `id`). "
            f"Maksimal **{MAX_ROWS} baris** dan ukuran berkas **1 MB**. "
            "Pemisah koma (`,`) dan titik-koma (`;`) didukung secara ketat."
        )

        col_dl1, col_dl2 = st.columns(2)
        template_csv = 'id,ulasan\n1,"Staf administrasi loket sangat ramah dan proses berkas cepat."\n2,"WiFi di kelas gedung B sering lambat dan terputus."\n3,"Buku referensi di perpustakaan sangat lengkap dan ruangan bersih."\n'
        with col_dl1:
            st.download_button(
                "📥 Unduh Template CSV",
                data=template_csv.encode('utf-8'),
                file_name="template_ulasan.csv",
                mime="text/csv",
                help="Gunakan template ini sebagai panduan format data masukan."
            )

        with col_dl2:
            if os.path.exists(DEMO_PATH):
                with open(DEMO_PATH, 'rb') as f_demo:
                    demo_bytes = f_demo.read()
                st.download_button(
                    "📂 Unduh Data Demo (100 Ulasan Unseen)",
                    data=demo_bytes,
                    file_name="demo_unseen.csv",
                    mime="text/csv",
                    help="Unduh file demo 100 ulasan independen yang belum pernah dilihat oleh model saat pelatihan."
                )

        if os.path.exists(DEMO_PATH) and st.button("🚀 Gunakan Data Demo (100 Ulasan Independen)", help="Muat data demo eksternal tanpa perlu upload manual"):
            with open(DEMO_PATH, 'rb') as f_demo:
                df_demo_parsed = parse_csv_strict(f_demo.read())
                st.session_state['active_batch_df'] = df_demo_parsed
                st.session_state['batch_source'] = "demo_unseen.csv (External Unseen Demo)"
                st.session_state['batch_sig'] = "demo_unseen_signature"
                st.session_state['batch_result'] = None

        uploaded_file = st.file_uploader(
            "Atau unggah berkas CSV sendiri:",
            type=['csv'],
            key="batch_uploader"
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            sig = hashlib.sha256(file_bytes).hexdigest()
            # F-01 & F-03: Hanya perbarui state dan reset hasil jika berkas baru
            if st.session_state.get('batch_sig') != sig:
                try:
                    df_parsed = parse_csv_strict(file_bytes)
                    st.session_state['active_batch_df'] = df_parsed
                    st.session_state['batch_source'] = uploaded_file.name
                    st.session_state['batch_sig'] = sig
                    st.session_state['batch_result'] = None
                except CsvError as e:
                    st.error(f"⛔ {e}")
                    st.session_state['active_batch_df'] = None
                    st.session_state['batch_result'] = None

        # Tampilkan data aktif dan proses batch
        if 'active_batch_df' in st.session_state and st.session_state['active_batch_df'] is not None:
            df_active = st.session_state['active_batch_df']
            source_label = st.session_state.get('batch_source', 'Data Batch')

            st.markdown(f"**Pratinjau Data: {source_label} ({len(df_active)} ulasan)**")
            st.dataframe(df_active.head(5), width="stretch")

            if st.button("🔍 Proses Klasifikasi Batch", type="primary", width="stretch"):
                with st.spinner("Memproses seluruh ulasan (termasuk penyamaran PII)..."):
                    df_hasil, stats = predict_batch(
                        df_active, vectorizer, model_sentimen, model_kategori
                    )
                    st.session_state['batch_result'] = df_hasil
                    st.session_state['batch_stats'] = stats

            if 'batch_result' in st.session_state and st.session_state['batch_result'] is not None:
                df_hasil = st.session_state['batch_result']
                st.success(f"✅ {len(df_hasil)} ulasan berhasil diklasifikasikan dengan aman.")
                st.dataframe(df_hasil, width="stretch")

                # Visualisasi distribusi dengan grafik bawaan Streamlit (PERF-03)
                st.subheader("📊 Distribusi Hasil Prediksi")
                c_dist1, c_dist2 = st.columns(2)

                with c_dist1:
                    st.caption("Distribusi Sentimen")
                    s_counts = df_hasil['sentimen'].value_counts()
                    st.bar_chart(s_counts)

                with c_dist2:
                    st.caption("Distribusi Kategori Layanan")
                    k_counts = df_hasil['kategori'].value_counts()
                    st.bar_chart(k_counts)

                # Ekspor CSV aman dari formula injection dengan UTF-8 BOM (SEC-03)
                safe_csv_bytes = to_safe_csv(df_hasil, as_bytes=True)
                st.download_button(
                    "⬇️ Unduh Hasil Klasifikasi (CSV Aman & UTF-8 BOM)",
                    data=safe_csv_bytes,
                    file_name="hasil_klasifikasi_ulasan_kampus.csv",
                    mime="text/csv",
                    help="Dilengkapi UTF-8 BOM dan netralisasi formula injection (=, +, -, @) untuk keamanan pembacaan di Excel."
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Transparansi Hitungan (XAI)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔬 Transparansi Hitungan & Explainability")
    st.caption("Inspeksi tahapan pemrosesan teks, penyamaran PII, dan kontribusi fitur berbasis Log-Odds (U-01).")

    if 'last_result' not in st.session_state:
        st.info("💡 Silakan lakukan analisis ulasan pada **Tab 🔍 Uji Ulasan Langsung** terlebih dahulu.")
    else:
        res = st.session_state['last_result']
        prep = res['prep_result']

        # 1. Teks Asli (SEC-07: gunakan st.text atau st.code, jangan st.markdown f-string)
        st.subheader("1️⃣ Teks Masukan Asli")
        st.text(prep['teks_asli'])

        # 2. Penyamaran PII (SEC-02)
        st.subheader("2️⃣ Hasil Penyamaran PII")
        if prep['pii_ditemukan']:
            st.warning("⚠️ Data pribadi (PII) terdeteksi dan disamarkan:")
        else:
            st.success("✅ Tidak ditemukan data pribadi (PII).")
        st.code(prep['teks_masked_pii'], language=None)

        # 3. Pembersihan & Stopword
        st.subheader("3️⃣ Pembersihan Teks & Penandaan Negasi")
        col_clean1, col_clean2 = st.columns(2)
        with col_clean1:
            st.caption("Lowercase & Pembersihan Karakter Khusus:")
            st.code(prep['teks_bersih'], language=None)
        with col_clean2:
            st.caption("Stopword Filtering & Penandaan Negasi (NOT_):")
            st.code(prep['teks_no_stop'], language=None)

        # 4. Token Aktif
        st.subheader("4️⃣ Token Aktif yang Dianalisis")
        if prep['tokens']:
            st.caption(f"Ditemukan {prep['jumlah_token']} token:")
            st.code(", ".join(prep['tokens']), language=None)
        else:
            st.warning("Tidak ada token bermakna yang tersisa setelah filtering.")

        st.divider()

        # 5. Top Kata Pemicu (Log-Odds vs Kelas Lain) (U-01)
        st.subheader("5️⃣ Kata / Frasa Pemicu Keputusan (Log-Odds)")
        st.caption(
            "Skor log-odds mengukur seberapa kuat suatu kata mendukung kelas target "
            "dibandingkan rata-rata kelas lainnya. Nilai positif menunjukkan preferensi kuat."
        )

        def display_top_features(model, class_label, tokens):
            top_feats = get_top_features(model, vectorizer, class_label, tokens, top_n=5)
            if not top_feats:
                st.info("ℹ️ Tidak ada kata dari input yang dikenali dalam kosakata model.")
                return
            df_feat = pd.DataFrame(top_feats, columns=['Fitur / Kata', 'Log-Odds vs Kelas Lain'])
            st.dataframe(df_feat, width="stretch", hide_index=True)

        col_xai1, col_xai2 = st.columns(2)
        with col_xai1:
            pred_s = res['sentimen']
            st.markdown(f"**Pemicu Sentimen: {pred_s}**")
            if pred_s != "Tidak dapat diklasifikasikan":
                display_top_features(model_sentimen, pred_s, prep['tokens'])
            else:
                st.caption("Tidak dapat menganalisis pemicu pada ulasan tanpa klasifikasi.")

        with col_xai2:
            pred_k = res['kategori']
            st.markdown(f"**Pemicu Kategori: {pred_k}**")
            if pred_k != "Tidak dapat diklasifikasikan":
                display_top_features(model_kategori, pred_k, prep['tokens'])
            else:
                st.caption("Tidak dapat menganalisis pemicu pada ulasan tanpa klasifikasi.")

        st.divider()
        st.info(
            "ℹ️ **Catatan Explainability:** Model Naive Bayes mengasumsikan independensi kondisional. "
            "Skor log-odds di atas bersifat indikatif dari kontribusi kosakata, bukan penalaran semantik penuh."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Evaluasi & Responsible AI Audit
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📊 Evaluasi Model & Audit Responsible AI")

    # Pembagian Data
    st.subheader("📂 Pembagian Dataset")
    c_data1, c_data2, c_data3 = st.columns(3)
    c_data1.metric("Total Dataset (dataset.csv)", f"{ds_info.get('total_rows', 100)} ulasan")
    c_data2.metric("Data Latih (80%)", f"{ds_info.get('train_rows', 80)} ulasan")
    c_data3.metric("Data Uji Holdout (20%)", f"{ds_info.get('test_rows', 20)} ulasan")
    st.caption(
        "Model dilatih menggunakan Stratified Split 80/20 (compound key kategori × sentimen) dengan `random_state=42`. "
        "Evaluasi holdout dilakukan murni pada 20 sampel uji yang tidak pernah dilihat saat pembentukan kosakata."
    )

    st.divider()

    # Fungsi render evaluasi holdout & CV
    def render_model_evaluation(model_key: str, title: str):
        st.subheader(title)

        m_data = metadata['metrics'][model_key]
        hold = m_data['holdout']
        cv = m_data['cv_repeated_kfold']

        # Metrik Holdout
        st.markdown("**Performa Holdout Test Set (n=20):**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{hold['accuracy']*100:.1f}%")
        m2.metric("Precision (Macro)", f"{hold['precision_macro']*100:.1f}%")
        m3.metric("Recall (Macro)", f"{hold['recall_macro']*100:.1f}%")
        m4.metric("F1-Score (Macro)", f"{hold['f1_macro']*100:.1f}%")

        # Metrik CV Berulang (5x5)
        st.caption(
            f"**Validasi Silang Berulang (Repeated Stratified 5-Fold, 25 iterasi):** "
            f"Akurasi Rata-rata = **{cv['accuracy_mean']*100:.1f}% (±{cv['accuracy_std']*100:.1f}%)**, "
            f"Macro F1 Rata-rata = **{cv['f1_macro_mean']*100:.1f}% (±{cv['f1_macro_std']*100:.1f}%)**"
        )

        st.caption(
            "⚠️ *Catatan Evaluasi Ilmiah:* Pada sampel uji kecil ($n=20$), estimasi titik memiliki interval "
            "ketidakpastian lebar. Nilai validasi silang berulang (5x5) memberikan gambaran generalisasi yang lebih robust."
        )

        # Confusion Matrix & Per-Class Table
        col_cm, col_pc = st.columns([1, 1.2])

        with col_cm:
            st.markdown("**Confusion Matrix (Holdout Test Set):**")
            class_labels = metadata['classes'][model_key]
            cm_df = pd.DataFrame(
                hold['conf_matrix'],
                index=[f"Aktual {c}" for c in class_labels],
                columns=[f"Pred {c}" for c in class_labels]
            )
            st.dataframe(cm_df, width="stretch")

        with col_pc:
            st.markdown("**Metrik Per Kelas (Holdout Test Set):**")
            pc_df = pd.DataFrame(hold['per_class'])
            if 'Precision' in pc_df.columns:
                pc_df['Precision'] = pc_df['Precision'].map(lambda x: f"{x*100:.1f}%")
                pc_df['Recall'] = pc_df['Recall'].map(lambda x: f"{x*100:.1f}%")
                pc_df['F1-Score'] = pc_df['F1-Score'].map(lambda x: f"{x*100:.1f}%")
            st.dataframe(pc_df, width="stretch", hide_index=True)

    render_model_evaluation('sentimen', "🏷️ Model Sentimen (3 Kelas: Positif, Negatif, Netral)")
    st.divider()
    render_model_evaluation('kategori', "🏢 Model Kategori Layanan (4 Unit: Akademik, Fasilitas, Kemahasiswaan, Administrasi)")

    st.divider()

    # Responsible AI Audit
    st.subheader("🛡️ Responsible AI Framework Compliance")
    rai_data = [
        {"Pilar": "🔒 Privacy & Security", "Status": "Terpenuhi", "Implementasi": "Sanitasi regex PII linear (anti-ReDoS), masking otomatis NIM/Telepon/Email/Nama, netralisasi formula CSV."},
        {"Pilar": "⚖️ Fairness & Balance", "Status": "Seimbang", "Implementasi": "Dataset seimbang: 25 ulasan per kategori (4 unit) dan proporsi sentimen seimbang (34:33:33)."},
        {"Pilar": "🔍 Explainability", "Status": "Aktif", "Implementasi": "Transparansi log-odds antar kelas tanpa fallback kata global."},
        {"Pilar": "📢 Transparency", "Status": "Aktif", "Implementasi": "Distribusi probabilitas terbuka, pelaporan metrik CV dengan simpangan baku (tanpa overklaim)."},
        {"Pilar": "👤 Human Oversight", "Status": "Aktif", "Implementasi": "Peringatan ambang keyakinan rendah (< 60%) untuk peninjauan manual staf kampus."}
    ]
    st.dataframe(pd.DataFrame(rai_data), width="stretch", hide_index=True)

    st.markdown("")
    st.warning(
        "⚠️ **Keterbatasan Model yang Diketahui (Known Limitations):**\n\n"
        "- **Sarkasme & Bahasa Slang:** Pendekatan BoW menghitung bobot kata secara terpisah, sehingga kalimat sarkasme berisiko salah diklasifikasikan.\n"
        "- **Kata Asing / OOV:** Kosakata yang belum pernah ada pada data latih tidak memberi kontribusi fitur.\n"
        "- **Decision Support Only:** Sistem ini berfungsi sebagai pendukung keputusan (decision support), bukan penentu kebijakan otomatis tanpa tinjauan manusia."
    )

    st.divider()
    st.caption("Sistem Klasifikasi Ulasan Kampus v2.0 (Audit-Hardened) | NLP Campus Analytics")
