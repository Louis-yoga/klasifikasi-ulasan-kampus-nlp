"""
app.py
------
Antarmuka Streamlit 3-Tab untuk Sistem Klasifikasi Ulasan Kampus.
Sesuai FR-05 PRD: Panel 1 (Prediksi), Panel 2 (Explainability), Panel 3 (Evaluasi & RAI).
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import csv

# Pastikan folder src bisa diimport
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessor import preprocess_pipeline, tokens_to_string
from src.feature_extractor import (
    build_vocabulary, preprocess_corpus,
    transform_texts, transform_single,
    save_vectorizer, load_vectorizer,
)
from src.classifier import (
    build_and_train, predict_single, predict_batch,
    save_model, load_model,
    LABEL_SENTIMEN, LABEL_KATEGORI,
    EMOJI_SENTIMEN, EMOJI_KATEGORI,
    LOW_CONF_THRESHOLD,
)
from src.evaluator import evaluate_both_models, get_top_features, get_misclassified

# ── Konfigurasi Path ──────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(__file__)
DATA_PATH       = os.path.join(BASE_DIR, 'data', 'dataset.csv')
MODEL_DIR       = os.path.join(BASE_DIR, 'models')
PATH_VEC        = os.path.join(MODEL_DIR, 'vectorizer.pkl')
PATH_MODEL_S    = os.path.join(MODEL_DIR, 'model_sentimen.pkl')
PATH_MODEL_K    = os.path.join(MODEL_DIR, 'model_kategori.pkl')

# ── Konfigurasi Halaman ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Ulasan Kampus",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Kustom ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #d0d4db;
    }
    .metric-label { font-size: 0.85em; color: #555; margin-bottom: 4px; }
    .metric-value { font-size: 1.8em; font-weight: 700; color: #1a1a2e; }
    .metric-sub   { font-size: 0.8em; color: #888; }
    .token-pill {
        display: inline-block;
        background: #e8f4fd;
        border: 1px solid #aed9f5;
        border-radius: 20px;
        padding: 3px 10px;
        margin: 3px 2px;
        font-size: 0.88em;
        color: #1a73e8;
        font-family: monospace;
    }
    .low-conf-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 10px;
    }
    .rai-ok  { color: #28a745; font-weight: 600; }
    .rai-warn{ color: #fd7e14; font-weight: 600; }
    .stTabs [data-baseweb="tab"] { font-size: 1em; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


def parse_flexible_csv(file_bytes: bytes):
    """
    Parser CSV toleran untuk menangani berbagai format ekspor (Excel/Indonesian locale,
    delimiters koma/titik-koma/tab, dan teks berisi koma tanpa tanda petik).
    """
    text = None
    for enc in ['utf-8-sig', 'utf-8', 'latin1', 'cp1252']:
        try:
            text = file_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = file_bytes.decode('utf-8', errors='ignore')

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None, "File CSV kosong."

    header_raw = lines[0]
    
    # Deteksi pemisah (delimiter)
    count_semi = header_raw.count(';')
    count_tab = header_raw.count('\t')
    count_comma = header_raw.count(',')
    if count_semi > count_comma and count_semi >= count_tab:
        sep = ';'
    elif count_tab > count_comma and count_tab > count_semi:
        sep = '\t'
    else:
        sep = ','

    # 1. Coba standard csv.reader (standar kutip terpenuhi)
    rows = []
    try:
        reader = csv.reader(io.StringIO(text), delimiter=sep)
        for r in reader:
            if r:
                rows.append(r)
    except Exception:
        rows = []

    if rows and len(rows) > 1:
        header = [c.strip().strip('"\'') for c in rows[0]]
        header_lower = [c.lower() for c in header]
        
        if any(h in ['ulasan', 'review', 'text', 'komentar'] for h in header_lower):
            u_idx = next(i for i, h in enumerate(header_lower) if h in ['ulasan', 'review', 'text', 'komentar'])
            id_idx = next((i for i, h in enumerate(header_lower) if h in ['id', 'no', 'nomor']), None)
            
            clean_rows = []
            mismatch_found = False
            for r in rows[1:]:
                if len(r) == len(header):
                    row_data = {'ulasan': r[u_idx].strip()}
                    if id_idx is not None and id_idx < len(r):
                        row_data['id'] = r[id_idx].strip()
                    clean_rows.append(row_data)
                elif len(r) > len(header) and id_idx is not None:
                    mismatch_found = True
                    # Rekonstruksi baris jika ada koma tak bertanda kutip di dalam teks ulasan
                    if id_idx == 0:
                        clean_rows.append({'id': r[0].strip(), 'ulasan': sep.join(r[1:]).strip()})
                    else:
                        clean_rows.append({'id': r[-1].strip(), 'ulasan': sep.join(r[:-1]).strip()})
                elif len(r) > 0:
                    clean_rows.append({'id': '', 'ulasan': sep.join(r).strip()})
            
            df = pd.DataFrame(clean_rows)
            note = "Format baris CSV dengan tanda koma bebas berhasil direkonstruksi otomatis." if mismatch_found else None
            return df, note

    # 2. Fallback parsing baris manual jika header cocok
    header_parts = [p.strip().strip('"\'') for p in header_raw.split(sep)]
    header_lower = [p.lower() for p in header_parts]
    clean_rows = []
    if any(h in ['ulasan', 'review', 'text', 'komentar'] for h in header_lower):
        u_idx = next(i for i, h in enumerate(header_lower) if h in ['ulasan', 'review', 'text', 'komentar'])
        id_idx = next((i for i, h in enumerate(header_lower) if h in ['id', 'no', 'nomor']), None)
        for line in lines[1:]:
            parts = line.split(sep)
            if len(parts) >= 2 and id_idx is not None:
                if id_idx == 0:
                    clean_rows.append({'id': parts[0].strip(), 'ulasan': sep.join(parts[1:]).strip().strip('"\'')})
                else:
                    clean_rows.append({'id': parts[-1].strip(), 'ulasan': sep.join(parts[:-1]).strip().strip('"\'')})
            else:
                clean_rows.append({'id': '', 'ulasan': line.strip().strip('"\'')})
        return pd.DataFrame(clean_rows), "File CSV dibaca dengan toleransi baris bebas."

    # 3. Fallback jika kolom ulasan tidak bernama eksplisit: jadikan setiap baris sebagai ulasan
    for line in lines[1:] if len(lines) > 1 else lines:
        clean_rows.append({'id': '', 'ulasan': line.strip().strip('"\'')})
    return pd.DataFrame(clean_rows), "Header ulasan tidak ditemukan; seluruh baris dibaca sebagai teks ulasan."


# ══════════════════════════════════════════════════════════════════════════════
# FUNGSI STARTUP — Muat atau Latih Model
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="⚙️ Memuat / melatih model, harap tunggu...")
def load_or_train_models():
    """
    Cek apakah model .pkl sudah ada.
    - Jika YA : muat dari file (cepat, tanpa training ulang).
    - Jika TIDAK: latih dari dataset.csv lalu simpan ke models/.
    """
    models_ready = all([
        os.path.exists(PATH_VEC),
        os.path.exists(PATH_MODEL_S),
        os.path.exists(PATH_MODEL_K),
    ])

    df = pd.read_csv(DATA_PATH)

    if models_ready:
        vectorizer     = load_vectorizer(PATH_VEC)
        model_sentimen = load_model(PATH_MODEL_S)
        model_kategori = load_model(PATH_MODEL_K)

        # Rekonstruksi test_df dengan seed yang sama
        from src.classifier import prepare_data
        _, test_df = prepare_data(df, random_state=42)
        source = "loaded"
    else:
        os.makedirs(MODEL_DIR, exist_ok=True)
        train_df, test_df, vectorizer, model_sentimen, model_kategori = build_and_train(df)

        save_vectorizer(vectorizer, PATH_VEC)
        save_model(model_sentimen, PATH_MODEL_S)
        save_model(model_kategori, PATH_MODEL_K)
        source = "trained"

    # Evaluasi pada test set
    eval_results = evaluate_both_models(
        model_sentimen, model_kategori, test_df, vectorizer
    )

    return {
        'df'            : df,
        'test_df'       : test_df,
        'vectorizer'    : vectorizer,
        'model_sentimen': model_sentimen,
        'model_kategori': model_kategori,
        'eval_results'  : eval_results,
        'source'        : source,
        'vocab_size'    : len(vectorizer.vocabulary_),
    }


# ── Muat semua resource ───────────────────────────────────────────────────────
resources = load_or_train_models()
df             = resources['df']
test_df        = resources['test_df']
vectorizer     = resources['vectorizer']
model_sentimen = resources['model_sentimen']
model_kategori = resources['model_kategori']
eval_results   = resources['eval_results']


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Logo_UIN_Maulana_Malik_Ibrahim.png/200px-Logo_UIN_Maulana_Malik_Ibrahim.png", width=80)
    st.title("🎓 Klasifikasi Ulasan Kampus")
    st.caption("NLP Mini Project — Teknik Informatika UIN Malang")

    st.divider()
    st.subheader("📊 Info Dataset")
    col1, col2 = st.columns(2)
    col1.metric("Total Data", len(df))
    col2.metric("Vocabulary", resources['vocab_size'])
    col1.metric("Training Set", f"{len(df) - len(test_df)}")
    col2.metric("Test Set", len(test_df))

    st.divider()
    st.subheader("🤖 Status Model")
    status_icon = "✅ Dimuat dari file" if resources['source'] == "loaded" else "🏋️ Baru dilatih"
    st.success(status_icon)

    acc_s = eval_results['sentimen']['accuracy']
    acc_k = eval_results['kategori']['accuracy']
    st.metric("Akurasi Sentimen", f"{acc_s*100:.1f}%",
              delta="✅ ≥ 85%" if acc_s >= 0.85 else "⚠️ < 85%")
    st.metric("Akurasi Kategori", f"{acc_k*100:.1f}%",
              delta="✅ ≥ 85%" if acc_k >= 0.85 else "⚠️ < 85%")

    st.divider()
    st.caption(
        "**Metode:** Multinomial Naive Bayes + Laplace Smoothing\n\n"
        "**Fitur:** Bag-of-Words (BoW)\n\n"
        "**Split:** Stratified 80/20\n\n"
        f"**Low Conf. Threshold:** < {int(LOW_CONF_THRESHOLD*100)}%"
    )


# ══════════════════════════════════════════════════════════════════════════════
# HEADER UTAMA
# ══════════════════════════════════════════════════════════════════════════════
st.title("🎓 Sistem Klasifikasi Ulasan Layanan Kampus")
st.markdown(
    "Klasifikasi otomatis **sentimen** dan **unit layanan** dari ulasan mahasiswa "
    "menggunakan NLP berbasis **Multinomial Naive Bayes**."
)

# ── 3 Tab Utama ───────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Uji Ulasan Langsung",
    "🔬 Transparansi Hitungan",
    "📊 Evaluasi & Responsible AI",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Uji Ulasan Langsung  [FR-05 Panel 1]
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
            placeholder="Contoh: WiFi di ruang kelas sering putus dan sangat mengganggu proses belajar...",
            height=120,
        )

        if st.button("🔍 Analisis Ulasan", type="primary", use_container_width=True):
            # ── Validasi Input ───────────────────────────────────────────────
            if not teks_input or not teks_input.strip():
                st.error("⛔ Ulasan tidak boleh kosong. Masukkan minimal 3 kata.")
            else:
                prep_check = preprocess_pipeline(teks_input)
                if prep_check['jumlah_token'] < 3:
                    st.error(
                        "⛔ Teks terlalu pendek setelah dibersihkan. "
                        f"Hanya ditemukan **{prep_check['jumlah_token']} token**. "
                        "Tambahkan deskripsi lebih lengkap."
                    )
                else:
                    # ── Prediksi ─────────────────────────────────────────────
                    result = predict_single(
                        teks_input, vectorizer, model_sentimen, model_kategori
                    )

                    # Simpan ke session_state untuk Tab 2
                    st.session_state['last_result'] = result
                    st.session_state['last_teks']   = teks_input

                    # ── Tampilkan Hasil ───────────────────────────────────────
                    st.subheader("📋 Hasil Prediksi")
                    c1, c2 = st.columns(2)

                    # Sentimen Card
                    if result['sentimen'] == 'Positif':
                        senti_color = "#d4edda"
                        senti_border = "#28a745"
                    elif result['sentimen'] == 'Negatif':
                        senti_color = "#f8d7da"
                        senti_border = "#dc3545"
                    else:  # Netral
                        senti_color = "#e2e3e5"
                        senti_border = "#6c757d"

                    with c1:
                        st.markdown(f"""
                        <div class="metric-card" style="border-left: 5px solid {senti_border}; background: {senti_color}">
                            <div class="metric-label">{EMOJI_SENTIMEN.get(result['sentimen'], 'Sentimen')}</div>
                            <div class="metric-value">{result['sentimen'].upper()}</div>
                            <div class="metric-sub">Keyakinan: <b>{result['conf_sentimen']*100:.1f}%</b></div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Kategori Card
                    with c2:
                        st.markdown(f"""
                        <div class="metric-card" style="border-left: 5px solid #1a73e8;">
                            <div class="metric-label">🏢 Kategori Layanan</div>
                            <div class="metric-value" style="font-size:1.3em">{EMOJI_KATEGORI.get(result['kategori'], result['kategori'])}</div>
                            <div class="metric-sub">Keyakinan: <b>{result['conf_kategori']*100:.1f}%</b></div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Bar probabilitas
                    st.markdown("")
                    st.markdown("**Distribusi Probabilitas:**")
                    col_s, col_k = st.columns(2)

                    with col_s:
                        st.caption("Sentimen")
                        for label, prob in sorted(result['proba_sentimen'].items(),
                                                   key=lambda x: x[1], reverse=True):
                            st.progress(prob, text=f"{label}: {prob*100:.1f}%")

                    with col_k:
                        st.caption("Kategori Layanan")
                        for label, prob in sorted(result['proba_kategori'].items(),
                                                   key=lambda x: x[1], reverse=True):
                            st.progress(prob, text=f"{label}: {prob*100:.1f}%")

                    # Low Confidence Flag
                    if result['low_confidence']:
                        st.markdown(f"""
                        <div class="low-conf-box">
                            {result['low_conf_message']}
                        </div>
                        """, unsafe_allow_html=True)

                    st.info("💡 Lihat **Tab 🔬 Transparansi Hitungan** untuk detail lengkap preprocessing dan kata pemicu.")

    # ── Sub-mode B: Upload CSV Batch ─────────────────────────────────────────
    else:
        st.markdown("**Format CSV:** Kolom wajib `ulasan`, opsional `id`. Maksimal 500 baris. Mendukung pemisah koma (`,`), titik-koma (`;`), maupun tab.")
        
        # Opsi unduh template contoh CSV & File Demo Book2.csv
        col_dl1, col_dl2 = st.columns(2)
        template_csv = 'id,ulasan\n1,"Staf loket sangat ramah dan proses administrasi cepat"\n2,"WiFi di kelas sering putus dan lambat"\n3,"Buku referensi di perpustakaan sangat lengkap"\n4,"Server KRS sering error saat pengisian semester baru"\n'
        with col_dl1:
            st.download_button(
                "📥 Unduh Template CSV Ringkas",
                data=template_csv,
                file_name="contoh_format_ulasan.csv",
                mime="text/csv",
                help="Unduh file template CSV 4 baris ini sebagai acuan format ulasan yang benar."
            )
        
        path_b2 = os.path.join(BASE_DIR, 'data', 'Book2.csv')
        with col_dl2:
            if os.path.exists(path_b2):
                with open(path_b2, 'rb') as fb2:
                    st.download_button(
                        "📂 Unduh Demo Book2.csv (100 Ulasan)",
                        data=fb2.read(),
                        file_name="Book2.csv",
                        mime="text/csv",
                        help="Unduh file Book2.csv berisi 100 ulasan tanpa label khusus untuk demo batch prediction."
                    )

        # Opsi cepat: Load Book2.csv langsung
        if os.path.exists(path_b2):
            if st.button("🚀 Gunakan File Demo Book2.csv (100 Ulasan)", help="Muat langsung 100 ulasan dari data/Book2.csv tanpa perlu upload manual"):
                with open(path_b2, 'rb') as fb2:
                    df_b2_loaded, note_b2 = parse_flexible_csv(fb2.read())
                    st.session_state['active_batch_df'] = df_b2_loaded
                    st.session_state['batch_source'] = "Book2.csv (Demo Batch Data)"
                    st.session_state['batch_result'] = None

        uploaded_file = st.file_uploader("Atau Upload file CSV sendiri:", type=['csv'], key="batch_uploader")

        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.getvalue()
                df_upload, parse_note = parse_flexible_csv(file_bytes)
                if df_upload is None or df_upload.empty:
                    st.error("⛔ File CSV kosong atau tidak dapat diuraikan.")
                elif 'ulasan' not in df_upload.columns:
                    st.error("⛔ Kolom **'ulasan'** tidak ditemukan di dalam file. Pastikan header CSV memiliki kolom bernama `ulasan`.")
                else:
                    if parse_note:
                        st.info(f"ℹ️ {parse_note}")
                    st.session_state['active_batch_df'] = df_upload
                    st.session_state['batch_source'] = uploaded_file.name
            except Exception as e:
                st.error(f"⛔ Gagal membaca file: {e}")

        # Tampilkan data batch aktif dan tombol proses
        if 'active_batch_df' in st.session_state and st.session_state['active_batch_df'] is not None:
            df_active = st.session_state['active_batch_df']
            source_name = st.session_state.get('batch_source', 'Data Upload')

            if len(df_active) > 500:
                st.warning(f"⚠️ File memiliki {len(df_active)} baris. Hanya **500 baris pertama** yang diproses.")
                df_active = df_active.head(500)

            st.markdown(f"**Pratinjau {source_name} ({len(df_active)} ulasan terdeteksi):**")
            st.dataframe(df_active.head(5), use_container_width=True)

            if st.button("🔍 Proses Semua Ulasan", type="primary", use_container_width=True):
                with st.spinner("Memproses prediksi seluruh ulasan..."):
                    df_hasil = predict_batch(
                        df_active, vectorizer, model_sentimen, model_kategori
                    )
                st.session_state['batch_result'] = df_hasil

            # Tampilkan hasil jika sudah diproses
            if 'batch_result' in st.session_state and st.session_state['batch_result'] is not None:
                df_hasil = st.session_state['batch_result']
                st.success(f"✅ {len(df_hasil)} ulasan berhasil diklasifikasikan.")
                st.dataframe(df_hasil, use_container_width=True)

                # Ringkasan distribusi
                st.subheader("📊 Ringkasan Distribusi Prediksi")
                cr1, cr2 = st.columns(2)
                with cr1:
                    fig_s, ax_s = plt.subplots(figsize=(4, 3))
                    sentimen_counts = df_hasil['sentimen'].value_counts()
                    color_palette = {'Positif': '#28a745', 'Negatif': '#dc3545', 'Netral': '#6c757d'}
                    pie_colors = [color_palette.get(c, '#007bff') for c in sentimen_counts.index]
                    ax_s.pie(sentimen_counts, labels=sentimen_counts.index,
                             autopct='%1.1f%%', colors=pie_colors)
                    ax_s.set_title("Distribusi Sentimen")
                    st.pyplot(fig_s, use_container_width=True)
                    plt.close()

                with cr2:
                    fig_k, ax_k = plt.subplots(figsize=(4, 3))
                    kat_counts = df_hasil['kategori'].value_counts()
                    ax_k.barh(kat_counts.index, kat_counts.values,
                              color='#1a73e8')
                    ax_k.set_xlabel("Jumlah")
                    ax_k.set_title("Distribusi Kategori")
                    plt.tight_layout()
                    st.pyplot(fig_k, use_container_width=True)
                    plt.close()

                # Download button
                csv_out = df_hasil.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    "⬇️ Download Hasil CSV",
                    data=csv_out,
                    file_name="hasil_klasifikasi.csv",
                    mime="text/csv",
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Transparansi Hitungan  [FR-05 Panel 2]
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔬 Transparansi Hitungan")
    st.caption("Panel ini menampilkan setiap langkah proses dari teks mentah hingga keputusan model.")

    if 'last_result' not in st.session_state:
        st.info("💡 Silakan masukkan ulasan di **Tab 🔍 Uji Ulasan Langsung** terlebih dahulu, lalu kembali ke tab ini.")
    else:
        result = st.session_state['last_result']
        prep   = result['prep_result']

        # ── 1. Teks Asli ─────────────────────────────────────────────────────
        st.subheader("1️⃣ Teks Asli")
        st.markdown(f"> *\"{prep['teks_asli']}\"*")

        # ── 2. Anonimisasi PII ───────────────────────────────────────────────
        st.subheader("2️⃣ Setelah Anonimisasi PII")
        if prep['pii_ditemukan']:
            st.warning(f"⚠️ PII terdeteksi dan disamarkan:\n\n> *\"{prep['teks_masked_pii']}\"*")
        else:
            st.success(f"✅ Tidak ada PII terdeteksi.\n\n> *\"{prep['teks_masked_pii']}\"*")

        # ── 3. Teks Bersih ───────────────────────────────────────────────────
        st.subheader("3️⃣ Setelah Cleaning & Stopword Removal")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Setelah lowercase + hapus tanda baca:**")
            st.code(prep['teks_bersih'])
        with col_b:
            st.markdown("**Setelah stopword removal:**")
            st.code(prep['teks_no_stop'])

        # ── 4. Token ─────────────────────────────────────────────────────────
        st.subheader("4️⃣ Token Aktif")
        if prep['tokens']:
            tokens_html = "".join(
                f'<span class="token-pill">{t}</span>' for t in prep['tokens']
            )
            st.markdown(
                f"{tokens_html}<br><small style='color:#888'>→ {prep['jumlah_token']} token aktif</small>",
                unsafe_allow_html=True
            )
        else:
            st.warning("Tidak ada token yang dihasilkan.")

        st.divider()

        # ── 5. Top-5 Kata Pemicu ─────────────────────────────────────────────
        st.subheader("5️⃣ Top-5 Kata Pemicu")

        def render_top_features(model, label, tokens, model_name):
            top_feats = get_top_features(model, vectorizer, label, tokens, top_n=5)
            if not top_feats:
                st.caption("Tidak ada kata dari input yang dikenali di vocabulary model.")
                return

            df_feat = pd.DataFrame(top_feats, columns=['Kata', 'Log-Probabilitas'])
            df_feat['Log-Prob (relatif)'] = (
                df_feat['Log-Probabilitas'] - df_feat['Log-Probabilitas'].min()
            )
            max_val = df_feat['Log-Prob (relatif)'].max()
            if max_val == 0:
                max_val = 1

            for _, row in df_feat.iterrows():
                bar_w = row['Log-Prob (relatif)'] / max_val
                st.markdown(
                    f"`{row['Kata']}` &nbsp; "
                    f"<span style='font-size:0.8em;color:#888'>{row['Log-Probabilitas']:.3f}</span>  \n"
                    f"<div style='background:#1a73e8;width:{bar_w*100:.0f}%;height:8px;border-radius:4px;'></div>",
                    unsafe_allow_html=True
                )

        col5a, col5b = st.columns(2)
        with col5a:
            pred_s = result['sentimen']
            st.markdown(f"**Sentimen — {EMOJI_SENTIMEN.get(pred_s, pred_s)}**")
            render_top_features(model_sentimen, pred_s, prep['tokens'], "Sentimen")

        with col5b:
            pred_k = result['kategori']
            st.markdown(f"**Kategori — {EMOJI_KATEGORI.get(pred_k, pred_k)}**")
            render_top_features(model_kategori, pred_k, prep['tokens'], "Kategori")

        st.divider()
        st.info(
            "ℹ️ **Disclaimer:** Model Bag-of-Words (BoW) tidak memahami urutan kata atau konteks kalimat. "
            "Kata pemicu di atas hanyalah kontribusi individual berdasarkan log-probabilitas — "
            "bersifat **indikatif**, bukan kausal. "
            "Kalimat sarkastis atau negasi kompleks mungkin salah diklasifikasikan."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Evaluasi & Responsible AI  [FR-05 Panel 3]
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📊 Evaluasi Model & Responsible AI Audit")

    # ── Info Split Data ───────────────────────────────────────────────────────
    st.subheader("📂 Pembagian Data")
    cds1, cds2, cds3 = st.columns(3)
    cds1.metric("Total Dataset", f"{len(df)} ulasan")
    cds2.metric("Training Set (80%)", f"{len(df)-len(test_df)} ulasan")
    cds3.metric("Test Set (20%)", f"{len(test_df)} ulasan")
    st.caption("Split stratifikasi menggunakan `dataset.csv` (100 ulasan) berdasarkan kombinasi **sentimen × kategori** (12 kombinasi kelas) dengan `random_state=42`. Evaluasi di bawah ini bebas dari *data leakage*, di mana metrik dan *Confusion Matrix* dihitung murni terhadap 20 ulasan pada *unseen test set* yang memiliki ground truth terverifikasi.")

    st.divider()

    # ── Fungsi render evaluasi ────────────────────────────────────────────────
    def render_eval_section(eval_dict: dict, title: str, label_names: list):
        st.subheader(title)

        # Metrik utama
        m1, m2, m3, m4 = st.columns(4)
        acc = eval_dict['accuracy']
        pre = eval_dict['precision_macro']
        rec = eval_dict['recall_macro']
        f1  = eval_dict['f1_macro']

        def mk(col, label, val, target):
            delta = "✅ Tercapai" if val >= target else "⚠️ Belum Tercapai"
            col.metric(label, f"{val*100:.1f}%", delta=delta)

        mk(m1, "Accuracy",        acc, 0.85)
        mk(m2, "Precision (Macro)", pre, 0.80)
        mk(m3, "Recall (Macro)",    rec, 0.85)
        mk(m4, "F1-Score (Macro)",  f1,  0.85)

        st.caption("Target: Accuracy ≥ 85%, Precision ≥ 80%, Recall ≥ 85%, F1-Score ≥ 85% (Macro Average)")

        # Layout confusion matrix + per-class
        cm_col, pc_col = st.columns([1, 1.2])

        with cm_col:
            st.markdown("**Confusion Matrix:**")
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(
                eval_dict['conf_matrix'],
                annot=True, fmt='d',
                xticklabels=label_names,
                yticklabels=label_names,
                cmap='Blues',
                linewidths=0.5,
                ax=ax_cm,
            )
            ax_cm.set_xlabel("Prediksi", fontsize=10)
            ax_cm.set_ylabel("Aktual", fontsize=10)
            plt.xticks(rotation=30, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_cm, use_container_width=True)
            plt.close()

        with pc_col:
            st.markdown("**Metrik Per Kelas:**")
            df_pc = eval_dict['per_class'].copy()
            df_pc['Precision'] = df_pc['Precision'].map(lambda x: f"{x*100:.1f}%")
            df_pc['Recall']    = df_pc['Recall'].map(lambda x: f"{x*100:.1f}%")
            df_pc['F1-Score']  = df_pc['F1-Score'].map(lambda x: f"{x*100:.1f}%")
            st.dataframe(df_pc, use_container_width=True, hide_index=True)

        # Analisis kasus gagal
        with st.expander("🔎 Analisis Kasus Salah Prediksi"):
            y_true = eval_dict['y_true']
            y_pred = eval_dict['y_pred']
            test_texts = test_df['ulasan'].tolist()
            df_miss = get_misclassified(y_true, y_pred, test_texts)
            if df_miss.empty:
                st.success("✅ Semua sampel test diklasifikasikan dengan benar!")
            else:
                st.warning(f"⚠️ {len(df_miss)} sampel salah diklasifikasikan:")
                st.dataframe(df_miss, use_container_width=True, hide_index=True)

    # ── Render kedua model ────────────────────────────────────────────────────
    render_eval_section(eval_results['sentimen'], "🏷️ Model Sentimen (Positif / Negatif / Netral)", LABEL_SENTIMEN)
    st.divider()
    render_eval_section(eval_results['kategori'], "🏢 Model Kategori Layanan (Administrasi / Akademik / Fasilitas / Kemahasiswaan)", LABEL_KATEGORI)

    st.divider()

    # ── Responsible AI Audit ──────────────────────────────────────────────────
    st.subheader("🛡️ Responsible AI Audit")
    st.markdown("""
    | Pilar | Status | Implementasi |
    |---|---|---|
    | 🔒 **Privacy** | <span class="rai-ok">✅ Aktif</span> | PII masking (NIM, Email, Telepon) di FR-01 sebelum pemrosesan |
    | ⚖️ **Fairness** | <span class="rai-ok">✅ Terpenuhi</span> | Dataset seimbang: 25 ulasan per kategori (4 kategori), 34:33:33 sentimen |
    | 🔍 **Explainability** | <span class="rai-ok">✅ Aktif</span> | Top-5 kata pemicu per kelas ditampilkan di Tab 🔬 |
    | 📢 **Transparency** | <span class="rai-ok">✅ Aktif</span> | Probabilitas per kelas ditampilkan untuk setiap prediksi |
    | 👤 **Human Oversight** | <span class="rai-ok">✅ Aktif</span> | Flag Low Confidence (< 60%) menyarankan tinjauan manual |
    """, unsafe_allow_html=True)

    st.markdown("")
    st.warning(
        "⚠️ **Keterbatasan Model yang Diketahui (Known Limitations):**\n\n"
        "- **Sarkasme & Negasi Kompleks:** BoW menghitung frekuensi kata secara independen tanpa memperhatikan "
        "urutan, sehingga kalimat seperti *\"Bagus sekali KRS error saat tenggat\"* berisiko salah diklasifikasikan.\n"
        "- **Kata Informal & Singkatan Baru:** Token yang tidak ada di vocabulary training hanya mendapat "
        "bobot Laplace seragam, menurunkan keyakinan prediksi.\n"
        "- **Dataset Sintetis:** Model dilatih pada 100 baris data simulasi — generalisasi ke data ulasan "
        "mahasiswa nyata memerlukan validasi lanjutan.\n"
        "- **Keputusan Akhir Oleh Manusia:** Output sistem berstatus *decision support*. "
        "Tindakan terkait evaluasi kinerja atau kebijakan WAJIB ditinjau oleh manusia yang berwenang."
    )

    st.divider()
    st.caption(
        "Sistem Klasifikasi Ulasan Kampus | "
        "Louis Ainur Rofiq Prayoga (240605110034) | "
        "S1 Teknik Informatika UIN Maulana Malik Ibrahim Malang | "
        "Mini Project AI — Pertemuan 15"
    )
