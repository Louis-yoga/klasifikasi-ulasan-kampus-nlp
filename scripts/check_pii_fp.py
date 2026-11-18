"""
scripts/check_pii_fp.py
-----------------------
Memeriksa potensi False Positive PII masking pada dataset kampus (SEC-02).
Memastikan istilah umum kampus (seperti nama gedung, mata kuliah, singkatan, angka semester)
tidak secara keliru disamarkan sebagai PII (NIM, TELEPON, EMAIL, NAMA).
"""

import os
import re
import sys

# Pastikan root workspace ada di sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

from src.preprocessor import mask_pii


def check_dataset_pii(csv_path: str):
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} tidak ditemukan.")
        return 0

    df = pd.read_csv(csv_path)
    total_reviews = len(df)
    masked_count = 0
    flagged = []

    for idx, row in df.iterrows():
        text = str(row['ulasan'])
        masked = mask_pii(text)

        # Cari tag PII yang tertempel
        pii_tags = re.findall(r'\[(EMAIL|TELEPON|NIM|NAMA)\]', masked)
        if pii_tags:
            masked_count += 1
            flagged.append({
                'id': row.get('id', idx + 1),
                'original': text,
                'masked': masked,
                'tags': pii_tags
            })

    print(f"\nAudit PII pada {csv_path}:")
    print(f"  Total ulasan: {total_reviews}")
    print(f"  Ulasan dengan deteksi PII: {masked_count}")
    if flagged:
        print("  Daftar deteksi:")
        for item in flagged:
            print(f"    - ID {item['id']}: Tags {item['tags']}")
            print(f"      Asli  : {item['original']}")
            print(f"      Masked: {item['masked']}")
    else:
        print("  [OK] 0 False Positive terdeteksi.")

    return len(flagged)


if __name__ == '__main__':
    print("=" * 60)
    print("AUDIT FALSE POSITIVE PII MASKING (SEC-02)")
    print("=" * 60)
    fp_dataset = check_dataset_pii(os.path.join("data", "dataset.csv"))
    fp_demo = check_dataset_pii(os.path.join("data", "demo_unseen.csv"))
    print(f"\nTotal False Positive terkonfirmasi: {fp_dataset + fp_demo}")
