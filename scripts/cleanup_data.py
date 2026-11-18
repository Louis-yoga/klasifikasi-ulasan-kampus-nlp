import os
import shutil

import pandas as pd


def main():
    # 1. Create data/demo_unseen.csv and data/external_test.csv from Data.csv
    if os.path.exists('Data.csv'):
        df_ext = pd.read_csv('Data.csv', sep=';')
        df_ext['ulasan'] = df_ext['ulasan'].astype(str).str.strip('"\t ')
        df_ext[['id', 'ulasan']].to_csv('data/demo_unseen.csv', index=False, encoding='utf-8')
        print(f"Created data/demo_unseen.csv ({len(df_ext)} rows)")
        df_ext.to_csv('data/external_test.csv', index=False, encoding='utf-8')
        print(f"Created data/external_test.csv ({len(df_ext)} rows)")
        os.remove('Data.csv')
        print("Removed Data.csv from root")

    # 2. Move .docx to docs/
    os.makedirs('docs', exist_ok=True)
    if os.path.exists('Klasifikasi Ulasan Kampus.docx'):
        shutil.move('Klasifikasi Ulasan Kampus.docx', os.path.join('docs', 'Klasifikasi Ulasan Kampus.docx'))
        print("Moved Klasifikasi Ulasan Kampus.docx to docs/")

    if os.path.exists('Audit Teknis.docx'):
        shutil.copy('Audit Teknis.docx', os.path.join('docs', 'Audit Teknis.docx'))
        print("Copied Audit Teknis.docx to docs/")

    # 3. Remove redundant / circular files
    to_remove = [
        'data/data_latih.csv',
        'data/hasil_klasifikasi.csv',
        'data/Book12.xlsx',
        'data/Book2.csv',
        'test_pipeline.py'
    ]
    for f in to_remove:
        if os.path.exists(f):
            os.remove(f)
            print(f"Removed redundant file: {f}")

if __name__ == '__main__':
    main()
