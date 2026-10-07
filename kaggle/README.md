# Kaggle 実行（任意・旧手順）

この notebook はパッケージソースを埋め込んだ self-contained 版です。現在の標準実行先は Kaggle ではなく Hugging Face Jobs です。HF Jobs の手順は project root の [README](../README.md) を参照してください。作業ディレクトリを project root にして実行してください。

## 実行手順

1. ソースを変更したら python scripts/build_kaggle_notebook.py で notebook を再生成します。
2. kaggle/kernel-metadata.json の id を自分の Kaggle username / slug に合わせます。別のユーザーが実行する場合は is_private も必要に応じて変更します。
3. GPU と Internet を有効にして kaggle kernels push -p kaggle を実行します。

Notebook はまず calibration data を準備します。空き VRAM が66 GiB以上の場合のみ teacher cache と GDN fitting を続け、未満なら teacher weights をダウンロードせず止まります。T4 × 2 の合計約29–32 GBは BF16 teacher pass の要件を満たしません。Hugging Face Hub へのネットワーク接続も必要です。

## 検証メモ

2026-10-06 の private kernel 実行では、Kaggle runtime が CPU-only PyTorch を報告し、Hugging Face Hub の DNS 解決にも失敗しました。calibration data / model weights のダウンロード前に停止しており、これはその実行環境での制約です。再実行前に GPU 割当と huggingface.co への接続を確認してください。標準の実行先は project root の README にある HF Jobs 手順です。
