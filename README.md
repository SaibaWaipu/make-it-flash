# make-it-flash

LLM-jp 4.1 32B-A3B thinking を対象に、attention を Gated DeltaNet (GDN) へ置き換えるための **1 層 pilot** です。データ準備、teacher activation の取得、単独 GDN block の fitting を分離し、小さく検証できるようにしています。

> **実験用コードです。** 生成するのは単独 GDN block と fitting metrics であり、ロード可能な hybrid Causal LM ではありません。

## Scope

既定の teacher は [llm-jp/llm-jp-4.1-32b-a3b-thinking](https://huggingface.co/llm-jp/llm-jp-4.1-32b-a3b-thinking)、校正データは [llm-jp/llm-jp-4.1-thinking-sft-data](https://huggingface.co/datasets/llm-jp/llm-jp-4.1-thinking-sft-data) です。初期実験では tokenizer、embedding、MoE experts/router、LM head、RMSNorm を変更しません。

実行する3段階:

1. **prepare** — SFT を streaming で読み、token quota に従って calibration JSONL を作成します。既定値は 100K tokens / sequence 長 2048 です。
2. **cache** — BF16 teacher の指定 attention 層について、正規化済み入力と attention 出力を保存します。重みを読む前に可視 GPU 全体で空き VRAM 66 GiB 以上を要求します。
3. **fit** — Transformers の Qwen3NextGatedDeltaNet を teacher forcing の MSE で局所 fitting します。

まだ含まないもの: 全32層の置換、24 GDN + 8 attention への統合、cache-aware generation、global calibration、perplexity / 生成品質評価、RLVR。

## Calibration mix

quota は最大剰余法で割り当て、実際に token 化した数と解決済み model/dataset revision を manifest に記録します。

| 分野 | 比率 | Dataset configs |
|---|---:|---|
| Japanese | 35% | jaster_v1.4.1, llmjp_extraction_wiki_ja_v0.3 |
| English | 25% | daring_anteater, flan |
| Math | 15% | logical_math_coding_wizard8x22b, nemotron_post_v3_math, nemotron3_sft_multilingual_v2_math_ja_stackoverflow |
| Code | 15% | synthetic_jp_en_coding |
| Tool / Agent | 10% | nemotron_agentic_mix_v0.1.1, nemotron_agentic_mix_v0.1.1_ja |

各 config の quota は区分内でほぼ均等に分けます。Tool role を tokenizer template が受け付けない場合、tool observation をラベル付き user turn として token 化します。元データの各 license 条件を確認してください。

## Requirements

- Python 3.10+
- prepare: Hugging Face Hub へのネットワーク接続（モデル tokenizer と dataset）。CUDA は不要です。
- cache / fit: CUDA 対応 PyTorch。BF16 32B teacher のため、cache は既定で **空き CUDA memory 66 GiB 以上**を確認します。T4 × 2 の約29–32 GBではこの BF16 pass は実行できません。A100 80 GB でも他プロセス等で空きが足りなければ preflight が停止します。

## Quick start

プロジェクトルートで:

    python -m pip install -e '.[test]'
    pytest

データ準備 (GPU 不要):

    make-it-flash prepare \
      --output-dir artifacts/data \
      --max-tokens 100000 \
      --max-seq-len 2048

Teacher cache と fitting (十分な GPU がある場合のみ):

    make-it-flash cache \
      --data-file artifacts/data/calibration.jsonl \
      --output-dir artifacts/cache \
      --layers 0

    make-it-flash fit \
      --cache-dir artifacts/cache \
      --output-dir artifacts/gdn \
      --layer 0 \
      --epochs 3 \
      --max-steps 200

prepare の出力 mix が目標に達しない場合は data_manifest.json の actual_tokens_by_category と skipped_rows_by_config を確認してください。上書きには各 stage の --overwrite を明示します。

## Hugging Face Jobs（標準の実行方法）

通常の GPU 実行には [scripts/run_hf_job.sh](scripts/run_hf_job.sh) を使います。ジョブ内で prepare → teacher cache → 1層の GDN fit を順に実行します。既定の teacher は `llm-jp/llm-jp-4.1-32b-a3b-thinking`（`MIF_MODEL` で変更可能）です。4.1向けの更新は [SaibaWaipu/make-it-flash](https://github.com/SaibaWaipu/make-it-flash) の `gdn-4.1-pilot` branch にpush済みです。実行例では `MIF_GIT_REF` でこのbranchを選びます。開始時にcloneしたcommit hashをジョブログへ表示します。

### 出力先の private repo ID

`MIF_OUTPUT_REPO` には Hugging Face の `namespace/repo-name` 形式の ID を指定します。たとえば `SaibaWaipu/make-it-flash-pilot` なら、保存先は `https://huggingface.co/SaibaWaipu/make-it-flash-pilot` です。**新規 repo はジョブが private として自動作成するため、事前作成は不要です。**既存 repo を指定する場合は、実行前に private であることを確認してください（既存 repo の公開設定はこのスクリプトでは変更しません）。

HF CLI にログインし、private model repo の作成・書き込み権限がある token を使って実行します。スクリプトの `--secrets HF_TOKEN` がログイン中の token をジョブに渡します。

    hf auth login
    MIF_GIT_REF=gdn-4.1-pilot MIF_OUTPUT_REPO=SaibaWaipu/make-it-flash-pilot bash scripts/run_hf_job.sh

既定は `a100-large`、timeout は4時間です。現在の目安は A100 80 GB が約 $2.50/時、4時間で最大約 $10 ですが、料金は変動するため起動直前に確認してください。ジョブは GDN checkpoint と fit metrics のみを private repo に upload します。calibration data と activation cache は upload しません。**この README を読むだけではジョブは起動せず、課金も発生しません。**

このスクリプトは既定で公開 Git remote の `main` を clone します。別の変更を使う場合は、その変更をpushした後、必要に応じて `MIF_GIT_URL` と `MIF_GIT_REF`（branch または tag）を指定してください。ジョブ内でソースコードが実行され、token も渡されるため、信頼できる remote/ref を指定してください。

## Kaggle（任意・旧手順）

[Kaggle notebook](kaggle/make_it_flash.ipynb) はソースを埋め込んだ self-contained 版として残していますが、標準の実行先ではありません。GPU と Internet を要求し、空き VRAM が66 GiB以上の場合だけ teacher cache と fitting に進みます。Kaggle を使う場合の push 手順と注意事項は [Kaggle runbook](kaggle/README.md) を参照してください。

## Outputs and privacy

- artifacts/data/: tokenized calibration.jsonl, data_manifest.json
- artifacts/cache/: sequence ごとの teacher input/output safetensors, cache_manifest.json
- artifacts/gdn/: 単独 GDN の safetensors と fit metrics JSON

大きな data、cache、weights、Hub cache は [.gitignore](.gitignore) で除外しています。データセット由来のデータや派生物を公開・再配布する場合は、元 dataset の条件を別途確認してください。

## Validation status

ローカル pytest は11件通過しました。既定 teacher を `llm-jp/llm-jp-4.1-32b-a3b-thinking` に切り替え、実データ100 tokens の prepare smoke test を実行し、35/25/15/15/10 の mix と解決済み model revision を確認しました。stream worker の終了警告は manifest に記録されていますが、全100 tokens が保存されています。hidden size 2560でGDN blockのCPU forwardと、合成activationを使った1 stepのCPU fitも通過しています。これは実teacher出力によるfitではありません。32B BF16 teacher cacheと実activationでのfit、HF Jobs上の完走は未検証です。

## License

このソースコードは [Apache License 2.0](LICENSE) です。モデルとデータセットはそれぞれの配布条件に従ってください。
