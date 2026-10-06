# make-it-flash

LLM-jp 4 32B-A3B を対象に、attention を Gated DeltaNet (GDN) へ置き換えるための **1 層 pilot** です。データ準備、teacher activation の取得、単独 GDN block の fitting を分離し、小さく検証できるようにしています。

> **実験用コードです。** 生成するのは単独 GDN block と fitting metrics であり、ロード可能な hybrid Causal LM ではありません。

## Scope

対象モデルは [llm-jp/llm-jp-4-32b-a3b-base](https://huggingface.co/llm-jp/llm-jp-4-32b-a3b-base)、データセットは [llm-jp/llm-jp-4.1-thinking-sft-data](https://huggingface.co/datasets/llm-jp/llm-jp-4.1-thinking-sft-data) です。初期実験では tokenizer、embedding、MoE experts/router、LM head、RMSNorm を変更しません。

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

## Kaggle

[Kaggle notebook](kaggle/make_it_flash.ipynb) はこの repository のソースを埋め込んだ self-contained 版です。GPU と Internet を要求します。データ準備を行い、空き VRAM が66 GiB以上の場合だけ teacher cache と fitting に進みます。Kaggle アカウントが異なる場合は [kaggle/kernel-metadata.json](kaggle/kernel-metadata.json) の id を変更してください。push 手順と実行時の確認事項は [Kaggle runbook](kaggle/README.md) を参照してください。

## Hugging Face Jobs

Kaggle の GPU / Hub 接続が使えない場合の有料 fallback は [scripts/run_hf_job.sh](scripts/run_hf_job.sh) です。既定のソース remote はこの public repository です。HF write token を HF_TOKEN としてログイン済み CLI に渡し、出力先だけ指定します:

    MIF_OUTPUT_REPO=<your-hf-namespace>/make-it-flash-pilot \
      bash scripts/run_hf_job.sh

既定は a100-large、4時間 timeout です。現在の hardware list では A100 80 GB は $2.50/時（4時間で最大約$10）ですが、料金は変動するため起動直前に確認してください。ジョブは指定名の **private model repo** を作成し、GDN checkpoint と fit metrics だけを upload します。calibration data と activation cache は upload しません。**この repository から有料 job は自動起動しません。**

## Outputs and privacy

- artifacts/data/: tokenized calibration.jsonl, data_manifest.json
- artifacts/cache/: sequence ごとの teacher input/output safetensors, cache_manifest.json
- artifacts/gdn/: 単独 GDN の safetensors と fit metrics JSON

大きな data、cache、weights、Hub cache は [.gitignore](.gitignore) で除外しています。データセット由来のデータや派生物を公開・再配布する場合は、元 dataset の条件を別途確認してください。

## Validation status

ローカルでは pytest 8件と、実データ100 tokens の streaming smoke test（35/25/15/15/10 mix）を確認しました。Kaggle 上の検証は Hub DNS 解決失敗と CPU-only image のため、データ取得前に停止しました。詳細は [Kaggle runbook](kaggle/README.md) に記録しています。BF16 teacher cache と実データでの GDN fit はまだ完了していません。

## License

このソースコードは [Apache License 2.0](LICENSE) です。モデルとデータセットはそれぞれの配布条件に従ってください。
