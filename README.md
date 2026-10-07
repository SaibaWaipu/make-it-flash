# make-it-flash

LLM-jp 4.1 32B-A3B thinking を対象に、attention を Gated DeltaNet (GDN) へ置き換えるための **1 層 pilot** です。データ準備、teacher activation の取得、単独 GDN block の fitting を分離し、小さく検証できるようにしています。

> **実験用コードです。** 生成するのは単独 GDN block と fitting metrics であり、ロード可能な hybrid Causal LM ではありません。

## Scope

既定の teacher は [llm-jp/llm-jp-4.1-32b-a3b-thinking](https://huggingface.co/llm-jp/llm-jp-4.1-32b-a3b-thinking) の revision `cda260706786758045e5e96bf4d738bbc01155b5`、校正データは [llm-jp/llm-jp-4.1-thinking-sft-data](https://huggingface.co/datasets/llm-jp/llm-jp-4.1-thinking-sft-data) です。初期実験では tokenizer、embedding、MoE experts/router、LM head、RMSNorm を変更しません。

実行する3段階:

1. **prepare** — SFT を streaming で読み、token quota に従って calibration JSONL を作成します。既定値は 100K tokens / sequence 長 2048 です。
2. **cache** — BF16 teacher の指定 attention 層について、正規化済み入力と attention 出力を保存します。重みを読む前に可視 GPU 全体で空き VRAM 66 GiB 以上を要求します。
3. **fit** — Transformers の Qwen3NextGatedDeltaNet を teacher forcing の MSE で局所 fitting します。GDN head layout はQwen3.8-Flash-Next参照値のQK 16 / V 48、head dim 128、sigmoid output gateです。

限定的な `Qwen3MoeAttention` adapterで、tiny Qwen3-MoEへのno-cache graftをCPU検証しています。maskなし・2-D binary maskに加え、標準的な4-D causal maskからleft/right paddingを抽出できます。nonstandard/sparse mask、hybrid cache、attention weights出力は未対応で、明示的に拒否します。

まだ含まないもの: 全32層の置換、24 GDN + 8 Qwen Sparse Attention (QSA) への統合、ロード可能なhybrid checkpoint、Gated Residual、N-gram/PLE embeddings、cache-aware generation、global calibration、perplexity / 生成品質評価、RLVR。

## Qwen3.8-Flash-Next reference

[公式HF model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) と [config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json) を確認しました。この名前のrepoは `model_type=qwen4_exp` / `Qwen4ExpForConditionalGeneration` のマルチモーダルpreviewです。言語部は48層で **3 Gated DeltaNet + 1 QSA** を繰り返し、36 GDN + 12 QSA（75/25）。32層のLLM-jpへ比率だけ移す候補は **24 GDN + 8 QSA** です。model cardによるとQSAは個々のtokenではなくmicro-block単位でselectionします。configではその12層が `full_attention` と記載されますが、[Qwen4Exp attention実装](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)では非linear layerのattentionがQSA indexerの選択maskを適用します。したがってこれらはdense full attentionではありません。

参照形状はGDNがQK 16 / V 48 heads × 128、sigmoid output gate、QSAが24 Q / 2 KV heads × 256。QSA indexerは4 query heads / 1 shared key head × 128、512 micro-blocks（2048 tokens）budget、compression ratio 4です。Qwen3.8は48層・262K contextのvision/videoモデルで、125B core (6B activated)に約51B n-gram/PLE embeddingsと4B MTPを持ち、MoEは512 routed expertsからtop-10に加えてshared expertを使います。さらに4-stream Gated Residual (rank 320)を備えます。これらの巨大な追加機構・MoE重みを移植するのではなく、初期方針どおりLLM-jpのtokenizer・128 experts/top-8・日本語能力を保持します。

このpilotはQwen3-NextのGDN実装を使い、16/48 head layoutとsigmoid output-gate RMSNormを合わせた近似です。Qwen4Expのhyper-connectionsやQSA blockそのものを使ってはいません。Qwen3.8 configの `transformers_version` は `5.8.0.dev0` で、ローカルの5.5.0では `qwen4_exp` を認識しません。将来QSAを統合する段階では、Qwen4Exp対応のTransformers版または検証済み実装を別途固定する必要があります。

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
- cache / fit: CUDA 対応 PyTorch。cache は既定で **空き CUDA memory 66 GiB 以上**を確認します。固定した4.1 revisionには13 safetensors（64.28 GB / 59.87 GiB）があり、これはweightのstorage sizeのみで実際のVRAM要件ではありません。T4 × 2 の約29–32 GBではこのBF16 passは実行できず、A100 80 GBでも空きが66 GiB未満ならpreflightが停止します。

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

通常の GPU 実行には [scripts/run_hf_job.sh](scripts/run_hf_job.sh) を使います。ジョブ内で prepare → teacher cache → 1層の GDN fit を順に実行します。既定teacherは `llm-jp/llm-jp-4.1-32b-a3b-thinking` と固定revision `cda260706786758045e5e96bf4d738bbc01155b5`（`MIF_MODEL` / `MIF_MODEL_REVISION` で変更可能）です。別モデルに変更する際は対応するrevisionも指定してください。4.1向けの更新は [SaibaWaipu/make-it-flash](https://github.com/SaibaWaipu/make-it-flash) の `gdn-4.1-pilot` branch にpush済みです。実行例では `MIF_GIT_REF` でこのbranchを選びます。開始時にcloneしたcommit hashをジョブログへ表示します。

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

ローカル pytest は14件通過しました。固定teacher revision `cda260706786758045e5e96bf4d738bbc01155b5` で実データ100 tokens のprepare smoke testを実行し、35/25/15/15/10のmixを確認しました。stream worker終了警告はmanifestに記録されていますが、全100 tokensが保存されています。hidden size 2560・QK 16/V 48・sigmoid gateのGDN CPU forwardと、合成activationを使った1 step CPU fitも通過しています（実teacher出力によるfitではありません）。さらにtiny Qwen3-MoE decoder/modelにGDN adapterを挿し、left/right paddingを含む標準causal maskでno-cache CPU forwardを検証しました。GDN safetensors再読込後も出力が一致し、MoE/Normは保持されます。hybrid cacheとnonstandard/sparse maskは未対応です。32B BF16重みのGPU load、実activation fit、HF Jobs完走は未検証です。Qwen3.8本体の `qwen4_exp` はローカルTransformers 5.5.0では未対応です。

## License

このソースコードは [Apache License 2.0](LICENSE) です。モデルとデータセットはそれぞれの配布条件に従ってください。
