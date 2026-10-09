# make-it-flash

LLM-jp 4.1 32B-A3B thinking を対象に、全32層のattentionを **24層のGated DeltaNet (GDN)＋8層のQSA** へ置き換える実験用コードです。単層pilotから始まり、データ準備、teacher activation取得、各層の局所fit、overlay組立、フル重みの統合まで扱います。品質の改善や実用的な高速化を確認したモデルではありません。

> **実験用コードです。** 2026-10-09の段階実行で全32層の局所fitとoverlay組立は完了し、13個のSafetensors shardを統合したフル重みsnapshotも作成しました。overlayは別途base modelを必要とし、フル重みsnapshotには`trust_remote_code=True`を要するカスタムローダーが付属します。**約66 GBのフルチェックポイントは実際にロード・生成・品質評価しておらず、正常動作や性能向上は未確認**です。tiny synthetic checkpointでのload/save/cache generationと、フルindex/key/shapeの静的照合は実施しました。既知の推論・キャッシュ・export上の制限は下記「既知の問題」を参照してください。

## Scope

既定の teacher は [llm-jp/llm-jp-4.1-32b-a3b-thinking](https://huggingface.co/llm-jp/llm-jp-4.1-32b-a3b-thinking) の revision `cda260706786758045e5e96bf4d738bbc01155b5`、校正データは [llm-jp/llm-jp-4.1-thinking-sft-data](https://huggingface.co/datasets/llm-jp/llm-jp-4.1-thinking-sft-data) です。初期実験では tokenizer、embedding、MoE experts/router、LM head、RMSNorm を変更しません。

実行する3段階:

1. **prepare** — SFT を streaming で読み、token quota に従って calibration JSONL を作成します。既定値は 100K tokens / sequence 長 2048 です。
2. **cache** — BF16 teacher の指定 attention 層について、正規化済み入力と attention 出力を保存します。QSA fitting用に、指定した1層だけdense teacher attentionを計算してmicro-blockごとのmassへ集約し、圧縮保存できます。重みを読む前に可視 GPU 全体で空き VRAM 66 GiB 以上を要求します。
3. **fit / fit-qsa** — GDNはteacher forcingのMSE、QSAは出力MSE＋正のweightを持つteacher attention block-mass selector lossで、それぞれ1層ずつ局所fittingします。QSA selectorの独立validation lossが改善しない、または出力MSEが悪化した成果はoverlay assemblyが拒否します。`full-run` はこれを32層に適用しますが、GPU時間と品質は保証しません。

GDN/QSA attention adapterとhybrid cacheをCPU tiny-modelで検証しています。GDN recurrent state・QSA indexer raw-key state・通常KVを持つ `FlashNextDynamicCache` は `create_flash_next_cache` で作成し、標準条件でのprefill→incremental decode parityを確認しました。ただし、キャッシュの`crop()`はGDNの再帰状態を巻き戻せず、使用すると以後の出力が不正になります。`crop()`を使うspeculative decoding等は避けてください。GDN-onlyには `configure_hybrid_cache` も使えます。static/offloaded cacheや最適化済みlong-context kernelは未対応です。

GDNとblock-indexed QSAは統合attention置換の対象です。別に4-stream Gated Residual、compact hashed PLE/ngram、tied-embedding MTP head、shared-expert add-onの再利用可能なprototypeも実装していますが、これらは変換対象モデルに統合されていません。全32層のfit済みoverlayは組立済みで、`scripts/export_full_staged_hf_artifact.py` がpinned baseのattention重みをGDN/QSAへ置換したfull-weight snapshotと、`trust_remote_code=True`用のcustom loaderを公開します。synthetic tiny checkpointのstrict round-trip/cache generationとfull index/key/shapeは検証済みですが、66GB full checkpoint load/generationは未実施です。まだ含まないもの: PLE/Gated Residualの学習・decoder統合、global calibration、Japanese perplexity / generation品質評価、long-context performance検証、vision/video、RLVR。

## モデル系譜と参照アーキテクチャ

**対象モデルそのものはQwenではありません。** teacher/baseは国立情報学研究所（NII）開発の `llm-jp/llm-jp-4.1-32b-a3b-thinking` です。`qwen3_moe` / `Qwen3MoeForCausalLM` は、このLLM-jp checkpointが公開設定と重み形式で採用するQwen3-MoE互換の実装インターフェースを指します。モデルの出自・学習主体がQwenという意味ではありません。

本実験ではLLM-jpの32層のうち、元のattentionを24層のGated DeltaNet（GDN）と8層のblock-indexed QSAに置き換えます。GDNはTransformersの **Qwen3-Next** `Qwen3NextGatedDeltaNet` 実装を再利用し、QK 16 / V 48 heads × 128と独自のsigmoid gated RMSNormを使います。したがって **Qwen3-NextはGDN実装の依存先であり、teacherでも変換後モデルのアーキテクチャ名でもありません**。QSAはこのプロジェクト内の独自実装です。層の割当は0–2 GDN、3 QSAを4層単位で反復します。

参照比率・形状はQwen3.8-Flash-Nextの公開資料（[model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)、[固定config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json)、[attention実装](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)）を**設計参考として部分的に参照**しています。そこにある3 GDN + 1 QSAの反復、GDN/QSA head dimensions、micro-block selectionを参考にしましたが、Qwen3.8-Flash-Next checkpointや重みを移植したものではありません。Qwen3.8固有のMoE・PLE/ngram・MTP・Gated Residual・vision/video構成は本モデルに含めていません。変換後もLLM-jp由来のtokenizer、128 experts/top-8のMoE、embedding、router、MLP、LM head、norm類を維持する設計です。

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
- cache / fit: CUDA 対応 PyTorch。cache は40文字の固定commit SHAを要求し、`AutoConfig` のrepo ID/commitが要求値と一致することをweights load前に検証したうえで、既定で **空き CUDA memory 66 GiB 以上**を確認します。固定した4.1 revisionには13 safetensors（64.28 GB / 59.87 GiB）があり、これはweightのstorage sizeのみで実際のVRAM要件ではありません。T4 × 2 の約29–32 GBではこのBF16 passは実行できず、A100 80 GBでも空きが66 GiB未満ならpreflightが停止します。

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

QSAのselector学習では、dense teacher attentionをGPU上で計算した後、micro-block massだけを圧縮保存します（dense行列計算のため一度に1層のみ）。既定budgetは2048 tokens、compression ratioは4なので、top-k pruningが起きる最短系列長は2052 tokensです。Assemblyには十分な長さのtrain例と独立validation例の両方が必要です。`cache --attention-layers` はGPU確認/teacher weight load前にeligible系列が2本未満なら失敗し、`fit-qsa` はtrain/独立validation双方の条件不足時にoptimizer開始前に失敗します。例えば `--max-seq-len 4096` でprepareし、train/validation双方に実際に長い例が分割されるだけのtoken数を用意してください。

    make-it-flash prepare \
      --output-dir artifacts/qsa-data \
      --max-tokens 100000 \
      --max-seq-len 4096

    make-it-flash cache \
      --data-file artifacts/qsa-data/calibration.jsonl \
      --output-dir artifacts/qsa-cache-layer-03 \
      --layers 3 \
      --attention-layers 3

    make-it-flash fit-qsa \
      --cache-dir artifacts/qsa-cache-layer-03 \
      --output-dir artifacts/qsa \
      --layer 3

QSA fitは1層のみの局所fitです。長い独立train/validation例がない場合は学習開始前に拒否します。cacheの `--overwrite` は書込開始前に旧manifestを無効化し、成功時に古い余剰shardを削除します。途中失敗後のshard群はmanifestがないためfitに使えません。実際のteacher cache取得にはGPUが必要ですが、このREADME更新では有料Jobを実行していません。

prepare の出力 mix が目標に達しない場合は data_manifest.json の actual_tokens_by_category と skipped_rows_by_config を確認してください。上書きには各 stage の --overwrite を明示します。

## Hybrid cache（GDN recurrent + attention KV）

GDN graft 後に Transformers `DynamicCache` を使う場合は、最初の `generate(..., use_cache=True)` より前に対象 layer index を設定します。GDN layer は conv/recurrent state、他の dense/QSA layer は通常の KV cache を持ちます。

```python
from make_it_flash.model import configure_hybrid_cache

configure_hybrid_cache(model, gdn_layers=[0, 3, 6])
```

この試作では標準 causal mask と binary padding mask の DynamicCache decode を検証しています。static cache・独自 sparse mask はまだ fail-closed です。

## 全32層の逐次処理（段階fit実行済み、統合モデルの推論未検証）

`make-it-flash full-run` は固定revisionの1つの校正JSONLで train/validation の長文分割をGPU load前に確定し、最初のQSA teacher cache取得時に24 GDN層もまとめてcaptureします。続いて残る7 QSA層を別々にcaptureし、32層の局所fit、厳格な `assemble` を実行します。つまり**teacherを8回ロード**し、QSAでは4096 tokensのdense attentionを各層ごとに作るので、GPU時間・メモリ負荷は大きくなります。このコマンドの`--resume`はローカルの既存cache/fitを検査して続行しますが、HF Jobsのephemeral filesystemをまたいで再開できません。段階JobのHub保存・再開とは別の経路です。GPUのない環境ではdry-runのみ使ってください。

    make-it-flash prepare --output-dir artifacts/full-data --max-tokens 100000 --max-seq-len 4096
    make-it-flash full-run --data-file artifacts/full-data/calibration.jsonl --output-dir artifacts/full-run --dry-run
    make-it-flash full-run --data-file artifacts/full-data/calibration.jsonl --output-dir artifacts/full-run

独立データでの `evaluate` は品質を主張する前に必要です。`full-run` は成功時にoverlayへ校正token列のSHA-256集合を付記し、評価データのtoken列との重複を検知します。古い単層fit由来overlayでSHA集合が無い場合、この追加の重複検査は行われません。staged training repoにはadapter/checkpoint/metricsのみを置き、原文・teacher activationはアップロードしません。別のfull-checkpoint exporterはpinned baseの重みとfit済みattentionを統合したprivate snapshotとcustom `auto_map` loaderを作ります。Hubから使う際は`trust_remote_code=True`が必要です。tiny synthetic loader testと全weight index/key/adapter shapeは確認しましたが、full 66GB weightsのload/generation・品質は未検証です。最適化long-context kernelも未実装です。

HF Jobs向けには [scripts/run_full_hf_job.sh](scripts/run_full_hf_job.sh) を用意しました。**既定はdry-runで、有料ジョブを起動しません。** 既存の単層runnerは変更していません。起動には更新コードを信頼できるGit remoteへpushして40桁commitを固定し、private output repoの現在SHA、今回の追加予算上限（最大$9）を指定する必要があります。開始前に対象repoとsourceを必ず確認してください。ジョブが実行する `prepare → full-run → evaluate → upload` は、独立評価dataset/splitの指定を必須とし（評価なしアップロードは `MIF_ALLOW_UNEVALUATED_UPLOAD=1` を明示）、現時点では32B実機未検証です。timeout到達やOOM、品質ゲート拒否なら完成overlayはありません。Job filesystemは終了時に消えるため途中成果は維持されません。

料金の読み取り専用確認（`hf jobs hardware --json`、2026-10-08）ではA100 80GB `a100-large` が **$2.50/時**。追加上限$9なら理論上3.6時間、余裕を取ってtimeout `3h` なら最大$7.50です。H200 141GBは$5/時（1.8時間）、RTX PRO 6000 96GBは$2.75/時（約3.27時間）。8回の32B teacher load、データ準備、各fit、評価・uploadまでこれに収まる実測根拠はなく、**$9で全層完成できるとの見積もりではありません**。実測の速度と試料数が揃うまではlaunchを推奨しません。

## 層ごとにHF repoへ保存する段階実行

固定校正データは指定に合わせて日本語 `llmjp_extraction_wiki_ja_v0.3` / 英語 `daring_anteater` / コード `synthetic_jp_en_coding` を40/30/30で選び、100,000 tokens・4096 max sequenceのtokenized JSONLにしました。各カテゴリでstable train/validationそれぞれ最低1件の2052-token長文を予約しています。upstream revisionは `cb4210190af6a0000fd91cb5bb8361ad6d6ade01`、teacherは `cda260706786758045e5e96bf4d738bbc01155b5` です。private校正repo [RemydreScarlet/llm-jp-4.1-flash-next-calibration](https://huggingface.co/datasets/RemydreScarlet/llm-jp-4.1-flash-next-calibration) の固定revisionは `22c44496a7d708bca4986f80f1478e51aebad67a` (private) です。JSONL SHA-256 `aa0500f9d21a4251d9dbf9e7158857cec78f39ae4b902a986c7a5abac6fde706`、manifest SHA-256 `a728725277d56c6d2a7d639d1e4ffb9677bd907474a2e6906bf134096bc4c452`。splitは78 train/8 validation、QSA長文例は13/3 (日本語2・英語7・コード7) です。source全体のライセンスではなく選定configごとの条件を同repo cardに記載しています。

段階保存のメイン入口は [`scripts/run_staged_hf_job.sh`](scripts/run_staged_hf_job.sh)（[`scripts/run_all_staged_hf_job.sh`](scripts/run_all_staged_hf_job.sh)へ委譲）です。**1つのHF Jobの1回の実行で層0〜31を順番にfit**し、fit/品質検証が終わるたびにcheckpoint・metrics・recordをmodel repoへatomic commitしてから次へ進みます。既存の共有capture最適化によりteacher loadは8回（24 GDN + 初回QSAを一度にcapture、その後7 QSA）です。Jobが途中終了してもcommit済み層は残り、同じrun IDと最新repo SHAで `MIF_RESUME=1` とすれば不足分を再開できます。実行コードを更新して再開する場合は元の `MIF_RUN_SOURCE_COMMIT` を維持し、現在のコードSHAは `execution_commit` に別記します。保存先は `staged/<run-id>/layers/...` で、層成果は単独ではロードできず、全32件後に別途組立て・held-out評価が必要です。`staged-layer` コマンドは局所テスト用で、Jobを32回に分割するメイン経路ではありません。

段階Jobの入力は毎回**同じ固定校正JSONL**でなければなりません。`prepare` のmanifestには作成時刻が入り、毎Job再生成してもSHAが異なります。今回のtokenized corpusは利用権・private再配布確認後、`scripts/publish_calibration_dataset.py` でprivate dataset repoへ保存し、Hubから固定revisionで再取得してSHAを照合しました。次回corpusを差し替える場合も新しい親commitとデータSHAを固定します。calibration JSONLやteacher cacheは出力モデルrepoへは保存しません。段階Jobはそのprivate dataset repoの40桁commitと校正2ファイルのSHAを照合します。SFTデータ由来のtoken列のHubアップロード許諾が確認できない場合はこの段階方式を起動せず、共有してよい独自校正データまたは承認済みprivate bucketの方式を使ってください。

    make-it-flash staged-layer --data-file artifacts/data/calibration.jsonl --output-dir artifacts/staged-layer-03 --layer 3 --dry-run
    MIF_RUN_ID=pilot-32 bash scripts/run_staged_hf_job.sh  # 32層を1つのJob内で実行、既定dry-run

有料実行は `MIF_LAUNCH_HF_JOB=1`、レビュー済みGit commit、private校正repo SHA、校正JSONL/manifest SHA、model repo初期SHA、予算と請求済み額の入力が必要です。新runner既定は `a100-large` / `3h` で、時給$2.50なら**最大$7.50**。これはtimeoutまでの上限計算で、32層が3時間以内に完了する保証はありません。途中終了時はその時点までの層commitが残り、同じrun IDと最新model repo HEADで `MIF_RESUME=1` とすれば不足分を再開できます。2026-10-09のrun `flashnext-32-run01` は、初期revision不整合と2回のVRAM guard自動終了を段階commit保持で復旧し、最終Job `6ac8695f095c578089301c03` が層0〜7を復元、層8〜31を逐次fit/commitして `COMPLETED` しました。短命な子プロセスにteacher captureを隔離してCUDA contextを解放します。これは校正activationを取得し各層を局所fitした実行記録であり、統合済み32Bモデルの生成・品質評価を意味しません。最終private staged repo revisionは `58c691aff794b1a1c458421f8beb152d5e11c619`、最終Job実費は約$6.22、失敗Job分込み累計概算は約$8.67（承認上限$9以内）です。32件のrecord/checkpoint/metricsを固定revisionから検証し、ローカルoverlayも組立済みです。full-weight snapshotは別途private model repo [RemydreScarlet/llm-jp-4.1-flash-prepreview](https://huggingface.co/RemydreScarlet/llm-jp-4.1-flash-prepreview) のrevision `acd90cac02c3d543dd3cbd44bddcef5cb4a46023` に統合し、同revisionにAutoConfig/AutoModel用custom loaderを追加しました。13 shard、index、全SHA-256、remote code SHA-256、private設定を検証済みです。tiny synthetic modelでload/cache generation smoke testと、全32層meta constructionでMoE変換後のkey set・全custom attention keysの一致を確認しましたが、66GBのweightsを読み込んだ通常推論や品質評価は行っていません。

## Base-model overlay assembly

32層分の各fit成果を揃えた後、`assemble` はbase configのrepo ID/commit SHA、各fit checkpointのSHA-256、独立validationでの全体/selector loss改善、QSAのtrain/validation双方でのtop-k pruning実例、全layer共通の校正JSONL/data manifest SHA-256、およびTransformers/package/implementation fingerprintを照合し、3-GDN/1-QSA scheduleのadapter weightsとmanifestだけを一時directoryからatomicにpublishします。LLM-jp本体のweights/tokenizerは複製せず、overlayの利用時は同じ固定revisionのbase modelを別途指定します。フルcheckpointが必要な場合は `scripts/export_full_staged_hf_artifact.py` が13 shardをstream rewriteし、既存attentionを置換してfull-weight snapshotを作ります。fit tensorはbase dtypeのBF16へ変換し、indexとtotal_sizeを更新します。private full checkpoint repoにはcustom `auto_map` loaderを同梱し、tiny synthetic modelのload/cache generationを確認しましたが、66GB full checkpointのloadability・品質は未検証です。

    make-it-flash assemble \
      --gdn-dir artifacts/gdn-all \
      --qsa-dir artifacts/qsa-all \
      --output-dir artifacts/flash-next-overlay

Pythonの `load_flash_next_overlay(model, overlay_dir, base_model_id=..., base_model_revision=...)` は完全なscheduleを厳密ロードしてgraftし、GDN/QSA用 `FlashNextDynamicCache` を返します。最初のcached forward/`generate`にはこのcacheを明示的に `past_key_values` として渡してください。Transformers標準の自動生成cacheはQSA indexer stateを持たず、意図的に非対応です。loaderはruntimeの `config.layer_types` を変更しますが、base modelのMLP/router/embedding/LM head/tokenizerを保存・書換しません。これはadapter overlayであり、`AutoModelForCausalLM.from_pretrained(overlay_dir)` だけでロードできるmerged checkpointではありません。synthetic 4層CausalLMでgraft・generation・cache decodeを検証済みです。full checkpoint repo側には別のcustom `auto_map` loaderがあり、`AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)`で読み込む設計です。synthetic tiny round-trip/cache generationは通過しましたが、実32層base checkpoint自体はload/generateしていません。

## Independent Japanese evaluation

`make-it-flash evaluate` は同じ固定base model/tokenizerを別々にロードして、held-out tokenized JSONLのbase対hybrid next-token perplexityを比較します。category別PPL、tokenizer class/vocab/special IDs、固定した日本語文のencode/decode結果とchat-template token IDs、base revision、overlay hash、evaluation data hashをreportに記録します。校正dataと同一のJSONL/manifest、同じdataset+split、model/tokenizer revision mismatch、vocab外token IDsは拒否します。既定で少なくとも1件のsequence長がQSA pruning threshold（通常2052 tokens）以上であることをGPU/model load前に確認し、短文PPLだけを取る場合は明示的に `--allow-no-qsa-pruning` を指定します。例: `make-it-flash evaluate --data-file artifacts/japanese-heldout/calibration.jsonl --overlay-dir artifacts/flash-next-overlay --output-file artifacts/evaluation/japanese.json`。32B base/hybridを順番にloadし、既定で空きGPU memory 66 GiBを要求します。これはローカル実行用で、HF Jobをsubmitしません。評価は未実施です（2026-10-09時点の作業環境にはCUDA GPUがなく、独立held-out JSONLもありません）。評価を進めるには適切なheld-out dataとGPU環境が必要です。

## Hugging Face Jobs（標準の実行方法）

通常の GPU 実行には [scripts/run_hf_job.sh](scripts/run_hf_job.sh) を使います。ジョブ内で prepare → teacher cache → 1層の GDN fit を順に実行します。既定teacherは `llm-jp/llm-jp-4.1-32b-a3b-thinking` と固定revision `cda260706786758045e5e96bf4d738bbc01155b5`（`MIF_MODEL` / `MIF_MODEL_REVISION` で変更可能）です。別モデルに変更する際は対応するrevisionも指定してください。4.1向けの更新は [SaibaWaipu/make-it-flash](https://github.com/SaibaWaipu/make-it-flash) の `gdn-4.1-pilot` branch にpush済みです。実行例では `MIF_GIT_REF` でこのbranchを選びます。開始時にcloneしたcommit hashをジョブログへ表示します。

### 出力先の private repo ID

`MIF_OUTPUT_REPO` には Hugging Face の `namespace/repo-name` 形式の ID を指定します。たとえば `SaibaWaipu/make-it-flash-pilot` なら、保存先は `https://huggingface.co/SaibaWaipu/make-it-flash-pilot` です。**新規 repo はジョブが private として自動作成するため、事前作成は不要です。**既存 repo を指定する場合は、実行前に private であることを確認してください（既存 repo の公開設定はこのスクリプトでは変更しません）。

HF CLI にログインし、private model repo の作成・書き込み権限がある token を使って実行します。スクリプトの `--secrets HF_TOKEN` がログイン中の token をジョブに渡します。

    hf auth login
    # Default is a cost-estimated dry run; no job is submitted.
    MIF_GIT_REF=gdn-4.1-pilot bash scripts/run_hf_job.sh

Launchは個別の明示承認後に限ります。実行時には per-job 上限 `MIF_APPROVED_BUDGET_USD` に加えて、pilotを含む請求確認済みの累計 `MIF_CONFIRMED_CUMULATIVE_SPENT_USD` を必須入力とし、per-job上限を加えても累計 $9 を超えないことを確認します。launch前にsource commitとteacher revisionが40桁SHAであることも拒否条件つきで検証します。$9は設計・統合枠であり、fine-tuning/post-training予算は別枠で現在$0です。pilot費用はログ上約 $0.27と推定しましたが請求額は未確認なので、その推定値だけでlaunchしないでください。

既定は `a100-large`、timeout 3時間、10K tokens・seq len 1024・1 epoch・最大20 stepsです。runnerは毎回 `hf jobs hardware --json` でrateを取得し、timeoutまで動いた場合の最大compute costを計算します。現時点のHF CLI表示はA100 80GBが $2.50/時で、3時間上限は約 $7.50です。価格変更時はlive rateで再計算します。ジョブはGDN checkpointとfit metricsのみをprivate repoへuploadし、calibration dataとactivation cacheはuploadしません。**dry-runが既定で、このREADMEのコマンドではジョブは起動しません。**

ジョブは公開Git remoteの`gdn-4.1-pilot`をcloneし、必須の`MIF_GIT_COMMIT`と一致することを確認してから学習します。選択imageにgitがなければaptで追加します。ジョブ内でsource codeが実行されHF tokenも渡されるため、信頼できるremote/ref/commitを指定してください。

## Kaggle（任意・旧手順）

[Kaggle notebook](kaggle/make_it_flash.ipynb) はソースを埋め込んだ self-contained 版として残していますが、標準の実行先ではありません。GPU と Internet を要求し、空き VRAM が66 GiB以上の場合だけ teacher cache と fitting に進みます。Kaggle を使う場合の push 手順と注意事項は [Kaggle runbook](kaggle/README.md) を参照してください。

## Outputs and privacy

- artifacts/data/: tokenized calibration.jsonl, data_manifest.json
- artifacts/cache/: sequence ごとの teacher input/output safetensors、任意のQSA block-mass targets、cache_manifest.json
- artifacts/gdn/・artifacts/qsa/: 単独 GDN/QSA の safetensors と fit metrics JSON
- artifacts/flash-next-overlay/: 24+8（32層時）adapter weights、revision/config manifest、SHA-256。base model/tokenizer weightsは含まない

大きな data、cache、weights、Hub cache は [.gitignore](.gitignore) で除外しています。データセット由来のデータや派生物を公開・再配布する場合は、元 dataset の条件を別途確認してください。

## 既知の問題・使用上の制限

以下はコードレビューと小型モデルで再現した問題であり、学習コード全体や公開済みフル重みが破損していると確認したものではありません。修正・回帰テスト・実フルモデル評価を終えるまでは、本snapshotを実運用の推論モデルや品質保証済みモデルとして扱わないでください。

- **推論（QSA・eager attention）:** 左パディングの完全にマスクされたquery行に加算マスクを重ねると、softmaxにNaNが生じます。小型Qwenモデルの`eager`経路でパディング位置のNaNを再現しました（通常のSDPA経路では再現せず、実トークン位置のNaNも未確認）。[実装](src/make_it_flash/flash_next.py)。
- **推論（hybrid cache）:** `FlashNextDynamicCache.crop()`ではQSAのKV/indexerは短縮される一方、GDNのconv/recurrent stateは巻き戻らず、続くトークンの出力が新規prefillと食い違います。現状で`crop()`による巻き戻しは使用しないでください。未初期化cacheの`reset()`も例外になります。直接QSAアダプターを`position_ids=None`のまま逐次呼び出すと、キャッシュ内の位置IDが毎回0になります。[cache実装](src/make_it_flash/hybrid_cache.py)。
- **フル重みexportの再開:** `scripts/export_full_staged_hf_artifact.py`の進捗記録はoverlayのmanifest SHAや全モジュールのhashに紐付いていません。途中で別overlayを指定して再開すると、旧・新overlay由来のshardが混在し得ます。また、出力repoに既存のtokenizer等があっても、固定base revisionとの内容照合なしにスキップします。**別overlayでの再開や、出力repoの既存ファイルを確認せずに再実行することは避けてください。**[export実装](scripts/export_full_staged_hf_artifact.py)。
- **検証の範囲:** fit/assemblyでの局所validation改善は、32層を統合したモデルのperplexity・日本語生成品質・速度を保証しません。フル重みの実ロード、生成、held-out評価、長文推論はいずれも未実施です。

## Validation status

`PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider` で既存テスト106件が通過しました。2026-10-09の段階実行では全32層の局所fit・overlay組立を完了していますが、フル32Bチェックポイントの実ロードと推論は未実施です。固定teacher revision `cda260706786758045e5e96bf4d738bbc01155b5` で実データ100 tokens のprepare smoke testを実行し、35/25/15/15/10のmixを確認しました。tiny Qwen3-MoEでGDN cached decode parityを確認し、2層QSA+GDNのfull/cached decode parityを合成pruning境界（token budget 4、compress ratio 2）越しに検証、prefillとcached next-token双方でtop-k selectorの選択数もassertしました。QSA block selection・dense/compact teacher block-mass loss parity・選択的attention capture・QSA local fitも検証しました。4層synthetic `Qwen3MoeForCausalLM` overlayでgraft後の非attention state（MoE/router、embedding、LM head）保持、明示cacheでの `generate` とincremental decode parity、default Transformers cacheのfail-closed動作を確認しました。overlay assemblyはfit loss改善・独立validationでのselector loss・train/validation両方のpruning実例・checkpoint hash/runtime fingerprintも検証します。cache manifestは校正JSONL・data manifestのSHA-256とdataset ID/revisionを保持し、各shard・fit checkpoint/metricsにも伝播します。assembleは全layerが同じ校正data provenanceを使った場合のみ受け入れます。cacheは短いQSA dataならteacher load前に、両fittersは各shardのmodel ID/revision/calibration hashがmanifestと異なる場合にfailします。fit-qsaはさらに独立train/validation例が不足すればoptimizer前にfailします。Gated Residual、compact PLE/ngram、shared expert、MTP prototypesもCPU shape/gradient testsを通過しています。

2026-10-07のA100 large HF pilotでは、32B teacher重みをloadし、固定revisionから10K tokens / max seq 1024でteacher activationを取得、19 train + 2 validation sequencesでlayer 0を19 steps fittingしました。validation MSEは初期 `5.5507e-4` から `4.9177e-4` に低下（約11.4%）。GDN block safetensors（231,835,448 bytes）とmetricsは[private Hub repo](https://huggingface.co/RemydreScarlet/llm-jp-41-gdn-layer0-pilot-2f9031b)に保存されています。これは初期の単層pilotの記録です。後続の32層局所fit・フル重み統合については上記段階実行節を参照してください。いずれも統合済みモデルの日本語生成品質/PPLの評価ではありません。

## License

このソースコードは [Apache License 2.0](LICENSE) です。モデルとデータセットはそれぞれの配布条件に従ってください。
