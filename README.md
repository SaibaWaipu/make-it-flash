# make_it_flash

LLM-jp 4 32B-A3B の全 Attention を一度に置き換えるのではなく、**小さな calibration mix → teacher の局所 activation cache → 1 層の Gated DeltaNet (GDN) fitting** を分けて試すパイプラインです。元の MoE や tokenizer は書き換えません。

## 重要な範囲

このリポジトリが生成するものは、選択した decoder layer の **単独 GDN block と teacher attention 出力への局所 alignment** です。まだ hybrid Causal LM 全体には統合せず、元モデルをそのままロードできる checkpoint でもありません。GDN + attention の全モデル化、cache-aware generation、global calibration、Harbor RLVR は後段です。

## 1 層 pilot の流れ

1. LLM-jp 4.1 SFT を streaming で読み、下記の mix から最大 100K tokens を chat template で token 化する。元 dataset の複製はリポジトリに入れない。
2. 十分な GPU VRAM がある場合だけ BF16 の teacher を読み、指定 attention 層の正規化済み入力と attention 出力を safetensors に保存する。
3. Transformers の Qwen3NextGatedDeltaNet だけを MSE teacher forcing で学習する。teacher と他層は freeze のまま。

初期 mix は want メモの目標値に合わせています: Japanese 35%、English 25%、Math 15%、Code 15%、Tool/Agent 10%。各区分の出典 config は src/make_it_flash/data.py に固定し、split・seed・model/dataset revision を manifest に記録します。複数のソースからなる区分では、その token quota をソース間でほぼ均等に割ります。

### 必要環境

- Python 3.10+
- pip install -e '.[test]'
- activation cache と fitting は CUDA GPU が必要。既定で cache 前に **66 GiB 以上の空き GPU memory** を検査し、足りない場合は teacher weights をダウンロードせず停止します。32B BF16 teacher のための安全側 preflight で、実際の空き領域や attention backend により A100 80GB でも調整が必要な場合があります。
- 手元の GPU が不足する場合は Kaggle GPU notebook (kaggle/) を先に試してください。Kaggle 側も必要な VRAM を満たさなければ cache は意図的に停止します。HF Jobs の A100 は有料なので、起動前に最新料金と出力先を確認してください。ここから有料 job は自動起動しません。

## Local / Kaggle 実行

    cd make_it_flash
    python -m pip install -e '.[test]'

    # 最初は小さく、data-only の準備
    make-it-flash prepare --max-tokens 100000 --max-seq-len 2048

    # activation cache: 初期は layer 0 だけ。VRAM preflight が必ず走る。
    make-it-flash cache --layers 0

    # GDN local alignment
    make-it-flash fit --layer 0 --epochs 3 --max-steps 200

出力は artifacts/data/, artifacts/cache/, artifacts/gdn/ に作られます。大きな data/activation/checkpoint artifact は .gitignore 対象です。Kaggle 用 notebook を再生成するときは python scripts/build_kaggle_notebook.py、Kaggle に送るときは kaggle/kernel-metadata.json の id を自分の Kaggle username に直し、kaggle kernels push -p kaggle を実行します。Notebook は GPU + Internet を要求し、空き VRAM が 66 GiB 以上のときだけ teacher cache と GDN fitting を実行します。それ未満では重みを読まずにデータ準備で止まります。

## Hugging Face Jobs fallback

Kaggle GPU が 66 GiB preflight を通らない場合の有料 fallback です。ローカル Git remote を用意してから、MIF_GIT_URL と MIF_OUTPUT_REPO (出力先は新規作成を含め private model repo) を設定し、bash scripts/run_hf_job.sh を実行します。既定 flavor は a100-large、timeout は 4h です。実行前に最新料金を必ず確認してください。HF_TOKEN は Jobs secret として渡し、calibration/cache は upload せず、GDN checkpoint と metrics だけを private repo へ送ります。この repository から Jobs は自動起動しません。

## 出力と再開

- calibration.jsonl: category/config/sample id/token ids を持つ短い系列
- data_manifest.json: dataset/model revision と token quota
- sample_*.safetensors: sequence ごとの teacher input/output (BF16) と ID metadata
- cache_manifest.json: 対象 layer と GDN dimension 設定
- gdn_layer_NN.safetensors: local-fit した単独 GDN block
- fit_layer_NN.json: train/validation MSE、steps、モデル revision

データの出典ごとに個別 license があるため、使用・公開前に元 dataset のライセンス条件を確認してください。Activation cache と fitted weight も自動アップロードしません。

## まず行う検証

    pytest
    python -m make_it_flash.cli --help

fit の CPU 実行は小さな unit test 用だけです。実モデルで CPU fit しないでください。モデルの生成能力や perplexity を計測する評価段階はまだ含まず、この 1 層 pilot の MSE は architecture conversion の成功判定そのものではありません。
