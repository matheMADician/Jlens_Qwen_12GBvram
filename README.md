# This is Outdated.

# JLens-Qwen Audio

本專案嘗試將 Anthropic 的 Jacobian Lens（JLens）應用到
Qwen2-Audio 語音語言模型，分析音訊與文字輸入在 language model
不同層的 residual representation。

目前專案仍在開發中，但已支援 JLens fitting、套用 JLens/Logit Lens、
輸出 top-k CSV，以及比較一個或兩個 JLens checkpoint 的 layer similarity
heatmap。

## 專案架構

```text
Qwen2AudioForConditionalGeneration
        |
        v
model/instance.py
  Qwen2-Audio -> JLens LensModel adapter
        |
        v
Jlens/Jlens.py
  JLens fitting / loading / applying
        |
        v
model/model.py
  高層模型 facade 與 lens output cache
        |
        v
Master.py
  預計作為實驗流程 wrapper
```

## 目前已完成

### JLens adapter

[`model/instance.py`](model/instance.py) 目前負責：

- 載入 Qwen2-Audio 模型與 processor
- 提供 JLens 所需的：
  - `n_layers`
  - `d_model`
  - `layers`
  - `tokenizer`
  - `encode()`
  - `forward()`
  - `unembed()`
- 解析 JSONL prompt
- 依照 `audio_path` 載入音訊
- 將 processor 產生的 audio features 暫存在 instance
- 以 audio feature cache 支援 JLens 後續重複執行 language model
- 提供 `load_data()` 設定資料集根目錄

目前 `Instance` 一次綁定一筆音訊。JLens fitting 需要複製 batch
時，會將同一筆音訊 feature expand 到 batch 維度。

### JLens wrapper

[`Jlens/Jlens.py`](Jlens/Jlens.py) 已改為直接使用 `Instance`：

- `Jlens` 建構子接收一個已初始化的 `Instance`
- `calc_lens()` 會先呼叫 `Instance.load_data()`
- JLens fitting 使用 `Instance.encode()`、`Instance.forward()` 與
  `Instance.unembed()`
- `apply_lens()` 支援 Jacobian Lens 與 vanilla Logit Lens 模式
- 支援保存與載入 lens checkpoint

### Model wrapper

[`model/model.py`](model/model.py) 已開始作為高層 facade：

- 透過 `Instance` 建立 Qwen2-Audio 模型
- 保存 raw Hugging Face model 與 processor
- 提供 JLens fit/load/apply 的入口
- 提供 `LensOutputCache`
- 分別保存最近一次的：
  - JLens 結果
  - Logit Lens 結果

每筆 cache 結果的格式為：

```python
(
    lens_logits,
    model_logits,
    input_ids,
)
```

## 資料格式

`data/get_data.py` 輸出的 JSONL 使用模型無關的 `messages` 對話格式，
並以相對於 dataset 根目錄的路徑引用音訊：

```json
{
  "id": "sample-001",
  "audio_path": "audio/sample.wav",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "audio", "path": "audio/sample.wav"},
        {"type": "text", "text": "Dataset-provided instruction"}
      ]
    }
  ]
}
```

`--prompt-column` 可指定 user message 使用的 dataset prompt 欄位；若未指定，
user message 的文字內容會是空字串。這樣可供後續 LLM 填入 prompt。
`--transcription` 指定頂層 `transcript` 欄位來源，會自動偵測常見的
`transcription`、`raw_transcription` 或 `transcript` 欄位。這和 prompt 欄位用途不同，
也避免自動把 emotion、gender 等分類標籤當作回答。
逐字稿只存在頂層 `transcript`，不會放進 `messages` 的 assistant 內容。
產生器保留頂層 `transcript` 欄位供目前的 model loader 使用，標準新格式以
`messages` 為準。

建議資料集結構如下：

```text
dataset/
├── prompts.jsonl
└── audio/
    └── sample.wav
```

其中 `audio_path` 是相對於 dataset 根目錄的路徑。

### 從 Hugging Face FLEURS 下載英文資料

`data/get_data.py` 可以直接下載 FLEURS。以下範例下載美式英文資料：

```bash
python data/get_data.py \
  --dataset google/fleurs \
  --config en_us \
  --split train \
  --target 100 \
  --output data/fleurs_en_us \
  --transcription transcription \
  --overwrite
```

其他英文 config 可使用 `en_gb`、`en_au`、`en_in` 或 `en_ng`。
每個 config 請輸出到不同資料夾。

JLens 在 fitting 或 applying 前，需要先讓 `Instance` 載入資料根目錄：

```python
instance.load_data("dataset")
```

在目前的 wrapper 設計中，這項工作由 `Jlens.calc_lens()` 或
`Jlens.apply_lens(data_root=...)` 負責。

## 使用方式

### 設定檔

`settings.json` 按照用途分組：

```json
{
  "model": {
    "id": "Qwen/Qwen2-Audio-7B-Instruct",
    "adapter": "qwen2_audio",
    "load_in_4bit": false
  },
  "data": {
    "root": "data",
    "jsonl_path": "data/esc50-50_fleurs_en_us-50.jsonl"
  },
  "training": {
    "max_seq_len": 300
  },
  "checkpoints": {
    "root": "output/checkpoints",
    "load_path": "output/checkpoints/esc50-50_fleurs_en_us-50/lens.pt"
  },
  "inference": {
    "jsonl_path": "data/prompts.jsonl",
    "max_length": 300,
    "output_root": "output/inference",
    "test_jsonl_path": "data/for_inference.jsonl",
    "top_k": 5
  }
}
```

執行前請確認 `data.root`、`data.jsonl_path` 和
`checkpoints.load_path` 指向目前要使用的資料與 checkpoint。
程式會在 `fitting.checkpoint_root`、`inference.output_root` 及實驗輸出根目錄下
自動加入模型 ID 最後一段作為子目錄，例如 `Qwen/Qwen2-Audio-7B` 會使用
`<output_root>/Qwen2-Audio-7B`。

### 建立模型

```python
from model.model import Model

model = Model(
    MODEL_ID="Qwen/Qwen2-Audio-7B-Instruct",
    adapter="qwen2_audio",
    do_4bit=False,
)
```

Model weights load lazily: `Model(...)` only configures the wrapper. Training and
inference load the model on first use and reuse that instance afterward. To
preload explicitly, call `model.load_model()`.

### Fitting JLens

```python
model.fit_Jlens(
    jsonl_path="dataset/prompts.jsonl",
    dim_batch=32,
    checkpoint_interval=5,
    MAX_SEQ_LEN=300,
    run_name="qwen2-audio-experiment",
)
```

### 套用 lens

```python
result = model.apply_lens(
    json_line=json_line,
    data_root="dataset",
    do_activate_Jacobian=True,
    MAX_SEQ_LEN=300,
)
```

結果預期為：

```python
lens_logits, model_logits, input_ids = result
```

設定 `do_activate_Jacobian=False` 時，可使用相同入口執行
vanilla Logit Lens baseline，並保存到另一份 cache。

### 使用 Master 執行

`main.py` 使用子命令選擇要呼叫的 `Master` 方法：

```bash
python main.py eval-jlens --lens-path path/to/lens.pt --prompts-path path/to/prompts.jsonl
python main.py generate --prompts-path path/to/prompts.jsonl --max-new-tokens 128 --run-name generated
python main.py info-jlens --lens-path path/to/lens.pt
python main.py comp-jlens --lens-path1 path/to/lens1.pt --lens-path2 path/to/lens2.pt
python main.py calc-jlens-experiment --run-name experiment
python main.py test-inference
python main.py test-training --run-name test-fit
python main.py test-lens
```

分別對應 `Master.eval_Jlens()`、`generate()`、`info_Jlens()`、`comp_Jlens()`、
`calc_Jlens_experiment()` 和三個 testing methods。評估 JLens 時，可以用
`--layer` 指定單一層、`--top-k` 設定 CSV 顯示的候選 token 數量，並用
`--run-name` 指定輸出名稱；不指定 layer 時會評估所有可用層。每個 prompt
只執行一次 `apply_lens`，同時計算 JLens 與 LogitLens 的 pass@k。CSV 同時包含兩種
預測；JSON 報告在頂層提供 JLens 指標，並在 `logit_lens` 欄位提供 LogitLens
指標。報告也會在 `by_layer` 下提供各層的獨立指標，並輸出
`<run-name>_passk_auc_by_layer.png` 比較 JLens 與 LogitLens 的 normalized AUC。
每層分數仍在該層的所有 prompt positions 中取最佳答案 token rank；因此這是
layer-wise、但仍採 oracle-position 的結果。

```bash
python main.py eval-jlens --run-name evaluate_run --layer 22 --top-k 10
```

`generate` 會使用模型的 greedy `generate()` 對每個音訊 prompt 產生文字，
並將 prompt、參考 transcript 和生成內容寫入
`output/inference/<model>/<run-name>/generated.csv`。未指定 prompt 路徑時使用
`inference.jsonl_path`；未指定生成長度時最多生成 128 個 token。

`test-training` 會執行測試 fitting，並且必須提供 `--run-name`。其他使用
`settings.json` 作為輸入的測試命令不需要額外參數。執行 `python main.py --help`
或 `python main.py <command> --help` 可查看完整選項。

若一個 Slurm job 需要連續執行多個 workflow，先建立只在該 job 使用的模型
設定副本，再將路徑傳給每個命令。原始 `settings.json` 不會被修改：

```bash
TEMP_SETTINGS="$(mktemp)"
trap 'rm -f "$TEMP_SETTINGS"' EXIT

python main.py create-model-settings \
  --model-name Qwen/Qwen2-Audio-7B \
  --output-settings "$TEMP_SETTINGS"

python main.py info-jlens \
  --settings-file "$TEMP_SETTINGS" \
  --run-name audio_analysis \
  --lens-path output/checkpoints/Qwen2-Audio-7B/train_1_no_SAKURA/experiment/lens_full.pt
python main.py eval-jlens \
  --settings-file "$TEMP_SETTINGS" \
  --run-name audio_eval \
  --lens-path output/checkpoints/Qwen2-Audio-7B/train_1_no_SAKURA/experiment/lens_full.pt \
  --prompts-path data/assembled/test_SAKURA_AnimalQA/prompts.jsonl
```

`Master.create_model_settings_file()` 會複製來源設定，只更新 `model.id` 與
`model.adapter`，並寫到指定路徑。`create-model-settings` CLI 命令提供相同功能；
model name 必須是程式目前支援的 adapter 名稱。

在 Slurm 環境中，也可以使用：

```bash
sbatch job.sh
squeue --me
scontrol show job <JOBID>
cat slurm-<JOBID>.out
```

### 繪製 JLens heatmap

`Master.draw_Jlens_heatmap()` 會比較 Jacobian matrices 本身，不是 activation
CKA。預設使用 linear CKA，也可以指定 `mode="cos"` 使用 Jacobian matrices
的 cosine similarity。單一 JLens 會產生 layer × layer 比較；指定第二個
checkpoint 時，會產生 `lens1 layer × lens2 layer` 比較圖。計算會在 CUDA
可用時自動使用 GPU，否則使用 CPU；也可以透過 `device` 參數指定裝置。
分析結果會放在
`output/experiments/<model_name>/single_lens/<run_name>/` 或
`output/experiments/<model_name>/two_lens/<run_name>/`。模型名稱會從
`model.id` 自動取得。`info_Jlens()` 的 cosine、CKA 和 SVD 圖都存於
`single_lens/`；`comp_Jlens()` 的比較表、JSON 報告、cosine similarity、
relative difference 和 subspace alignment 圖存於 `two_lens/`。pass@k 與
inference 輸出仍使用各自原本設定的輸出根目錄。

pass@k 會在每個 prompt 的所有輸入位置中尋找答案 token 的最低排名；指定
`layer` 時只評估該層，未指定時則在所有可用層中取最佳排名。執行
`python main.py eval-jlens --layer <layer>` 可指定層。

```python
from Master import Master

master = Master(run_name="heatmap")
heatmap = master.draw_Jlens_heatmap(
  lens_path1="output/checkpoints/esc50-50_fleurs_en_us-50/lens.pt",
  lens_path2="output/checkpoints/esc50-50_libri-50/lens.pt",
)
```

使用 cosine similarity：

```python
heatmap = master.draw_Jlens_heatmap(
  lens_path1="output/checkpoints/first/lens.pt",
  lens_path2="output/checkpoints/second/lens.pt",
  mode="cos",
)
```

回傳的 `heatmap` 是 shape 為
`[len(lens1_layers), len(lens2_layers)]` 的 PyTorch tensor。

### 中文 CSV 顯示

模型輸出會寫入 `inference.output_root`，CSV 使用 UTF-8 with BOM，方便
Excel 正確辨識中文。修改輸出編碼後，請重新產生 CSV；既有 CSV 不會自動
補上 BOM。

## 尚待完成與驗證

以下項目暫時保留，等雲端算力可用後再處理。

### 高優先度

- 在實際 Qwen2-Audio 模型上完成單筆：
  - `encode()`
  - `forward()`
  - `unembed()`
  - shape 與 dtype 驗證
- 確認 processor 產生的 audio token 數量與 projector output
  token 數量一致
- 確認目前 prompt 是否需要套用 Qwen2-Audio chat template
- 在 GPU 上測試 JLens fitting 的 batch、記憶體與 checkpoint resume
- 擴充不同模型的 `LensModel` adapter

### Model wrapper

- 完成 `get_model_info()`
- 明確區分 raw Hugging Face model 與 JLens `Instance`
- 完成一般語音 generation API
- 決定並實作 `run_inference()` 的正式語意
- 完成 `get_model_info()`

### Master 與啟動流程

- 完成更完整的 fitting/apply/inference CLI 選項
- 加入 heatmap 與 inference 的自動化實驗輸出

### 其他限制

- 目前主要針對 Qwen2-Audio，尚未抽象化成通用多模態模型 adapter
- 目前 `Instance` 一次只支援一筆不同的音訊
- 尚未支援平行處理多個不同資料根目錄
- 尚未建立完整測試套件

## 環境安裝

安裝腳本位於 [`dependencies.sh`](dependencies.sh)。

目前預期環境包括：

- Python 3.10
- PyTorch 2.5.1
- CUDA 12.1
- Transformers（source version）
- bitsandbytes
- accelerate
- librosa
- soundfile
- jacobian-lens

安裝腳本目前以 Linux、Conda 與 NVIDIA CUDA 環境為主要目標。

```bash
bash dependencies.sh
```

實際執行前請確認 CUDA、NVIDIA driver、Conda 與模型下載權限已設定完成。

## 開發原則

- 不修改第三方 `jacobian-lens` 的核心 API
- 讓 `Instance` 實作 JLens 的 `LensModel` protocol
- 將模型載入與模型能力集中在 `Model`
- 將實驗流程放在 `Master` 或未來新增的 experiment module
- 不在模型初始化時自動執行實驗
- 先完成單筆 smoke test，再進行完整 fitting
