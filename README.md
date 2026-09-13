# JLens-Qwen Audio

本專案嘗試將 Anthropic 的 Jacobian Lens（JLens）應用到
Qwen2-Audio 語音語言模型，分析音訊與文字輸入在 language model
不同層的 residual representation。

目前專案仍在開發中。底層的 JLens adapter 已完成主要設計，高層的
model wrapper 與實驗控制流程則會在取得雲端 GPU 算力後繼續補齊與驗證。

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

目前 JLens 的輸入是 JSONL。每行至少需要包含：

```json
{
  "id": "sample-001",
  "audio_path": "audio/sample.wav",
  "prompt": "請描述這段音訊的內容。"
}
```

建議資料集結構如下：

```text
dataset/
├── prompts.jsonl
└── audio/
    └── sample.wav
```

其中 `audio_path` 是相對於 dataset 根目錄的路徑。

JLens 在 fitting 或 applying 前，需要先讓 `Instance` 載入資料根目錄：

```python
instance.load_data("dataset")
```

在目前的 wrapper 設計中，這項工作由 `Jlens.calc_lens()` 或
`Jlens.apply_lens(data_root=...)` 負責。

## 目前預計使用方式

以下是目前設計的使用方向；完整流程尚未在雲端 GPU 上完成 smoke test。

### 建立模型

```python
from model.model import Model

model = Model(
    MODEL_ID="Qwen/Qwen2-Audio-7B-Instruct",
    do_4bit=False,
)
```

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
- 修正並驗證 `fit_Jlens()` 的實際呼叫流程
- 在 GPU 上測試 JLens fitting 的 batch、記憶體與 checkpoint resume
- 驗證 `do_activate_Jacobian=True/False` 兩種輸出是否都能正確保存

### Model wrapper

- 完成 `get_model_info()`
- 明確區分 raw Hugging Face model 與 JLens `Instance`
- 完成一般語音 generation API
- 決定並實作 `run_inference()` 的正式語意
- 補上 cache 的 `.detach().cpu()` 策略
- 補上 cache 讀取與清除的公開 API
- 修正 package import 路徑並確認不同啟動方式都能使用

### Master 與啟動流程

- 完成 `Master` 的 initialization 與實驗控制 API
- 修正 `main.py` 目前與 `Master` 類別名稱不一致的問題
- 從 `settings.json` 讀取資料集、checkpoint 與實驗參數
- 決定 fitting、JLens apply、Logit Lens apply 與一般 generation
  的執行順序
- 加入實驗結果輸出格式

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

