# stroop：聲音 vs. 語意的 J lens

負責人：RaTa0314　｜　branch：`stroop`　｜　進度見 [docs/TODO.md](docs/TODO.md)

## 問題

Stroop 效應：用藍色墨水寫「紅」，字義和顏色互相衝突。
LALM 的版本：**女生的聲音說「he」或「我是男生」**。

J lens 是在某個資料分佈上平均出來的 Jacobian。換一種任務去 fit，讀出的方向可能不同。

> 同一批音訊，用「問性別」和「轉寫」兩種 prompt 各 fit 一個 lens。
> 在模型沒被問到性別時，**性別 lens** 讀出聲音的性別、**轉寫 lens** 讀出內容的性別嗎？

## 實驗

| # | 做什麼 | 看什麼 |
|---|---|---|
| 0 | Pilot：用主幹現有的 lens，**只看 audio 位置**，用單跳題問性別 | 讀不讀得到性別。讀不到就停 |
| 1 | 用 SAKURA 音訊 fit 性別 lens 和轉寫 lens，各自再切 A/B 兩半當 noise baseline | — |
| 2 | 逐層比較兩個 lens 的 male/female 方向 cosine | 差異有沒有大過 noise |
| 3 | LibriSpeech 衝突刺激加上轉寫 prompt，兩個 lens 各讀一次 | 一個讀到聲音、一個讀到內容？ |
| 4 | 因果 swap：分別沿兩個 lens 的方向改 | 只有對應的答案改變？ |

## 資料

| 資料 | 用途 |
|---|---|
| **SAKURA GenderQA**（500 筆） | Pilot；fit 兩個 lens（同音訊、不同 prompt） |
| **LibriSpeech** 2×2 | 衝突測試和 swap（說話者性別 ≠ 內容性別，例如女聲念「he said」） |

SAKURA 在這裡被拿來 fit，所以 stroop 的結果**不能**跟主幹在 SAKURA 上的分數直接比。

## 執行

在 repo 根目錄執行：`python -m stroop.<script>`。

- `data_prep/`：產生實驗資料的程式（資料本身存在 `data/datasets/stroop/`，不 commit）
- `experiments/`：實驗程式（結果存在 `output/stroop/`，不 commit）

## 依賴主幹的程式

`data/get_data.py`、`Model._token_directions`、`Model.run_swap_inference`、`Model.apply_lens`、`audio_jlens/subset_fitting.py`。
**改到這些函式時請通知我。**

## 文件（`docs/`）

- [TODO.md](docs/TODO.md)：待辦清單和結果紀錄，每項都寫了用意。
- [AGENTS.md](docs/AGENTS.md)：給 AI 看的完整細節（背景論文、主幹 API、設計考量、各步驟規格）。人不用讀。

## 學長feedback
suggest some paper to survey
swap exp is important to make differecne between linear probe.