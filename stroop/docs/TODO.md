# stroop TODO

格式：`- [ ] 事情（負責人，日期）-> 用意`，做完打勾，結果寫一行在底下。

## 現在

- [x] 寫 `get_stroop_libri.py`：LibriSpeech + 說話者性別 + 內容性別 -> 取得「聲音和內容性別不同」的測資
- [ ] 在 server 上跑 `get_stroop_libri.py`（需要 `SPEAKERS.TXT`）-> 確認每一格都收得滿
- [x] 寫 `get_sakura_gender.sh` + `make_sakura_sets.py` -> 步驟 0、1 的資料（pilot 兩份、fit 兩種任務 × all/A/B）
- [ ] 在 server 上跑上面兩支 -> 確認 500 筆、A/B 各 250 筆、男女平衡（看 `metadata.json`）
- [x] 確認 SAKURA GenderQA 有沒有單跳題的欄位 -> pilot 要直接問性別，不要多跳題
  - 有：`single_instruction`（選擇題 (a) Male (b) Female），500 筆，只有 test split
- [ ] 問同學：server 上跑過哪些 lens 和 pass@k？有沒有 `S2T` 模式的 lens？ -> 避免重做，看能沿用什麼

## 實驗

名詞：**主幹 lens** = 主幹現有的 lens（只用在 pilot）；**性別 lens** / **轉寫 lens** = 用 SAKURA 同一批音訊，分別配「問性別」和「轉寫」prompt fit 的 lens。

- [ ] 0. Pilot -> 看聲音的性別讀不讀得到，讀不到就換方向（先確認跟吳席綸工作的重疊部分）
  - [x] 寫 `experiments/pilot_readout.py`
  - [ ] 在 server 上先用 `--max-prompts 20` 試跑，再跑完整的 -> 確認程式在真模型上能動
  - 做法：SAKURA GenderQA，直接問性別。用主幹 lens，**只在 audio 位置**看 male/female 的 rank
  - 判斷：audio 位置排名高 → 繼續；只有文字位置讀得到 → 可能只是讀到答案
- [ ] 1. Fit 性別 lens 和轉寫 lens -> 兩個 lens 只差在任務（先確認跟吳席綸工作的重疊部分）
  - 做法：SAKURA 500 筆，配問性別 prompt fit 一個、配轉寫 prompt fit 一個，都用 `S2T` 模式
  - 問性別的 prompt 用 `prompts/ask_gender.txt`，**不用** SAKURA 原本的選擇題（題目裡有 Male/Female 字樣，lens 可能學到題目文字，而不是聲音）
  - 再把 500 筆切成 A、B 兩半（250/250），兩種 lens 各 fit A、B -> 量「同任務 fit 兩次」本來就有多少差異（noise baseline）
- [ ] 2. `compare_directions.py` -> 判斷兩個 lens 拆不拆得開，決定題目可不可行（先確認跟吳席綸工作的重疊部分）
  - 做法：取 male/female/man/woman 的 J vector，逐層算 cos（性別 lens vs. 轉寫 lens），和 cos（A vs. B）比
  - 判斷：前者明顯比後者低 → 拆得開；差不多 → 兩種性別是同一個概念
- [ ] 3. `conflict_readout.py` -> 驗證一個 lens 讀到聲音、另一個讀到內容
  - 做法：LibriSpeech 女聲念「he...」這類音訊，prompt 只要求轉寫（不問性別），兩個 lens 在 audio 位置各讀一次
  - 判斷：性別 lens 讀出 female、轉寫 lens 讀出 male → 假設成立
- [ ] 4. Swap -> 證明是因果，不只是讀得出來
  - 做法：沿性別 lens 的方向把 female 換成 male，再沿轉寫 lens 的方向做一次
  - 判斷：前者只改變「問性別」的答案，後者只改變轉寫內容 → 成立

## 待決定

- [ ] 用哪個模型：Qwen2-Audio 還是 Qwen2.5-Omni？ -> 所有比較的 lens 都必須來自同一個模型
- [ ] 什麼時候整合回主幹？ -> 等結果穩定，避免跟同學的改動打架

## 結果紀錄

（日期：一句話結論）
