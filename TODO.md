## 程式碼庫工作

### 增加資料集
加入以下資料集:

   https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0/tree/main
   https://huggingface.co/datasets/facebook/voxpopuli
   https://huggingface.co/datasets/bernardocecchetto/
   https://huggingface.co/datasets/jspaulsen/esd
   https://huggingface.co/datasets/renumics/emodb

- 先做:
   1. 人類語音分析
      https://huggingface.co/datasets/jspaulsen/esd
      這個資料集有分 中/英  男/女 情緒 和 speaker id
      將這些都標註後訓練，做 t-SNE，然後對同一張圖做不同coloring
   2. 環境音分析
      https://zenodo.org/records/4060432
      因為標籤特別多，因此幫環境音和動物個別挑選一些，然後訓練
      之後可以用那些label做 t-SNE

- 步驟:
   1. 修改 get_data (完成)
   2. 修改 assemble_json.py (完成)
   3. 修改有用到 "transcript" 行的 main flow (完成)
      - t-SNE embedding 保留每筆資料的 label，供後續查閱 (完成)
      - pass@k 可在 CLI 選擇 transcript 或 scalar label 欄位 (完成；目前維持 first-token 評估)
      - 其餘部分如果也有用到，則再討論


### 視覺化 prediction
對單一prompt的每個TOKEN位置和layer都做Jlens預測。然後用那個prompt對應的label在各處的預測機率畫成熱力圖。完成後觀察它的特性，並決定實驗有甚麼可以修改的。

### 雜項
1. 整合曹祖恩的工作(他push到github，但本地沒有事先上傳)

2. 擴充 `create-model-settings`，讓它涵蓋更多設定。

3. 整理程式碼庫。

4. 整理日誌輸出，並加入進度條。

## 待進行的實驗
### 更多dataset
- 可以考慮先等pass@k和其他指標出來後再決定，或是放在一邊跑
- 之前有想過要復刻原文中資料數目對pass@k的影響，但資料集分布的問題比想像中大，這部分就先不做了。

### comp_Jlens
1. 針對目前訓練出的不同dataset做各種比較

### pass@k
1. 先完成雜項的第一點，然後交叉比對不同dataset的Jlens在自己以及在其他工作的表現，交叉比對
- 文字 vs 環境音
- 不同語言
2. 一些考量:
   目前pass@k 計算時有加文字prompt，但很可能會有引導作用。比如說，當文字出現"emotion"之類的文字時，各種emotion的分數都會被拉高。或者，某些golden的文字可能在某一刻出現在Next-token prediction的選項內，儘管他沒有被讀出。因此我想到有兩種解法
- 不加文字prompt，但某些模型不熟悉的工作(emotion, 音檔為英文/中文時的language)可能讀不出
- 加文字prompt，但要對其他錯誤答案做某種標準化
3. 目前規劃:
   1. 先想出一個量化 pass@k 的標準。原文的 pass@k 是取某個位置所有layer的最高值，但我們認為這在語音不太可行。目前的 pass@k 是一個逐層的折線圖，但沒辦法寫成矩陣比較。因此可能要用某種能適當獎勵淺層讀出的總和方法。目前想到幾種。
   - 用(總層數-層數) x pass@kAUC的總和來比較
   - 用pass@kAUC首次達到0.5的層數
   - 其他(可補充)

   2. evaluate方法: 先不加文字prompt，並棄用 SAKURA。要做一個 train-lens × test-dataset matrix，分別幫 1. 文字 vs 環境音 2. 不同語言 整理出一個表，然後把它填齊。
4. 未來:
   1. 雖然文字prompt造成的影響很小，但也可以把它加入前面的比較。也可以思考一下文字prompt的語言如何選取和解釋。
   2. 目前比較兩個lens有很多指標。可以思考如何把他們整理成單一一個，可供判斷兩個lens是否有差異。

### Copilot remarks

1. 先明確研究問題：目前的 pass@k 主要衡量「Jlens 是否能讀出與任務相關的資訊」，不等同於模型完成任務的正確率。若要主張任務表現，應另報任務層級指標。
2. 主要比較前先確認訓練與測試資料沒有來源重疊；若 SAKURA 確實含有訓練資料來源，建議只列為可能有重疊的診斷結果，不作為主要泛化測試。可行時以錄音來源或說話者為單位切分。
3. 文字／語音與環境音、不同語言分開整理矩陣；任務及答案形式不同時，不直接合併原始分數。
4. 主要實驗固定簡短、各 lens 共用的任務提示，避免在提示中列出候選答案；無文字提示可作為配對的提示敏感度測試，不預設它一定較公平。
5. 逐層曲線應保留為主要結果。若需要單一矩陣數值，先定義並正規化深度加權的逐層 pass@k AUC；「首次達到 0.5 的層數」較適合作為次要指標，並說明未達門檻時如何處理。
6. 完整填表前，先用少數 lens 和乾淨測試資料做 pilot，確認提示、答案 token 化及摘要指標都合理；若可行，按說話者／錄音來源估計不確定性。
