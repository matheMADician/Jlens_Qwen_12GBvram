# For research
確認 dataset 對 Jlens 的影響後再繼續。

# For codebase
Jlens的讀出效果明顯比Qwen2的差
1. (DONE) 用 Qwen2.5-Omni 做小規模對照 fit：沿用 Qwen2 實驗使用的固定 text prompt；其他設定、音訊樣本及樣本順序盡量保持一致，透過 `fitting.experiment.max_prompts` 限制樣本數，並使用全新的 checkpoint。
2. 用相同音訊子集和 fit 設定重訓 Qwen2，改用 Qwen2 原生 chat template；與先前未使用 chat template 的結果比較，隔離 prompt 格式的影響。
3. 若前兩項結果仍有差異，再用相同音訊子集比較 Qwen2 與 Qwen2.5-Omni 的固定 prompt 和 Llama-generated prompt；每次只改變 prompt 條件，並使用獨立 checkpoint。
4. 比較各組的讀出效果及 Jacobian cosine/CKA 數值矩陣；記錄各模型實際使用的 chat template 和 token 序列。將熱圖視為相似性診斷，不單憑區塊形狀推論 Transformer 結構或原因。

# Miscellaneous
1. (難) 將dataset抽樣寫入master
2. (中) 實作 Linear probing