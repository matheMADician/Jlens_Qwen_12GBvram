# For research
1. 和助教開會
    1. 看起來文字模態對Jlens的影響巨大，要如何寫prompt比較好是個問題。
    2. 又或者現在的方法其實比較好，因為變量只有Audio，或許就是為甚麼會有好的讀出效果?

# For codebase
1. (易)把settings.json整理成用工作分類
2. (難)將dataset抽樣併入master的方法中
3. (中)實作ablation(如果官方有實作，難度為中，如果沒有，難度為難)
4. (中)實作CKA相似度(CKA的計算函數已經在tools/math_tools)。
    決定某一個(唯一的)測試測資，用他各層的activation作CKA_heatmap
5. (DONE)實作奇異值分解。需要做的有(deepseek)
    1. 把每層的奇異值譜畫成熱力圖（縱軸層 × 橫軸奇異值索引）
    2. 鄰近層主奇異向量(top-k?)相似度的比較
6. (DONE)修改Jlens的創建，使得不需要每次都load model(重要)
7. (重要)寫一個greedy decoding的推理
8. 嘗試讓模型詳細描述音訊內容，並在Jlens訓練時應用。
    -這樣attention路徑可以分成三類 1. Audio 2. Prompt 3. Inference
9. (易)用entropy畫圖(entropy的計算已經有了)
10. (易)仿造別人論文，算中間層經過Jlens映射後和最終層的相似度。

# Miscellaneous

