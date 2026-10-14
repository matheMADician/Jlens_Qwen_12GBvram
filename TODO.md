# For research
確認 dataset 對 Jlens 的影響後再繼續。

# For codebase
1. (DONE) 完成 data/TODO.md
2. (DONE) 修改 trainer和Inference 成支援 chat template
2.5 (DONE) 換成 Qwen2.5-Omni
3. (DONE) 訓練時產生並保存兩組 JacobianLens，研究資料數量對 JLens 的影響。
    A. 每個 dataset 各自一個完整 dataset lens。
    B. 對資料固定 seed 打亂後，按資料數量逐步累積並保存 lens（例如
       1、2、5、10、20、50、...）；數量清單需可設定。
    實作步驟：
    1. 明確固定 Jacobian 計算設定：fit mode、source/target token 範圍、
       source/target layers、序列長度上限與 token averaging 規則。
    2. 保留每筆樣本的 sample id 與 dataset name，並明確定義 B 組的抽樣
       範圍（全體資料或各 dataset 分別）、各數量的含義，以及超過可用
       樣本數時的處理方式。
    3. 以固定 seed 對預定範圍內的樣本做可重現的 shuffle；各數量使用同一
       shuffled order 的前 N 筆，形成巢狀子集，讓不同 N 的 lens 可比較。
    4. 一次計算每筆有效樣本的 Jacobian，增量更新對應 dataset accumulator
       與樣本數 accumulator；在指定 N 保存資料量實驗 lens，並保存每個
       dataset 的完整 lens。lens 的 n_prompts 必須是實際成功納入的樣本數。
    5. 輸出需記錄 sample ids / dataset names、shuffle seed、樣本數設定、
       實際納入數、模型與 fitting 設定；對被跳過或失敗的樣本明確記錄，
       避免請求的 N 和實際納入數混淆。
    6. 支援中斷後續跑且不重複累計；保存足夠的進度狀態，確保 dataset
       lens 與不同 N 的 lens 來自相同的成功樣本及既定順序。
    7. 加測試：shuffle 可重現、不同 N 子集巢狀、樣本數/metadata 正確、
       跳過與 resume 不重複，以及 per-dataset lens 可用
       JacobianLens.merge() 組合。先估算 dataset lens 的儲存需求。

4. (中) 實作pass@k AUC

5. (中) 根據 Short Horizons and Sparse Concepts. 先用平均和標準差建構一個高斯 
   分布，然後計算 KL散度(之後可以加入更多估計方法)

* Random projection 暫緩，先與學長討論投影方法與實驗設計；討論完成後
   再決定是否實作 per-sample projected Jacobian 輸出。

# Miscellaneous
1. (難) 將dataset抽樣寫入master
2. (中) 實作 Linear probing