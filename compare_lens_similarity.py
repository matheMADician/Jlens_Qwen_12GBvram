"""
compare_lens_similarity.py
=============================
比較三個已經 fit 好的 J-Lens（Pure ASR / Mixed / Pure QA）在每一層的相似度。

純 CPU 運算，不需要載入 Qwen2-Audio 模型，幾分鐘內就能跑完全部 31 層 x 3 組配對。

三種指標，回答不同層次的問題：
  1. cosine similarity（矩陣攤平成向量算）—— 主要指標，忽略整體尺度差異，
     只看「方向」像不像。
  2. 相對 Frobenius 距離 —— 輔助指標，對尺度敏感，僅供交叉參考。
  3. top-k 奇異子空間的 principal angles —— 機制性指標，只看矩陣「真正有
     影響力」的前 k 個奇異方向是否重疊，比整體 cosine similarity 更能
     回答「這兩個 lens 是不是在乎同一組方向」。

需要先 `pip install scipy`（用來算 principal angles，如果還沒裝的話）。

執行：
    python compare_lens_similarity.py
"""

import argparse
import torch
import numpy as np
from scipy.linalg import subspace_angles

from jlens.lens import JacobianLens


def cosine_sim(A: torch.Tensor, B: torch.Tensor) -> float:
    a = A.flatten().double()
    b = B.flatten().double()
    return torch.dot(a, b).item() / (a.norm().item() * b.norm().item())


def relative_frobenius(A: torch.Tensor, B: torch.Tensor) -> float:
    return (A - B).norm().item() / A.norm().item()


def cosine_sim_to_identity(A: torch.Tensor) -> float:
    """J_l 跟單位矩陣 I 的 cosine similarity。因為 I 攤平後只有對角線是 1、
    其餘全 0，這個內積只會抓到 A 的對角線元素，數學上等於
    trace(A) / (sqrt(D) * ||A||_F)——衡量這個 transform 離『完全不做事』
    （也就是傳統 logit-lens 的隱含假設 J_l=I）有多遠。
    不用子空間比較（principal angle）跟 I 比：I 的所有奇異值都相等，
    沒有『前 k 個主要方向』這種東西，比子空間角度沒有意義。"""
    D = A.shape[0]
    trace = torch.diagonal(A).double().sum().item()
    return trace / ((D ** 0.5) * A.double().norm().item())


def diagonal_energy_ratio(A: torch.Tensor) -> float:
    """輔助指標：這個矩陣的『總能量』(Frobenius norm^2) 有多少比例落在對角線上，
    不管對角線的值是不是接近 1。跟 cosine_sim_to_identity 的差異在於：
    這個只看『對角線 vs 非對角線』的能量分布，不管對角線本身的值對不對齊到 1。"""
    diag_energy = torch.diagonal(A).double().pow(2).sum().item()
    total_energy = A.double().pow(2).sum().item()
    return diag_energy / total_energy



def top_k_subspace_angles(A: torch.Tensor, B: torch.Tensor, k: int) -> np.ndarray:
    """回傳兩個矩陣前 k 個右奇異向量（V，也就是「輸入方向」）張成的子空間之間
    的 principal angles（弧度），由小到大排序。角度越接近 0，代表子空間重疊
    程度越高；角度接近 pi/2 代表兩個子空間幾乎正交（完全不同方向）。"""
    # svd_lowrank 比完整 SVD 快很多，對只需要前 k 個方向的情境很合適
    _, _, Va = torch.svd_lowrank(A.double(), q=k + 10, niter=4)
    _, _, Vb = torch.svd_lowrank(B.double(), q=k + 10, niter=4)
    Va = Va[:, :k].numpy()
    Vb = Vb[:, :k].numpy()
    return subspace_angles(Va, Vb)  # 長度 k 的陣列，弧度


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-lens", default="checkpoints/phase4_asr_lens.pt")
    parser.add_argument("--mixed-lens", default="checkpoints/phase4_mixed_lens.pt")
    parser.add_argument("--qa-lens", default="checkpoints/phase4_qa_lens.pt")
    parser.add_argument("--top-k", type=int, default=50,
                         help="比較 principal angles 時取前幾個奇異方向")
    parser.add_argument("--out-csv", default="outputs/phase6/lens_similarity.csv")
    args = parser.parse_args()

    lenses = {
        "ASR": JacobianLens.load(args.asr_lens),
        "Mixed": JacobianLens.load(args.mixed_lens),
        "QA": JacobianLens.load(args.qa_lens),
    }
    for name, lens in lenses.items():
        print(f"[{name}] {lens!r}")

    layers = sorted(set.intersection(*(set(l.source_layers) for l in lenses.values())))
    pairs = [("ASR", "Mixed"), ("ASR", "QA"), ("Mixed", "QA")]

    rows = []
    print(f"\n{'layer':>5} | " + " | ".join(f"{a}-{b} cos" for a, b in pairs) +
          " | " + " | ".join(f"{a}-{b} relFrob" for a, b in pairs) +
          f" | 前{args.top_k}方向平均principal angle(度)" +
          " | cos-to-I (ASR/Mixed/QA)")
    print("-" * 160)

    for layer in layers:
        row = {"layer": layer}
        cos_vals, frob_vals, angle_vals = [], [], []

        for a, b in pairs:
            Ja = lenses[a].jacobians[layer]
            Jb = lenses[b].jacobians[layer]

            cos = cosine_sim(Ja, Jb)
            frob = relative_frobenius(Ja, Jb)
            row[f"cos_{a}_{b}"] = cos
            row[f"relfrob_{a}_{b}"] = frob
            cos_vals.append(cos)
            frob_vals.append(frob)

            angles = top_k_subspace_angles(Ja, Jb, args.top_k)
            mean_angle_deg = np.degrees(angles.mean())
            row[f"meanangle_deg_{a}_{b}"] = mean_angle_deg
            angle_vals.append(mean_angle_deg)

        # 每個 lens 各自跟單位矩陣 I 比較：這個 transform 離「什麼都不做」多遠
        identity_cos_vals = []
        for name in lenses:
            J = lenses[name].jacobians[layer]
            cos_i = cosine_sim_to_identity(J)
            diag_ratio = diagonal_energy_ratio(J)
            row[f"cos_to_I_{name}"] = cos_i
            row[f"diag_energy_ratio_{name}"] = diag_ratio
            identity_cos_vals.append(cos_i)

        rows.append(row)
        cos_str = " | ".join(f"{v:.3f}" for v in cos_vals)
        frob_str = " | ".join(f"{v:.3f}" for v in frob_vals)
        angle_str = " | ".join(f"{v:.1f}°" for v in angle_vals)
        identity_str = " | ".join(f"{v:.3f}" for v in identity_cos_vals)
        print(f"{layer:>5} | {cos_str} | {frob_str} | {angle_str} | {identity_str}")

    # 存成 CSV，方便之後畫圖或進一步分析
    import csv
    import os
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n結果存到 {args.out_csv}")

    print("\n解讀提示：")
    print("  - cosine similarity 接近 1 代表矩陣整體方向高度一致；接近 0 代表幾乎無關")
    print("  - principal angle 接近 0 度代表兩個 lens 最重要的方向高度重疊；")
    print("    接近 90 度代表最重要的方向幾乎正交（完全不同）")
    print("  - 如果整體 cosine similarity 不高、但 principal angle 很小，代表兩個矩陣")
    print("    在『無關緊要的方向』上有差異，但在『真正影響輸出的方向』上高度一致——")
    print("    這種情況能直接解釋『矩陣看起來不太一樣，但下游準確率卻很像』的現象")
    print("  - cos_to_I 接近 1 代表這個 lens 幾乎等於『什麼都不做』（等同傳統 logit-lens）；")
    print("    越接近 0（或負值）代表這個 lens 做了越多實質的座標轉換。")
    print("    預期：淺層需要的轉換較大（cos_to_I 較低），深層應該逐漸接近 1")
    print("    ——如果深層 cos_to_I 仍然很低，值得回頭檢查 fit 過程是否有異常")


if __name__ == "__main__":
    main()
