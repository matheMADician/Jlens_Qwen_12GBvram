"""
case_study.py
================
展示 Pure ASR / Mixed / Pure QA 三個 J-Lens，加上傳統 logit-lens baseline，
在同一筆 held-out 樣本的同一組位置上，各自讀出什麼 top-k token。

用途：把「zero-shot 泛化」「沒有災難性遺忘」這些行為層級的統計發現，
具體變成「這裡有一個活生生的例子長這樣」的質化案例，適合直接放進報告當 case study。

放在跟 phase4_fit.py 同一個資料夾底下執行：
    python case_study.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

from phase4_fit import Qwen2AudioLensModel, load_jsonl_lines
from jlens.lens import JacobianLens

DATA_ROOT = "g_data"
MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"

LENS_PATHS = {
    "Pure ASR": "checkpoints/phase4_asr_lens.pt",
    "Mixed 50/50": "checkpoints/phase4_mixed_lens.pt",
    "Pure QA": "checkpoints/phase4_qa_lens.pt",
}

# 每個 case 可以用 "index"（第幾筆）或 "keyword"（搜尋 transcript 裡包含這個關鍵字的
# 第一筆，方便挑一個內容比較有記憶點的例子，不用盲猜 index）。
CASE_STUDIES = [
    {"jsonl": "jlens_val_dataset/val_prompts_asr.jsonl", "index": 0, "label": "ASR held-out example"},
    {"jsonl": "jlens_val_dataset/val_prompts_qa.jsonl", "index": 0, "label": "QA held-out example"},
    # 範例：想找 transcript 裡有 "dog" 的 QA 樣本，可以改成：
    # {"jsonl": "jlens_val_dataset/val_prompts_qa.jsonl", "keyword": "dog", "label": "QA: dog bark"},
]

LAYERS_TO_SHOW = [0, 8, 16, 20, 24, 28, 30]
TOP_K = 3
MAX_SEQ_LEN = 300


def find_example(records, index=None, keyword=None):
    if keyword is not None:
        for line in records:
            record = json.loads(line)
            if keyword.lower() in str(record.get("transcript", "")).lower():
                return line
        raise ValueError(f"在這份 jsonl 裡找不到 transcript 含有關鍵字 {keyword!r} 的樣本")
    return records[index if index is not None else 0]


def decode_top_tokens(tokenizer, logits_row, k):
    topk = torch.topk(logits_row, k)
    return [tokenizer.decode([tid]).strip() for tid in topk.indices.tolist()]


def main():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    full_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map={"": 0},
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    full_model.eval()
    for p in full_model.parameters():
        p.requires_grad_(False)

    lenses = {name: JacobianLens.load(path) for name, path in LENS_PATHS.items()}
    for name, lens in lenses.items():
        print(f"[{name}] {lens!r}")

    # 三個 lens 的 source_layers 理論上應該一致（都是 fit 全部 32 層），
    # 這裡still 用交集，避免哪個 lens 剛好層數不同時直接爆掉。
    common_layers = sorted(set.intersection(*(set(l.source_layers) for l in lenses.values())))
    layers = [l for l in LAYERS_TO_SHOW if l in common_layers]
    missing = [l for l in LAYERS_TO_SHOW if l not in common_layers]
    if missing:
        print(f"[注意] 這些層不在三個 lens 的共同 source_layers 裡，略過: {missing}")

    for case in CASE_STUDIES:
        jsonl_path = os.path.join(DATA_ROOT, case["jsonl"])
        records = load_jsonl_lines(jsonl_path)
        json_line = find_example(records, index=case.get("index"), keyword=case.get("keyword"))
        record = json.loads(json_line)

        print("\n" + "=" * 90)
        print(f"{case['label']}  |  id={record.get('id')}  |  answer={record.get('transcript')!r}")
        print("=" * 90)

        adapter = Qwen2AudioLensModel(
            full_model=full_model, processor=processor, data_root=DATA_ROOT,
            device="cuda", model_dtype=torch.bfloat16,
        )

        # 四種讀出方式：三個 lens 的 J-Lens 讀出 + 傳統 logit-lens baseline。
        # baseline 用哪個 lens 物件呼叫都一樣，因為 use_jacobian=False 時完全不會用到 J_l。
        readouts = {}
        model_logits = None
        input_ids = None
        for name, lens in lenses.items():
            logits, model_logits, input_ids = lens.apply(
                adapter, json_line, layers=layers, positions=None,
                max_seq_len=MAX_SEQ_LEN, use_jacobian=True,
            )
            readouts[name] = logits

        baseline_logits, _, _ = list(lenses.values())[0].apply(
            adapter, json_line, layers=layers, positions=None,
            max_seq_len=MAX_SEQ_LEN, use_jacobian=False,
        )
        readouts["Baseline (logit-lens)"] = baseline_logits
        readout_names = list(lenses.keys()) + ["Baseline (logit-lens)"]

        tokenizer = adapter.tokenizer
        seq_len = input_ids.shape[1]
        input_tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

        # 挑序列尾端幾個位置看——最接近實際答案內容的地方，最能看出各 lens 的差異
        positions_to_show = sorted(set([seq_len - 5, seq_len - 3, seq_len - 2, seq_len - 1]))
        positions_to_show = [p for p in positions_to_show if 0 <= p < seq_len]

        for pos in positions_to_show:
            print(f"\n--- position {pos}  (輸入 token: {input_tokens[pos]!r}) ---")
            actual_top = decode_top_tokens(tokenizer, model_logits[pos], TOP_K)
            print(f"  模型真實預測 (最後一層): {actual_top}")
            for layer in layers:
                parts = []
                for name in readout_names:
                    top = decode_top_tokens(tokenizer, readouts[name][layer][pos], TOP_K)
                    parts.append(f"{name}: {top}")
                print(f"  layer {layer:>2} | " + "  |  ".join(parts))


if __name__ == "__main__":
    main()
