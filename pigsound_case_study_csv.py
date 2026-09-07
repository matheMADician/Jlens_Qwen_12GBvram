"""
case_study.py
================
展示 Pure ASR / Mixed / Pure QA 三個 J-Lens，加上傳統 logit-lens baseline，
在同一筆 held-out 樣本的同一組位置上，各自讀出什麼 top-k token，
並加上 Top-K 的機率總和 (Probability Sum) 顯示信心度。

用途：把「zero-shot 泛化」「沒有災難性遺忘」這些行為層級的統計發現，
具體變成「這裡有一個活生生的例子長這樣」的質化案例，適合直接放進報告當 case study。

放在跟 phase4_fit.py 同一個資料夾底下執行：
    python case_study.py
"""

import os
import sys
import json
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
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

CASE_STUDIES = [
    {"jsonl": "jlens_val_dataset/test.jsonl", "index": 3, "label": "QA held-out example"},
]

LAYERS_TO_SHOW = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
TOP_K = 5
MAX_SEQ_LEN = 600
OUTPUT_CSV = "justpig_case_study_results.csv"

def find_example(records, index=None, keyword=None):
    if keyword is not None:
        for line in records:
            record = json.loads(line)
            if keyword.lower() in str(record.get("transcript", "")).lower():
                return line
        raise ValueError(f"在這份 jsonl 裡找不到 transcript 含有關鍵字 {keyword!r} 的樣本")
    return records[index if index is not None else 0]

def decode_top_tokens_and_probs(tokenizer, logits_row, k):
    """
    將 logits 轉換為機率，取出 Top-K Tokens，並計算這 K 個 token 的機率總和
    回傳格式例如: "hello | world | test (sum: 0.85)"
    """
    probs = F.softmax(logits_row, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, k)
    
    tokens = [tokenizer.decode([tid]).strip() for tid in topk_indices.tolist()]
    prob_sum = topk_probs.sum().item()
    
    token_str = " | ".join(tokens)
    result_str = f"{token_str} (sum: {prob_sum:.2f})"
    
    return result_str

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

    common_layers = sorted(set.intersection(*(set(l.source_layers) for l in lenses.values())))
    layers = [l for l in LAYERS_TO_SHOW if l in common_layers]
    missing = [l for l in LAYERS_TO_SHOW if l not in common_layers]
    if missing:
        print(f"[注意] 這些層不在三個 lens 的共同 source_layers 裡，略過: {missing}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as csvfile:
        csv_writer = csv.writer(csvfile)
        
        header = [
            "Case Label", "Case ID", "Transcript", 
            "Position", "Input Token", "Model True Prediction (Top K + Prob Sum)", 
            "Layer", "Baseline (logit-lens)", "Pure ASR", "Mixed 50/50", "Pure QA"
        ]
        csv_writer.writerow(header)

        for case in CASE_STUDIES:
            jsonl_path = os.path.join(DATA_ROOT, case["jsonl"])
            records = load_jsonl_lines(jsonl_path)
            json_line = find_example(records, index=case.get("index"), keyword=case.get("keyword"))
            record = json.loads(json_line)

            case_label = case['label']
            case_id = record.get('id', 'N/A')
            transcript = record.get('transcript', '')

            print("\n" + "=" * 90)
            print(f"Processing: {case_label}  |  id={case_id}  |  answer={transcript!r}")
            print("=" * 90)

            adapter = Qwen2AudioLensModel(
                full_model=full_model, processor=processor, data_root=DATA_ROOT,
                device="cuda", model_dtype=torch.bfloat16,
            )

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
            
            tokenizer = adapter.tokenizer
            seq_len = input_ids.shape[1]
            input_tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

            audio_start_idx = None
            audio_end_idx = None

            # 1. 尋找音訊區間的起訖點
            for i, token in enumerate(input_tokens):
                if "<|audio_bos|>" in token:
                    audio_start_idx = i
                elif "<|audio_eos|>" in token:
                    audio_end_idx = i

            # 2. 均勻抽取 10 個音訊位置
            if audio_start_idx is not None and audio_end_idx is not None:
                start = audio_start_idx + 1
                end = audio_end_idx - 1
                audio_len = end - start + 1
                
                if audio_len >= 10:
                    step = (audio_len - 1) / 9.0
                    positions_to_show = [int(start + i * step) for i in range(10)]
                else:
                    positions_to_show = list(range(start, end + 1))
                
                
                print(f"\n🎧 偵測到音訊區間: 索引 {audio_start_idx} ~ {audio_end_idx}")
                print(f"📌 均勻抽取 {len(positions_to_show)} 個觀察位置: {positions_to_show}")

            else:
                fallback_start = min(5, seq_len - 1)
                fallback_end = min(15, seq_len)
                positions_to_show = list(range(fallback_start, fallback_end))
                print(f"\n⚠️ 未偵測到音訊邊界，盲測前段位置: {positions_to_show}")
            
            #ans_positions_to_show = sorted(set([seq_len - 5, seq_len - 4, seq_len - 3, seq_len - 2, seq_len - 1]))
            #positions_to_show = positions_to_show + ans_positions_to_show
                
            positions_to_show = sorted(set(range(audio_start_idx+1, audio_end_idx-1, 5)))
            
            positions_to_show = [p for p in positions_to_show if 0 <= p < seq_len]

            for pos in positions_to_show:
                input_token_str = input_tokens[pos]
                actual_top_str = decode_top_tokens_and_probs(tokenizer, model_logits[pos], TOP_K)
                
                print(f"  --> Pos {pos} (Token: {input_token_str!r}) processing...")

                for layer in layers:
                    row = [
                        case_label,
                        case_id,
                        transcript,
                        pos,
                        input_token_str,
                        actual_top_str,
                        layer
                    ]
                    
                    for col_name in ["Baseline (logit-lens)", "Pure ASR", "Mixed 50/50", "Pure QA"]:
                        top_str = decode_top_tokens_and_probs(tokenizer, readouts[col_name][layer][pos], TOP_K)
                        row.append(top_str)
                    
                    csv_writer.writerow(row)
    
    print(f"\n✅ 所有結果已成功輸出至 {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
