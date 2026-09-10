"""
test1 : evn, model, vram
prepare================================================
已依「環境建構建議」安裝好 CUDA / PyTorch / transformers(source)等套件
準備一段測試音訊檔

How to use================================================
    python phase1_env_and_model_load_test.py --test-audio-path <test audio> 
"""

import os
import sys
import time
import argparse

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"


# ----------------------------------------------------------------------
# Step 0. 依賴檢查
# ----------------------------------------------------------------------
def check_dependencies():
    print("=" * 70)
    print("[Step 0] 依賴檢查")
    print("=" * 70)

    missing = []
    versions = {}

    try:
        import torch
        versions["torch"] = torch.__version__
        if not torch.cuda.is_available():
            print("[FATAL] 找不到可用的 CUDA GPU，請確認驅動與 CUDA 安裝。")
            sys.exit(1)
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  GPU: {gpu_name} | 總 VRAM: {total_vram_gb:.1f} GB")
    except ImportError:
        missing.append("torch")

    for pkg in ["transformers", "accelerate", "librosa", "soundfile"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"[FATAL] 缺少套件: {missing}")
        sys.exit(1)

    print("  已安裝套件版本：")
    for k, v in versions.items():
        print(f"    - {k}: {v}")
    print()


# ----------------------------------------------------------------------
# Step 1. 準備測試音訊（若無現成檔案，合成一段 smoke-test 用的音訊）
# ----------------------------------------------------------------------
def ensure_test_audio(path: str, duration_sec: float = 5.0, sr: int = 16000):
    if os.path.exists(path):
        print(f"[Step 1] 使用既有測試音訊: {path}")
        return path

    print(f"[Step 1] 找不到 {path}，合成一段 {duration_sec}s 正弦波作為管線 smoke test（非語義測試）")
    import numpy as np
    import soundfile as sf

    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    tone = 0.1 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    sf.write(path, tone, sr)
    return path


# ----------------------------------------------------------------------
# Step 2. 載入模型
# ----------------------------------------------------------------------
def load_model():
    print("=" * 70)
    print("[Step 2] 載入 Qwen2-Audio-7B-Instruct")
    print("=" * 70)

    import torch
    from transformers import (
        Qwen2AudioForConditionalGeneration,
        AutoProcessor,
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # attn_implementation="flash_attention_2" 可選用（需另外安裝 flash-attn，
    # RTX 3060 屬 Ampere 架構相容）；若未安裝則保留預設 "sdpa" 即可。
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()

    load_time = time.time() - t0
    vram_after_load = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"  模型載入完成，耗時 {load_time:.1f}s")
    print(f"  載入後 VRAM 峰值: {vram_after_load:.2f} GB")
    print()

    return model, processor


# ----------------------------------------------------------------------
# Step 3. 最小可行推論測試（音訊 + 文字指令 -> 文字回應）
# ----------------------------------------------------------------------
def run_inference_smoke_test(model, processor, audio_path: str):
    print("=" * 70)
    print("[Step 3] 音訊 + 文字指令推論測試")
    print("=" * 70)

    import torch
    import librosa

    audio, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": "Describe the audio content"},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )

    inputs = processor(
        text=text_prompt,
        audio=[audio],
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=128)

    gen_time = time.time() - t0
    vram_after_gen = torch.cuda.max_memory_allocated() / (1024 ** 3)

    generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(f"  生成耗時: {gen_time:.1f}s")
    print(f"  推論階段 VRAM 峰值: {vram_after_gen:.2f} GB")
    print(f"  模型輸出: {response!r}")
    print()

    return vram_after_gen


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="test 1")
    parser.add_argument("--test-audio-path", default="test_audio.wav")
    args = parser.parse_args()
    
    check_dependencies()
    audio_path = ensure_test_audio(args.test_audio_path)
    model, processor = load_model()
    peak_vram = run_inference_smoke_test(model, processor, audio_path)

    print("=" * 70)
    print("[結論] test 1 完成")
    print("=" * 70)
    print(f"  本次推論階段 VRAM 峰值: {peak_vram:.2f} GB")


if __name__ == "__main__":
    main()
