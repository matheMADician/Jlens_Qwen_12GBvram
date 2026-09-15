import os
import json
import gc
import io
import subprocess
import sys
import traceback
import librosa
import soundfile as sf
from datasets import Audio, load_dataset
import warnings

# 忽略 librosa 的一些警告
warnings.filterwarnings("ignore")

# ==========================================
# 參數設定
# ==========================================
TARGET_EN = 50       # 英文需求數量
TARGET_ZH = 50       # 中文需求數量
MIN_DUR = 2.0        # 最短 2 秒
MAX_DUR = 5.0        # 最長 5 秒
TARGET_SR = 16000    # Qwen2-Audio 的標準採樣率

OUT_DIR = "jlens_dataset"
AUDIO_DIR = os.path.join(OUT_DIR, "audio")
JSONL_PATH = os.path.join(OUT_DIR, "prompts.jsonl")

os.makedirs(AUDIO_DIR, exist_ok=True)

PROMPT_TEMPLATE = "<|audio_bos|><|AUDIO|><|audio_eos|>請寫出這段音訊的逐字稿："

def save_sample(idx, lang, audio_array, sr, text):
    """將音訊存成本地 wav，並回傳 metadata 字典"""
    if sr != TARGET_SR:
        audio_array = librosa.resample(y=audio_array, orig_sr=sr, target_sr=TARGET_SR)
    
    file_name = f"{lang}_{idx:03d}.wav"
    file_path = os.path.join(AUDIO_DIR, file_name)
    
    sf.write(file_path, audio_array, TARGET_SR)
    
    return {
        "id": str(idx),
        "language": lang,
        "audio_path": os.path.relpath(file_path, OUT_DIR),
        "transcript": text,
        "prompt": PROMPT_TEMPLATE
    }

def decode_audio(audio):
    """Decode one sample with soundfile instead of datasets/torchcodec."""
    if audio.get("bytes") is not None:
        audio_array, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    elif audio.get("path") is not None:
        audio_array, sr = sf.read(audio["path"], dtype="float32")
    else:
        raise ValueError("FLEURS audio sample has neither bytes nor path")
    return audio_array, sr

def run_worker(language: str, target: int, id_offset: int = 0):
    configs = {
        "EN": ("en_us", "美式英文"),
        "ZH": ("cmn_hans_cn", "簡體中文"),
    }
    config, label = configs[language]
    metadata_path = os.path.join(OUT_DIR, f"metadata_{language}.jsonl")
    records = []

    print(f"\n[{language}] 正在串流搜尋 FLEURS ({label})...", flush=True)
    dataset = load_dataset("google/fleurs", config, split="train", streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    count = 0
    for sample in dataset:
        if count >= target:
            break

        sample_record = dict(sample)
        audio, sr = decode_audio(sample_record["audio"])
        text = sample_record["transcription"]
        duration = len(audio) / sr

        if MIN_DUR <= duration <= MAX_DUR:
            record = save_sample(id_offset + count, language, audio, sr, text)
            records.append(record)
            count += 1
            clean_text = text.replace("\n", " ")
            print(
                f"  ✓ 取得 {language} 樣本 {count}/{target} "
                f"(長度: {duration:.1f}s) | 內容: {clean_text[:15]}...",
                flush=True,
            )

        del sample, sample_record, audio
        gc.collect()

    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        for record in records:
            metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[{language}] 完成，共 {len(records)} 筆。", flush=True)


def run_main():
    print("=" * 60)
    print("🎧 Phase 2: Qwen2-Audio J-Lens 雙語資料集 (FLEURS 多樣化百科)")
    print("=" * 60)

    for language, target, offset in (("EN", TARGET_EN, 0), ("ZH", TARGET_ZH, TARGET_EN)):
        subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker", language, str(target), str(offset)],
            check=True,
        )

    records = []
    for language in ("EN", "ZH"):
        metadata_path = os.path.join(OUT_DIR, f"metadata_{language}.jsonl")
        with open(metadata_path, encoding="utf-8") as metadata_file:
            records.extend(json.loads(line) for line in metadata_file if line.strip())

    with open(JSONL_PATH, "w", encoding="utf-8") as metadata_file:
        for record in records:
            metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("=" * 60)
    print(f"🎉 資料集建立完成！共 {len(records)} 筆。")
    print(f"📁 音訊檔已儲存至: {AUDIO_DIR}/")
    print(f"📄 清單檔已儲存至: {JSONL_PATH}")
    print("=" * 60)


def main():
    if len(sys.argv) >= 5 and sys.argv[1] == "--worker":
        try:
            run_worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
        except Exception:
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)

        # The datasets/torchcodec native cleanup can abort during interpreter
        # finalization. The worker has already flushed its output files here.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    else:
        run_main()

if __name__ == "__main__":
    main()