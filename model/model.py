import Jlens.lens as Lens, Jlens.Jlens as J, Jlens.LogitLens as L
import instance as Inst
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from loguru import logger
import os, sys, time, torch, librosa, soundfile

class Model:
    def __init__(self, lens_path: str = None, do_4bit: bool = False, 
                 MODEL_ID="Qwen/Qwen2-Audio-7B-Instruct"):
        self.model_id = MODEL_ID
        self.do_4bit = do_4bit
        """
        Automatically loads the model.
        lens_path: Default None, automatically apply LogitLens. Auto-switch to Jlens after calc_lens() is called.
        """
        if do_4bit:
            try:
                self.model, self.processor = load_model_4bit(MODEL_ID)
                print("Loading model using 4bit")
            except Exception as e:
                print("Failed Loading model({e}). Exiting...")
                exit(1)
        else:
            try:
                self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.bfloat16,
                    device_map={"": 0},
                )
                self.processor = AutoProcessor.from_pretrained(MODEL_ID)
            except Exception as e:
                print("Failed Loading model({e}). Exiting...")
                exit(1)

        self.model.eval()
        for p in full_model.parameters():
            p.requires_grad_(False)
        n_requires_grad = sum(p.requires_grad for p in full_model.parameters())
        logger.info("requires_grad=True 參數數量: %d（預期為 0）", n_requires_grad)
        assert n_requires_grad == 0

        if lens_path is None:
            self.lens = L.LogitLens(self.model)
        else:
            self.lens = J.Jlens(self.model)

    def get_model(self):
        return self.model, self.processor

    def test_model(self, audio_path: str = "test_audio.wav"):
        run_inference_smoke_test(self.model, self.processor, audio_path)

    def load_lens(self, lens_path: str):
        pass

    def run(self, audio, sampling_rate: int):
        instance = Inst.Instance(self.model, self.processor, audio, sampling_rate)
        pass

def load_model_4bit(MODEL_ID: str):
    print("=" * 70)
    print("[Step 2] 以 4-bit (NF4) 量化載入 Qwen2-Audio-7B-Instruct")
    print("=" * 70)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # attn_implementation="flash_attention_2" 可選用（需另外安裝 flash-attn，
    # RTX 3060 屬 Ampere 架構相容）；若未安裝則保留預設 "sdpa" 即可。

    #TODO: 加入相容其他模型的選項
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    load_time = time.time() - t0
    vram_after_load = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"  模型載入完成，耗時 {load_time:.1f}s")
    print(f"  載入後 VRAM 峰值: {vram_after_load:.2f} GB")
    print()

    return model, processor

def run_inference_smoke_test(model, processor, audio_path: str):
    print("=" * 70)
    print("[Step 3] 音訊 + 文字指令推論測試")
    print("=" * 70)

    audio, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": "請描述這段音訊的內容。"},
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