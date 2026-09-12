import Jlens.lens as Lens, Jlens.Jlens as J, Jlens.LogitLens as L
import instance as Inst
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from loguru import logger
import os, sys, time, torch, librosa, soundfile

class Model:
    def __init__(self, do_4bit: bool = False, MODEL_ID="Qwen/Qwen2-Audio-7B-Instruct"):
        self.model_id = MODEL_ID
        self.do_4bit = do_4bit
        self.lens = L.LogitLens()
        """
        Master class of all model methods.
        Automatically loads the model.
        LogitLens is the default. To load Jlens, use load_lens(), and to calculate Jlens, use calc_Jlens().
        """
        if do_4bit:
            try:
                self.model, self.processor = load_model_4bit(MODEL_ID)
                print("Loading model using 4bit")
            except Exception as e:
                raise RuntimeError("Failed Loading model({e}). Exiting...")
                
        else:
            try:
                self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.bfloat16,
                    device_map={"": 0},
                )
                self.processor = AutoProcessor.from_pretrained(MODEL_ID)
            except Exception as e:
                raise RuntimeError("Failed Loading model({e}). Exiting...")

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        n_requires_grad = sum(p.requires_grad for p in self.model.parameters())
        logger.info("requires_grad=True 參數數量: %d（預期為 0）", n_requires_grad)
        assert n_requires_grad == 0

    def get_model(self):
        return self.model, self.processor

    def test_model(self, audio_path: str = "test_audio.wav"):
        #TODO Add a default testing audio path!
        if not os.path.isfile(audio_path):
            raise RuntimeError("Audio path does not exist.")
        run_inference_smoke_test(self.model, self.processor, audio_path)

    def load_lens(self, lens_path: str | None = None, name: str | None = None):
        self.lens = J.Jlens(self.model, self.processor)
        self.lens.load_lens(path = lens_path, name = name)

    def calc_Jlens(self, jsonl_path: str, dim_batch: int | None = None,
                   checkpoint_interval: int | None = None, MAX_SEQ_LEN: int | None = None,
                   do_replace: bool = False, checkpoint_save_path: str | None = None,
                   run_name: str | None = None):
        """
        * Calls the calc_lens() function in Jlens.
        * See ~/Jlens/Jlens.py for detail.
        * jsonl_path accepts the relative path to jsonl files.
        """
        # Parse the jsonl_path into data path(parent directory) and jsonl name
        if not os.path.isfile(jsonl_path):
            raise RuntimeError(f"Jsonl file({jsonl_path}) does not exist.")
        parent_dir, jsonl_name = os.path.split(jsonl_path)

        self.lens = J.Jlens(self.model, self.processor)
        self.lens.calc_lens(self, data_path= parent_dir, jsonl_name= jsonl_name, model_name= self.model_id, do_replace= do_replace, 
                            checkpoint_path= checkpoint_save_path, run_name= run_name, dim_batch= dim_batch,
                            MAX_SEQ_LEN= MAX_SEQ_LEN, checkpoint_every= checkpoint_interval)

    def encode_audio(self, audio_path: str):
        pass

    def apply_lens(self, audio_path):
        instance = Inst.Instance(self.model, self.processor)
        pass

    def run_inference(self):
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