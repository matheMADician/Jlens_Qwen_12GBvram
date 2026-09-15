from jacobian_lens.jlens.protocol import LensModel
import parser as J
import json, os, logging, time
import librosa, torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class Instance(LensModel):
    """
    Qwen2-Audio 的 adapter，給 JLens 使用。
    職責：
      - encode(json_line)：把 prompts.jsonl 的一行變成 input_ids，
        並把該樣本的 audio features 快取起來。
      - forward(input_ids)：跑 LLM residual stack，回傳 last_hidden_state。
      - unembed(residual)：final norm + lm_head，把 residual 解碼成 logits。

    注意：這個 adapter 一次只綁定「一個」音檔（batch=1）。
    forward() 裡的 .expand(dim_batch, ...) 是把「同一個音檔」複製到
    整個 batch，不是處理多個不同音檔。
    """
    def __init__(self, MODEL_ID: str, do_4bit: bool = False ):
        # Root directory of data. Jlens methods rely on the absolute path, while the
        # input json uses relative path
        self.data_root = None

        # Load Model
        self.hf_model, self.processor = load_model(MODEL_ID= MODEL_ID, do_4bit= do_4bit)
        self.LALM_model = getattr(self.hf_model, "model", self.hf_model)

        # Model parameters
        self.sampling_rate = self.processor.feature_extractor.sampling_rate
        self.device = self.LALM_model.get_input_embeddings().weight.device
        self.hf_model_dtype = self.LALM_model.get_input_embeddings().weight.dtype
        lm_config = self.LALM_model.language_model.config

        # Audio feature cache
        self.cached_audio_features = None
        
        # Variable for Anthropic's protocol
        self.n_layers = lm_config.num_hidden_layers
        self.d_model = lm_config.hidden_size
        self.layers = self.LALM_model.language_model.layers
        self.tokenizer = self.processor.tokenizer
        
    def clear_audio_cache(self):
        self.cached_audio_features = None

    def get_model(self):
        return self.hf_model, self.processor

    def get_model_info(self):
        pass

    def load_data(self, data_root: str):
        """
        * Accepts relative path to data_root.
        """
        if not data_root:
            raise ValueError("data_root 不能是空字串")
        self.data_root = os.path.abspath(os.path.expanduser(data_root))
        if not os.path.isdir(self.data_root):
            raise FileNotFoundError(
                f"資料根目錄不存在：{self.data_root}"
            )

    # ------------------------------------------------------------------
    # JLens 介面：encode / forward / unembed
    # ------------------------------------------------------------------
    def encode(self, text: str, *, max_length: int = 300) -> torch.Tensor:
        """
        * Loads one line from the jsonl file, and parse all the audio.
        * Calls the processor to encode the audio files.
        * Returns the tokenized (Audio + text prompt)
        """
        if self.data_root is None:
            raise RuntimeError("No data root directory specified. Please call load_data() first.")

        self.clear_audio_cache()
        parser = J.Parser()

        sample_id, audio_rel_path, text_prompt = parser.parse_json_line(text)
        if not sample_id.isdecimal():
            raise ValueError(f"Sample id must be decimal({sample_id})")
        sample_id = int(sample_id)
        
        audio_path = os.path.join(self.data_root, audio_rel_path)
        audio_array = parser.load_audio(sample_id= sample_id,
            audio_path= audio_path, sampling_rate= self.sampling_rate)

        inputs = self.processor( text=text_prompt, audio=[audio_array],
            sampling_rate=self.sampling_rate, return_tensors="pt" )
        input_ids = inputs["input_ids"]

        if input_ids.shape[1] > max_length:
            raise ValueError(
                f"樣本 {sample_id} 長度 {input_ids.shape[1]} 超過 "
                f"max_length={max_length}，安全跳過。"
            )

        input_features = inputs["input_features"]
        feature_attention_mask = inputs["feature_attention_mask"]

        input_ids = input_ids.to(self.device)
        input_features = input_features.to(self.device, dtype=self.hf_model_dtype)
        feature_attention_mask = feature_attention_mask.to(self.device)

        self.compute_audio_features_once(
            input_ids_batch1=input_ids,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )
        return input_ids

    #TODO Chat template NOT implemented! The current version relies on simple text prompt!
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        * Runs the residual stack.
        * Returns the last hidden states.
        """
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids 預期為 [batch, seq_len]，"
                f"實際 shape={tuple(input_ids.shape)}"
            )

        if self.cached_audio_features is None:
            raise RuntimeError(
                "forward() 被呼叫前必須先呼叫 encode()，"
                "因為 audio features 尚未快取。"
            )

        dim_batch = input_ids.shape[0]

        text_embeds = self.LALM_model.get_input_embeddings()(input_ids)
        audio_features = self.cached_audio_features.expand(
            dim_batch, -1, -1
        ).to(text_embeds.dtype)

        special_audio_mask = (
            input_ids == self.hf_model.config.audio_token_id
        ).unsqueeze(-1)
        inputs_embeds = text_embeds.masked_scatter(
            special_audio_mask, audio_features
        )

        outputs = self.LALM_model.language_model(
            inputs_embeds=inputs_embeds,
            use_cache=False,
        )
        return outputs.last_hidden_state

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """
        * Maps the residual stream into logits.
        """
        target_dtype = self.LALM_model.language_model.norm.weight.dtype
        norm_device = self.LALM_model.language_model.norm.weight.device
        residual = residual.to(device=norm_device, dtype=target_dtype)
        normed = self.LALM_model.language_model.norm(residual)
        return self.hf_model.lm_head(normed)

    # ------------------------------------------------------------------
    # 音訊特徵快取
    # ------------------------------------------------------------------
    def compute_audio_features_once(self, input_ids_batch1: torch.Tensor,
        input_features: torch.Tensor, feature_attention_mask: torch.Tensor ):
        """
        以 batch=1、no_grad 跑一次完整 forward，用 forward hook 攔截
        multi_modal_projector 的輸出（就是最終要合併進 inputs_embeds 的
        audio_features），完全重用官方的合併邏輯。

        結果存到 self.cached_audio_features，後續 forward() 重複使用。
        """
        captured = {}

        def _hook(module, inputs, output):
            captured["audio_features"] = output

        handle = self.LALM_model.multi_modal_projector.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                self.LALM_model(
                    input_ids=input_ids_batch1,
                    input_features=input_features,
                    feature_attention_mask=feature_attention_mask,
                    use_cache=False,
                )
        finally:
            handle.remove()

        if "audio_features" not in captured:
            raise RuntimeError(
                "multi_modal_projector 的 hook 沒有被觸發，"
                "請確認這次 forward 真的有音訊輸入。"
            )

        self.cached_audio_features = captured["audio_features"].detach()

        if self.cached_audio_features.ndim != 3:
            raise ValueError(
                f"audio_features 預期為 [batch, tokens, d_model]，"
                f"實際 shape={tuple(self.cached_audio_features.shape)}"
            )

        if self.cached_audio_features.shape[0] != 1:
            raise ValueError(
                f"Instance 目前只支援 batch=1 音訊，"
                f"實際 batch={self.cached_audio_features.shape[0]}"
            )

        if self.cached_audio_features.shape[-1] != self.d_model:
            raise ValueError(
                f"audio feature hidden size={self.cached_audio_features.shape[-1]}，"
                f"預期 d_model={self.d_model}"
            )

def load_model(MODEL_ID: str, do_4bit: bool = False):
    if do_4bit:
        try:
            hf_model, processor = load_model_4bit(MODEL_ID)
            print("Loading model using 4bit")
        except Exception as e:
            raise RuntimeError("Failed Loading model({e}). Exiting...")
            
    else:
        try:
            hf_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
            )
            processor = AutoProcessor.from_pretrained(MODEL_ID)
        except Exception as e:
            raise RuntimeError("Failed Loading model({e}). Exiting...")

    hf_model.eval()
    for p in hf_model.parameters():
        p.requires_grad_(False)
    n_requires_grad = sum(p.requires_grad for p in hf_model.parameters())
    logger.info("requires_grad=True 參數數量: %d（預期為 0）", n_requires_grad)
    assert n_requires_grad == 0

    return hf_model, processor

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
    hf_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    hf_model.eval()

    load_time = time.time() - t0
    vram_after_load = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"  模型載入完成，耗時 {load_time:.1f}s")
    print(f"  載入後 VRAM 峰值: {vram_after_load:.2f} GB")
    print()

    return hf_model, processor

        