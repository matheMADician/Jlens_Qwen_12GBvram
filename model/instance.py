from jlens.protocol import LensModel
from json_parser import Parser, ParsedSample
import os, logging, time
import torch
from typing import Protocol, cast
from transformers import (
    BitsAndBytesConfig,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

logger = logging.getLogger(__name__)


class _ThinkerGenerator(Protocol):
    def generate(self, **kwargs: object) -> torch.Tensor: ...


ABLATION_MODES = ("ablation", "jablation", "injection")


class AblationSpec:
    """
    Tensor-only description of a residual-stream intervention.
    Directions are computed by the caller (see Model), so Instance never
    depends on a lens.

    * ablation:  set the residual to zero.
    * jablation: h -> h - beta * (h . d) d. beta=1 removes the component, beta=2 reverses it.
    * injection: h -> h - (h . d) d + (h . d) d2.
    ``directions`` / ``directions2`` map layer -> unit vector of shape [d_model].
    """
    def __init__(
        self,
        mode: str,
        layers: list[int],
        token_pos: list[int],
        directions: dict[int, torch.Tensor] | None = None,
        beta: float = 1.0,
        directions2: dict[int, torch.Tensor] | None = None,
    ):
        if mode not in ABLATION_MODES:
            raise ValueError(f"Unknown ablation mode {mode!r}; expected one of {ABLATION_MODES}")
        if mode != "ablation" and directions is None:
            raise ValueError(f"mode={mode!r} requires directions.")
        if mode == "injection" and directions2 is None:
            raise ValueError("mode='injection' requires directions2.")
        for name, table in (("directions", directions), ("directions2", directions2)):
            if table is not None:
                missing = sorted(set(layers) - set(table))
                if missing:
                    raise ValueError(f"{name} is missing layers {missing}.")
        self.mode = mode
        self.layers = list(layers)
        self.token_pos = list(token_pos)
        self.directions = directions
        self.beta = beta
        self.directions2 = directions2


class Instance(LensModel):
    """
    Qwen2.5-Omni Thinker 的 adapter，給 JLens 使用。
    職責：
      - encode(json_line)：把 prompts.jsonl 的一行變成 input_ids，
        並快取該樣本的 processor inputs。
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
        self.LALM_model = self.hf_model
        self.text_model = self.hf_model.model

        # Model parameters
        self.sampling_rate = self.processor.feature_extractor.sampling_rate
        embedding_weight = self.text_model.embed_tokens.weight
        self.device = embedding_weight.device
        self.hf_model_dtype = embedding_weight.dtype
        lm_config = self.text_model.config

        # Cache processor inputs; Omni's model handles audio encoding and merging.
        self.cached_model_inputs: dict[str, torch.Tensor] | None = None
        
        # Variable for Anthropic's protocol
        self.n_layers = lm_config.num_hidden_layers
        self.d_model = lm_config.hidden_size
        self.layers = list(self.text_model.layers)
        self.tokenizer = self.processor.tokenizer
        
    def clear_audio_cache(self):
        self.cached_model_inputs = None

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
    def encode(self, text: str, *, max_length: int = 3000) -> torch.Tensor:
        """
        * Accepts a single raw Json line.
        * Calls the processor to encode the audio files.
        * Returns the tokenized (Audio + text prompt)
        * ``max_length`` is retained for interface compatibility; prompts are
          not rejected or truncated based on their encoded length.
        """
        if self.data_root is None:
            raise RuntimeError("No data root directory specified. Please call load_data() first.")

        self.clear_audio_cache()
        
        sample: ParsedSample = Parser.parse_json_line(
            text,
            audio_root=self.data_root,
        )
        
        if not sample.sample_id.isdecimal():
            raise ValueError(f"Sample id must be decimal({sample.sample_id})")
        sample_id = int(sample.sample_id)
        
        audio_path = sample.audio_path
        audio_array = Parser.load_audio(
            sample_id= sample_id,
            audio_path= audio_path,
            sampling_rate= self.sampling_rate
        )

        text_prompt = Parser.build_chat_prompt(self.processor, sample.prompt, audio_path)
        inputs = self.processor(
            text=text_prompt,
            audio=[audio_array],
            audio_kwargs={"sampling_rate": self.sampling_rate},
        )
        inputs.convert_to_tensors("pt")
        input_ids = inputs["input_ids"]

        model_inputs = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                dtype = self.hf_model_dtype if key == "input_features" else value.dtype
                value = value.to(device=self.device, dtype=dtype)
            model_inputs[key] = value
        self.cached_model_inputs = model_inputs
        return model_inputs["input_ids"]

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

        if self.cached_model_inputs is None:
            raise RuntimeError(
                "forward() 被呼叫前必須先呼叫 encode()，"
                "因為 processor inputs 尚未快取。"
            )

        cached_input_ids = self.cached_model_inputs["input_ids"]
        if input_ids.shape[1] != cached_input_ids.shape[1]:
            raise ValueError("input_ids sequence length does not match the encoded prompt.")

        model_inputs = {}
        for key, value in self.cached_model_inputs.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1:
                value = value.expand(input_ids.shape[0], *value.shape[1:])
            model_inputs[key] = value
        model_inputs["input_ids"] = input_ids
        model_inputs["use_cache"] = False

        captured: dict[str, torch.Tensor] = {}

        def capture_residual(module, args, output):
            captured["residual"] = output if torch.is_tensor(output) else output[0]

        handle = self.layers[-1].register_forward_hook(capture_residual)
        try:
            self.LALM_model(**model_inputs)
        finally:
            handle.remove()

        if "residual" not in captured:
            raise RuntimeError("The Thinker text decoder did not produce a final-layer residual.")
        return captured["residual"]

    def ablation_forward(
        self,
        input_ids: torch.Tensor,
        ablation_spec: AblationSpec | None = None,
    ) -> torch.Tensor:
        """
        * Runs forward() with the ablation applied to the output of each block in ablation_spec.layers.
        * Returns the final block's residual (before the final norm), which unembed() maps to logits.
          This matches the model logits used by JacobianLens.apply().
        * ablation_spec=None gives the unmodified baseline.
        """
        final_layer = self.n_layers - 1
        captured: dict[str, torch.Tensor] = {}
        handles = []

        def _capture(module, inputs, output):
            captured["residual"] = (output if torch.is_tensor(output) else output[0]).detach()

        try:
            # Ablation hooks must be registered before the capture hook so that
            # ablating the last layer is visible to it.
            if ablation_spec is not None:
                for layer in ablation_spec.layers:
                    if not 0 <= layer < self.n_layers:
                        raise ValueError(f"layer {layer} out of range for a {self.n_layers}-layer model")
                    handles.append(
                        self.layers[layer].register_forward_hook(
                            self._make_ablation_hook(layer, ablation_spec)
                        )
                    )
            handles.append(self.layers[final_layer].register_forward_hook(_capture))
            with torch.no_grad():
                self.forward(input_ids)
        finally:
            for handle in handles:
                handle.remove()

        return captured["residual"]

    @torch.no_grad()
    def greedy_forward(self, input_ids: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        * Greedy decoding with a KV cache. Prefills the whole prompt once, then feeds one token per step.
        * max_len is the maximum number of NEW tokens. Stops early at an EOS token.
        * Returns the generated token ids only (prompt excluded), shape [1, n_new].
        """
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"input_ids 預期為 [1, seq_len]，實際 shape={tuple(input_ids.shape)}")
        if self.cached_model_inputs is None:
            raise RuntimeError("greedy_forward() 被呼叫前必須先呼叫 encode()。")

        generator = cast(_ThinkerGenerator, self.hf_model)
        generated = generator.generate(
            **self.cached_model_inputs,
            max_new_tokens=max_len,
            do_sample=False,
        )
        return generated[:, input_ids.shape[1]:]

    @staticmethod
    def _make_ablation_hook(layer: int, spec: AblationSpec):
        def hook(module, inputs, output):
            hidden = output if torch.is_tensor(output) else output[0]
            seq_len = hidden.shape[1]
            pos = [p % seq_len for p in spec.token_pos]
            selected = hidden[:, pos, :].float()

            if spec.mode == "ablation": # Ablation
                new = torch.zeros_like(selected)
            else:
                assert spec.directions is not None
                d = spec.directions[layer].to(selected.device, torch.float32)
                coeff = (selected @ d).unsqueeze(-1)
                if spec.mode == "jablation": # Jlens Ablation
                    new = selected - spec.beta * coeff * d
                else:
                    assert spec.directions2 is not None # Change directioin
                    d2 = spec.directions2[layer].to(selected.device, torch.float32)
                    new = selected - coeff * d + coeff * d2

            patched = hidden.clone()
            patched[:, pos, :] = new.to(hidden.dtype)
            return patched if torch.is_tensor(output) else (patched, *output[1:])
        return hook

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """
        * Maps the residual stream into logits.
        """
        target_dtype = self.text_model.norm.weight.dtype
        norm_device = self.text_model.norm.weight.device
        residual = residual.to(device=norm_device, dtype=target_dtype)
        normed = self.text_model.norm(residual)
        return self.hf_model.lm_head(normed)

def load_model(MODEL_ID: str, do_4bit: bool = False):
    if do_4bit:
        try:
            hf_model, processor = load_model_4bit(MODEL_ID)
            print("Loading model using 4bit")
        except Exception as e:
            raise RuntimeError(f"Failed loading 4-bit model: {e}") from e
            
    else:
        try:
            hf_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
        except Exception as e:
            raise RuntimeError(f"Failed loading model: {e}") from e

    hf_model.eval()
    for p in hf_model.parameters():
        p.requires_grad_(False)
    n_requires_grad = sum(p.requires_grad for p in hf_model.parameters())
    logger.info("requires_grad=True 參數數量: %d（預期為 0）", n_requires_grad)
    assert n_requires_grad == 0

    return hf_model, processor

def load_model_4bit(MODEL_ID: str):
    print("=" * 70)
    print("[Step 2] Loading Qwen2.5-Omni Thinker using 4-bit (NF4)")
    print("=" * 70)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)

    # attn_implementation="flash_attention_2" 可選用（需另外安裝 flash-attn，
    # RTX 3060 屬 Ampere 架構相容）；若未安裝則保留預設 "sdpa" 即可。

    hf_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
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
