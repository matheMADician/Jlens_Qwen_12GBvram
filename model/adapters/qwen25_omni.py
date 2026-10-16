import logging
import time
from typing import Any

import torch
from transformers import (
    BitsAndBytesConfig,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

from json_parser import ParsedSample, Parser
from model.adapters.Instance import Instance as BaseInstance

logger = logging.getLogger(__name__)


class Instance(BaseInstance):
    """Qwen2.5-Omni Thinker implementation of the shared JLens adapter."""

    def _initialize_model(self, model_id: str, do_4bit: bool) -> None:
        hf_model, processor = load_model(model_id, do_4bit)
        self.hf_model = hf_model
        self.processor = processor
        self.LALM_model = hf_model
        self.text_model = hf_model.model

        self.sampling_rate = processor.feature_extractor.sampling_rate
        embedding_weight = self.text_model.embed_tokens.weight
        self.device = embedding_weight.device
        self.hf_model_dtype = embedding_weight.dtype
        self.cached_model_inputs: dict[str, torch.Tensor] | None = None

        self.n_layers = self.text_model.config.num_hidden_layers
        self.d_model = self.text_model.config.hidden_size
        self.layers = list(self.text_model.layers)
        self.tokenizer = processor.tokenizer

    def clear_model_input_cache(self) -> None:
        self.cached_model_inputs = None

    def encode(self, text: str, *, max_length: int = 3000) -> torch.Tensor:
        if self.data_root is None:
            raise RuntimeError("No data root directory specified. Please call load_data() first.")
        self.clear_model_input_cache()
        
        # Get model
        _, processor = self.get_model()
        # Parse the input text(a raw jsonl line)
        sample: ParsedSample = Parser.parse_json_line(
            text,
            audio_root=self.data_root,
        )
        if not sample.sample_id.isdecimal():
            raise ValueError(f"Sample id must be decimal({sample.sample_id})")
        # Load the audio
        audio_array = Parser.load_audio(
            sample_id=int(sample.sample_id),
            audio_path=sample.audio_path,
            sampling_rate=self.sampling_rate,
        )
        # Build the chat model and tokenize it with the processor
        text_prompt = Parser.build_chat_prompt(
            processor, sample.prompt, sample.audio_path
        )
        inputs = processor(
            text=text_prompt,
            audio=[audio_array],
            audio_kwargs={"sampling_rate": self.sampling_rate},
        )
        inputs.convert_to_tensors("pt")
        # Convert the inputs to PyTorch tensors and move them to the appropriate device
        model_inputs: dict[str, Any] = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                dtype = self.hf_model_dtype if key == "input_features" else value.dtype
                value = value.to(device=self.device, dtype=dtype)
            model_inputs[key] = value
        # Cache the processed model inputs.
        self.cached_model_inputs = model_inputs
        return model_inputs["input_ids"]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        * Performs a forward pass. input_ids expect [batch, seq_len].
        * Returns the final layer residual. This is used for Jlens fitting.
        """
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids expected [batch, seq_len], got {tuple(input_ids.shape)}")
        
        # Retrieve the cached model inputs.
        assert self.cached_model_inputs is not None, "forward() requires encode() to cache processor inputs."
        cached_input_ids = self.cached_model_inputs["input_ids"]
        if input_ids.shape[1] != cached_input_ids.shape[1]:
            raise ValueError("input_ids sequence length does not match the encoded prompt.")
        # Expand cached model inputs to match the batch size of input_ids.
        model_inputs = {}
        for key, value in self.cached_model_inputs.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1:
                value = value.expand(input_ids.shape[0], *value.shape[1:])
            model_inputs[key] = value
        model_inputs["input_ids"] = input_ids
        model_inputs["use_cache"] = False

        # Capture the final-layer residual
        captured: dict[str, torch.Tensor] = {}
        def capture_residual(_module, _args, output):
            captured["residual"] = output if torch.is_tensor(output) else output[0]

        handle = self.layers[-1].register_forward_hook(capture_residual)
        try:
            self.LALM_model(**model_inputs)
        finally:
            handle.remove()

        if "residual" not in captured:
            raise RuntimeError("The Thinker text decoder did not produce a final-layer residual.")
        return captured["residual"]

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """
        * Unembeds the final layer residual into the vocabulary space.
        * Returns the logits for each token in the vocabulary.
        """
        hf_model, _ = self.get_model()
        target_dtype = self.text_model.norm.weight.dtype
        norm_device = self.text_model.norm.weight.device
        residual = residual.to(device=norm_device, dtype=target_dtype)
        return hf_model.lm_head(self.text_model.norm(residual))

    def get_unembedding_direction(self, token_id: int) -> torch.Tensor:
        hf_model, _ = self.get_model()
        if not 0 <= token_id < hf_model.lm_head.weight.shape[0]:
            raise ValueError(f"token_id {token_id} is outside the vocabulary")
        return (
            hf_model.lm_head.weight[token_id].float()
            * self.text_model.norm.weight.float()
        )


def load_model(
    model_id: str, do_4bit: bool = False
) -> tuple[Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor]:
    if do_4bit:
        try:
            hf_model, processor = load_model_4bit(model_id)
        except Exception as exc:
            raise RuntimeError(f"Failed loading 4-bit Qwen2.5-Omni: {exc}") from exc
    else:
        try:
            hf_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
        except Exception as exc:
            raise RuntimeError(f"Failed loading Qwen2.5-Omni: {exc}") from exc

    hf_model.eval()
    for parameter in hf_model.parameters():
        parameter.requires_grad_(False)
    trainable_parameters = sum(
        parameter.requires_grad for parameter in hf_model.parameters()
    )
    logger.info("Trainable parameters: %d (expected 0)", trainable_parameters)
    if trainable_parameters:
        raise RuntimeError(
            f"Expected all model parameters frozen, found {trainable_parameters} trainable"
        )

    return hf_model, processor


def load_model_4bit(
    model_id: str,
) -> tuple[Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor]:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    start_time = time.perf_counter()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
    hf_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    hf_model.eval()
    logger.info(
        "Loaded 4-bit Qwen2.5-Omni model in %.1f seconds",
        time.perf_counter() - start_time,
    )
    return hf_model, processor
