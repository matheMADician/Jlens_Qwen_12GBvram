"""
fit_general.py
How to use=====================================
python fit_general.py \
  --jsonl-name <資料集路徑(under root)例如 jlens_dataset_mixed/prompts.jsonl> \
  --checkpoint-path <輸出 .ckpt 路徑> \
  2>&1 | tee <log 路徑>

note===========================================
本檔案假設你本地有一個 `jlens` 套件
凍結全部參數 + 讓 ActivationRecorder 的 start_graph_at 機制生效
音訊特徵只算一次、快取後給所有 dim_batch 複製重複使用
unembed() 裡的 dtype 轉換修正

parameter(not complosive)======================
--data-root : defult = g_data : file for data
--dim-batch : defult = 32
--max-seq-len : defult = 256
--checkpoint-every : defult = 5 : make checkpoint every n data
      
structure======================================
1. Adapter
Qwen2AudioLensModel : the model
    _compute_audio_features_once : audio feature -> audio embedding (run for once)
    encoder : word -> token
    forward : embeding + fitting
    unembed : unembed

2. JSONL Loader
load_jsonl_lines
filter_by_length

3. Main

"""

import os
import json
import argparse
import logging

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

from jlens.fitting import fit as jlens_fit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fit_general")


# 1. Adapter
# ---------------------------------------------------------------------------
class Qwen2AudioLensModel:
    def __init__(self, full_model, processor, data_root: str, device: str = "cuda",
                 model_dtype: torch.dtype = torch.bfloat16):
        self._model = full_model
        self._inner = full_model.model
        self._processor = processor
        self._sampling_rate = processor.feature_extractor.sampling_rate
        self._data_root = data_root
        self._device = device
        self._model_dtype = model_dtype

        self.tokenizer = processor.tokenizer
        self.n_layers = self._inner.language_model.config.num_hidden_layers
        self.d_model = self._inner.language_model.config.hidden_size
        self.layers = self._inner.language_model.layers

        self._cached_audio_features = None

    def _compute_audio_features_once(self, input_ids_batch1, input_features, feature_attention_mask):
        """只在 batch=1 時跑一次完整 forward，把 projector 輸出的 audio embedding 快取起來。
        之後 forward() 裡對這份快取做 .expand(dim_batch, ...)，
        避免每個 batch step 都重新跑一次昂貴的音訊 encoder。"""
        captured = {}

        def _hook(module, inputs, output):
            captured["audio_features"] = output

        handle = self._inner.multi_modal_projector.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                self._inner(
                    input_ids=input_ids_batch1,
                    input_features=input_features,
                    feature_attention_mask=feature_attention_mask,
                    use_cache=False,
                )
        finally:
            handle.remove()
        self._cached_audio_features = captured["audio_features"].detach()

    def encode(self, json_line: str, *, max_length: int = 300):
        #解析 prompts.jsonl 的其中一行，動態載入對應 wav，算好音訊特徵並快取。
        record = json.loads(json_line)
        audio_path = os.path.join(self._data_root, record["audio_path"])
        #{"id": "", "audio_path": "","prompt": ""}
        prompt = record["prompt"]
        
        #確認data
        safe_text = prompt.replace('\n', '\\n') 
        if len(safe_text) > 40:
            display_text = f"<{safe_text[:20]}>...<{safe_text[-20:]}>"
        else:
            display_text = f"<{safe_text}>"
            
        logger.info(f"📥 載入樣本 [{record.get('id', 'N/A')}]: {display_text}")
        
        # 5 秒以內的音訊，載入成本不高，直接同步讀取即可
        audio_array, _ = librosa.load(audio_path, sr=self._sampling_rate)

    # 1. 保持最乾淨的 Processor 呼叫（不加任何 padding，保護文字長度）
        inputs = self._processor(
            text=prompt,
            audio=[audio_array],
            sampling_rate=self._sampling_rate,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"]
        if input_ids.shape[1] > max_length:
            raise ValueError(
                f"樣本 {record.get('id', '<unknown>')} 長度 {input_ids.shape[1]} "
                f"超過 max_length={max_length}，安全跳過。"
            )

        
        input_features = inputs["input_features"]
        feature_attention_mask = inputs["feature_attention_mask"]

        input_ids = input_ids.to(self._device)
        input_features = input_features.to(self._device, dtype=self._model_dtype)
        feature_attention_mask = feature_attention_mask.to(self._device)
        
        self._compute_audio_features_once(
            input_ids_batch1=input_ids,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )

        return input_ids

    def forward(self, input_ids):
        dim_batch = input_ids.shape[0]
        
        text_embeds = self._inner.get_input_embeddings()(input_ids)
        audio_features = self._cached_audio_features.expand(dim_batch, -1, -1).to(text_embeds.dtype)

        special_audio_mask = (input_ids == self._model.config.audio_token_id).unsqueeze(-1)
        inputs_embeds = text_embeds.masked_scatter(special_audio_mask, audio_features)

        outputs = self._inner.language_model(
            inputs_embeds=inputs_embeds,
            use_cache=False,
        )
        return outputs.last_hidden_state

    def unembed(self, residual):
        # transport()/select() 回傳的 residual 是 float32（JacobianLens 為了精度
        # 刻意這麼做），但 norm/lm_head 的權重是 bf16，dtype 對不上會直接報錯，
        # 所以這裡先 cast 回模型實際在用的 dtype 再送進去。
        target_dtype = self._inner.language_model.norm.weight.dtype

        residual = residual.to(device=self._device, dtype=target_dtype)

        normed = self._inner.language_model.norm(residual.to(target_dtype))
        return self._model.lm_head(normed)
        
# ---------------------------------------------------------------------------
# 2. JSONL Loader
# ---------------------------------------------------------------------------
def load_jsonl_lines(path: str):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                lines.append(raw)
    return lines


def filter_by_length(records, processor, data_root: str, sampling_rate: int, max_length: int):
    kept, dropped = [], []
    for line in records:
        record = json.loads(line)
        audio_path = os.path.join(data_root, record["audio_path"])
        audio_array, _ = librosa.load(audio_path, sr=sampling_rate)
        
        prompt = record["prompt"]

        inputs = processor(
            text=prompt,
            audio=[audio_array],
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        length = inputs["input_ids"].shape[1]
        if length > max_length:
            logger.warning(
                "跳過樣本 %s：長度 %d 超過 max_length=%d",
                record.get("id", "<unknown>"), length, max_length,
            )
            dropped.append(record.get("id", "<unknown>"))
        else:
            kept.append(line)

    logger.info(
        "長度過濾完成：保留 %d / %d 筆%s",
        len(kept), len(records),
        f"，捨棄: {dropped}" if dropped else "",
    )
    return kept


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="General fit J-Lens")
    parser.add_argument("--data-root", default="g_data",
                        help="工作目錄，內含 jlens_dataset/prompts.jsonl 與 jlens_dataset/audio/")
    parser.add_argument("--jsonl-name", required=True, help="輸入的 JSONL : --jsonl-name")
    parser.add_argument("--model-name", default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--checkpoint-path", required=True, help="輸出的 ckpt : --checkpoint-path")
    parser.add_argument("--dim-batch", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)

    jsonl_path = os.path.join(args.data_root, args.jsonl_name)
    logger.info("Loading metadata from %s", jsonl_path)
    records = load_jsonl_lines(jsonl_path)
    logger.info("Loaded %d samples", len(records))

    logger.info("Loading model %s in bf16 (no quantization)...", args.model_name)
    full_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )

    processor = AutoProcessor.from_pretrained(args.model_name)

    logger.info("Filtering samples by length (max_seq_len=%d) before fitting...", args.max_seq_len)
    records = filter_by_length(
        records,
        processor=processor,
        data_root=args.data_root,
        sampling_rate=processor.feature_extractor.sampling_rate,
        max_length=args.max_seq_len,
    )
    if not records:
        raise RuntimeError("長度過濾後沒有剩下任何樣本，請檢查 max_seq_len 設定或資料本身。")

    # 手動凍結全部參數：整個模型只做 forward，不需要梯度，
    full_model.eval()
    for p in full_model.parameters():
        p.requires_grad_(False)
    n_requires_grad = sum(p.requires_grad for p in full_model.parameters())
    logger.info("requires_grad=True 參數數量: %d（預期為 0）", n_requires_grad)
    assert n_requires_grad == 0

    adapter = Qwen2AudioLensModel(
        full_model=full_model,
        processor=processor,
        data_root=args.data_root,
        device=args.device,
        model_dtype=torch.bfloat16,
    )

    n_layers = adapter.n_layers  # 應為 32
    source_layers = list(range(n_layers - 1))  # 0..30，共 31 層
    target_layer = n_layers - 1  # 31

    logger.info(
        "Fitting all %d layers together: source_layers=%s..%s, target_layer=%s",
        n_layers, source_layers[0], source_layers[-1], target_layer,
    )
    logger.info(
        "dim_batch=%d, max_seq_len=%d, checkpoint_every=%d, checkpoint_path=%s",
        args.dim_batch, args.max_seq_len, args.checkpoint_every, args.checkpoint_path,
    )


    lens = jlens_fit(
        model=adapter,
        prompts=records,
        source_layers=source_layers,
        target_layer=target_layer,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=args.checkpoint_path,
        checkpoint_every=args.checkpoint_every,
    )


    logger.info("fitting finished. %r", lens)


if __name__ == "__main__":
    main()
