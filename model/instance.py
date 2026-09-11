import json, os, logger, librosa
import torch


# 可綁定多個audio
class Instance:
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