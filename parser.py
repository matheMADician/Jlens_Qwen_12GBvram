from logging import logger
import os, json, librosa

class Parser:
    """
    * Handles all load and save operations.
    * Includes all data protocols in this project.
    """
    def __init__(self):
        pass

    def parse_prompt(self, json_line: str, max_prompt_size: int = 40) -> tuple[str, str, str]:
        """
        * The protocol of the multimodal prompt is as follows:
        {
            "id": "sample-001",
            "audio_path": "audio/sample.wav", (The relative path from ~/data/*whatever folder*/)
            "prompt": "請描述這段音訊。"
        }
        * Returns in order: id, audio_rel_path, text_prompt
        * If text_prompt is longer than max_prompt_size(default 40), only the first and last
        """
        record = json.loads(json_line)
        
        audio_rel_path = record.get("audio_path")
        text_prompt = record.get("prompt")
        id = record.get("id", "<unknown>")

        if audio_rel_path is None or text_prompt is None:
            raise ValueError(f"樣本 {id} 缺少 audio_path 或 prompt 欄位")

        safe_text = text_prompt.replace("\n", "\\n")
        if len(safe_text) > max_prompt_size:
            from math import floor
            retain_token_num = floor(max_prompt_size / 2)
            display_text = f"<{safe_text[:retain_token_num]}>...<{safe_text[-retain_token_num:]}>"
        else:
            display_text = f"<{safe_text}>"
        logger.info("📥 載入樣本 [%s]: %s", id, display_text)

        return id, audio_rel_path, safe_text

    def load_audio(self, sample_id: int, audio_path: str, sampling_rate: int):
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"樣本 {sample_id} 的音檔不存在：{audio_path}")
        audio_array, _ = librosa.load(audio_path, sr= sampling_rate)
        return audio_array


    