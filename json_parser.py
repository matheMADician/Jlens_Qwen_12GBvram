import os, json, librosa
from pathlib import Path

from dataclasses import dataclass

@dataclass(frozen=True)
class ParsedSample:
    sample_id: str
    audio_path: str
    prompt: str
    transcript: str | None
    dataset_name: str | None
    
class Parser:
    """
    * Handles all load and save operations.
    * Includes all data protocols in this project.
    """
    def __init__(self):
        pass

    @staticmethod
    def load_jsonl_lines(path: str) -> list[str]:
        """
        * Accepts the absolute path to the jsonl file.
        * Parses the jsonl file into lines.
        """
        if not os.path.isfile(path):
            raise RuntimeError(f"Json file does not exist({path})")
        lines = []
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if raw:
                    lines.append(raw)
        return lines

    @staticmethod
    def parse_json_line(
        json_line: str,
        text_start_token: str = "",
        text_end_token: str = "",
        audio_root: str | None = None,
    ) -> ParsedSample:
        """
        This is a dataset parser. Parses a single JSON line representing an audio-text sample.
        The expected format is:
        {
            "id": "0",
            "source_index": 298,
            "audio_path": "datasets/SAKURA/AnimalQA/audio/sample-000298.wav",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "path": "datasets/SAKURA/AnimalQA/audio/sample-000298.wav"
                        }, 
                        {
                            "type": "text",
                            "text": "Based on the sound in the audio, which of the following actions best represents the behavior typically seen in the type of animal making the provided sound? (a) Chewing cud (b) Flapping its wings (c) Leaping between surfaces (d) Flying between trees"
                        }
                    ]
                }
            ],
            "transcript": "(c) Leaping between surfaces",
            "source_id": "sample-000298",
            "dataset_name": "SAKURA_AnimalQA",
            "source_dataset": "SAKURA/AnimalQA"
        }
        """
        record = json.loads(json_line)
        
        id = record.get("id")
        audio_rel_path = record.get("messages")[0].get("content")[0].get("path")
        if not isinstance(audio_rel_path, str) or not audio_rel_path.strip():
            raise ValueError(f"樣本 {id} 缺少有效的音檔路徑")
        if audio_root is not None:
            audio_rel_path = str((Path(audio_root) / audio_rel_path).resolve())
        if record.get("messages")[0].get("role") != "user":
            raise ValueError(f"樣本 {id} 的第一條消息角色不是 user")
        text_prompt = record.get("messages")[0].get("content")[1].get("text")
        text_prompt = text_prompt.strip()
        
        # Add text start and end tokens while consulting the model.
        if text_start_token:
            text_prompt = text_start_token + text_prompt
        if text_end_token:
            text_prompt = text_prompt + text_end_token
        safe_text = text_prompt.replace("\n", "\\n")
        
        parsed_sample = ParsedSample(
            sample_id=id,
            audio_path=audio_rel_path,
            prompt=safe_text,
            transcript=record.get("transcript"),
            dataset_name=record.get("dataset_name"),
        )

        return parsed_sample

    @staticmethod
    def build_chat_prompt(processor, text_prompt: str, audio_path: str) -> str:
        """
        * Wraps the prompt in the model's chat template (one user turn with one audio, then the
          generation prompt of the assistant turn).
        * A legacy prompt that carries the raw audio placeholder has the placeholder removed, and the
          rest is used as the instruction. The template inserts the audio placeholder itself.
        * A prompt that is already templated (contains <|im_start|>) is returned unchanged.
        """
        # Only compatible with Qwen2.5-Omni
        conversation = [{
            "role": "user",
            "content": [
                {"type": "audio", "path": audio_path},
                {"type": "text", "text": text_prompt},
            ],
        }]
        return processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

    @staticmethod
    def load_audio(sample_id: int, audio_path: str, sampling_rate: int):
        """
        * Accepts the absolute path to the audio.
        """
        audio_path = str(Path(audio_path).resolve())
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"樣本 {sample_id} 的音檔不存在：{audio_path}")
        audio_array, _ = librosa.load(audio_path, sr= sampling_rate)
        return audio_array


    