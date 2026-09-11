import os
import numpy as np
import soundfile as sf

class Data:
    #TODO: 這裡應該要有: 1.聲音預處理 2.資料集載入
    def __init__(self, path: str):
        self.path = path
        pass

    def parse_audio(self, path: str):
        pass

    def ensure_test_audio(self, path: str, duration_sec: float = 5.0, sr: int = 16000):
        if os.path.exists(path):
            print(f"[Step 1] 使用既有測試音訊: {path}")
            return path

        print(f"[Step 1] 找不到 {path}，合成一段 {duration_sec}s 正弦波作為管線 smoke test（非語義測試）")
        
        t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
        tone = 0.1 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
        sf.write(path, tone, sr)
        return path