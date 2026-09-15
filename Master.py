import model.model as M
import data.data as data
import parser
from Jlens import Jlens as J
import sys, torch, json, logging

#TODO: Apply this
"""
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
args = parser.parse_args()"""


class Master:
    def __init__(self):
        self.check_dependencies()
        self.parser = parser.Parser()
        self.logger = logging.getLogger()
        self.settings = json.load(open("settings.json", "r", encoding="utf-8"))
        self.model = M.Model(logger= self.logger,
            do_4bit= self.settings["do_4bit"],
            MODEL_ID= self.settings["Model_ID"])

    def test_inference(self):
        text_line = "" #TODO
        logits = self.model.run_inference(jsonl_line= text_line, data_root= "")
        pass

    def test_training(self):
        pass
    
    def test_lens(self):
        pass

    def calc_JLens(self):
        pass

    def apply_Jlens(self):
        pass

    def apply_Llens(self):
        pass
    
    def check_dependencies(self):
        print("=" * 70)
        print("[Step 0] 依賴檢查")
        print("=" * 70)

        missing = []
        versions = {}

        try:
            versions["torch"] = torch.__version__
            if not torch.cuda.is_available():
                print("[FATAL] 找不到可用的 CUDA GPU，請確認驅動與 CUDA 安裝。")
                sys.exit(1)
            gpu_name = torch.cuda.get_device_name(0)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"  GPU: {gpu_name} | 總 VRAM: {total_vram_gb:.1f} GB")
            if total_vram_gb < 11:
                print("  [警告] 偵測到的 VRAM 低於 11GB，本腳本的記憶體預算可能不適用，請調整 batch/長度設定。")
        except ImportError:
            missing.append("torch")

        for pkg in ["transformers", "bitsandbytes", "accelerate", "librosa", "soundfile"]:
            try:
                mod = __import__(pkg)
                versions[pkg] = getattr(mod, "__version__", "unknown")
            except ImportError:
                missing.append(pkg)

        if missing:
            print(f"[FATAL] 缺少套件: {missing}")
            print("  請參考環境建構建議安裝，例如：")
            print("  pip install git+https://github.com/huggingface/transformers")
            print("  pip install bitsandbytes accelerate librosa soundfile")
            sys.exit(1)

        print("  已安裝套件版本：")
        for k, v in versions.items():
            print(f"    - {k}: {v}")
