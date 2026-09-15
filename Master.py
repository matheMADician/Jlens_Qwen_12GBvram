import model.model as M
import parser, os, sys, torch, json, logging

class Master:
    def __init__(self, run_name: str | None = None):
        self.check_dependencies()
        self.parser = parser.Parser()
        self.logger = logging.getLogger()
        self.settings = json.load(open("settings.json", "r", encoding="utf-8"))
        self.model = M.Model(logger= self.logger,
            do_4bit= self.settings["do_4bit"],
            MODEL_ID= self.settings["Model_ID"])
        self.run_name = run_name

    def test_inference(self):
        data_root = self.settings.get("Inference_data_root")
        jsonl_path = self.settings.get("Inference_jsonl")
        max_length = self.settings.get("Inference_max_length", 300)

        if not data_root or not jsonl_path:
            raise ValueError(
                "請在 settings.json 設定 Inference_data_root 與 Inference_jsonl"
            )

        data_root = os.path.abspath(data_root)
        jsonl_path = os.path.abspath(jsonl_path)
        lines = self.parser.load_jsonl_lines(jsonl_path)
        if not lines:
            raise ValueError(f"JSONL 沒有可推論的樣本：{jsonl_path}")

        logits = self.model.run_inference(
            jsonl_line=lines[0],
            data_root=data_root,
            MAX_LENGTH=max_length,
        )
        if logits.ndim != 3:
            raise RuntimeError(
                f"推論結果預期為 [batch, seq_len, vocab]，實際 shape={tuple(logits.shape)}"
            )

        next_token_logits = logits[0, -1]
        top_ids = torch.topk(next_token_logits, k=5).indices.tolist()
        top_tokens = self.model.model.tokenizer.convert_ids_to_tokens(top_ids)
        print(f"推論完成：logits shape={tuple(logits.shape)}")
        print(f"下一個 token 候選：{top_tokens}")
        return logits

    def test_training(self):
        if not self.run_name:
            raise ValueError("test_training() 需要提供 --run-name")

        run_directory = os.path.join(
            self.settings["Jlens_checkpoint_path"],
            self.run_name
        )
        checkpoint_path = os.path.join(run_directory, "fit.pt")
        print(f"Checkpoint path: {checkpoint_path}")
        os.makedirs(run_directory, exist_ok=True)
            
        self.model.fit_Jlens(
            jsonl_path= self.settings["Inference_jsonl"],
            checkpoint_save_path= checkpoint_path,
            run_name= self.run_name,
            is_test=True,
        )
    
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
