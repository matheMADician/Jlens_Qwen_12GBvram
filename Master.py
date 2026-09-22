import model.model as M
from Jlens.Jlens import Jlens
import parser, os, sys, torch, json, logging

class Master:
    def __init__(self, run_name: str):
        self.check_dependencies()
        self.parser = parser.Parser()
        self.logger = logging.getLogger()
        self.settings = json.load(open("settings.json", "r", encoding="utf-8"))
        self.model = M.Model(logger= self.logger,
            do_4bit= self.settings["model"]["load_in_4bit"],
            MODEL_ID= self.settings["model"]["id"])
        self.run_name = run_name

    # *可以跑*
    def test_inference(self):
        data_root = self.settings["data"]["root"]
        jsonl_path = self.settings["inference"]["jsonl_path"]
        max_length = self.settings["inference"].get("max_length", 300)

        if not data_root or not jsonl_path:
            raise ValueError(
                "請在 settings.json 設定 data.root 與 inference.jsonl_path"
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
        top_tokens = self.model.model.tokenizer.decode(
            top_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        print(f"推論完成：logits shape={tuple(logits.shape)}")
        print(f"下一個 token 候選：{top_tokens}")
        return logits

    # *可以跑*
    def test_training(self):
        if not self.run_name:
            raise ValueError("test_training() 需要提供 --run-name")

        run_directory = os.path.join(
            self.settings["checkpoints"]["root"],
            self.run_name
        )
        checkpoint_path = os.path.join(run_directory, "fit.pt")
        print(f"Checkpoint path: {checkpoint_path}")
        os.makedirs(run_directory, exist_ok=True)
            
        self.model.fit_Jlens(
            jsonl_path= self.settings["data"]["jsonl_path"],
            checkpoint_save_path= checkpoint_path,
            run_name= self.run_name,
            is_test=True,
        )
    
    # *可以跑*
    def test_lens(self):
        data_root = self.settings["data"]["root"]
        jsonl_path = self.settings["data"]["jsonl_path"]
        max_length = self.settings["inference"].get("max_length", 300)
        
        if not data_root or not jsonl_path:
            raise ValueError(
                "請在 settings.json 設定 data.root 與 data.jsonl_path"
            )
        
        data_root = os.path.abspath(data_root)
        jsonl_path = os.path.abspath(jsonl_path)
        lines = self.parser.load_jsonl_lines(jsonl_path)
        if not lines:
            raise ValueError(f"JSONL 沒有可推論的樣本：{jsonl_path}")
        
        self.model.apply_lens(
            json_line= lines[0],
            run_name= self.run_name,
            data_root= data_root,
            use_Jlens= False,
        )
        
        logits= self.model.lens_cache.get(
            lens_type= "logit_lens",
            key= "lens_logits",
            layer= 30
        )
        
        print(f"Success! Logits: {logits}")

    # *可以跑*
    def calc_JLens(self):
        checkpoint_root = self.settings["checkpoints"]["root"]
        checkpoint_name = self.run_name or "default"
        checkpoint_directory = os.path.join(checkpoint_root, checkpoint_name)
        checkpoint_path = os.path.join(checkpoint_directory, "fit.pt")
        os.makedirs(checkpoint_directory, exist_ok=True)

        self.model.fit_Jlens(
            jsonl_path= self.settings["data"]["jsonl_path"],
            checkpoint_save_path=checkpoint_path,
            run_name= self.run_name
        )

    def apply_Jlens(self):
        self.inference_jsonl_path = self.settings["inference"]["jsonl_path"]
        json_lines = self.parser.load_jsonl_lines(self.inference_jsonl_path)
        jlens_path = self.settings["checkpoints"]["load_path"]
        self.model.load_Jlens(lens_path= jlens_path)
        for i in range(len(json_lines)):
            self.model.apply_lens(
                json_line = json_lines[i],
                data_root= self.settings["data"]["root"],
                save_path= self.settings["inference"]["output_root"],
                top_k= self.settings["inference"]["top_k"],
                use_Jlens= True,
                run_name= self.run_name,
            )
    
    def comp_dataset_choice(self):
        # TODO do research on the dimensionality of datasets
        pass
    
    def draw_Jlens_heatmap(
        self,
        heatmap_path: str,
        lens_path1: str,
        lens_path2: str | None = None,
    ) -> torch.Tensor:
        """Draw a heatmap using cosine similarity between layers.
        The returned tensor is indexed as ``[lens1_layer, lens2_layer]``.
        
        The y axis represents the first len, while the x axis does for the second.

        Args:
            heatmap_path: Path where the PNG heatmap will be written.
            lens_path1 (str): Path to the Jlens checkpoint.
            lens_path2 (str | None, optional): If not specified, compare lens1 to itself.
        """
        lens1 = Jlens(self.logger, self.model.model)
        lens1.load_lens(path=lens_path1)
        lens2 = lens1 if lens_path2 is None else Jlens(self.logger, self.model.model)
        if lens_path2 is not None:
            lens2.load_lens(path=lens_path2)

        assert lens1.lens is not None
        assert lens2.lens is not None
        layers1 = lens1.lens.source_layers
        layers2 = lens2.lens.source_layers

        from tools.math_tools import cos_sim_for_sets, draw_heatmap

        heatmap = cos_sim_for_sets(
            l1=lens1.lens.jacobians,
            l2=lens2.lens.jacobians,
            d1=layers1,
            d2=layers2,
        )
        draw_heatmap(
            data=heatmap,
            png_path=os.path.join(heatmap_path, self.run_name),
            x_labels=layers2,
            y_labels=layers1,
            title="JLens Jacobian cosine similarity",
        )
        self.logger.info("Saved JLens heatmap to %s", heatmap_path)
        return heatmap
    
    def comp_Jlens(self, lens_path1, lens_path2):
        # Compares two Jlens
        Jlens1= Jlens(self.logger, self.model.model)
        Jlens2= Jlens(self.logger, self.model.model)
        
        Jlens1.load_lens(lens_path1)
        Jlens2.load_lens(lens_path2)
        
        # Phase 1
        # Draw a heatmap between layers using two similarity functions
            # 1. cosine similarity
            # 2. CKA similarity
        
        # Phase 2
        # Calculate the overall 
            
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
