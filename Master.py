import os, sys, torch, json, logging

import model.model as M
from audio_jlens.jlens import Jlens
from json_parser import Parser
from tools.experiments import Experiments

class Master:
    def __init__(self, run_name: str):
        Master.check_dependencies()
        self.logger = logging.getLogger()
        self.settings = json.load(open("settings.json", "r", encoding="utf-8"))
        self.model = M.Model(logger= self.logger,
            do_4bit= self.settings["model"]["load_in_4bit"],
            MODEL_ID= self.settings["model"]["id"])
        self.run_name = run_name

    def _configured_data_root(self) -> str:
        data_root = self.settings["fitting"].get("root")
        if not isinstance(data_root, str) or not data_root.strip():
            raise ValueError("請在 settings.json 設定有效的 fitting.root")
        return os.path.abspath(os.path.expanduser(data_root))

    # Testing methods
    # ---------------------------------------------------------------------------------------
    def test_inference(self):
        data_root = self._configured_data_root()
        jsonl_path = self.settings["inference"]["jsonl_path"]
        max_length = self.settings["inference"].get("max_length", 300)

        if not data_root or not jsonl_path:
            raise ValueError(
                "請在 settings.json 設定 fitting.root 與 inference.jsonl_path"
            )

        jsonl_path = os.path.abspath(jsonl_path)
        lines = Parser.load_jsonl_lines(jsonl_path)
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

    def test_training(self):
        if not self.run_name:
            raise ValueError("test_training() 需要提供 --run-name")

        run_directory = os.path.join(
            self.settings["fitting"]["checkpoint_root"],
            self.run_name
        )
        checkpoint_path = os.path.join(run_directory, "fit.pt")
        print(f"Checkpoint path: {checkpoint_path}")
        os.makedirs(run_directory, exist_ok=True)
            
        self.model.fit_Jlens_batch(
            jsonl_path= self.settings["fitting"]["jsonl_path"],
            data_root=self._configured_data_root(),
            checkpoint_save_path= checkpoint_path,
            run_name= self.run_name,
            dim_batch=self.settings["fitting"].get("dim_batch", 32),
            MAX_SEQ_LEN=self.settings["fitting"].get("max_seq_len", 300),
            checkpoint_interval=self.settings["fitting"].get(
                "checkpoint_interval", 5
            ),
            mode= None,
            is_test=True,
        )
    
    def test_lens(self):
        data_root = self._configured_data_root()
        jsonl_path = self.settings["fitting"]["jsonl_path"]
        max_length = self.settings["inference"].get("max_length", 300)
        
        if not data_root or not jsonl_path:
            raise ValueError(
                "請在 settings.json 設定 fitting.root 與 fitting.jsonl_path"
            )

        jsonl_path = os.path.abspath(jsonl_path)
        lines = Parser.load_jsonl_lines(jsonl_path)
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
    
    # Fitting methods
    # ---------------------------------------------------------------------------------------
    def calc_JLens_batch(self, mode: str | None = None):
        """Calls the model's fit_Jlens method to calculate JLens and save it to a checkpoint.
        
        Args:
            mode: Defines the source and target tokens for Jacobian matrix calculation.
                Can be one of the following:
                - "S2T"
                - "T2T"
                - "S&T2T"
                - "None" (Will use ordinary fit_jlens() function)
        """
        checkpoint_root = self.settings["fitting"]["checkpoint_root"]
        checkpoint_name = self.run_name or "default"
        checkpoint_directory = os.path.join(checkpoint_root, checkpoint_name)
        checkpoint_path = os.path.join(checkpoint_directory, "fit.pt")
        os.makedirs(checkpoint_directory, exist_ok=True)
        
        # Copy settings.json to the checkpoint directory for reproducibility
        settings_copy_path = os.path.join(checkpoint_directory, "settings.json")
        from shutil import copy
        copy("settings.json", settings_copy_path)

        self.model.fit_Jlens_batch(
            jsonl_path= self.settings["fitting"]["jsonl_path"],
            data_root=self._configured_data_root(),
            checkpoint_save_path=checkpoint_path,
            run_name= self.run_name,
            dim_batch=self.settings["fitting"].get("dim_batch", 32),
            MAX_SEQ_LEN=self.settings["fitting"].get("max_seq_len", 300),
            checkpoint_interval=self.settings["fitting"].get(
                "checkpoint_interval", 5
            ),
            mode= mode
        )

    def calc_Jlens_experiment(self):
        """Fit dataset-specific and cumulative-count JLens for an experiment.

        ``fitting.experiment`` settings:
          - ``seed``: deterministic shuffle seed (default: 42)
          - ``mode``: token subset mode, or null for ordinary fitting
          - ``save_prompt_count``: cumulative successful-sample counts
          - ``dim_batch``: Jacobian output batch size
          - ``checkpoint_interval``: fit-state save interval
        """
        fitting_settings = self.settings["fitting"]
        experiment_settings = fitting_settings.get("experiment", {})
        if not isinstance(experiment_settings, dict):
            raise ValueError("settings.json 'fitting.experiment' must be an object")

        seed = experiment_settings.get("seed", 42)
        if not isinstance(seed, int):
            raise ValueError("settings.json 'fitting.experiment.seed' must be an integer")

        milestone_counts = experiment_settings.get(
            "save_prompt_count", [1, 2, 5, 10, 20, 50, 100, 200, 500]
        )
        if (
            not isinstance(milestone_counts, list)
            or any(not isinstance(count, int) or count <= 0 for count in milestone_counts)
        ):
            raise ValueError(
                "settings.json 'fitting.experiment.save_prompt_count' must be a list "
                "of positive integers"
            )

        checkpoint_name = self.run_name or "default"
        experiment_directory = os.path.join(
            fitting_settings["checkpoint_root"],
            checkpoint_name,
            "experiment",
        )
        
        # Ensure the experiment directory exists
        os.makedirs(experiment_directory, exist_ok=True)
        checkpoint_path = os.path.join(experiment_directory, "fit.pt")
        
        # Copy settings.json to the experiment directory for reproducibility
        settings_copy_path = os.path.join(experiment_directory, "settings.json")
        from shutil import copy
        copy("settings.json", settings_copy_path)

        # Fit
        self.logger.info(
            "Starting experiment JLens fit: seed=%d, milestones=%s, output=%s",
            seed,
            milestone_counts,
            experiment_directory,
        )
        return self.model.fit_Jlens_for_experiment(
            jsonl_path=os.path.abspath(fitting_settings["jsonl_path"]),
            data_root=self._configured_data_root(),
            seed=seed,
            dim_batch=fitting_settings.get("dim_batch", 32),
            MAX_SEQ_LEN=fitting_settings.get("max_seq_len", 300),
            mode=experiment_settings.get("mode"),
            save_prompt_count=milestone_counts,
            checkpoint_save_path=checkpoint_path,
            run_name=checkpoint_name,
            is_test=experiment_settings.get("is_test", False),
            checkpoint_interval=fitting_settings.get("checkpoint_interval", 5),
        )
    
    # Experiment methods
    # ---------------------------------------------------------------------------------------
    def ablation(self):
        Experiments.ablation()
    
    def comp_dataset_choice(self):
        Experiments.comp_dataset_choice()
    
    def draw_Jlens_heatmap(
        self,
        lens_path1: str,
        lens_path2: str | None = None,
    ) -> torch.Tensor:
        return Experiments.draw_Jlens_cka_heatmap(
            logger=self.logger,
            run_name=self.run_name,
            heatmap_path=os.path.join(
                self.settings["experiment"]["matrix_analysis"]["output_root"],
                self.run_name or "default",
            ),
            lens_path1=lens_path1,
            lens_path2=lens_path2,
        )

    def draw_Jlens_svd_heatmaps(
        self,
        lens_path: str,
        top_k: int | None = None,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[str]]:
        return Experiments.draw_Jlens_svd_heatmaps(
            logger=self.logger,
            run_name=self.run_name,
            output_dir=self.settings["experiment"]["matrix_analysis"]["output_root"],
            lens_path=lens_path,
            top_k=top_k or int(self.settings["experiment"]["matrix_analysis"]["SVD_top_k"]),
            device=device,
        )
    
    def comp_Jlens(self, lens_path1, lens_path2):
        return Experiments.comp_Jlens(
            logger=self.logger,
            lens_path1=lens_path1,
            lens_path2=lens_path2,
        )
    
    #TODO
    def eval_Jlens(
        self,
        lens_path: str,
        prompts_path: str,
    ):
        """First calls the inference function, calculate the .csv file.
        Then calls the evaluation function to compute the pass@k AUC.

        Args:
            lens_path (str): Path to the Jlens file.
            prompts_path (str): Path to the prompt JSONL file.

        Returns:
            str: Path to the output directory containing the inference results.
        """
    
    # Inference methods
    # ---------------------------------------------------------------------------------------
    def apply_Jlens(
        self,
        lens_path: str,
        prompts_path: str,
        top_k: int | None = None,
        ) -> str:
        """Inference the Jlens on the given prompt set.

        Args:
            lens_path (str): Path to the Jlens file.
            prompts_path (str): Path to the prompt JSONL file.
            top_k (int | None, optional): Number of top predictions to consider. 
                The Default is at settings["inference"]["top_k"].
        """
        json_lines = Parser.load_jsonl_lines(prompts_path)
        self.model.load_Jlens(lens_path= lens_path)
        
        output_root = os.path.join(
            self.settings["inference"]["output_root"], self.run_name
        )
        os.makedirs(output_root, exist_ok=True)
        for i in range(len(json_lines)):
            self.model.apply_lens(
                json_line = json_lines[i],
                data_root=self._configured_data_root(),
                save_path= output_root,
                top_k= top_k or self.settings["inference"]["top_k"],
                use_Jlens= True,
                run_name= self.run_name,
            )
        return output_root

    
    @staticmethod
    def check_dependencies():
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
