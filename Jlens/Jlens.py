from jlens.fitting import fit as jlens_fit
from jlens.lens import JacobianLens
import sys, os, json, librosa, logging
import torch
from Jlens import lens as l
from Jlens.Subset_fitting import subset_fit
from model.instance import Instance

#TODO update the fit function for subset fitting

class Jlens(l.Lens):
    """
    Master class of all Jlens related methods.
    """
    def __init__(self, logger: logging.Logger, model: Instance):
        self.model = model
        self.lens = None
        self.logger = logger

    def _load_data(self, data_root: str):
        if not data_root:
            raise ValueError("No data root passed to Jlens.")
        self.model.load_data(data_root)

    def calc_lens(self, data_path: str, jsonl_name: str, do_replace: bool = False,
                  checkpoint_path: str | None = None, run_name: str | None = None,
                  dim_batch: int = 32, MAX_SEQ_LEN: int = 300, checkpoint_every: int = 5,
                  is_test: bool = False):
        """
        * Calculates Jlens for the specified model using data in the given data_path.
        * If run_name is specified, the function ignores the path and looks for it in ~/Jlens/lens_checkpoints/*name*
        * If a checkpoint named *run_name* is already in there, automatically try run_name_1 or more.
        * If more than 10 is present, it'll be named as *run_name*_1919810😢.
        * Replaces the original if replace = True is specified.
        * WARNING: No duplicate protection when full path is specified!!!
        """
        import parser
        parser = parser.Parser()
    
        jsonl_path = os.path.abspath(os.path.join(data_path, jsonl_name))
        self._load_data(data_path)
        self.logger.info("Loading metadata from %s", jsonl_path)
        records = parser.load_jsonl_lines(jsonl_path)
        if is_test: records = [records[0]]    
        self.logger.info("Loaded %d samples", len(records))
        print("Loaded %d samples", len(records))
    
        self.logger.info("Filtering samples by length (max_seq_len=%d) before fitting...", MAX_SEQ_LEN)
        records = process_audio(
            records = records,
            logger = self.logger,
            processor=self.model.processor,
            data_root=data_path,
            sampling_rate=self.model.sampling_rate,
            max_length=MAX_SEQ_LEN,
        )
        if not records:
            raise ValueError("長度過濾後沒有剩下任何樣本，請檢查 max_seq_len 設定或資料本身。Exiting...")
    
        n_layers = self.model.n_layers  # 應為 32
        source_layers = list(range(n_layers - 1))  # 0..30，共 31 層
        target_layer = n_layers - 1  # 31
        
        if is_test: source_layers = [n_layers - 2]
    
        self.logger.info(
            "Fitting all %d layers together: source_layers=%s..%s, target_layer=%s",
            n_layers, source_layers[0], source_layers[-1], target_layer,
        )
        self.logger.info(
            "dim_batch=%d, max_seq_len=%d, checkpoint_every=%d, checkpoint_path=%s",
            dim_batch, MAX_SEQ_LEN, checkpoint_every, checkpoint_path,
        )
    
        self.lens = jlens_fit(
            model=self.model,
            prompts=records,
            source_layers=source_layers,
            target_layer=target_layer,
            dim_batch=dim_batch,
            max_seq_len=MAX_SEQ_LEN,
            checkpoint_path=checkpoint_path,
            checkpoint_every=checkpoint_every,
        )
    
        self.logger.info("fitting finished. %r", self.lens)
        self.logger.info("Saving lens...")

        save_dir = ""
        if run_name is not None:
            save_directory_dir = os.path.join(os.getcwd(), "Jlens", "lens_checkpoints")
            save_dir = os.path.join(save_directory_dir, run_name)
            if not do_replace and os.path.exists(save_dir):
                save_success = False
                for i in range(10):
                    if not os.path.exists(os.path.join(save_directory_dir, run_name + f"_{i}")):
                        self.logger.warning(f"File {save_dir} already exists. Saving as {run_name}_{i}")
                        save_dir = os.path.join(save_directory_dir, run_name + f"_{i}")
                        save_success = True
                        break
                if not save_success:
                    self.logger.error(f"File {save_dir} have 10+ checkpoint with the same name. Check your code 😭.")
                    save_dir = os.path.join(save_directory_dir, run_name + f"_{1919810}")
        elif checkpoint_path is not None:
            save_dir = checkpoint_path  
        else:
            from random import choices
            from string import digits, ascii_uppercase

            alphabet = digits + ascii_uppercase
            name = ''.join(choices(alphabet, k=16))
            self.logger.warning(f"No checkpoint path or name specified in calc_lens. Saving as {name} to lens_checkpoints.")
            save_dir = os.path.join(os.getcwd(), "Jlens", "lens_checkpoints", name)
        
        parent = os.path.dirname(save_dir)
        if parent:
            os.makedirs(parent, exist_ok=True)

        self.save_lens(path = save_dir)

    #TODO: This is Qwen-specific. Has to be changed to work for more models.
    def apply_lens(
            self, 
            do_activate_Jacobian: bool = True,
            json_line: str | None = None,
            layers_available: list[int] | None = None,
            positions: list[int] | None = None,
            top_k: int = 5,
            MAX_SEQ_LEN: int = 300,
            data_root: str | None = None ):
        """
        * Applies Lens to model. To use LogitLens, use do_activate_Jacobian = False
        * json_line takes raw json data.
        * Returns in order: lens_logits, model_logits, input_ids
        * The dimension of the returned data is
        
        lens_logits[layer].shape
            =[len(positions), vocab_size]
        model_logits.shape
            =[len(positions), vocab_size]
        input_ids.shape
            =[1, sequence_length]
        """
        if self.lens is None:
            raise RuntimeError("Jlens not yet calculated, please call calc_lens() before applying.")
        if data_root is not None:
            self._load_data(data_root)

        run_layers = []
        if layers_available is None:
            if do_activate_Jacobian:
                run_layers = list(range(self.model.n_layers - 1))
            else:
                run_layers = list(range(self.model.n_layers))
        else:
            run_layers = layers_available
        
        if json_line is None:
            raise ValueError("Missing Json line.")
        
        lens_logits, model_logits, input_ids = self.lens.apply(
            model= self.model,
            prompt= json_line,
            layers= run_layers,
            positions= positions,
            max_seq_len= MAX_SEQ_LEN,
            use_jacobian= do_activate_Jacobian,
        )

        return lens_logits, model_logits, input_ids

    def save_lens(self, path: str | None = None):
        """
        Saves the Jlens to the specified path.
        WARNING: No duplicate protection implemented. Just use the auto-save feature from calc_lens😡
        """
        if self.lens is None:
            self.logger.error("No lens to save.")
            return 1
        if path is None:
            path = os.path.join(os.getcwd(), "Jlens", "lens_checkpoints")
        try:
            self.lens.save(path)
            self.logger.info(f"lens saved at {path}")
            return 0
        except Exception as e:
            fallback = os.getcwd()
            fallback_path = os.path.join(fallback, os.path.basename(path))
            self.logger.error(f"[Lens] failed to save to {path}({e}). Saved to {fallback_path}")
            try:
                self.lens.save(fallback_path)
                return 0
            except Exception as e1:
                self.logger.critical(f"Failed to save lens({e1}).")
                return 1
                
    def load_lens(self, path: str | None = None, name: str | None = None):
        """
        * If name is specified, the function ignores the path and looks for it in ~/Jlens/lens_checkpoints/*name*
        """
        if name is not None:
            checkpoint_path = os.path.join(os.getcwd(), "Jlens", "lens_checkpoints", name)
        elif path is not None:
            checkpoint_path = path
        else:
            raise ValueError("No checkpoint path or name specified in load_lens. Exiting...")

        try:
            self.lens = JacobianLens.load(checkpoint_path)
            self.logger.info(f"Successfully loaded lens from {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Failed loading lens from {checkpoint_path}({e}).")

# TODO: Decide on a default value for max_length! Require testing on a physical device.
def process_audio(records, processor, logger: logging.Logger, data_root: str, sampling_rate: int, max_length: int):
    """
    * Processes the audio.
    * Filters all samples with length(after encoding) longer than max_length
    """
    kept, dropped = [], []
    for line in records:
        record = json.loads(line)
        audio_rel = record.get("audio_path")
        prompt = record.get("prompt")
        if audio_rel is None or prompt is None:
            logger.warning("跳過缺少欄位的樣本：%s", record.get("id", "<unknown>"))
            dropped.append(record.get("id", "<unknown>"))
            continue

        try:
            audio_array, _ = librosa.load(os.path.join(data_root, audio_rel), sr=sampling_rate)
        except Exception as e:
            logger.warning("跳過載入失敗的樣本 %s：%s", record.get("id", "<unknown>"), e)
            dropped.append(record.get("id", "<unknown>"))
            continue
        
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