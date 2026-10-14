from jlens.lens import JacobianLens
import os, logging
from audio_jlens import lens as l
from audio_jlens.subset_fitting import subset_fit
from model.instance import Instance
from json_parser import Parser

class Jlens(l.Lens):
    """
    Master class of all Jlens related methods.
    """
    def __init__(self, logger: logging.Logger, model: Instance | None = None):
        self.model = model
        self.lens: JacobianLens | None = None
        self.logger = logger

    def init_zero(self):
        """Initialize the lens to zero."""
        if self.lens is not None:
            raise RuntimeError("Lens is already initialized.")
        
        self.lens = JacobianLens(
            jacobians={},
            n_prompts=0,
            d_model=self._require_model().d_model,
        )

    def _require_model(self) -> Instance:
        if self.model is None:
            raise RuntimeError(
                "A model instance is required for fitting or applying a JLens."
            )
        return self.model

    def _load_data(self, data_root: str):
        if not data_root:
            raise ValueError("No data root passed to Jlens.")
        self._require_model().load_data(data_root)
 
    def calc_lens_batch(
        self,
        data_path: str,
        jsonl_name: str,
        mode: str | None = None,
        checkpoint_path: str | None = None,
        checkpoint_every: int = 5,
        run_name: str | None = None,
        dim_batch: int = 32,
        MAX_SEQ_LEN: int = 300,
        is_test: bool = False,
        data_root: str | None = None,
        ):
        """
        * This is the basic fitting function.
        * Calculates Jlens using the JSONL in data_path and audio paths relative
          to data_root (or data_path when data_root is omitted).
        * If run_name is specified, the function ignores the path and looks for it in output/checkpoints/*name*/lens.pt
        * If a checkpoint named *run_name* is already in there, automatically try run_name_1 or more.
        * Replaces the original if replace = True is specified.
        * No duplicate protection when full path is specified!!!
        """
        # Get model
        model = self._require_model()
        
        # Load JSONL and audio data
        jsonl_path = os.path.abspath(os.path.join(data_path, jsonl_name))
        audio_root = data_root if data_root is not None else data_path
        self._load_data(audio_root)
        self.logger.info("Loading metadata from %s", jsonl_path)
        
        # Passes the raw JSONL records directly to the parser
        records = Parser.load_jsonl_lines(jsonl_path)
        if is_test: records = [records[i] for i in range(min(5, len(records)))]
        self.logger.info("Loaded %d samples", len(records))

        # The source and target layers for Jlens fitting.
        n_layers = model.n_layers
        source_layers = list(range(n_layers - 1))
        target_layer = n_layers - 1
        
        # If in test mode, only use the second-to-last layer as the source layer.
        if is_test: source_layers = [n_layers - 2]
    
        self.logger.info(
            "Fitting all %d layers together: source_layers=%s..%s, target_layer=%s",
            n_layers, source_layers[0], source_layers[-1], target_layer,
        )
        self.logger.info(
            "dim_batch=%d, max_seq_len=%d, checkpoint_every=%d, checkpoint_path=%s",
            dim_batch, MAX_SEQ_LEN, checkpoint_every, checkpoint_path,
        )
        
        # Fit
        self.logger.info("fitting started. %r", self.lens)
        self.lens = subset_fit(
            model=model,
            prompts=records,
            processor=model.processor,
            data_root=audio_root,
            mode=mode,
            source_layers=source_layers,
            target_layer=target_layer,
            dim_batch=dim_batch,
            max_seq_len=MAX_SEQ_LEN,
            checkpoint_path=checkpoint_path,
            checkpoint_every=checkpoint_every,
        )
        self.logger.info("fitting finished. %r", self.lens)
        self.logger.info("Saving lens...")

        # Rigorously ensuring the lens is actually saved.
        if checkpoint_path is not None:
            # Keep the resumable fit checkpoint and final lens together by task.
            save_dir = os.path.join(os.path.dirname(checkpoint_path), "lens.pt")
        elif run_name is not None:
            save_dir = os.path.join(os.getcwd(), "output", "checkpoints", run_name, "lens.pt")
        else:
            from random import choices
            from string import digits, ascii_uppercase

            alphabet = digits + ascii_uppercase
            name = ''.join(choices(alphabet, k=16))
            self.logger.warning(f"No checkpoint path or name specified in calc_lens. Saving as {name} to output/checkpoints.")
            save_dir = os.path.join(os.getcwd(), "output", "checkpoints", name, "lens.pt")
        
        parent = os.path.dirname(save_dir)
        if parent:
            os.makedirs(parent, exist_ok=True)

        self.logger.info("Saving lens to %s", save_dir)
        self.logger.info("*"+"-"*20+"*")
        self.save_lens(path = save_dir)

    def add_prompt(
        self,
        
    ):
        """
        * Calculate the Jlens for the single prompt given, filter it using the top_k most significant directions.
            the same filtering logic as in fitting.py, and adds it to the lens.
        * Also initilizes the lens if it hasn't been created yet.
        """
        from audio_jlens.subset_fitting import subset_jacobian_for_prompt
        self.init_zero()

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

        model = self._require_model()
        run_layers = []
        if layers_available is None:
            if do_activate_Jacobian:
                run_layers = list(range(model.n_layers - 1))
            else:
                run_layers = list(range(model.n_layers))
        else:
            run_layers = layers_available
        
        if json_line is None:
            raise ValueError("Missing Json line.")
        
        lens_logits, model_logits, input_ids = self.lens.apply(
            model=model,
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
        * If name is specified, the function ignores the path and looks for it in output/checkpoints/*name*/lens.pt
        """
        if name is not None:
            checkpoint_path = os.path.join(
                os.getcwd(), "output", "checkpoints", name, "lens.pt"
            )
        elif path is not None:
            checkpoint_path = path
        else:
            raise ValueError("No checkpoint path or name specified in load_lens. Exiting...")

        try:
            self.lens = JacobianLens.load(checkpoint_path)
            self.logger.info(f"Successfully loaded lens from {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Failed loading lens from {checkpoint_path}({e}).")

    def get_J_vector(self, layer: int, token: int):
        assert self.lens is not None
        if token > self.lens.jacobians[layer].shape[0]:
            raise ValueError(f"Token index {token} exceeds model dimension {self.lens.jacobians[layer].shape[0]}")
        return self.lens.jacobians[layer][token]