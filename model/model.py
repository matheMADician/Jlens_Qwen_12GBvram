import os, logging, json, csv
from pathlib import Path
import torch
from torch import Tensor

from audio_jlens import jlens as J, lens, logitLens as L
from model.adapters.Instance import AblationSpec, Instance
from model.adapters.qwen25_omni import Instance as Qwen25OmniInstance
from model.adapters.qwen2_audio import Instance as Qwen2AudioInstance
from jacobian_lens.jlens.lens import JacobianLens


MODEL_ADAPTERS: dict[str, type[Instance]] = {
    "Qwen/Qwen2.5-Omni-7B": Qwen25OmniInstance,
    "Qwen/Qwen2-Audio-7B": Qwen2AudioInstance,
}


class LensOutputCache:
    """
    * Cache for the latest JLens and LogitLens outputs.
    * Each output is stored as:
      [lens_logits, model_logits, input_ids]
    """
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.jlens_data: tuple[
            dict[int, Tensor],
            Tensor,
            Tensor,
        ] | None = None
        self.logit_lens_data: tuple[
            dict[int, Tensor],
            Tensor,
            Tensor,
        ] | None = None

    def _get_data(
        self,
        lens_type: str,
    ) -> tuple[dict[int, Tensor], Tensor, Tensor] | None:
        """
        * Returns the full data.
        * Use get() instead, unless necessary.
        """
        if lens_type == "jlens":
            return self.jlens_data
        if lens_type == "logit_lens":
            return self.logit_lens_data
        raise ValueError(f"Unknown lens type: {lens_type}")

    def get(
        self,
        lens_type: str,
        key: str,
        layer: int | None = None,
    ) -> Tensor:
        """
        * Input for lens_type: jlens or logit_lens.
        * Input for key: lens_logits, model_logits, input_ids.
        * For lens_logits, layer is required to prevent too much data loaded at once.
        """
        lens_data = self._get_data(lens_type)
        if lens_data is None:
            raise RuntimeError(f"No cached output for {lens_type}.")

        if key == "lens_logits":
            if layer is None:
                raise ValueError("No layer specified.")
            return lens_data[0][layer]
        elif key == "model_logits":
            return lens_data[1]
        elif key == "input_ids":
            return lens_data[2]
        else:
            raise ValueError("No such data type in cache.")

    def save(self, save_path: str, run_name: str, top_k: int = 5):
        """Append cached logits to a CSV file, grouped by token position."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        output_path = Path(save_path)
        if output_path.exists() and output_path.is_dir():
            output_path /= run_name + ".csv"
        elif output_path.suffix.lower() != ".csv":
            output_path = output_path.with_suffix(".csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cached = []
        if self.jlens_data is not None:
            cached.append(("jlens", self.jlens_data))
        if self.logit_lens_data is not None:
            cached.append(("logit_lens", self.logit_lens_data))
        if not cached:
            raise RuntimeError("No cached lens output to save.")

        prepared = {}
        layer_names = set()
        for lens_type, (layer_logits, model_logits, input_ids) in cached:
            input_ids = input_ids.detach().cpu().reshape(-1)
            layer_rows = [(str(layer), logits) for layer, logits in layer_logits.items()]
            layer_rows.append(("model", model_logits))
            prepared_layers = []
            for layer, logits in layer_rows:
                logits = logits.detach().float().cpu()
                if logits.ndim == 1:
                    logits = logits.unsqueeze(0)
                actual_top_k = min(top_k, logits.shape[-1])
                probabilities = torch.softmax(logits, dim=-1)
                entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
                top_probs, top_ids = probabilities.topk(actual_top_k, dim=-1)
                prepared_layers.append((layer, top_probs, top_ids, entropy))
                layer_names.add(layer)
            prepared[lens_type] = (input_ids, prepared_layers)

        fieldnames = [
            "position",
            "layer",
            "input_token",
            "top_k_Jlens",
            "top_k_LogitLens",
            "other_info_Jlens",
            "other_info_LogitLens",
        ]

        write_header = not output_path.exists() or output_path.stat().st_size == 0
        # utf-8-sig keeps the file UTF-8 while allowing Excel to detect it.
        with output_path.open("a", newline="", encoding="utf-8-sig") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            input_ids = next(iter(prepared.values()))[0]
            prepared_by_layer = {
                lens_type: {
                    layer: (top_probs, top_ids, entropy)
                    for layer, top_probs, top_ids, entropy in prepared_layers
                }
                for lens_type, (_, prepared_layers) in prepared.items()
            }
            n_positions = max(
                tensor.shape[0]
                for _, prepared_layers in prepared.values()
                for _, tensor, _, _ in prepared_layers
            )
            for position in range(n_positions):
                input_token = ""
                if position < len(input_ids):
                    input_token = self._decode_token(int(input_ids[position]))
                for layer in sorted(
                    layer_names,
                    key=lambda value: (value == "model", int(value) if value != "model" else 0),
                ):
                    row = {
                        "position": position,
                        "layer": layer,
                        "input_token": input_token,
                    }
                    for lens_type, column_suffix in (
                        ("jlens", "Jlens"),
                        ("logit_lens", "LogitLens"),
                    ):
                        values = prepared_by_layer.get(lens_type, {}).get(layer)
                        if values is None or position >= values[1].shape[0]:
                            row[f"top_k_{column_suffix}"] = ""
                            row[f"other_info_{column_suffix}"] = ""
                            continue
                        top_probs, top_ids, entropy = values
                        token_text = [
                            self._decode_token(token_id)
                            for token_id in top_ids[position].tolist()
                        ]
                        row[f"top_k_{column_suffix}"] = json.dumps(
                            token_text,
                            ensure_ascii=False,
                        )
                        row[f"other_info_{column_suffix}"] = json.dumps(
                            {
                                "top_tokens": token_text,
                                "top_k_probability": float(top_probs[position].sum()),
                                "top_probability": float(top_probs[position, 0]),
                                "entropy": float(entropy[position]),
                            },
                            ensure_ascii=False,
                        )
                    writer.writerow(row)

        return str(output_path)

    def _decode_token(self, token_id: int) -> str:
        if self.tokenizer is None:
            return str(token_id)
        try:
            return self.tokenizer.decode(
                [token_id], clean_up_tokenization_spaces=False
            ).replace("\n", "\\n")
        except (AttributeError, TypeError, ValueError):
            return str(token_id)
    
    def put(
        self,
        lens_type: str,
        data: tuple[dict[int, Tensor], Tensor, Tensor],
    ):
        if lens_type == "jlens":
            self.jlens_data = (
                {
                    layer: logits.detach().cpu()
                    for layer, logits in data[0].items()
                },
                data[1].detach().cpu(),
                data[2].detach().cpu(),
            )
        elif lens_type == "logit_lens":
            self.logit_lens_data = (
                {
                    layer: logits.detach().cpu()
                    for layer, logits in data[0].items()
                },
                data[1].detach().cpu(),
                data[2].detach().cpu(),
            )
        else:
            raise ValueError(f"Unknown lens type: {lens_type}")

    def clear(self):
        self.jlens_data = None
        self.logit_lens_data = None

    def clear_jlens(self):
        self.jlens_data = None

    def clear_logit_lens(self):
        self.logit_lens_data = None
        
    #TODO
    def find_token(self, lens: str, token_id: int) -> tuple[list[int], list[int]]:
        """Find the position where a token appears in the predictions for a specific lens type.

        Args:
            lens (str): The type of lens to search in ("jlens" or "logit_lens").
            token_id (int): The ID of the token to find in the predictions.

        Returns:
            max_posision[layer], :
                - First list contains the token positions where the token has the highest probability for the given token_id.
                - The second list contains the highest rank of each token across token positions.
        """
        return [], []

class Model:
    """
    Master class of all model methods.
    Loads the model on demand; repeated load_model() calls reuse the same instance.
    LogitLens is the default. To load Jlens, use load_lens(), and to calculate Jlens, use calc_Jlens().
    """
    def __init__(self,
        logger: logging.Logger,
        MODEL_ID: str,
        adapter: str,
        do_4bit: bool = False,
    ):
        
        self.logger = logger
        self.model_id = MODEL_ID
        self.do_4bit = do_4bit
        self.adapter_name = adapter
        try:
            self.adapter_class: type[Instance] = MODEL_ADAPTERS[adapter]
        except KeyError as exc:
            raise ValueError(
                f"Unknown model adapter {adapter!r}; "
                f"available adapters: {sorted(MODEL_ADAPTERS)}"
            ) from exc
        # A model is fixed to a single Jlens. fit_Jlens_for_experiment
        #   will create multiple sub-lens, but only the combined one is binded.
        self.lens: lens.Lens | None = None
        self._model: Instance | None = None
        self.lens_cache = LensOutputCache()

    def load_model(self) -> Instance:
        """Load the model once, reusing it on subsequent calls."""
        if self._model is None:
            model = self.adapter_class(
                MODEL_ID=self.model_id,
                do_4bit=self.do_4bit,
            )
            self._model = model
            self.lens_cache.tokenizer = model.tokenizer
        return self._model

    def scoped_output_root(self, output_root: str) -> str:
        """Append the model ID's final path component to an output root."""
        if not output_root:
            raise ValueError("output_root must not be empty")
        model_name = self.model_id.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]
        if not model_name or model_name in {".", ".."}:
            raise ValueError(f"Cannot derive a model name from {self.model_id!r}")

        root = Path(os.path.expanduser(output_root))
        if root.name == model_name:
            return str(root)
        return str(root / model_name)

    @property
    def model(self) -> Instance:
        return self.load_model()

    @property
    def hf_model(self):
        return self.load_model().hf_model

    @property
    def processor(self):
        return self.load_model().processor

    def get_model(self):
        model = self.load_model()
        return model, model.processor

    def get_lens_output(self) -> LensOutputCache:
        return self.lens_cache

    def load_Jlens(self, lens_path: str | None = None, lens_name: str | None = None):
        """
        * If name is specified, the function ignores the path and looks for it in output/checkpoints/*name*/lens.pt
        """
        self.lens = J.Jlens(logger=self.logger)
        self.lens.load_lens(path= lens_path, name= lens_name)
        self.lens_cache.clear_jlens()
 
    def fit_Jlens_batch(
            self, jsonl_path: str,
            dim_batch: int = 32,
            checkpoint_interval: int = 5,
            MAX_SEQ_LEN: int = 300,
            mode: str | None = None,
            checkpoint_save_path: str | None = None,
            run_name: str | None = None,
            is_test: bool = False,
            data_root: str | None = None,
        ):
        """
        * Calls the calc_lens() function in Jlens.
        * See ~/audio_jlens/Jlens.py for detail.
        * jsonl_path accepts the relative path to jsonl files.
        * data_root is the root used to resolve audio paths inside each JSONL record.
          Defaults to the JSONL's parent directory for backwards compatibility.
        * WARNING: This cleans the cache.
        * Audio paths in JSONL records are resolved relative to data_root.
        * jsonl_path is located independently of data_root.
        """
        # Parse the jsonl_path into data path(parent directory) and jsonl name
        if not os.path.isfile(jsonl_path):
            raise RuntimeError(f"Jsonl file({jsonl_path}) does not exist.")
        parent_dir, jsonl_name = os.path.split(jsonl_path)
        audio_root = data_root if data_root is not None else parent_dir

        model = self.load_model()
        self.lens = J.Jlens(logger=self.logger, model=model)
        self.lens.calc_lens_batch(
            data_path= parent_dir,
            data_root= audio_root,
            jsonl_name= jsonl_name,
            mode= mode,
            checkpoint_path= checkpoint_save_path,
            run_name= run_name,
            dim_batch= dim_batch,
            MAX_SEQ_LEN= MAX_SEQ_LEN,
            checkpoint_every= checkpoint_interval,
            is_test= is_test
            )
        self.lens_cache.clear_jlens()

    def fit_Jlens_for_experiment(
        self,
        jsonl_path: str,
        seed: int,
        dim_batch: int = 32,
        MAX_SEQ_LEN: int = 300,
        mode: str | None = None,
        save_prompt_count: list[int] | None = None,
        checkpoint_save_path: str | None = None,
        run_name: str | None = None,
        is_test: bool = False,
        data_root: str | None = None,
        checkpoint_interval: int = 5,
        max_prompts: int | None = None,
    ) -> tuple[JacobianLens, dict[str, JacobianLens], dict[int, JacobianLens]]:
        """Fit and save Jacobian lenses for data-size and per-dataset experiments.

        `max_prompts`, when set, caps the deterministically shuffled input
        records before fitting.

        Saves:
          1. One JacobianLens for each dataset.
          2. Progressive cumulative JacobianLens snapshots at each milestone count
             in `save_prompt_count` (e.g. [1, 2, 5, 10, 20, 50, ...]).
          3. Final combined lens across all processed samples.

        Binds the combined lens to `self.lens`.
        """
        from json_parser import Parser
        from audio_jlens.subset_fitting import subset_fit_experiment

        if not os.path.isfile(jsonl_path):
            raise RuntimeError(f"Jsonl file({jsonl_path}) does not exist.")

        parent_dir, _ = os.path.split(jsonl_path)
        audio_root = data_root if data_root is not None else parent_dir

        model = self.load_model()
        model.load_data(audio_root)

        records_str: list[str] = Parser.load_jsonl_lines(jsonl_path)
        if is_test:
            records_str = records_str[: min(5, len(records_str))]
        self.logger.info("Loaded %d raw sample records from %s", len(records_str), jsonl_path)

        n_layers = model.n_layers
        source_layers = list(range(n_layers - 1))
        target_layer = n_layers - 1
        if is_test:
            source_layers = [n_layers - 2]

        # Determine output directory
        if checkpoint_save_path is not None:
            output_dir = os.path.dirname(checkpoint_save_path)
            ckpt_file = checkpoint_save_path
        elif run_name is not None:
            output_dir = os.path.join(
                self.scoped_output_root(os.path.join(os.getcwd(), "output", "checkpoints")),
                run_name,
            )
            ckpt_file = os.path.join(output_dir, "fit_checkpoint.pt")
        else:
            output_dir = os.path.join(
                self.scoped_output_root(os.path.join(os.getcwd(), "output", "checkpoints")),
                f"experiment_seed_{seed}",
            )
            ckpt_file = os.path.join(output_dir, "fit_checkpoint.pt")

        os.makedirs(output_dir, exist_ok=True)

        combined_lens, dataset_lenses, milestone_lenses = subset_fit_experiment(
            model=model,
            prompts=records_str,
            seed=seed,
            mode=mode,
            save_prompt_counts=save_prompt_count,
            source_layers=source_layers,
            target_layer=target_layer,
            dim_batch=dim_batch,
            max_seq_len=MAX_SEQ_LEN,
            output_dir=output_dir,
            checkpoint_path=ckpt_file,
            checkpoint_every=checkpoint_interval,
            resume=os.path.exists(ckpt_file),
            max_prompts=max_prompts,
            processor=model.processor,
            data_root=audio_root,
        )

        # Bind the combined lens to self.lens
        jlens_wrapper = J.Jlens(logger=self.logger, model=model)
        jlens_wrapper.lens = combined_lens
        self.lens = jlens_wrapper
        self.lens_cache.clear_jlens()

        return combined_lens, dataset_lenses, milestone_lenses

    def apply_lens(
            self,
            json_line: str,
            data_root: str,
            run_name: str,
            save_path: str | None = None,
            use_Jlens: bool = True,
            top_k: int = 5,
            layers_available: list[int] | None = None,
            MAX_SEQ_LEN: int = 300,
            ):
        """
        * Auto-save the data into model cache.
        * If save_path is specified, creates and auto-saves into the .csv file.
        """
        if self.lens is None and use_Jlens:
            raise ValueError("No lens in model. Call fit_Jlens() / load_Jlens() first.")
        model = self.load_model()
        if self.lens is None:
            self.lens = L.LogitLens(
                logger= self.logger,
                model=model
            )
        elif isinstance(self.lens, J.Jlens):
            self.lens.model = model
        
        # Do both Jlens and LogitLens when not testing
        runs = [True, False] if use_Jlens else [False]
        for do_activate_Jacobian in runs:
            result = self.lens.apply_lens(
                do_activate_Jacobian= do_activate_Jacobian,
                json_line= json_line,
                layers_available= layers_available,
                positions= None,
                top_k= top_k,
                MAX_SEQ_LEN= MAX_SEQ_LEN,
                data_root= data_root
            )
            self.lens_cache.put(
                "jlens" if do_activate_Jacobian else "logit_lens",
                result,
            )
        
        if save_path is not None:
            self.lens_cache.save(save_path= save_path, run_name= run_name, top_k= top_k)

    def encode_prompt(self, json_line: str, MAX_LENGTH: int = 300):
        return self.load_model().encode(text=json_line, max_length=MAX_LENGTH)
        
    def run_inference(
        self,
        jsonl_line: str,
        data_root: str,
        MAX_LENGTH: int = 300,
    ):
        """
        * Returns the last hidden state
        """
        model = self.load_model()
        model.load_data(data_root=data_root)
        return model.unembed(
            model.forward(
                self.encode_prompt(jsonl_line, MAX_LENGTH=MAX_LENGTH)
            )
        )
         
    def _token_directions(self, token_id: int, layers: list[int]) -> dict[int, torch.Tensor]:
        """
        * Unit direction of token_id in the residual space of each layer: d_l = J_l^T u,
          where u is the token's unembedding row scaled by the final norm weight.
        * Moving a residual along d_l changes the lens logit of that token.
        """
        jacobian_lens = getattr(self.lens, "lens", None)
        if not isinstance(self.lens, J.Jlens) or jacobian_lens is None:
            raise RuntimeError("J-vector ablation needs a Jlens. Call fit_Jlens() / load_Jlens() first.")
        missing = sorted(set(layers) - set(jacobian_lens.source_layers))
        if missing:
            raise ValueError(f"layers {missing} are not in the lens source_layers.")

        instance = self.load_model()
        unembed_row = instance.get_unembedding_direction(token_id)
        directions = {}
        for layer in layers:
            jacobian = jacobian_lens.jacobians[layer].to(unembed_row.device, torch.float32)
            direction = unembed_row @ jacobian
            directions[layer] = direction / direction.norm().clamp_min(1e-8)
        return directions

    # TODO: test this!
    def run_ablation_inference(
        self,
        jsonl_line: str,
        data_root: str,
        mode: str,
        layers: list[int],
        token_pos: list[int],
        target_token: int | None = None,
        beta: float | None = None,
        target_token2: int | None = None,
        read_positions: list[int] | None = None,
        MAX_LENGTH: int = 300,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        * Returns (baseline_logits, ablated_logits), each [len(read_positions), vocab] on the CPU.
        * read_positions defaults to the last position. Python indexing, negatives allowed.
        * token_pos: positions whose residual is modified, at every layer in layers.
        * Modes:
            - "ablation":  set the residual to zero.
            - "jablation": remove the target_token J-vector component. beta=1 removes it (default), beta=2 reverses it.
            - "injection": do jablation(beta=1) on target_token, then add the removed magnitude along target_token2.
        """
        if mode in ("jablation", "injection") and target_token is None:
            raise ValueError(f"mode={mode!r} requires target_token.")
        if mode == "injection" and target_token2 is None:
            raise ValueError("mode='injection' requires target_token2.")

        directions = directions2 = None
        if target_token is not None and mode in ("jablation", "injection"):
            directions = self._token_directions(target_token, layers)
        if target_token2 is not None and mode == "injection":
            directions2 = self._token_directions(target_token2, layers)

        spec = AblationSpec(
            mode=mode,
            layers=layers,
            token_pos=token_pos,
            directions=directions,
            beta=1.0 if beta is None else beta,
            directions2=directions2,
        )

        instance = self.load_model()
        instance.load_data(data_root=data_root)
        input_ids = self.encode_prompt(jsonl_line, MAX_LENGTH=MAX_LENGTH)
        read = [-1] if read_positions is None else read_positions

        def logits_at(residual: torch.Tensor) -> torch.Tensor:
            return instance.unembed(residual[0][read]).float().cpu()

        baseline = logits_at(instance.ablation_forward(input_ids))
        ablated = logits_at(instance.ablation_forward(input_ids, spec))
        return baseline, ablated
