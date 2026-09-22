from Jlens import lens, Jlens as J, LogitLens as L
from model import instance as Inst
import os, logging
import json

class Model:
    """
    Master class of all model methods.
    Automatically loads the model on creation.
    LogitLens is the default. To load Jlens, use load_lens(), and to calculate Jlens, use calc_Jlens().
    """
    #TODO: This is Qwen-specific, change this!
    def __init__(self,
        logger: logging.Logger,
        do_4bit: bool = False,
        MODEL_ID="Qwen/Qwen2-Audio-7B-Instruct"):
        
        self.logger = logger
        self.model_id = MODEL_ID
        self.do_4bit = do_4bit
        self.lens: lens.Lens | None = None

        self.model = Inst.Instance(
            MODEL_ID=MODEL_ID,
            do_4bit=do_4bit,
        )
        self.hf_model, self.processor = self.model.get_model()

        self.lens_cache = LensOutputCache(tokenizer=self.model.tokenizer)

    def get_model(self):
        return self.model, self.processor

    def get_model_info(self):
        #TODO
        pass

    def load_Jlens(self, lens_path: str | None = None, lens_name: str | None = None):
        """
        * If name is specified, the function ignores the path and looks for it in ~/Jlens/lens_checkpoints/*name*
        """
        self.lens = J.Jlens(logger = self.logger, model= self.model)
        self.lens.load_lens(path= lens_path, name= lens_name)
        self.lens_cache.clear_jlens()

    def fit_Jlens(
            self, jsonl_path: str,
            dim_batch: int = 32,
            checkpoint_interval: int = 5,
            MAX_SEQ_LEN: int = 300,
            do_replace: bool = False,
            checkpoint_save_path: str | None = None,
            run_name: str | None = None,
            is_test: bool = False
            ):
        """
        * Calls the calc_lens() function in Jlens.
        * See ~/Jlens/Jlens.py for detail.
        * jsonl_path accepts the relative path to jsonl files.
        * WARNING: This cleans the cache.
        * Your structure should look like this:
        * *parent directory*/jsonl_file
        * *parent directory*/audio_path_in_jsonl
        """
        # Parse the jsonl_path into data path(parent directory) and jsonl name
        if not os.path.isfile(jsonl_path):
            raise RuntimeError(f"Jsonl file({jsonl_path}) does not exist.")
        parent_dir, jsonl_name = os.path.split(jsonl_path)

        self.lens = J.Jlens(logger= self.logger, model= self.model)
        self.lens.calc_lens(
            data_path= parent_dir,
            jsonl_name= jsonl_name,
            do_replace= do_replace, 
            checkpoint_path= checkpoint_save_path,
            run_name= run_name,
            dim_batch= dim_batch,
            MAX_SEQ_LEN= MAX_SEQ_LEN,
            checkpoint_every= checkpoint_interval,
            is_test= is_test
            )
        self.lens_cache.clear_jlens()

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
        elif self.lens is None:
            self.lens = L.LogitLens(
                logger= self.logger,
                model= self.model
            )
        
        # Do both Jlens and LogitLens when not testing
        runs = [True, False] if use_Jlens else [False]
        for do_activate_Jacobian in runs:
            result = self.lens.apply_lens(
                do_activate_Jacobian= do_activate_Jacobian,
                json_line= json_line,
                layers_available= layers_available,
                # TODO: Support selected token positions; save() currently assumes all positions.
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
        return self.model.encode(text= json_line, max_length= MAX_LENGTH)
        
    def run_inference(
        self,
        jsonl_line: str,
        data_root: str,
        MAX_LENGTH: int = 300,
    ):
        """
        * Returns the last hidden state
        """
        self.model.load_data(data_root= data_root)
        return self.model.unembed(
            self.model.forward(
                self.encode_prompt(jsonl_line, MAX_LENGTH=MAX_LENGTH)
            )
        )
            

import csv
from pathlib import Path
import torch
from torch import Tensor


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