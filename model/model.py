from Jlens import lens, Jlens as J
from model import instance as Inst
import os, logging

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

        self.lens_cache = LensOutputCache()

    def get_model(self):
        return self.model, self.processor

    def get_model_info(self):
        #TODO
        pass

    def load_Jlens(self, lens_path: str | None = None, lens_name: str | None = None):
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
            use_Jlens: bool = True,
            layers_available: list[int] | None = None,
            MAX_SEQ_LEN: int | None = None,
            ):
        """
        * Auto-save the data into model cache.
        """
        if self.lens is None:
            raise ValueError("No lens in model. Call fit_Jlens() / load_Jlens() first.")
        
        result = self.lens.apply_lens(
            do_activate_Jacobian= use_Jlens,
            json_line= json_line,
            layers_available= layers_available,
            MAX_SEQ_LEN= MAX_SEQ_LEN,
            data_root= data_root
        )
        self.lens_cache.put(
            "jlens" if use_Jlens else "logit_lens",
            result,
        )
        return result

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
            

from torch import Tensor
class LensOutputCache:
    """
    * Cache for the latest JLens and LogitLens outputs.
    * Each output is stored as:
      [lens_logits, model_logits, input_ids]
    """
    def __init__(self):
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