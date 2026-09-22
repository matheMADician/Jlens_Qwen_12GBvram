from jlens.lens import JacobianLens
from torch import Tensor
import logging
from Jlens import lens as l
from model.instance import Instance

class LogitLens(l.Lens):
    def __init__(self, logger: logging.Logger, model: Instance):
        self.lens = JacobianLens({}, n_prompts= 1, d_model= model.d_model)
        self.model = model
        
    def _load_data(self, data_root: str):
        if not data_root:
            raise ValueError("No data root passed to Jlens.")
        self.model.load_data(data_root)
        
    def apply_lens(
        self,
        do_activate_Jacobian: bool = False,
        json_line: str | None = None,
        layers_available: list[int] | None = None,
        positions: list[int] | None = None,
        top_k: int = 5,
        MAX_SEQ_LEN: int = 300,
        data_root: str | None = None ):
        
        if data_root is not None:
            self._load_data(data_root)
        
        run_layers = []
        if layers_available is None:
            if do_activate_Jacobian:
                #? Does Anthropic's code accept the last layer in Jlens functions? Needs investigation.
                #  If not, just use Logitlens instead.
                run_layers = list(range(self.model.n_layers - 1))
            else:
                run_layers = list(range(self.model.n_layers))
        else:
            run_layers = layers_available
        
        if json_line is None:
            raise ValueError("Missing Json line.")
        
        lens_logits, model_logits, input_ids = self.lens.apply(
            self.model, json_line, layers=run_layers, positions=None,
            max_seq_len=MAX_SEQ_LEN, use_jacobian=False,
        )

        return lens_logits, model_logits, input_ids
        pass
        