from abc import ABC, abstractmethod
import logging
from model.instance import Instance

class Lens(ABC):
    @abstractmethod
    def __init__(self, logger: logging.Logger, model: Instance):
        pass

    from torch import Tensor
    @abstractmethod
    def apply_lens(self,
        do_activate_Jacobian: bool,
        json_line: str,
        layers_available,
        positions: list[int] | None,
        top_k: int,
        MAX_SEQ_LEN: int,
        data_root) -> tuple[dict[int, Tensor], Tensor, Tensor]:
        pass