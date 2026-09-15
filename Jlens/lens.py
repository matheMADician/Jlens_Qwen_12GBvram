from abc import ABC, abstractmethod
class Lens(ABC):
    @abstractmethod
    def __init__(self):
        pass

    from torch import Tensor
    @abstractmethod
    def apply_lens(self,
        do_activate_Jacobian: bool,
        json_line: str,
        layers_available,
        MAX_SEQ_LEN,
        data_root) -> tuple[dict[int, Tensor], Tensor, Tensor]:
        pass