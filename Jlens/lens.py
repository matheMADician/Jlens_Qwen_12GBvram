from abc import ABC, abstractmethod
class Lens(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def apply(self, do_activate_Jacobian: bool, json_line: str, layers_available, MAX_SEQ_LEN):
        pass