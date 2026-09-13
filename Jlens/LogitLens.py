from Jlens import lens as l
class LogitLens(l.Lens):
    def __init__(self, model = None, processor = None):
        super().__init__()

    def apply(self, do_activate_Jacobian: bool = True, json_line: str = None, layers_available = None, MAX_SEQ_LEN: int = 300):
        """
        * Applying LogitLens to model.
        * Returns in order: lens_logits, model_logits, input_ids
        """
        if layers_available == None:
            run_layers = list(range(self.model.model.language_model.config.num_hidden_layers))
        else:
            run_layers = layers_available

        if json_line is None:
            raise ValueError("Missing Json line.")

        lens_logits, model_logits, input_ids = self.lens.apply(
            self.model, json_line, layers=run_layers, positions=None,
            max_seq_len=MAX_SEQ_LEN, use_jacobian=False,
        )

        return lens_logits, model_logits, input_ids