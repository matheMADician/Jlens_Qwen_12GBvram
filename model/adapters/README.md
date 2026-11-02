# Adding a model adapter

Adapters connect a Hugging Face model and processor to the shared JLens
interface. Keep model-specific loading and input formatting in this folder;
register each supported model ID in `model/model.py`'s `MODEL_ADAPTERS`.

## Implement the adapter

Create `model/adapters/<model_name>.py` with an `Instance` subclass of
`model.adapters.Instance.Instance`. Use an existing adapter as a reference and
implement:

- `_initialize_model(model_id, do_4bit)`: load the model and processor, then set
  `hf_model`, `processor`, `layers`, `n_layers`, `d_model`, and `tokenizer`.
  Store any model-specific text decoder, input cache, device, and audio sampling
  rate needed by other methods.
- `encode(text, max_length=...)`: parse the JSONL record with
  `Parser.parse_json_line(..., audio_root=self.data_root)`, load any media,
  format the model's prompt, and cache every processor tensor needed by the
  forward pass. Return the `input_ids` tensor.
- `clear_model_input_cache()`: release the cached processor inputs.
- `forward(input_ids)`: run the model with the cached inputs and return the
  decoder residuals used by JLens. Preserve the cached audio/features while
  substituting the supplied `input_ids`; remove any temporary hooks even if
  forward raises.
- `unembed(residual)`: apply the model's final normalization and output head to
  produce vocabulary logits.
- `get_unembedding_direction(token_id)`: return the corresponding output
  direction for ablation and related analyses.

The base class implements greedy `generate()` using the cached inputs, so the
adapter's `encode()` must leave a complete, reusable input cache. Follow the
base-class device and dtype conventions, freeze model parameters, and call
`eval()` after loading.

## Register and configure

Import the new adapter in `model/model.py` and add a `MODEL_ADAPTERS` entry
mapping the exact Hugging Face model ID to its `Instance` class. This registry is
also used to validate IDs in `create-model-settings`.

Create a separate settings file for the new model instead of overwriting the
shared settings:

```bash
python main.py create-model-settings \
  --model-name organization/model-name \
  --output-settings /tmp/model-settings.json
```

The generated settings set both `model.id` and `model.adapter` to the registered
ID. Model-scoped outputs use the final component of the model ID, so a new model
gets a separate checkpoint and inference directory.

## Validate

Before fitting a lens, verify the adapter on a small real input:

1. Generate settings and load the model and processor.
2. Encode a JSONL record and check that `input_ids` and cached media tensors
   have compatible batch and sequence dimensions.
3. Run `forward()` and `unembed()`; confirm finite residuals/logits and the
   expected layer, hidden, and vocabulary dimensions.
4. Run one short inference or fitting smoke test. For a fine-tuned or instruct
   checkpoint, confirm that its processor/chat template and module paths match
   the base architecture; do not assume weight compatibility means its lens is
   compatible with another checkpoint.