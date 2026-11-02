## Codebase changes

### More datasets
Current training data includes Clotho, ESC-50, Fleurs (en_us), and CREMA-D.
After the current t-SNE run finishes, use its dataset behavior to assess whether
this mix is appropriate. Then decide whether to add more variety (for example,
additional languages) and evaluate the effect of any changes.

### JLens evaluation

1. Senpai said SVD analysis could act on the actual `d_vocab` J vectors (after the
   final projection matrix `W`) instead of the `d_model` J-matrix columns.

## Miscellaneous

1. `Experiments.eval_passk()` could accept an input specifying which part of the
   prompt to evaluate (e.g. speech, user, ...).

2. Make create-model-settings cover more settings.

3. The evaluation set is SAKURA, which is a curated set from multiple different other datasets. There is a chance our training set contains the same data as the testing set. Though the probability is super low, it's better to check for this.

4. After reviewing the current t-SNE results and deciding on the dataset mix,
   evaluate Qwen2-Audio-7B-Instruct. Smoke-test model loading and inference,
   then fit a separate JLens for its weights before comparing it with the
   existing Qwen2-Audio lens.

## Experiments to do
### Evaluation
Compare the old Qwen2-Audio and Qwen2.5-Omni lenses on all four SAKURA splits.
Interpret the current pass@k target as probing the audio attribute label in
`transcript` (its first token), not as multiple-choice answer accuracy. The
per-layer metric also takes the best rank across prompt positions.
### passk
Copilot:
- I’d compare paired per-prompt results next: evaluate the same target and a matched set of positions/layers for both lenses, then report the distribution of JLens-minus-LogitLens gains—not just the aggregate AUC. That should help distinguish a genuinely stronger Omni readout from an advantage due to the larger search space.
- [ ] Choose a readout layer using validation data, then report that fixed layer on
  a separate held-out test set; do not select the best layer on the final test set.
- [ ] Consider a separate fixed-position evaluation to measure readout without
  searching over prompt positions.