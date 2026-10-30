## Codebase changes

### Random projection and t-SNE

Goal: add an optional random-projection workflow to JLens fitting. Use one
projection dimension and seed per run, compare settings across separate runs, and
generate a final t-SNE visualization without changing existing fitting behavior
when the feature is disabled.

1. **Fix the projection and artifact contract.**
   - Use a projection matrix `P` with shape `[d_model, projection_dim]` and project
     each fitted layer Jacobian as `P.T @ J_layer @ P`.
   - Use Gaussian entries scaled by `1 / sqrt(projection_dim)`.
   - Read one `projection_dim` and one `projection_seed` from `settings.json` per
     run. Use proposed defaults `128` and `42`, respectively.
   - The same dimension, seed, and projection method must reproduce the same `P`;
     reuse that `P` across prompts, layers, and datasets within the run.
   - Compare dimensions `64`, `128`, and `256` and additional seeds by changing
     the single configured values and running separate experiments; do not add
     list-based sweep configuration.
   - Preserve full-size per-dataset JLens artifacts; save cumulative/single-lens
     artifacts as projections only. Verify this policy against the fitting outputs
     before implementation.

2. **Implement and test the projection helper.**
   - Add a reusable helper that validates a square input Jacobian, projection
     dimensions, seed, dtype, and device.
   - Return a `[projection_dim, projection_dim]` matrix for each input layer.
   - Verify output shape, determinism for a fixed dimension and seed, different
     results for different seeds/dimensions, finite values, and behavior at
     dimensions `64`, `128`, and `256`.
   - Record enough projection metadata to recreate the projection and identify
     projected artifacts: original dimension, projected dimension, seed, and
     projection method.

3. **Add an opt-in fitting configuration.**
   - Add a disabled-by-default random-projection switch under
     `fitting.experiment`.
   - Add one integer `projection_dim` (default `128`) and one integer
     `projection_seed` (default `42`); do not accept lists of dimensions or seeds.
   - Validate configuration early and report clear errors for invalid dimensions
     or unsupported projection methods.
   - Confirm that disabling the switch leaves current fitting outputs and behavior
     unchanged.

4. **Integrate projection with fitting and artifact saving.**
   - Keep full-size running sums during fitting; project finalized mean Jacobians
     at the save boundary so the current per-prompt fitting and accumulation logic
     remains unchanged.
   - Apply the one configured seeded projection consistently across all layers and
     corresponding lenses in the run.
   - Also project each per-prompt Jacobian using the same `P`; retain its projected
     layer features and dataset label for the t-SNE stage.
   - Save projected lenses with correct `d_model`, `n_prompts`, source layers, and
     projection metadata.
   - Preserve the artifact policy confirmed in step 1. Ensure resume/checkpoint
     state is not accidentally confused with projected output lenses.
   - Handle the combined lens, cumulative milestone lenses, and per-dataset lenses
     consistently with that policy.

5. **Validate downstream use of projected lenses.**
   - Check that projected lenses can be loaded and analyzed by the existing
     single-/two-lens comparison tools.
   - Ensure comparisons use the same projection matrix when comparing lenses;
     fail clearly if projection metadata is incompatible.
   - Keep full-size checkpoint comparisons working as before.

6. **Create the t-SNE input and visualization.**
   - Represent each prompt as one point. Build its feature vector by flattening
     and concatenating its projected Jacobian matrices across source layers.
   - Mark the dataset for each prompt using a distinct plot color and/or marker;
     include a legend.
   - Produce one t-SNE embedding for the configured projection dimension/run.
   - Use configurable output path and visualization parameters; handle small sample
     counts and invalid perplexity.
   - Save the plot and embedding coordinates, prompt identifiers, dataset labels,
     projection dimension, seed, and t-SNE parameters so the visualization can be
     inspected or reproduced without refitting.

7. **Add end-to-end validation and documentation.**
   - Test that the disabled path matches existing behavior.
   - Test that enabled fitting produces the expected projected shapes and metadata
     for the configured dimension and each artifact type.
   - Test deterministic projections, compatible/incompatible lens comparisons,
     t-SNE output generation, and small-input error handling.
   - Document configuration, output artifacts, runtime/storage implications, and
     how to run the projection-size experiment.

### More datasets
Currently, the training data contains Clotho, esc50 and fleurs_en_us.
Consider adding more variety(like emotion data, more languages...).
Do this after the random projection part is done, and evaluate the effect of data.

### JLens evaluation

1. Senpai said SVD analysis could act on the actual `d_vocab` J vectors (after the
   final projection matrix `W`) instead of the `d_model` J-matrix columns.

## Miscellaneous

1. `Experiments.eval_passk()` could accept an input specifying which part of the
   prompt to evaluate (e.g. speech, user, ...).

2. Make create-model-settings cover more settings.

3. The evaluation set is SAKURA, which is a curated set from multiple different other datasets. There is a chance our training set contains the same data as the testing set. Though the probability is super low, it's better to check for this.

4. Using Qwen2-Audio-Instruct is better than Qwen2-Audio.

## Experiments to do
### Evaluation
Though the random projection is not currently done, there are still dataset-specific Jlens available for comparison. Do this.
### passk
Copilot:
- I’d compare paired per-prompt results next: evaluate the same target and a matched set of positions/layers for both lenses, then report the distribution of JLens-minus-LogitLens gains—not just the aggregate AUC. That should help distinguish a genuinely stronger Omni readout from an advantage due to the larger search space.
- [ ] Choose a readout layer using validation data, then report that fixed layer on
  a separate held-out test set; do not select the best layer on the final test set.
- [ ] Consider a separate fixed-position evaluation to measure readout without
  searching over prompt positions.