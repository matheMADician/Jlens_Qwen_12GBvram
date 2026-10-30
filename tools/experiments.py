import os, math
from dataclasses import dataclass

import torch
from torch import Tensor, linalg

from audio_jlens.jlens import Jlens
from json_parser import Parser
from tools.math_tools import MathTools
from tools.matrix_tools import MatrixTools, LayerSVDMetrics
from model.model import Model

@dataclass
class passk_cache:
    """Per-prompt pass@k observations accumulated during evaluation."""

    prompt_cnt: int
    best_token_pos: list[int]
    best_k_values: list[int]
    best_probability: list[float]
    best_layers: list[int]
    layer: int | None = None

#TODO write docstring for this class
@dataclass
class JlensComparison:
    """Holds comparison results between two Jlens instances."""
    cosine_similarity: float
    symmetric_relative_frobenius_dist: float
    frobenius_norm_ratio: float
    operator_norm_diff: float
    largest_sval_value_diff: float
    relative_largest_sval_diff: float
    sval_threshold: float
    left_subspace_rank1: int
    left_subspace_rank2: int
    left_principal_cosines: list[float]
    left_mean_principal_cosine: float
    right_subspace_rank1: int
    right_subspace_rank2: int
    right_principal_cosines: list[float]
    right_mean_principal_cosine: float
    
    
class Experiments:
    @staticmethod
    def ablation():
        return
    
    @staticmethod
    def comp_dataset_choice():
        pass
    
    @staticmethod
    def draw_Jlens_cka_heatmap(
        logger,
        run_name: str,
        heatmap_path: str,
        lens_path1: str,
        lens_path2: str | None = None,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Draw a heatmap using cka similarity between layers.
        The returned tensor is indexed as ``[lens1_layer, lens2_layer]``.
        
        The y axis represents the first lens, while the x axis does for the second.

        Args:
            lens_path1 (str): Path to the first Jlens checkpoint.
            lens_path2 (str | None, optional): If not specified, compare lens1 to itself.
            device: Device for CKA calculation. Defaults to CUDA when available.
        """
        os.makedirs(heatmap_path, exist_ok=True)
        
        lens1 = Jlens(logger)
        lens1.load_lens(path=lens_path1)
        lens2 = lens1 if lens_path2 is None else Jlens(logger)
        if lens_path2 is not None:
            lens2.load_lens(path=lens_path2)

        assert lens1.lens is not None
        assert lens2.lens is not None
        layers1 = lens1.lens.source_layers
        layers2 = lens2.lens.source_layers

        heatmap = MathTools.cka_similarity(
            l1=lens1.lens.jacobians,
            l2=lens2.lens.jacobians,
            d1=layers1,
            d2=layers2,
            device=device,
        )
        logger.info("Computed CKA heatmap on %s", device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        MathTools.draw_heatmap(
            data=heatmap,
            png_path=os.path.join(heatmap_path, run_name or "heatmap"),
            x_labels=layers2,
            y_labels=layers1,
            title="JLens Jacobian cka similarity",
        )
        logger.info("Saved JLens heatmap to %s", heatmap_path)
        return heatmap

    @staticmethod
    def draw_Jlens_cosine_heatmap(
        logger,
        run_name: str,
        heatmap_path: str,
        lens_path1: str,
        lens_path2: str | None = None,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Draw pairwise cosine similarities between Jacobian matrices."""
        os.makedirs(heatmap_path, exist_ok=True)
        compute_device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        lens1 = Jlens(logger)
        lens1.load_lens(path=lens_path1)
        lens2 = lens1 if lens_path2 is None else Jlens(logger)
        if lens_path2 is not None:
            lens2.load_lens(path=lens_path2)

        assert lens1.lens is not None
        assert lens2.lens is not None
        layers1 = lens1.lens.source_layers
        layers2 = lens2.lens.source_layers
        jacobians1 = {
            layer: lens1.lens.jacobians[layer].to(
                device=compute_device, dtype=torch.float32
            )
            for layer in layers1
        }
        if lens1 is lens2:
            jacobians2 = jacobians1
        else:
            jacobians2 = {
                layer: lens2.lens.jacobians[layer].to(
                    device=compute_device, dtype=torch.float32
                )
                for layer in layers2
            }

        heatmap = MathTools.sim_for_sets(
            l1=jacobians1,
            l2=jacobians2,
            d1=layers1,
            d2=layers2,
            func=MathTools.cos_similarity,
        )
        logger.info("Computed cosine heatmap on %s", compute_device)
        MathTools.draw_heatmap(
            data=heatmap,
            png_path=os.path.join(heatmap_path, f"{run_name or 'heatmap'}_cosine"),
            x_labels=layers2,
            y_labels=layers1,
            title="JLens Jacobian cosine similarity",
        )
        logger.info("Saved JLens cosine heatmap to %s", heatmap_path)
        return heatmap

    @staticmethod
    def draw_Jlens_svd_heatmaps(
        logger,
        run_name: str,
        output_dir: str,
        lens_path: str,
        top_k: int,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[str]]:
        """Save per-layer sval spectra and adjacent-layer subspace heatmaps."""

        output_dir = os.path.join(output_dir, "SVD_analysis")
        lens = Jlens(logger)
        lens.load_lens(path=lens_path)
        assert lens.lens is not None

        spectra, similarities, layers, layer_pairs = MatrixTools.draw_svd_heatmaps(
            jacobians=lens.lens.jacobians,
            output_dir=output_dir,
            name=run_name or "jlens",
            top_k=top_k,
            device=device,
        )
        logger.info("Saved JLens SVD heatmaps to %s", output_dir)
        return spectra, similarities, layers, layer_pairs

    @staticmethod
    def compare_jacobians(
        jacobian1: Tensor,
        jacobian2: Tensor,
        *,
        sval_threshold: float = 0.5,
        device: torch.device | str = "cpu",
    ) -> JlensComparison:
        #TODO This is directly from Copilot. Consult 學長.
        """
        Compare matched Jacobians by similarity, dist, and sval subspaces.
        Jacobians must have the same shape.
        For Jacobians from different models, some comparison metrics may not be meaningful.
        If so, call the lower-level comparison functions.
        """
        # Validate inputs
        MatrixTools._validate_sval_threshold(sval_threshold)
        matrix1 = MatrixTools._validate_jacobian(jacobian1, device=device)
        matrix2 = MatrixTools._validate_jacobian(jacobian2, device=device)
        if matrix1.shape != matrix2.shape:
            raise ValueError(
                "Jacobians must have matching shapes, got "
                f"{tuple(matrix1.shape)} and {tuple(matrix2.shape)}."
            )
        # Compute SVD
        svd_metrics1: LayerSVDMetrics = MatrixTools.SVD_decomp(matrix1)
        svd_metrics2: LayerSVDMetrics = MatrixTools.SVD_decomp(matrix2)
        if svd_metrics1.sval[0] == 0 or svd_metrics2.sval[0] == 0:
            raise ValueError(
                "Jacobian comparison metrics are undefined for an all-zero matrix."
            )
        # Compute frobenius dist
        matrix1_norm = linalg.vector_norm(matrix1)
        matrix2_norm = linalg.vector_norm(matrix2)
        mean_frobenius_norm = (matrix1_norm + matrix2_norm) / 2
        delta = matrix1 - matrix2
        relative_frobenius_dist = (
            linalg.vector_norm(delta) / mean_frobenius_norm
        )
        # Largest singular value of the difference
        operator_norm_diff = linalg.matrix_norm(delta, ord=2)
        # Normalized difference of largest singular
        largest_sval_diff = abs(svd_metrics1.sval[0] - svd_metrics2.sval[0])
        mean_operator_norm = (svd_metrics1.sval[0] + svd_metrics2.sval[0]) / 2
        relative_largest_sval_diff = largest_sval_diff / mean_operator_norm
        # Masks for strong singular values
        strong_mask1 = svd_metrics1.sval >= sval_threshold * svd_metrics1.sval[0]
        strong_mask2 = svd_metrics2.sval >= sval_threshold * svd_metrics2.sval[0]
        # Extract only the impactful left and right subspaces
        left_basis1 = svd_metrics1.left_svec[:, strong_mask1]
        left_basis2 = svd_metrics2.left_svec[:, strong_mask2]
        right_basis1 = svd_metrics1.right_svec[:, strong_mask1]
        right_basis2 = svd_metrics2.right_svec[:, strong_mask2]
        # Principal cosines of the two subspaces
        # Since SVD decomposition works as M -> U S Vh, the below effectively
        #   does MN^T -> U S Vh, that is, (U-1 M)(Vh-1^T N)^T -> S
        #   This maximizes the alignment between the subspace vectors after a linear transformation.
        # This is much better than directly comparing them. Copilot is the absolute goat.
        # See https://web.ma.utexas.edu/users/vandyke/notes/deep_learning_presentation/presentation.pdf
        left_principal_cosines = linalg.svdvals(
            left_basis1.T @ left_basis2
        ).clamp_(0, 1)
        right_principal_cosines = linalg.svdvals(
            right_basis1.T @ right_basis2
        ).clamp_(0, 1)

        def mean_or_zero(values: Tensor) -> float:
            return float(values.mean().item()) if values.numel() else 0.0

        return JlensComparison(
            cosine_similarity=MathTools.cos_similarity(matrix1, matrix2),
            symmetric_relative_frobenius_dist=float(
                relative_frobenius_dist.item()
            ),
            frobenius_norm_ratio=float((matrix2_norm / matrix1_norm).item()),
            operator_norm_diff=float(operator_norm_diff.item()),
            largest_sval_value_diff=float(
                largest_sval_diff.item()
            ),
            relative_largest_sval_diff=float(
                relative_largest_sval_diff.item()
            ),
            sval_threshold=sval_threshold,
            left_subspace_rank1=int(strong_mask1.sum().item()),
            left_subspace_rank2=int(strong_mask2.sum().item()),
            left_principal_cosines=left_principal_cosines.cpu().tolist(),
            left_mean_principal_cosine=mean_or_zero(left_principal_cosines),
            right_subspace_rank1=int(strong_mask1.sum().item()),
            right_subspace_rank2=int(strong_mask2.sum().item()),
            right_principal_cosines=right_principal_cosines.cpu().tolist(),
            right_mean_principal_cosine=mean_or_zero(right_principal_cosines),
        )

    @staticmethod
    def info_Jlens(
        logger,
        jacobians: dict[int, Tensor],
        layers: list[int],
        svd_top_k: int = 8,
        device: str | torch.device | None = None,
        svd_device: str | torch.device = "cpu",
    ) -> dict[str, object]:
        """Compute single-lens cosine, CKA, and singular-value analyses."""
        if not layers:
            raise ValueError("Cannot analyze a JLens with no source layers.")
        if len(set(layers)) != len(layers) or any(
            layer not in jacobians for layer in layers
        ):
            raise ValueError("Source layers must be unique keys in jacobians.")

        compute_device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        cosine_jacobians = {
            layer: jacobians[layer].to(device=compute_device, dtype=torch.float32)
            for layer in layers
        }
        cosine_heatmap = MathTools.sim_for_sets(
            l1=cosine_jacobians,
            l2=cosine_jacobians,
            d1=layers,
            d2=layers,
            func=MathTools.cos_similarity,
        )
        cka_heatmap = MathTools.cka_similarity(
            l1=jacobians,
            l2=jacobians,
            d1=layers,
            d2=layers,
            device=device,
        )
        spectra, adjacent_cosines, svd_layers, layer_pairs = (
            MatrixTools.analyze_layer_svd(
                jacobians,
                top_k=svd_top_k,
                device=svd_device,
            )
        )
        logger.info(
            "Computed single-lens cosine, CKA, and SVD analysis for layers %s",
            layers,
        )
        return {
            "layers": layers,
            "cosine_heatmap": cosine_heatmap,
            "cka_heatmap": cka_heatmap,
            "svd_spectra": spectra,
            "svd_adjacent_cosines": adjacent_cosines,
            "svd_layers": svd_layers,
            "svd_layer_pairs": layer_pairs,
        }

    @staticmethod
    def passk_Jlens(
        logger,
        model: Model,
        run_name: str,
        lens_path: str,
        prompts_path: str,
        data_root: str,
        top_k: int,
        k_values: list[int],
        layer: int | None = None,
    ) -> dict[str, object]:
        """Run JLens and LogitLens pass@k evaluation for all prompts.

        Applies both lenses once per prompt, then reports the first transcript
        token's best rank across positions/layers and independently at each layer.
        """
        if top_k < 1: raise ValueError("top_k must be at least 1")
        # To match the original paper,
        # this is expected to be a geometric or pseudo-geometric sequence.
        sorted_k_values = sorted(set(k_values))
        json_lines = Parser.load_jsonl_lines(prompts_path)
        # Load lens
        model.load_Jlens(lens_path=lens_path)
        passk_results = {
            lens_name: passk_cache(
                prompt_cnt=0,
                best_token_pos=[],
                best_k_values=[],
                best_probability=[],
                best_layers=[],
            )
            for lens_name in ("jlens", "logit_lens")
        }
        prediction_rows: list[dict[str, object]] = []
        passk_summaries: dict[
            str, dict[str, float | int | dict[int, float] | list[float]]
        ] = {}
        per_layer_caches: dict[str, dict[int, passk_cache]] | None = None
        per_layer_summaries: dict[
            str, dict[int, dict[str, float | int | dict[int, float] | list[float]]]
        ] = {}
        # Main loop(loop through prompts)
        for json_line in json_lines:
            model.apply_lens(
                json_line=json_line,
                data_root=data_root,
                top_k=top_k,
                use_Jlens=True,
                run_name=run_name,
                layers_available=[layer] if layer is not None else None,
            )
            prediction_rows.extend(model.lens_cache.to_rows(top_k=top_k))
            # Parse JSON line
            sample = Parser.parse_json_line(json_line)
            transcript = sample.transcript
            if transcript is None:
                logger.warning(f"Sample {sample.sample_id} has no transcript to evaluate")
                continue
            transcript = transcript.strip()
            # Remove leading (a) (b) .etc in case it's part of multiple-choice formatting
            if transcript.startswith("(") and ")" in transcript:
                transcript = transcript.split(")", 1)[1].strip()

            processor = model.processor
            assert processor is not None, "The configured model adapter has no processor."
            
            target_ids = processor.tokenizer(
                transcript,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"]
            if target_ids.numel() == 0:
                raise ValueError(
                    f"Transcript for sample {sample.sample_id} produced no tokens"
                )
            ans_token = int(target_ids[0, 0].item())
            for lens_name, lens_data in (
                ("jlens", model.lens_cache.jlens_data),
                ("logit_lens", model.lens_cache.logit_lens_data),
            ):
                if lens_data is None:
                    raise RuntimeError(
                        f"{lens_name} output was not cached for sample {sample.sample_id}"
                    )
                passk_summaries[lens_name] = Experiments.eval_passk(
                    logger=logger,
                    run_name=f"{run_name} {lens_name}",
                    lens_data=lens_data,
                    passk_cache=passk_results[lens_name],
                    ans_token=ans_token,
                    k_values=sorted_k_values,
                    layer=layer,
                )
            jlens_data = model.lens_cache.jlens_data
            logit_lens_data = model.lens_cache.logit_lens_data
            if jlens_data is None or logit_lens_data is None:
                raise RuntimeError(
                    f"Both lens outputs are required for sample {sample.sample_id}"
                )
            available_layers = sorted(jlens_data[0])
            logit_layers = sorted(logit_lens_data[0])
            if available_layers != logit_layers:
                raise ValueError(
                    "JLens and LogitLens layer outputs do not match: "
                    f"{available_layers!r} != {logit_layers!r}."
                )
            selected_layers = [layer] if layer is not None else available_layers
            if per_layer_caches is None:
                per_layer_caches = {
                    lens_name: {
                        layer_idx: passk_cache(
                            prompt_cnt=0,
                            best_token_pos=[],
                            best_k_values=[],
                            best_probability=[],
                            best_layers=[],
                        )
                        for layer_idx in selected_layers
                    }
                    for lens_name in ("jlens", "logit_lens")
                }
                per_layer_summaries = {
                    lens_name: {} for lens_name in ("jlens", "logit_lens")
                }
            elif any(
                sorted(per_layer_caches[lens_name]) != selected_layers
                for lens_name in ("jlens", "logit_lens")
            ):
                raise ValueError(
                    "The available lens layers changed during pass@k evaluation."
                )

            for lens_name, lens_data in (
                ("jlens", jlens_data),
                ("logit_lens", logit_lens_data),
            ):
                for layer_idx in selected_layers:
                    per_layer_summaries[lens_name][layer_idx] = (
                        Experiments.eval_passk(
                            logger=logger,
                            run_name=f"{run_name} {lens_name} layer {layer_idx}",
                            lens_data=lens_data,
                            passk_cache=per_layer_caches[lens_name][layer_idx],
                            ans_token=ans_token,
                            k_values=sorted_k_values,
                            layer=layer_idx,
                        )
                    )

        if not passk_summaries:
            raise RuntimeError("Pass@k evaluation did not process any prompts")

        def build_lens_report(lens_name: str) -> dict[str, object]:
            cache = passk_results[lens_name]
            return {
                **passk_summaries[lens_name],
                "best_answer_ranks": cache.best_k_values,
                "answer_probabilities": cache.best_probability,
                "target_positions": cache.best_token_pos,
                "best_layers": cache.best_layers,
            }

        by_layer: dict[str, dict[str, dict[str, object]]] = {}
        if per_layer_caches is not None:
            for lens_name, layer_caches in per_layer_caches.items():
                by_layer[lens_name] = {}
                for layer_idx, cache in layer_caches.items():
                    by_layer[lens_name][str(layer_idx)] = {
                        **per_layer_summaries[lens_name][layer_idx],
                        "best_answer_ranks": cache.best_k_values,
                        "answer_probabilities": cache.best_probability,
                        "target_positions": cache.best_token_pos,
                        "best_layers": cache.best_layers,
                    }

        return {
            "k_values": sorted_k_values,
            "auc_x_axis": "log(k)",
            "layer": layer,
            "position_policy": "best rank over prompt positions",
            "layer_policy": (
                "selected layer only"
                if layer is not None
                else "all layers pooled for overall metrics; each layer scored independently in by_layer"
            ),
            # Keep the original JLens report fields at top level for compatibility.
            **build_lens_report("jlens"),
            "logit_lens": build_lens_report("logit_lens"),
            "by_layer": by_layer,
            "prediction_rows": prediction_rows,
        }
    
    @staticmethod
    def eval_passk(
        logger,
        run_name: str,
        lens_data: tuple[dict[int, Tensor], Tensor, Tensor],
        passk_cache: passk_cache,
        ans_token: int,
        k_values: list[int],
        layer: int | None = None,
    ) -> dict[str, float | int | dict[int, float] | list[float]]:
        """Accumulate one example and summarize pass@k over processed examples.
        Returns:
            Current aggregate prompt pass rates and normalized trapezoidal AUC
            integrated over log(k). Each prompt contributes its lowest answer-token
            rank across the selected layers and all prompt positions.
        """
        if not k_values or any(not isinstance(k, int) or k <= 0 for k in k_values):
            raise ValueError("k_values must contain positive integers")
        sorted_k_values = sorted(set(k_values))
        # Check if settings match
        if passk_cache.layer is not None and passk_cache.layer != layer:
            raise ValueError("Cannot mix layer settings in one passk_cache")
        passk_cache.layer = layer

        logits_by_layer, _, input_ids = lens_data
        if not logits_by_layer:
            raise ValueError("Cached JLens output has no layer logits")
        # If layer is not specified, consider all available layers
        #     pass@k then takes the best result over all layers
        if layer is not None:
            if layer not in logits_by_layer:
                raise ValueError(
                    f"Layer {layer} is not present in cached JLens predictions; "
                    f"available layers: {sorted(logits_by_layer)}"
                )
            selected_layers = [layer]
        else:
            selected_layers = sorted(logits_by_layer)

        prompt_length = int(input_ids.reshape(-1).numel())
        if prompt_length == 0:
            raise ValueError("Cached input_ids are empty")

        best_rank: int | None = None
        best_probability: float | None = None
        best_position: int | None = None
        best_layer: int | None = None
        # Iterate over each selected layer
        for layer_idx in selected_layers:
            layer_logits_tensor = logits_by_layer[layer_idx]
            if layer_logits_tensor.ndim != 2:
                raise ValueError(
                    f"Layer {layer_idx} logits must have shape [positions, vocab], "
                    f"got {tuple(layer_logits_tensor.shape)}"
                )
            if prompt_length > layer_logits_tensor.shape[0]:
                raise ValueError(
                    f"Prompt length {prompt_length} exceeds available positions in "
                    f"layer {layer_idx} logits"
                )
            if ans_token < 0 or ans_token >= layer_logits_tensor.shape[-1]:
                raise ValueError(
                    f"Answer token ID {ans_token} is outside the logits vocabulary"
                )
            # This counts how many logits are greater than the answer token's logits
            # instead of sorting the whole vocabulary
            position_logits = layer_logits_tensor[:prompt_length].float()
            answer_logits = position_logits[:, ans_token].unsqueeze(1)
            position_ranks = (position_logits > answer_logits).sum(dim=1) + 1
            # Determine the best rank and position for this layer
            # Yes, currently we are not considering the effect of token positions.
            # Future implementations can seperate the effect of different modalities
            layer_best_rank, layer_best_position = position_ranks.min(dim=0)
            rank = int(layer_best_rank.item())
            # Update the best rank
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_position = int(layer_best_position.item())
                best_layer = layer_idx
                best_probability = float(
                    torch.softmax(position_logits[best_position], dim=-1)[
                        ans_token
                    ].item()
                )

        if (
            best_rank is None
            or best_probability is None
            or best_position is None
            or best_layer is None
        ):
            raise RuntimeError("Could not determine answer-token rank")

        passk_cache.prompt_cnt += 1
        passk_cache.best_token_pos.append(best_position)
        passk_cache.best_k_values.append(best_rank)
        passk_cache.best_probability.append(best_probability)
        passk_cache.best_layers.append(best_layer)
        # Compute pass@k rates for the current prompt
        pass_rates = {
            k: sum(rank <= k for rank in passk_cache.best_k_values)
            / passk_cache.prompt_cnt
            for k in sorted_k_values
        }
        log_k_values = [math.log(k) for k in sorted_k_values]
        # Compute the Area Under the Curve (AUC) using the trapezoidal rule on log(k) values
        if len(sorted_k_values) == 1:
            normalized_auc = next(iter(pass_rates.values()))
        else:
            log_span = log_k_values[-1] - log_k_values[0]
            area = sum(
                (pass_rates[left] + pass_rates[right])
                * (log_right - log_left)
                / 2
                for left, right, log_left, log_right in zip(
                    sorted_k_values[:-1],
                    sorted_k_values[1:],
                    log_k_values[:-1],
                    log_k_values[1:],
                    strict=True,
                )
            )
            normalized_auc = area / log_span

        logger.info(
            "%s pass@k progress: %d prompts, normalized log(k) AUC=%.6f",
            run_name,
            passk_cache.prompt_cnt,
            normalized_auc,
        )
        return {
            "prompt_count": passk_cache.prompt_cnt,
            "normalized_auc": normalized_auc,
            "pass_rates": pass_rates,
            "log_k_values": log_k_values,
        }


        