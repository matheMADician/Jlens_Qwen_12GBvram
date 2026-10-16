import os
import math
from venv import logger
from dataclasses import dataclass
import torch

from audio_jlens.jlens import Jlens
from tools.math_tools import MathTools
from model.model import LensOutputCache

@dataclass
class passk_cache:
    """Per-prompt pass@k observations accumulated during evaluation."""

    prompt_cnt: int
    best_token_pos: list[int]
    best_k_values: list[int]
    best_probability: list[float]
    layer: int | None = None

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
        """Save per-layer singular spectra and adjacent-layer subspace heatmaps."""

        output_dir = os.path.join(
            output_dir,
            "SVD_analysis",
            run_name or "default",
        )
        lens = Jlens(logger)
        lens.load_lens(path=lens_path)
        assert lens.lens is not None

        from tools.matrix_tools import MatrixTools

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
    def comp_Jlens(
        logger,
        lens_path1: str,
        lens_path2: str,
    ):
        # Compares two Jlens
        Jlens1= Jlens(logger)
        Jlens2= Jlens(logger)
        
        Jlens1.load_lens(path=lens_path1)
        Jlens2.load_lens(path=lens_path2)
        
        # Phase 1
        # Draw a heatmap between layers using two similarity functions
            # 1. cosine similarity
            # 2. CKA similarity
        
        # Phase 2
        # Calculate the overall 
    
    #TODO
    @staticmethod
    def eval_passk(
        logger,
        run_name: str,
        lens_cache: LensOutputCache,
        passk_cache: passk_cache,
        ans_token: int,
        k_values: list[int],
        layer: int | None = None,
    ) -> dict[str, float | int | dict[int, float] | list[float]]:
        """Accumulate one example and summarize pass@k over processed examples.

        Args:
            logger: Logger instance for logging messages.
            run_name (str): Name of the current run.
            lens_cache (LensOutputCache): Precomputed predictions for the Jlens.
            passk_cache: Accumulator updated in place with this example.
            ans_token: Token ID of the target answer.
            k_values (list[int]): List of k values for pass@k evaluation.
            layer: If specified, evaluate only this layer; otherwise use the
                best (lowest) answer-token rank across available lens layers.

        Returns:
            Current aggregate prompt pass rates and normalized trapezoidal AUC
            integrated over log(k).
        """
        if not k_values or any(not isinstance(k, int) or k <= 0 for k in k_values):
            raise ValueError("k_values must contain positive integers")
        sorted_k_values = sorted(set(k_values))
        if passk_cache.layer is not None and passk_cache.layer != layer:
            raise ValueError("Cannot mix layer settings in one passk_cache")
        passk_cache.layer = layer

        layer_logits = lens_cache.jlens_data
        if layer_logits is None:
            raise RuntimeError("No JLens predictions are cached for this prompt")
        logits_by_layer, _, input_ids = layer_logits
        if not logits_by_layer:
            raise ValueError("Cached JLens output has no layer logits")
        if layer is not None:
            if layer not in logits_by_layer:
                raise ValueError(
                    f"Layer {layer} is not present in cached JLens predictions; "
                    f"available layers: {sorted(logits_by_layer)}"
                )
            selected_layers = [layer]
        else:
            selected_layers = sorted(logits_by_layer)

        target_position = int(input_ids.reshape(-1).numel()) - 1
        if target_position < 0:
            raise ValueError("Cached input_ids are empty")

        best_rank: int | None = None
        best_probability: float | None = None
        for layer_idx in selected_layers:
            layer_logits_tensor = logits_by_layer[layer_idx]
            if layer_logits_tensor.ndim != 2:
                raise ValueError(
                    f"Layer {layer_idx} logits must have shape [positions, vocab], "
                    f"got {tuple(layer_logits_tensor.shape)}"
                )
            if target_position >= layer_logits_tensor.shape[0]:
                raise ValueError(
                    f"Final input position {target_position} is missing from "
                    f"layer {layer_idx} logits"
                )
            if ans_token < 0 or ans_token >= layer_logits_tensor.shape[-1]:
                raise ValueError(
                    f"Answer token ID {ans_token} is outside the logits vocabulary"
                )

            position_logits = layer_logits_tensor[target_position].float()
            answer_logit = position_logits[ans_token]
            rank = int((position_logits > answer_logit).sum().item()) + 1
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_probability = float(
                    torch.softmax(position_logits, dim=-1)[ans_token].item()
                )

        if best_rank is None or best_probability is None:
            raise RuntimeError("Could not determine answer-token rank")

        passk_cache.prompt_cnt += 1
        passk_cache.best_token_pos.append(target_position)
        passk_cache.best_k_values.append(best_rank)
        passk_cache.best_probability.append(best_probability)

        pass_rates = {
            k: sum(rank <= k for rank in passk_cache.best_k_values)
            / passk_cache.prompt_cnt
            for k in sorted_k_values
        }
        log_k_values = [math.log(k) for k in sorted_k_values]
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

        