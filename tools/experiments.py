import os
from venv import logger
import torch

from audio_jlens.jlens import Jlens
from tools.math_tools import MathTools

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
    ) -> torch.Tensor:
        """Draw a heatmap using cka similarity between layers.
        The returned tensor is indexed as ``[lens1_layer, lens2_layer]``.
        
        The y axis represents the first lens, while the x axis does for the second.

        Args:
            lens_path1 (str): Path to the first Jlens checkpoint.
            lens_path2 (str | None, optional): If not specified, compare lens1 to itself.
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
        )
        
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
        lens_path: str,
        csv_path: str,
        k_values: list[int],
    ):
        """Then derives the pass@k AUC for a given Jlens and precomputed predictions.

        Args:
            logger: Logger instance for logging messages.
            run_name (str): Name of the current run.
            lens_path (str): Path to the Jlens file.
            csv_path (str): Path to the CSV file containing precomputed predictions.
            k_values (list[int]): List of k values for pass@k evaluation.
        """
        lens = Jlens(logger)
        lens.load_lens(path=lens_path)
        assert lens.lens is not None

        