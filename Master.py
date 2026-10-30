import os, sys, torch, json, logging, csv, tempfile
from dataclasses import asdict

import model.model as M
from audio_jlens.jlens import Jlens
from json_parser import Parser
from tools.experiments import Experiments, JlensComparison
from tools.math_tools import MathTools

class Master:
    def __init__(self, run_name: str, settings_path: str = "settings.json"):
        Master.check_dependencies()
        self.logger = logging.getLogger()
        with open(settings_path, "r", encoding="utf-8") as settings_file:
            self.settings = json.load(settings_file)
        self.model = M.Model(logger= self.logger,
            do_4bit= self.settings["model"]["load_in_4bit"],
            MODEL_ID= self.settings["model"]["id"],
            adapter=self.settings["model"]["adapter"])
        self.run_name = run_name

    @staticmethod
    def create_model_settings_file(
        model_name: str,
        output_path: str,
        source_path: str = "settings.json",
    ) -> str:
        """Write a temporary settings copy configured for the run."""
        if model_name not in M.MODEL_ADAPTERS:
            raise ValueError(
                f"Unsupported model {model_name!r}; available models: "
                f"{sorted(M.MODEL_ADAPTERS)}"
            )

        source_path = os.path.abspath(os.path.expanduser(source_path))
        output_path = os.path.abspath(os.path.expanduser(output_path))
        if source_path == output_path:
            raise ValueError("Output settings path must differ from source settings.")

        with open(source_path, "r", encoding="utf-8") as settings_file:
            settings = json.load(settings_file)
        if not isinstance(settings, dict) or not isinstance(settings.get("model"), dict):
            raise ValueError(f"Settings file has no valid 'model' object: {source_path}")

        settings["model"]["id"] = model_name
        settings["model"]["adapter"] = model_name

        output_directory = os.path.dirname(output_path)
        os.makedirs(output_directory, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_directory,
                prefix=".settings-",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = temporary_file.name
                json.dump(settings, temporary_file, indent=4)
                temporary_file.write("\n")
            os.replace(temporary_path, output_path)
            temporary_path = None
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.remove(temporary_path)

        return output_path

    def _configured_data_root(self) -> str:
        data_root = self.settings["fitting"].get("root")
        if not isinstance(data_root, str) or not data_root.strip():
            raise ValueError("請在 settings.json 設定有效的 fitting.root")
        return os.path.abspath(os.path.expanduser(data_root))

    def _model_output_root(self, output_root: str) -> str:
        if not isinstance(output_root, str) or not output_root.strip():
            raise ValueError("Output root must be a non-empty string.")
        return self.model.scoped_output_root(output_root)

    def _experiment_analysis_dir(self, analysis_type: str) -> str:
        if analysis_type not in {"single_lens", "two_lens"}:
            raise ValueError(
                f"Unknown analysis type {analysis_type!r}; "
                "expected 'single_lens' or 'two_lens'."
            )
        run_name = self.run_name or "default"
        experiment_root = self._model_output_root(
            self.settings["experiment"]["matrix_analysis"]["output_root"]
        )
        return os.path.join(experiment_root, analysis_type, run_name)

    # Testing methods
    # ---------------------------------------------------------------------------------------
    def test_inference(self):
        data_root = self._configured_data_root()
        jsonl_path = self.settings["inference"]["jsonl_path"]
        max_length = self.settings["inference"].get("max_length", 300)

        if not data_root or not jsonl_path:
            raise ValueError(
                "請在 settings.json 設定 fitting.root 與 inference.jsonl_path"
            )

        jsonl_path = os.path.abspath(jsonl_path)
        lines = Parser.load_jsonl_lines(jsonl_path)
        if not lines:
            raise ValueError(f"JSONL 沒有可推論的樣本：{jsonl_path}")

        logits = self.model.run_inference(
            jsonl_line=lines[0],
            data_root=data_root,
            MAX_LENGTH=max_length,
        )
        if logits.ndim != 3:
            raise RuntimeError(
                f"推論結果預期為 [batch, seq_len, vocab]，實際 shape={tuple(logits.shape)}"
            )

        next_token_logits = logits[0, -1]
        top_ids = torch.topk(next_token_logits, k=5).indices.tolist()
        tokenizer = self.model.model.tokenizer
        if tokenizer is None:
            raise RuntimeError("The configured model adapter has no tokenizer.")
        top_tokens = tokenizer.decode(
            top_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        print(f"推論完成：logits shape={tuple(logits.shape)}")
        print(f"下一個 token 候選：{top_tokens}")
        return logits

    def test_training(self):
        run_directory = os.path.join(
            self._model_output_root(self.settings["fitting"]["checkpoint_root"]),
            self.run_name
        )
        checkpoint_path = os.path.join(run_directory, "fit.pt")
        print(f"Checkpoint path: {checkpoint_path}")
        os.makedirs(run_directory, exist_ok=True)
            
        self.model.fit_Jlens_batch(
            jsonl_path= self.settings["fitting"]["jsonl_path"],
            data_root=self._configured_data_root(),
            checkpoint_save_path= checkpoint_path,
            run_name= self.run_name,
            dim_batch=self.settings["fitting"].get("dim_batch", 32),
            MAX_SEQ_LEN=self.settings["fitting"].get("max_seq_len", 300),
            checkpoint_interval=self.settings["fitting"].get(
                "checkpoint_interval", 5
            ),
            mode= None,
            is_test=True,
        )
    
    def test_lens(self):
        data_root = self._configured_data_root()
        jsonl_path = self.settings["fitting"]["jsonl_path"]
        max_length = self.settings["inference"].get("max_length", 300)
        
        if not data_root or not jsonl_path:
            raise ValueError(
                "請在 settings.json 設定 fitting.root 與 fitting.jsonl_path"
            )

        jsonl_path = os.path.abspath(jsonl_path)
        lines = Parser.load_jsonl_lines(jsonl_path)
        if not lines:
            raise ValueError(f"JSONL 沒有可推論的樣本：{jsonl_path}")
        
        self.model.apply_lens(
            json_line= lines[0],
            run_name= self.run_name,
            data_root= data_root,
            use_Jlens= False,
        )
        
        logits= self.model.lens_cache.get(
            lens_type= "logit_lens",
            key= "lens_logits",
            layer= 30
        )
        
        print(f"Success! Logits: {logits}")
    
    # Fitting methods
    # ---------------------------------------------------------------------------------------
    def calc_JLens_batch(self, mode: str | None = None):
        """Calls the model's fit_Jlens method to calculate JLens and save it to a checkpoint.
        
        Args:
            mode: Defines the source and target tokens for Jacobian matrix calculation.
                Can be one of the following:
                - "S2T"
                - "T2T"
                - "S&T2T"
                - "None" (Will use ordinary fit_jlens() function)
        """
        checkpoint_root = self._model_output_root(
            self.settings["fitting"]["checkpoint_root"]
        )
        checkpoint_name = self.run_name or "default"
        checkpoint_directory = os.path.join(checkpoint_root, checkpoint_name)
        checkpoint_path = os.path.join(checkpoint_directory, "fit.pt")
        os.makedirs(checkpoint_directory, exist_ok=True)
        
        # Copy settings.json to the checkpoint directory for reproducibility
        settings_copy_path = os.path.join(checkpoint_directory, "settings.json")
        from shutil import copy
        copy("settings.json", settings_copy_path)

        self.model.fit_Jlens_batch(
            jsonl_path= self.settings["fitting"]["jsonl_path"],
            data_root=self._configured_data_root(),
            checkpoint_save_path=checkpoint_path,
            run_name= self.run_name,
            dim_batch=self.settings["fitting"].get("dim_batch", 32),
            MAX_SEQ_LEN=self.settings["fitting"].get("max_seq_len", 300),
            checkpoint_interval=self.settings["fitting"].get(
                "checkpoint_interval", 5
            ),
            mode= mode
        )

    def calc_Jlens_experiment(self):
        """Fit dataset-specific and cumulative-count JLens for an experiment.

        ``fitting.experiment`` settings:
          - ``seed``: deterministic shuffle seed (default: 42)
          - ``mode``: token subset mode, or null for ordinary fitting
          - ``save_prompt_count``: cumulative successful-sample counts
          - ``max_prompts``: optional cap applied after deterministic shuffling
          - ``dim_batch``: Jacobian output batch size
          - ``checkpoint_interval``: fit-state save interval
        """
        fitting_settings = self.settings["fitting"]
        experiment_settings = fitting_settings.get("experiment", {})
        if not isinstance(experiment_settings, dict):
            raise ValueError("settings.json 'fitting.experiment' must be an object")

        seed = experiment_settings["seed"]
        if not isinstance(seed, int):
            raise ValueError("settings.json 'fitting.experiment.seed' must be an integer")

        milestone_counts = experiment_settings.get(
            "save_prompt_count", [1, 2, 5, 10, 20, 50, 100, 200, 500]
        )
        if (
            not isinstance(milestone_counts, list)
            or any(not isinstance(count, int) or count <= 0 for count in milestone_counts)
        ):
            raise ValueError(
                "settings.json 'fitting.experiment.save_prompt_count' must be a list "
                "of positive integers"
            )

        max_prompts = experiment_settings.get("max_prompts")
        if max_prompts is not None and (
            not isinstance(max_prompts, int)
            or isinstance(max_prompts, bool)
            or max_prompts <= 0
        ):
            raise ValueError(
                "settings.json 'fitting.experiment.max_prompts' must be "
                "a positive integer or null"
            )

        checkpoint_name = self.run_name or "default"
        experiment_directory = os.path.join(
            self._model_output_root(fitting_settings["checkpoint_root"]),
            checkpoint_name,
            "experiment",
        )
        
        # Ensure the experiment directory exists
        os.makedirs(experiment_directory, exist_ok=True)
        checkpoint_path = os.path.join(experiment_directory, "fit.pt")
        
        # Copy settings.json to the experiment directory for reproducibility
        settings_copy_path = os.path.join(experiment_directory, "settings.json")
        from shutil import copy
        copy("settings.json", settings_copy_path)

        # Fit
        self.logger.info(
            "Starting experiment JLens fit: seed=%d, max_prompts=%s, "
            "milestones=%s, output=%s",
            seed,
            max_prompts,
            milestone_counts,
            experiment_directory,
        )
        return self.model.fit_Jlens_for_experiment(
            jsonl_path=os.path.abspath(fitting_settings["jsonl_path"]),
            data_root=self._configured_data_root(),
            seed=seed,
            dim_batch=fitting_settings.get("dim_batch", 32),
            MAX_SEQ_LEN=fitting_settings.get("max_seq_len", 300),
            mode=experiment_settings.get("mode"),
            save_prompt_count=milestone_counts,
            checkpoint_save_path=checkpoint_path,
            run_name=checkpoint_name,
            is_test=experiment_settings.get("is_test", False),
            checkpoint_interval=fitting_settings.get("checkpoint_interval", 5),
            max_prompts=max_prompts,
        )
    
    # Experiment methods
    # ---------------------------------------------------------------------------------------
    def ablation(self):
        Experiments.ablation()
    
    def comp_dataset_choice(self):
        Experiments.comp_dataset_choice()
    
    def info_Jlens(
        self,
        lens_path: str,
        device: str | torch.device | None = None,
        svd_device: str | torch.device = "cpu",
    ) -> dict[str, object]:
        """Compute and save all single-lens cosine, CKA, and SVD heatmaps."""
        task_name = self.run_name or "default"
        matrix_settings = self.settings["experiment"]["matrix_analysis"]
        output_dir = self._experiment_analysis_dir("single_lens")
        os.makedirs(output_dir, exist_ok=True)
        
        svd_top_k = int(matrix_settings["SVD_top_k"])
        lens = Jlens(logger=self.logger)
        lens.load_lens(path=lens_path)
        if lens.lens is None:
            raise RuntimeError(f"JLens checkpoint did not load: {lens_path}")

        jacobians = lens.get_jacobian()
        layers = list(lens.lens.source_layers)
        analysis = Experiments.info_Jlens(
            logger=self.logger,
            jacobians=jacobians,
            layers=layers,
            svd_top_k=svd_top_k,
            device=device,
            svd_device=svd_device,
        )
        # Validate the structure and types of the analysis results
        layers = analysis["layers"]
        assert isinstance(layers, list) and all(isinstance(layer, int) for layer in layers)
        assert isinstance(analysis["cosine_heatmap"], torch.Tensor)
        assert isinstance(analysis["cka_heatmap"], torch.Tensor)
        assert isinstance(analysis["svd_spectra"], torch.Tensor)
        assert isinstance(analysis["svd_adjacent_cosines"], torch.Tensor)
        assert isinstance(analysis["svd_layers"], list) and all(isinstance(layer, int) for layer in analysis["svd_layers"])
        assert isinstance(analysis["svd_layer_pairs"], list) and all(isinstance(pair, str) for pair in analysis["svd_layer_pairs"])
        # Draw the heatmaps
        MathTools.draw_heatmap(
            data=analysis["cosine_heatmap"],
            png_path=os.path.join(output_dir, f"{task_name}_cosine.png"),
            x_labels=layers,
            y_labels=layers,
            title="JLens Jacobian cosine similarity",
        )
        MathTools.draw_heatmap(
            data=analysis["cka_heatmap"],
            png_path=os.path.join(output_dir, f"{task_name}_cka.png"),
            x_labels=layers,
            y_labels=layers,
            title="JLens Jacobian CKA similarity",
            colorbar_label="CKA similarity",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )

        svd_output_dir = os.path.join(
            output_dir,
            "SVD_analysis",
        )
        os.makedirs(svd_output_dir, exist_ok=True)
        spectra = analysis["svd_spectra"]
        log_spectra = spectra.clamp_min(
            torch.finfo(spectra.dtype).tiny
        ).log10()
        
        MathTools.draw_heatmap(
            data=log_spectra,
            png_path=os.path.join(svd_output_dir, f"{task_name}_spectrum.png"),
            x_labels=list(range(log_spectra.shape[1])),
            y_labels=analysis["svd_layers"],
            title="Jacobian singular-value spectra",
            x_axis_label="Singular value index",
            y_axis_label="Layer",
            colorbar_label="log10(singular value)",
            cmap="magma",
            vmin=None,
            vmax=None,
        )
        MathTools.draw_heatmap(
            data=analysis["svd_adjacent_cosines"],
            png_path=os.path.join(
                svd_output_dir,
                f"{task_name}_adjacent_subspace.png",
            ),
            x_labels=list(range(1, svd_top_k + 1)),
            y_labels=analysis["svd_layer_pairs"],
            title=f"Adjacent-layer top-{svd_top_k} left-singular subspaces",
            x_axis_label="Principal cosine rank",
            y_axis_label="Adjacent layer pair",
            colorbar_label="Principal cosine",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        self.logger.info("Saved single-lens heatmaps to %s", output_dir)
        return analysis
    
    def comp_Jlens(self, lens_path1: str, lens_path2: str) -> str:
        """Compare matching Jacobian layers and save metrics in the two-lens run folder."""
        lens1 = Jlens(logger=self.logger)
        lens2 = Jlens(logger=self.logger)
        lens1.load_lens(path=lens_path1)
        lens2.load_lens(path=lens_path2)

        jacobians1 = lens1.get_jacobian()
        jacobians2 = lens2.get_jacobian()
        layers1 = sorted(jacobians1)
        layers2 = sorted(jacobians2)
        if layers1 != layers2:
            raise ValueError(
                "JLens Jacobian layers do not match: "
                f"{layers1!r} != {layers2!r}."
            )
        if not layers1:
            raise ValueError("Cannot compare JLens checkpoints with no Jacobian layers.")

        comparisons = [
            Experiments.compare_jacobians(
                jacobian1=jacobians1[layer],
                jacobian2=jacobians2[layer],
            )
            for layer in layers1
        ]

        output_dir = self._experiment_analysis_dir("two_lens")
        os.makedirs(output_dir, exist_ok=True)
        lens1_name = os.path.splitext(os.path.basename(lens_path1))[0]
        lens2_name = os.path.splitext(os.path.basename(lens_path2))[0]
        output_name = f"comparison_{lens1_name}_vs_{lens2_name}"
        rows = [
            {"layer": layer, **asdict(comparison)}
            for layer, comparison in zip(layers1, comparisons)
        ]

        csv_path = os.path.join(output_dir, f"{output_name}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(value) if isinstance(value, list) else value
                        for key, value in row.items()
                    }
                )

        report_path = os.path.join(output_dir, f"{output_name}.json")
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(
                {
                    "lens1": lens_path1,
                    "lens2": lens_path2,
                    "layers": rows,
                },
                report_file,
                indent=2,
            )

        similarity_plot_path = os.path.join(
            output_dir,
            f"{output_name}_cosine_similarity.png",
        )
        MathTools.draw_line_graph(
            data={
                "Cosine similarity": [
                    comparison.cosine_similarity for comparison in comparisons
                ]
            },
            png_path=similarity_plot_path,
            x_values=layers1,
            title="Same-layer JLens Jacobian cosine similarity",
            x_axis_label="Layer",
            y_axis_label="Cosine similarity",
        )
        relative_difference_plot_path = os.path.join(
            output_dir,
            f"{output_name}_relative_differences.png",
        )
        MathTools.draw_line_graph(
            data={
                "Symmetric relative Frobenius distance": [
                    comparison.symmetric_relative_frobenius_dist
                    for comparison in comparisons
                ],
                "Relative largest singular-value difference": [
                    comparison.relative_largest_sval_diff
                    for comparison in comparisons
                ],
            },
            png_path=relative_difference_plot_path,
            x_values=layers1,
            title="Relative JLens Jacobian differences",
            x_axis_label="Layer",
            y_axis_label="Relative difference",
        )
        subspace_alignment_plot_path = os.path.join(
            output_dir,
            f"{output_name}_subspace_alignment.png",
        )
        MathTools.draw_line_graph(
            data={
                "Left mean principal cosine": [
                    comparison.left_mean_principal_cosine
                    for comparison in comparisons
                ],
                "Right mean principal cosine": [
                    comparison.right_mean_principal_cosine
                    for comparison in comparisons
                ],
            },
            png_path=subspace_alignment_plot_path,
            x_values=layers1,
            title="JLens Jacobian subspace alignment",
            x_axis_label="Layer",
            y_axis_label="Mean principal cosine",
        )
        self.logger.info("Saved JLens comparison CSV to %s", csv_path)
        self.logger.info("Saved JLens comparison report to %s", report_path)
        self.logger.info(
            "Saved JLens comparison cosine plot to %s",
            similarity_plot_path,
        )
        self.logger.info(
            "Saved JLens comparison relative-difference plot to %s",
            relative_difference_plot_path,
        )
        self.logger.info(
            "Saved JLens comparison subspace-alignment plot to %s",
            subspace_alignment_plot_path,
        )
        return output_dir

    def plot_comp_Jlens_csv(
        self,
        lens_path1: str,
        lens_path2: str,
        csv_path: str | None = None,
    ) -> list[str]:
        """Plot comparison metrics from an existing CSV without loading checkpoints."""
        lens1_name = os.path.splitext(os.path.basename(lens_path1))[0]
        lens2_name = os.path.splitext(os.path.basename(lens_path2))[0]
        output_name = f"comparison_{lens1_name}_vs_{lens2_name}"
        if csv_path is None:
            csv_path = os.path.join(
                self._experiment_analysis_dir("two_lens"),
                f"{output_name}.csv",
            )
        csv_path = os.path.abspath(os.path.expanduser(csv_path))

        required_columns = {
            "layer",
            "cosine_similarity",
            "symmetric_relative_frobenius_dist",
            "relative_largest_sval_diff",
            "left_mean_principal_cosine",
            "right_mean_principal_cosine",
        }
        with open(csv_path, newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            missing_columns = required_columns - set(reader.fieldnames or ())
            if missing_columns:
                raise ValueError(
                    f"Comparison CSV {csv_path} is missing required columns: "
                    f"{sorted(missing_columns)}"
                )

            layers: list[int] = []
            values: dict[str, list[float]] = {
                column: []
                for column in required_columns
                if column != "layer"
            }
            for row_number, row in enumerate(reader, start=2):
                try:
                    layers.append(int(row["layer"]))
                    for column in values:
                        values[column].append(float(row[column]))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid numeric value in comparison CSV row {row_number}: "
                        f"{exc}"
                    ) from exc

        if not layers:
            raise ValueError(f"Comparison CSV contains no data rows: {csv_path}")

        output_dir = os.path.dirname(csv_path)
        output_name = os.path.splitext(os.path.basename(csv_path))[0]
        plot_specs = [
            (
                f"{output_name}_cosine_similarity.png",
                {"Cosine similarity": values["cosine_similarity"]},
                "Same-layer JLens Jacobian cosine similarity",
                "Cosine similarity",
            ),
            (
                f"{output_name}_relative_differences.png",
                {
                    "Symmetric relative Frobenius distance": values[
                        "symmetric_relative_frobenius_dist"
                    ],
                    "Relative largest singular-value difference": values[
                        "relative_largest_sval_diff"
                    ],
                },
                "Relative JLens Jacobian differences",
                "Relative difference",
            ),
            (
                f"{output_name}_subspace_alignment.png",
                {
                    "Left mean principal cosine": values[
                        "left_mean_principal_cosine"
                    ],
                    "Right mean principal cosine": values[
                        "right_mean_principal_cosine"
                    ],
                },
                "JLens Jacobian subspace alignment",
                "Mean principal cosine",
            ),
        ]
        plot_paths: list[str] = []
        for filename, series, title, y_axis_label in plot_specs:
            plot_path = os.path.join(output_dir, filename)
            MathTools.draw_line_graph(
                data=series,
                png_path=plot_path,
                x_values=layers,
                title=title,
                x_axis_label="Layer",
                y_axis_label=y_axis_label,
            )
            plot_paths.append(plot_path)

        self.logger.info("Plotted JLens comparison metrics from %s", csv_path)
        return plot_paths
        
    def eval_Jlens(
        self,
        lens_path: str,
        prompts_path: str,
        top_k: int | None = None,
        layer: int | None = None,
    ) -> str:
        """Evaluate pass@k with both JLens and LogitLens predictions."""
        # Prepare evaluation and output settings.
        evaluation_settings = self.settings["experiment"]["pass_k_AUC"]
        k_values = evaluation_settings.get("k_values")
        csv_top_k = top_k or self.settings["inference"]["top_k"]
        task_name = self.run_name or "default"
        output_root = os.path.join(
            self._model_output_root(self.settings["inference"]["output_root"]),
            task_name,
        )

        # Evaluate both lenses on the prompt set.
        evaluation = Experiments.passk_Jlens(
            logger=self.logger,
            model=self.model,
            run_name=task_name,
            lens_path=lens_path,
            prompts_path=prompts_path,
            data_root=self._configured_data_root(),
            top_k=csv_top_k,
            k_values=k_values,
            layer=layer,
        )

        # Separate and validate prediction rows.
        prediction_rows = evaluation.pop("prediction_rows")
        if not isinstance(prediction_rows, list) or any(
            not isinstance(row, dict) for row in prediction_rows
        ):
            raise RuntimeError("JLens evaluation returned invalid prediction rows.")

        # Save token predictions to CSV.
        os.makedirs(output_root, exist_ok=True)
        csv_path = os.path.join(output_root, f"{task_name}.csv")
        write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=self.model.lens_cache.CSV_FIELDNAMES,
            )
            if write_header:
                writer.writeheader()
            writer.writerows(prediction_rows)

        # Save aggregate and per-layer metrics to JSON.
        report_path = os.path.join(
            output_root,
            f"{task_name}_passk_auc.json",
        )
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(evaluation, report_file, indent=2)

        # Validate layer results and align both lenses.
        per_layer = evaluation.get("by_layer")
        if not isinstance(per_layer, dict) or not per_layer:
            raise RuntimeError("JLens evaluation returned no per-layer pass@k results.")
        jlens_by_layer = per_layer.get("jlens")
        logit_lens_by_layer = per_layer.get("logit_lens")
        if not isinstance(jlens_by_layer, dict) or not isinstance(
            logit_lens_by_layer, dict
        ):
            raise RuntimeError("Per-layer results are missing JLens or LogitLens data.")
        layers = sorted(int(layer_name) for layer_name in jlens_by_layer)
        if [str(layer_idx) for layer_idx in layers] != sorted(
            logit_lens_by_layer, key=int
        ):
            raise RuntimeError("Per-layer JLens and LogitLens results do not align.")

        # Collect normalized AUC values in layer order.
        def layer_aucs(
            lens_results: dict[str, object],
            lens_name: str,
        ) -> list[float]:
            aucs: list[float] = []
            for layer_idx in layers:
                layer_result = lens_results[str(layer_idx)]
                if not isinstance(layer_result, dict):
                    raise RuntimeError(
                        f"Invalid {lens_name} result for layer {layer_idx}."
                    )
                auc = layer_result.get("normalized_auc")
                if not isinstance(auc, (int, float)):
                    raise RuntimeError(
                        f"Missing {lens_name} AUC for layer {layer_idx}."
                    )
                aucs.append(float(auc))
            return aucs

        # Plot JLens and LogitLens AUC by layer.
        layer_auc_plot_path = os.path.join(
            output_root,
            f"{task_name}_passk_auc_by_layer.png",
        )
        MathTools.draw_line_graph(
            data={
                "JLens": layer_aucs(jlens_by_layer, "JLens"),
                "LogitLens": layer_aucs(logit_lens_by_layer, "LogitLens"),
            },
            png_path=layer_auc_plot_path,
            x_values=layers,
            title="Pass@k AUC by layer",
            x_axis_label="Layer",
            y_axis_label="Normalized AUC over log(k)",
        )
        self.logger.info("Saved inference predictions to %s", csv_path)
        self.logger.info("Saved pass@k evaluation to %s", report_path)
        self.logger.info("Saved pass@k AUC-by-layer plot to %s", layer_auc_plot_path)
        return output_root
    
    # Inference methods
    # ---------------------------------------------------------------------------------------
    def apply_Jlens(
        self,
        lens_path: str,
        prompts_path: str,
        inference_cnt: int | None = None,
        top_k: int | None = None,
        layer: int | None = None,
        ) -> str:
        """Inference the Jlens on the given prompt set.

        Args:
            lens_path (str): Path to the Jlens file.
            prompts_path (str): Path to the prompt JSONL file.
            top_k (int | None, optional): Number of top predictions to consider. 
                The Default is all layers.
            inference_cnt (int | None, optional): Number of inferences to run. 
                The Default is all prompts.
            layer (int | None, optional): Layer of the model to apply the Jlens. 
                The Default is at settings["inference"]["layer"].
        """
        json_lines = Parser.load_jsonl_lines(prompts_path)
        self.model.load_Jlens(lens_path= lens_path)
        
        output_root = os.path.join(
            self._model_output_root(self.settings["inference"]["output_root"]),
            self.run_name,
        )
        os.makedirs(output_root, exist_ok=True)
        for i in range(len(json_lines)):
            if inference_cnt is not None and i >= inference_cnt:
                break
            self.model.apply_lens(
                json_line = json_lines[i],
                data_root=self._configured_data_root(),
                save_path= output_root,
                top_k= top_k or self.settings["inference"]["top_k"],
                layers_available= [layer] if layer is not None else None,
                use_Jlens= True,
                run_name= self.run_name,
            )
        return output_root

    def generate(
        self,
        prompts_path: str | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate responses for prompts and save them to a CSV file."""
        if prompts_path is None:
            prompts_path = self.settings["inference"].get("jsonl_path")
        if not isinstance(prompts_path, str) or not prompts_path.strip():
            raise ValueError("A valid prompts_path or inference.jsonl_path is required.")
        prompts_path = os.path.abspath(os.path.expanduser(prompts_path))
        json_lines = Parser.load_jsonl_lines(prompts_path)
        if not json_lines:
            raise ValueError(f"Prompt JSONL contains no samples: {prompts_path}")

        if max_new_tokens is None:
            max_new_tokens = int(
                self.settings["inference"].get("max_new_tokens", 128)
            )
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")

        data_root = self._configured_data_root()
        output_root = os.path.join(
            self._model_output_root(self.settings["inference"]["output_root"]),
            self.run_name or "default",
        )
        os.makedirs(output_root, exist_ok=True)
        output_path = os.path.join(output_root, "generated.csv")
        with open(output_path, "w", newline="", encoding="utf-8-sig") as output_file:
            fieldnames = ("sample_id", "prompt", "reference", "generated")
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for json_line in json_lines:
                sample = Parser.parse_json_line(json_line, audio_root=data_root)
                generated_text = self.model.generate(
                    jsonl_line=json_line,
                    data_root=data_root,
                    max_new_tokens=max_new_tokens,
                    max_input_length=int(
                        self.settings["inference"].get("max_input_length", 3000)
                    ),
                )
                writer.writerow(
                    {
                        "sample_id": sample.sample_id,
                        "prompt": sample.prompt,
                        "reference": sample.transcript or "",
                        "generated": generated_text,
                    }
                )
        self.logger.info("Saved generated responses to %s", output_path)
        return output_path

    @staticmethod
    def check_dependencies():
        print("=" * 70)
        print("[Step 0] 依賴檢查")
        print("=" * 70)

        missing = []
        versions = {}

        try:
            versions["torch"] = torch.__version__
            if not torch.cuda.is_available():
                print("[FATAL] 找不到可用的 CUDA GPU，請確認驅動與 CUDA 安裝。")
                sys.exit(1)
            gpu_name = torch.cuda.get_device_name(0)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"  GPU: {gpu_name} | 總 VRAM: {total_vram_gb:.1f} GB")
            if total_vram_gb < 11:
                print("  [警告] 偵測到的 VRAM 低於 11GB，本腳本的記憶體預算可能不適用，請調整 batch/長度設定。")
        except ImportError:
            missing.append("torch")

        for pkg in ["transformers", "bitsandbytes", "accelerate", "librosa", "soundfile"]:
            try:
                mod = __import__(pkg)
                versions[pkg] = getattr(mod, "__version__", "unknown")
            except ImportError:
                missing.append(pkg)

        if missing:
            print(f"[FATAL] 缺少套件: {missing}")
            print("  請參考環境建構建議安裝，例如：")
            print("  pip install git+https://github.com/huggingface/transformers")
            print("  pip install bitsandbytes accelerate librosa soundfile")
            sys.exit(1)

        print("  已安裝套件版本：")
        for k, v in versions.items():
            print(f"    - {k}: {v}")
