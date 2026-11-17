import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from Master import Master
from tools.experiments import Experiments


def test_master_evaluates_independent_lenses_and_saves_separate_outputs():
    with TemporaryDirectory() as output_dir:
        master = Master.__new__(Master)
        master.logger = Mock()
        master.run_name = "cross_eval"
        master.settings = {
            "experiment": {"pass_k_AUC": {"k_values": [1]}},
            "inference": {"top_k": 1, "output_root": output_dir},
            "fitting": {"root": output_dir},
        }
        master.model = SimpleNamespace(
            scoped_output_root=lambda _root: output_dir,
            lens_cache=SimpleNamespace(CSV_FIELDNAMES=("position",)),
        )

        per_lens_result = {
            "best_answer_ranks": [1],
            "by_layer": {"0": {"normalized_auc": 0.5}},
        }
        evaluation = {
            "label_results": {
                "emotion": {
                    "label": "emotion",
                    "lenses": {
                        "0": per_lens_result,
                        "1": per_lens_result,
                    },
                    "logit_lens": {
                        "by_layer": {"0": {"normalized_auc": 0.4}},
                    },
                },
            },
            "prediction_rows": {
                "0": [{"position": 0}],
                "1": [{"position": 1}],
            },
        }
        lens_a = SimpleNamespace()
        lens_b = SimpleNamespace()

        with (
            patch("Master.JacobianLens.load", side_effect=[lens_a, lens_b]),
            patch.object(Experiments, "passk_Jlens", return_value=evaluation) as evaluate,
            patch.object(Master, "_merge_lenses") as merge,
            patch("Master.MathTools.draw_line_graph") as draw_plot,
        ):
            master.eval_Jlens(
                lens_paths=["lens_esd.pt", "lens_fsd50k.pt"],
                prompts_path="prompts.jsonl",
                label_keys=["emotion"],
                evaluate_lenses_independently=True,
            )

        merge.assert_not_called()
        evaluate.assert_called_once()
        assert evaluate.call_args.kwargs["do_batch"] is True
        assert len(evaluate.call_args.kwargs["lens"]) == 2
        result_dir = Path(output_dir) / "cross_eval"
        assert set(result_dir.glob("cross_eval_lens_*.csv")) == {
            result_dir / "cross_eval_lens_esd.csv",
            result_dir / "cross_eval_lens_fsd50k.csv",
        }
        report_path = result_dir / (
            "cross_eval_lens_esd_label_emotion_passk_auc.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["lens_path"] == "lens_esd.pt"
        assert set(report["lenses"]) == {"0"}
        assert {
            Path(call.kwargs["png_path"])
            for call in draw_plot.call_args_list
        } == {
            result_dir / "emotion" / "esd.png",
            result_dir / "emotion" / "fsd50k.png",
        }
