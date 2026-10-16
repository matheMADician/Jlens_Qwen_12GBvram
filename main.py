from Master import Master
import argparse

parser = argparse.ArgumentParser(description="Run JLens model workflows.")
parser.add_argument(
    "--run-name",
    default=None,
    help="Name used for the JLens checkpoint run.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    master = Master(run_name=args.run_name)
    #master.calc_Jlens_experiment()
    #master.test_inference()
    #master.test_training()
    master.apply_Jlens(
        lens_path= "output/checkpoints/Qwen2-Audio/train_1_no_SAKURA/experiment/milestones/lens_count_50.pt",
        prompts_path= "data/assembled/test_SAKURA_AnimalQA/prompts.jsonl",
        top_k=20,
        inference_cnt=2,
    )
    """
    master.draw_Jlens_heatmap(
        lens_path1= "output/checkpoints/train_1_no_SAKURA/experiment/milestones/lens_count_500.pt",
        mode= "cka"
        #lens_path2= "output/checkpoints/esc50-50_libri-50/lens.pt"
    )
    """

    # master.calc_JLens(mode="T2T")
    """
    master.draw_Jlens_heatmap(
        lens_path1= "output/checkpoints/esc50-50_+_libri-50/lens.pt",
        lens_path2= "output/checkpoints/esc50-50_+_libri-50_T2T/lens.pt"
    )
    """
    """
    master.draw_Jlens_svd_heatmaps(
        lens_path= "output/checkpoints/esc50-50_+_libri-50_T2T/lens.pt",
    )
    """
    """
    master.eval_Jlens(
        lens_path= "output/checkpoints/train_1_no_SAKURA/experiment/lens_full.pt",
        prompts_path= "data/assembled/test_SAKURA_AnimalQA/prompts.jsonl",
        top_k=5,
        layer=22,
    )
    """
    
    