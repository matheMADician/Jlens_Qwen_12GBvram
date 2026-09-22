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
    master.apply_Jlens()
    """
    master.draw_Jlens_heatmap(
        heatmap_path= "tools/heatmaps",
        lens_path1= "Jlens/lens_checkpoints/esc50-50_fleurs_en_us-50",
        lens_path2= "Jlens/lens_checkpoints/esc50-50_libri-50"
    )
    """