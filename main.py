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
    #master.test_inference()
    #master.apply_Jlens(lens_path= "/home/c8763c8763/Jlens/Jlens/output/checkpoints/esc50-50_+_libri-50_S&T2T/lens.pt")
    """
    master.draw_Jlens_heatmap(
        heatmap_path= "output/heatmaps",
        lens_path1= "output/checkpoints/esc50-50_+_libri-50/lens.pt",
        #lens_path2= "output/checkpoints/esc50-50_libri-50/lens.pt"
    )
    """
    # master.test_training()
    # master.calc_JLens(mode="T2T")
    """
    master.draw_Jlens_heatmap(
        lens_path1= "output/checkpoints/esc50-50_+_libri-50/lens.pt",
        lens_path2= "output/checkpoints/esc50-50_+_libri-50_T2T/lens.pt"
    )
    """
    master.draw_Jlens_svd_heatmaps(
        lens_path= "output/checkpoints/esc50-50_+_libri-50_T2T/lens.pt",
    )
    
    
    