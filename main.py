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
    master.test_training()