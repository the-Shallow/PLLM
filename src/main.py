import argparse
import yaml
from src.runner.run_experiment import run_experiment

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, help="Path to experiment yaml")
    args = ap.parse_args()

    with open(args.experiment, "r") as f:
        cfg = yaml.safe_load(f)

    metrics = run_experiment(cfg)
    print(metrics)

if __name__ == "__main__":
    main()