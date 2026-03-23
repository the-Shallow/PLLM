import argparse
import yaml
from src.runner.run_experiment import run_experiment
from src.runner.logging import logger

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, help="Path to experiment yaml")
    args = ap.parse_args()

    logger.info(f"Loading experiment config from {args.experiment}")


    with open(args.experiment, "r") as f:
        cfg = yaml.safe_load(f)
        logger.info(f"Model : {cfg['model']['name']}")
        logger.info(f"Pruning enabled : {cfg.get('prune', {}).get('enabled', False)}")

    metrics = run_experiment(cfg)
    logger.info(f"Experiment completed. Metrics : {metrics}")

if __name__ == "__main__":
    main()