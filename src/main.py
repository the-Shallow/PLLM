import argparse
import yaml
from src.runner.run_experiment import run_experiment
from src.runner.logging import logger


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, help="Path to experiment yaml")
    ap.add_argument("--profile", required=False, help="Path to profile yaml")
    args = ap.parse_args()

    logger.info(f"Loading experiment config from {args.experiment}")
    exp_cfg = load_yaml(args.experiment)

    profile_cfg = {}
    if args.profile:
        logger.info(f"Loading profile config from {args.profile}")
        profile_cfg = load_yaml(args.profile)
    else:
        logger.info("No profile config provided")

    # with open(args.experiment, "r") as f:
    #     cfg = yaml.safe_load(f)
    logger.info(f"Model : {exp_cfg['model']['name']}")
    logger.info(f"Pruning enabled : {exp_cfg.get('prune', {}).get('enabled', False)}")

    if profile_cfg:
        logger.info(f"Profile loaded: {args.profile}")
        if "paths" in profile_cfg:
            logger.info(f"Profile paths: {profile_cfg['paths']}")

    metrics = run_experiment(exp_cfg, profile_cfg)
    logger.info(f"Experiment completed. Metrics : {metrics}")

if __name__ == "__main__":
    main()