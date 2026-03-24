import argparse
from pathlib import Path
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Running baseline experiment: {config['experiment_name']}")
    print(f"Forecast horizon: {config['horizon']}")
    print("Next step: connect this script to your TFT training pipeline.")


if __name__ == "__main__":
    main()
