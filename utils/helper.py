import os

import yaml


def load_config(config_path):
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            config = {}
        config["wandb_id"] = os.path.basename(config_path)[:-5]
        config["config_name"] = os.path.basename(config_path)[:-5]
    return config
