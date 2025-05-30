import copy
import os
import shutil
import yaml

config_dir = "configs/imagenet"
shutil.rmtree(config_dir, onerror=lambda a, b, c: None)
os.makedirs(f"{config_dir}/local", exist_ok=True)
os.makedirs(f"{config_dir}/cluster", exist_ok=True)

with open("configs/local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)

_base_config = {
    'dataset_name': 'imagenet',
    'wandb_api_key': local_config['wandb_api_key'],
    'wandb_project_name': 'attributing-clip-imagenet-0',
}

def store_local(config, config_name):
    config['ckpt_path'] = ""
    config['batch_size'] = 32
    config['data_path'] = local_config['imagenet_dir']

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def store_cluster(config, config_name):
    config['ckpt_path'] = ""
    config['batch_size'] = 64
    config['data_path'] = "/mnt/imagenet"

    with open(f"{config_dir}/cluster/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

for model_name in [
    "clip_vit_mobiles2_datacompdr",
    # ... add more models as needed
]:

    model_name = f"{model_name}"
    _base_config['model_name'] = model_name

    base_config = copy.deepcopy(_base_config)

    config = copy.deepcopy(base_config)
    config_name = f"{model_name}_{base_config['dataset_name']}"
    store_local(config, config_name)
    store_cluster(config, config_name)