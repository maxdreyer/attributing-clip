import copy
import os
import shutil

import yaml

config_dir = "configs/train_sae/imagenet"
shutil.rmtree(config_dir, onerror=lambda a, b, c: None)
os.makedirs(f"{config_dir}/for_train", exist_ok=True)
os.makedirs(f"{config_dir}/after_train", exist_ok=True)
os.makedirs(f"{config_dir}/cluster", exist_ok=True)

with open("configs/local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)

_base_config = {
    'dataset_name': 'imagenet',
    'wandb_api_key': local_config['wandb_api_key'],
    'wandb_project_name': 'attributing-clip-train-sae-0',
    'num_epochs': 30,
    'lr_scheduler': 'MultiStepLR',
    'milestones': [24, 28],
}


def store_local_for_training(config, config_name):
    config['ckpt_path'] = f""
    config['batch_size'] = 32
    config['data_path'] = local_config['imagenet_dir']

    with open(f"{config_dir}/for_train/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def store_local_after_training(config, config_name):
    config['ckpt_path'] = f""
    config['sae_ckpt_path'] = f"checkpoints/{config_name}/last.ckpt"
    config['batch_size'] = 32
    config['data_path'] = local_config['imagenet_dir']

    with open(f"{config_dir}/after_train/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


for model_name in [
    'clip_vit_b32_datacomp_xl_s13b_b90k',
    'clip_vit_b16_datacomp_xl_s13b_b90k',
    'clip_vit_l14_datacomp_xl_s13b_b90k',
    'clip_vit_h14_dfn5b',
    'clip_vit_l14_336_openai',
]:

    for lr in [2e-4]:
        model_name = f"{model_name}"
        _base_config['lr'] = lr
        _base_config['model_name'] = model_name

        for sae_layer_idx in [-1]:
            if model_name.startswith('clip_vit_l14_336_openai'):
                sae_layer_idx = -2
            else:
                sae_layer_idx = -1
            _base_config['sae_layer_idx'] = sae_layer_idx

            for sae_hidden_dim in [30000]:
                _base_config['sae_hidden_dim'] = sae_hidden_dim

                for sae_token in ['cls', 'spatial']:
                    _base_config['sae_token'] = sae_token

                    for sae_k in [64]:
                        _base_config['sae_k'] = sae_k

                        base_config = copy.deepcopy(_base_config)

                        config = copy.deepcopy(base_config)
                        config_name = f"{model_name}_{base_config['dataset_name']}_topk-{sae_token}-{sae_k}-{sae_hidden_dim}_layer-{sae_layer_idx}_lr_{lr}"
                        store_local_for_training(config, config_name)
                        store_local_after_training(config, config_name)
