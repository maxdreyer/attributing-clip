import argparse
import os
import socket

import torch as torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset
from model_training.sae import TopKSAE
from models import get_fn_model_loader
from utils.helper import load_config


def get_args():
    parser = argparse.ArgumentParser(description='Compute relevances and activations')
    parser.add_argument(
        '--config_file', type=str,
        default="configs/imagenet/local/clip_vit_mobiles2_datacomp_imagenet.yaml"
    )
    parser.add_argument('--class_id', type=int, default=-1)
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--sae_ckpt_path', type=str, default=None)
    return parser.parse_args()


def collect_latent_features(model_name,
                            dataset_name,
                            data_path,
                            split,
                            class_id,
                            batch_size,
                            sae_config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = get_dataset(dataset_name)(data_path=data_path, normalize_data=True, split=split)
    print(len(dataset))
    if class_id != -1:
        samples_of_class = torch.tensor([i for i in range(len(dataset)) if dataset.get_target(i) == class_id])
    else:
        samples_of_class = torch.tensor([i for i in range(len(dataset))])

    print("Dataset loaded", len(samples_of_class))

    model = get_fn_model_loader(model_name)()
    sae = TopKSAE.from_config(sae_config, model)
    model.add_sae(sae)
    torch.nn.utils.remove_weight_norm(model.sae.decoder)

    model = model.to(device)
    model.eval()
    print("Model loaded")
    dataset.transform = model.preprocess
    dataset_subset = torch.utils.data.Subset(dataset, samples_of_class)
    dataloader = DataLoader(dataset_subset, batch_size=batch_size, shuffle=False, num_workers=16)

    layer_names = ["sae.hook_hidden_post"]

    max_activation = {}
    mean_activation = {}
    max_sample_ids = {}
    mean_sample_ids = {}

    topk = 100
    j = 0
    # hook to model
    def hook_layer(layer_name):
        def hook_activations(module, input, output):
            if len(output.shape) == 4:
                max_activation[layer_name].append(output.detach().amax((2, 3)).to(torch.float16))
                mean_activation[layer_name].append(output.detach().mean((2, 3)).to(torch.float16))
            elif len(output.shape) == 3:
                max_activation[layer_name].append(output.detach().amax((1)).to(torch.float16))
                mean_activation[layer_name].append(output.detach().mean((1)).to(torch.float16))
            else:
                max_activation[layer_name].append((output.detach()).to(torch.float16))
                mean_activation[layer_name].append((output.detach()).to(torch.float16))

            max_sample_ids[layer_name].append(
                samples_of_class[(j * batch_size): ((j + 1) * batch_size)][:, None].expand_as(mean_activation[layer_name][-1]).to(device))
            mean_sample_ids[layer_name].append(
                samples_of_class[(j * batch_size): ((j + 1) * batch_size)][:, None].expand_as(mean_activation[layer_name][-1]).to(device))

        return hook_activations

    for n, m in model.named_modules():
        if n in layer_names:
            m.register_forward_hook(hook_layer(n))
            max_activation[n] = []
            mean_activation[n] = []
            max_sample_ids[n] = []
            mean_sample_ids[n] = []

    for i, (x, y) in enumerate(tqdm(dataloader)):
        model(x.to(device))
        j += 1

        if i % 50 == 0 or i == len(dataloader) - 1:
            for layer_name in layer_names:
                max_ids = torch.cat(max_activation[layer_name], 0).argsort(0, descending=True)[:topk]
                mean_ids = torch.cat(mean_activation[layer_name], 0).argsort(0, descending=True)[:topk]
                max_sample_ids[layer_name] = [torch.gather(torch.cat(max_sample_ids[layer_name], 0), dim=0, index=max_ids)]
                mean_sample_ids[layer_name] = [torch.gather(torch.cat(mean_sample_ids[layer_name], 0), dim=0, index=mean_ids)]
                max_activation[layer_name] = [torch.gather(torch.cat(max_activation[layer_name], 0), dim=0, index=max_ids)]
                mean_activation[layer_name] = [torch.gather(torch.cat(mean_activation[layer_name], 0), dim=0, index=mean_ids)]


    path = f"results/global_features_minimal/{dataset_name}/{model_name}"
    os.makedirs(path, exist_ok=True)

    save_file({k: torch.cat(v, 0)[:, ].cpu() for k, v in max_activation.items()},
              f"{path}/max_activations_{class_id}_{split}.safetensors")
    save_file({k: torch.cat(v, 0)[:, ].cpu() for k, v in mean_activation.items()},
              f"{path}/mean_activations_{class_id}_{split}.safetensors")
    save_file({k: torch.cat(v, 0)[:, ].cpu() for k, v in max_sample_ids.items()},
                f"{path}/max_sample_ids_{class_id}_{split}.safetensors")
    save_file({k: torch.cat(v, 0)[:, ].cpu() for k, v in mean_sample_ids.items()},
                f"{path}/mean_sample_ids_{class_id}_{split}.safetensors")



if __name__ == "__main__":
    args = get_args()

    config = load_config(args.config_file)

    model_name = config['model_name']
    dataset_name = config['dataset_name']
    class_id = args.class_id
    batch_size = config['batch_size'] * 0 + 8
    if socket.gethostname() == "dqg6mm2":
        batch_size = 12  # if dataset_name == 'chexpert' else 20
    data_path = config.get('data_path', None)
    split = args.split

    sae_keys = ["sae_input_dim", "sae_hidden_dim", "sae_k", "sae_layer_idx", "sae_token", "sae_ckpt_path"]
    sae_config = {key: config[key] for key in sae_keys if key in config}
    if args.sae_ckpt_path is not None:
        sae_config["sae_ckpt_path"] = args.sae_ckpt_path

    collect_latent_features(model_name, dataset_name, data_path, split, class_id, batch_size, sae_config)
