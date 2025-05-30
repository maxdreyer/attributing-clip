import argparse
import os
import torch as torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset
from models import get_fn_model_loader
from utils.helper import load_config


def get_args():
    parser = argparse.ArgumentParser(description='Compute visual embeddings')
    parser.add_argument(
        '--config_file', type=str,
        default="configs/imagenet/local/clip_vit_mobiles2_datacompdr_imagenet.yaml"
    )
    parser.add_argument('--class_id', type=int, default=-1)
    parser.add_argument('--split', type=str, default="train")
    return parser.parse_args()


def collect_latent_features(model_name,
                            dataset_name,
                            data_path,
                            split,
                            class_id,
                            batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = get_dataset(dataset_name)(data_path=data_path, normalize_data=True, split=split)
    print(len(dataset))
    if class_id != -1:
        samples_of_class = [i for i in range(len(dataset)) if dataset.get_target(i) == class_id]
    else:
        samples_of_class = [i for i in range(len(dataset))]

    print("Dataset loaded", len(samples_of_class))

    print("Loading CLIP model")
    model = get_fn_model_loader(model_name)(return_clip_model=True)
    model = model.to(device).eval()

    dataset.transform = model.preprocess
    dataset_subset = torch.utils.data.Subset(dataset, samples_of_class)
    print("Model loaded")

    dataloader = DataLoader(dataset_subset, batch_size=batch_size, shuffle=False, num_workers=16)

    outs = []
    for i, (x, y) in enumerate(tqdm(dataloader)):
        x = x.to(device)
        outs.append(model.encode_image(x).detach().cpu())

    path = f"results/global_features/{dataset_name}/{model_name}"
    os.makedirs(path, exist_ok=True)

    save_file({"embeddings": torch.cat(outs, dim=0)},
              f"{path}/features_{split}_{class_id}.safetensors")


if __name__ == "__main__":
    args = get_args()

    config = load_config(args.config_file)

    model_name = config['model_name']
    dataset_name = config['dataset_name']
    class_id = args.class_id
    batch_size = config['batch_size']
    data_path = config.get('data_path', None)
    split = args.split

    collect_latent_features(model_name, dataset_name, data_path, split, class_id, batch_size)
