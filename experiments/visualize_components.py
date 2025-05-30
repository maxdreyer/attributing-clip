from argparse import ArgumentParser, Namespace

import torchvision

from safetensors import safe_open

from torchvision.transforms import InterpolationMode

from torchvision.utils import make_grid

from datasets import get_dataset
from models import get_fn_model_loader

from utils.helper import load_config

import torch
import numpy as np
import matplotlib.pyplot as plt
import zennit.image as zimage


def get_parser():
    parser = ArgumentParser(
        description='Compute and display the most activating samples of components.', )
    parser.add_argument('--config_file',
                        default="configs/train_sae/imagenet/local/clip_vit_h14_dfn5b_imagenet_topk-spatial-64-30000_layer--1_lr_0.0002.yaml")
    parser.add_argument('--components',
                        default="24511, 5123, 16279, 17023, 22972, 20760")  #
    parser.add_argument('--split', default="train")  # split of the dataset used to compute the activations
    parser.add_argument('--class_id', default=-1, type=int)  # -1 for all classes
    parser.add_argument('--num_refimgs', default=6, type=int)  # number of reference images to show
    parser.add_argument('--pooling_mode', default="mean", type=str)  # pooling mode for the activations (mean, max)
    args = parser.parse_args()
    config_dict = load_config(args.config_file)
    return Namespace(**config_dict, **vars(args))


device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_parser()
model_name = args.model_name
dataset_name = args.dataset_name
class_id_filter = args.class_id
dataset_split = args.split
n_refimgs = args.num_refimgs
batch_size = args.batch_size
pooling_mode = args.pooling_mode
components = torch.tensor([int(i) for i in args.components.split(",")])

layer_name = 'sae.hook_hidden_post'


dataset = get_dataset(dataset_name)(data_path=args.data_path, normalize_data=False, split=dataset_split)

model = get_fn_model_loader(model_name)().to(device)
model.eval()

dataset.transform = torchvision.transforms.Compose(model.preprocess.transforms[:-1])


path = f"results/global_features_minimal/{dataset_name}/{model_name}"
with safe_open(f"{path}/{pooling_mode}_activations_{class_id_filter}_{dataset_split}.safetensors", framework='pt', device='cpu') as f:
    activations = f.get_tensor(layer_name)

with safe_open(f"{path}/{pooling_mode}_sample_ids_{class_id_filter}_{dataset_split}.safetensors", framework='pt', device='cpu') as f:
    sample_ids = f.get_tensor(layer_name)

topk = sample_ids[:n_refimgs, components]
topk_activating = activations[:n_refimgs, components]

resize_inp = torchvision.transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC)


fig, axs = plt.subplots(1, len(components), dpi=300, figsize=(len(components) * 2.5 / 1.5, 5.5 / 1.3))
for i, inds in enumerate(components):
    print(f"{i}: {topk_activating[:, i]}")

    ref_imgs = [dataset[x][0] for x in topk[:, i].cpu().numpy()]

    resize = torchvision.transforms.Resize((120, 120))

    NUM = n_refimgs
    grid = make_grid(
        [resize(k) for k in ref_imgs],
        nrow=2,
        padding=0)
    grid = np.array(zimage.imgify(grid.detach().cpu()))
    axs[i].imshow(grid)
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].set_ylabel("concept examples") if i == 0 else None
    axs[i].set_title(f"#{inds.item()}")
    axs[i].set_xlabel(f"mean activation:\n{topk_activating[:, i].mean().item():.2f}")
    attentions = []
plt.tight_layout()

plt.show()
