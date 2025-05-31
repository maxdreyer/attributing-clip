import matplotlib.pyplot as plt
import argparse

import torchvision
from numpy.lib._iotools import str2bool
from torchvision.utils import make_grid

import open_clip
import torch
from safetensors import safe_open
import numpy as np
from tqdm import tqdm
from zennit.image import imgify

from datasets import get_dataset

from datasets.imagenet import IN21k_class_labels
from utils.helper import load_config

from utils.interpretability_scores import clarity_score


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--config_file', type=str,
        default="configs/train_sae/imagenet/local/clip_vit_h14_dfn5b_imagenet_topk-spatial-64-30000_layer--1_lr_0.0002.yaml")
    parser.add_argument('--most_aligned', type=str2bool, default=True,)
    parser.add_argument('--custom_prompt', type=str, default="USA") # custom prompt for probing
    parser.add_argument('--pooling_mode', type=str, default="mean")  # pooling mode for the activations (mean, max)
    parser.add_argument('--min_activation', type=float, default=0)  # minimum activation value to consider
    parser.add_argument('--min_clarity', type=float, default=0.40)  # minimum clarity value to consider
    parser.add_argument('--max_clarity', type=float, default=1.0)  # maximum clarity value to consider
    return parser.parse_args()


TOPK = 20 # Number of reference samples to consider

args = get_args()
config_dict = load_config(args.config_file)

model_name = config_dict["model_name"]
dataset_name = config_dict["dataset_name"]
layer_name = "sae.hook_hidden_post"
dataset_split = "train"
pooling_mode = args.pooling_mode

path = f"results/global_features_minimal/{dataset_name}/{model_name}"
with safe_open(f"{path}/{pooling_mode}_activations_-1_{dataset_split}.safetensors", framework='pt', device='cpu') as f:
    activations = f.get_tensor(layer_name)

with safe_open(f"{path}/{pooling_mode}_sample_ids_-1_{dataset_split}.safetensors", framework='pt', device='cpu') as f:
    sample_ids = f.get_tensor(layer_name)

neuron_ids = torch.where(activations[:TOPK].mean(0) > args.min_activation)[0]

sample_ids_sae = sample_ids[:, neuron_ids]
mean_activations = activations.mean(dim=0)[neuron_ids]
path = f"results/global_features/{dataset_name}/clip_vit_mobiles2_datacompdr"
with safe_open(f"{path}/features_{dataset_split}_-1.safetensors", framework='pt', device='cpu') as f:
    clip_embeddings = f.get_tensor("embeddings")
clarity_score = clarity_score(clip_embeddings[sample_ids_sae[:TOPK].permute(1, 0)])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model, _, preprocess = open_clip.create_model_and_transforms("MobileCLIP-S2", pretrained="datacompdr")
clip_model.eval().to(device)
tokenizer = open_clip.get_tokenizer("MobileCLIP-S2")

sem_emb_sae = clip_embeddings[sample_ids_sae[:20, :]].mean(0)

templates = [
    "{}",
]

label_set = IN21k_class_labels if not args.custom_prompt else args.custom_prompt.split(';')
label_features_all = []
batch_size = 48

print(f"Encoding {len(label_set)} labels with {len(templates)} templates...")
for t in tqdm(templates):
    labels = tokenizer([t.format(label.replace("_", " ")) for label in label_set])
    label_features_all.append(
        torch.cat([clip_model.encode_text(labels[i * batch_size:(i + 1) * batch_size].to(device)).detach().cpu() for i in range(len(labels) // batch_size + 1)], dim=0))
label_features = torch.stack(label_features_all, dim=0).mean(0)

templates = [
    "",
]
empty = tokenizer(templates)
empty_features = clip_model.encode_text(empty.to(device)).mean(0)

concept_labels = np.array(label_set)
sem_emb_sae = sem_emb_sae.to(device)
label_features = label_features.to(device)
print(f"Computing semantic alignment scores...")
lab_sae = torch.cat([
    (torch.nn.functional.cosine_similarity(sem_emb_sae[10*i:10*(i+1)][None], label_features[:, None], dim=-1)
     -
     torch.nn.functional.cosine_similarity(sem_emb_sae[10*i:10*(i+1)], empty_features[None, :], dim=-1)
     )
    for i in range(len(sem_emb_sae)//10 + 1)
], dim=-1).detach().cpu()

dataset = get_dataset(dataset_name)(data_path=config_dict["data_path"], normalize_data=False, split="train")

## Filter components
min_activation = args.min_activation
min_clarity = args.min_clarity
max_clarity = args.max_clarity
filt = (mean_activations > min_activation) & (clarity_score > min_clarity) & (clarity_score < max_clarity)
ids_filt = torch.where(filt)[0][lab_sae.amax(0)[filt].topk(20, largest=args.most_aligned).indices]

resize = torchvision.transforms.Resize((120, 120))

print(f"Plotting concept examples...")
components_filtered = ids_filt[0:6]
fig, axs = plt.subplots(1, len(components_filtered), dpi=300, figsize=(len(components_filtered) * 2.5 / 1.5, 5.5 / 1.3))
for i, inds in enumerate(components_filtered):
    NUM = 6
    ref_imgs = [dataset[x][0] for x in sample_ids[:NUM, neuron_ids[inds]].cpu().numpy()]

    grid = make_grid(
        [resize(k) for k in ref_imgs],
        nrow=2,
        padding=0)
    grid = np.array(imgify(grid.detach().cpu()))
    axs[i].imshow(grid)
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].set_xlabel(f"mean act.: {mean_activations[inds]:.2f} \nclarity: {clarity_score[inds]:.2f} \nalignment: {lab_sae.amax(0)[inds]:.2f} \nlabel: {concept_labels[lab_sae.argmax(0)[inds]][:13]}")
    axs[i].set_title(f"#{neuron_ids[inds].item()}")
plt.tight_layout()
plt.show()
