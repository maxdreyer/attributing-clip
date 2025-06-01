from argparse import ArgumentParser, Namespace

import torchvision

from safetensors import safe_open

from torchvision.transforms import InterpolationMode

from torchvision.utils import make_grid
from transformers.image_utils import load_image

from datasets import get_dataset
from model_training.sae import TopKSAE
from models import get_fn_model_loader

from utils.helper import load_config

import torch
import numpy as np
import matplotlib.pyplot as plt
import zennit.image as zimage

def get_parser():
    parser = ArgumentParser(description='')
    parser.add_argument(
        '--config_file', type=str,
        default="configs/train_sae/imagenet/after_train/clip_vit_l14_datacomp_xl_s13b_b90k_imagenet_topk-cls-64-30000_layer--1_lr_0.0002.yaml")
    parser.add_argument('--img_url', type=str, default="your_img_url.jpg")  # custom prompt for probing
    parser.add_argument('--text_prompt', type=str, default="an image of a brown feathered bird")  # custom prompt for probing
    parser.add_argument('--split', type=str, default="train")  # split of the dataset used for concept examples
    parser.add_argument('--pooling_mode', type=str, default="mean")  # pooling mode for the activations (mean, max)
    args = parser.parse_args()
    config_dict = load_config(args.config_file)
    return Namespace(**config_dict, **vars(args))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_parser()
model_name = args.model_name
dataset_name = args.dataset_name
dataset_split = args.split
batch_size = args.batch_size
pooling_mode = args.pooling_mode
layer_name = 'sae.hook_hidden_post'

dataset = get_dataset(dataset_name)(data_path=args.data_path, normalize_data=False, split=dataset_split)

clip_model = get_fn_model_loader(model_name)(return_clip_model=True).to(device)
vis_model = clip_model.visual

sae = TopKSAE.from_config(config=args.__dict__, model=vis_model)
vis_model.add_sae(sae)
vis_model.eval().to(device)
clip_model.eval().to(device)

dataset.transform = torchvision.transforms.Compose(vis_model.preprocess.transforms)

sae_activations = []

def hook_activations(module, input, output):
    output.requires_grad_().retain_grad()
    sae_activations.append(output)
    return output


sae.hook_hidden_post.register_forward_hook(hook_activations)

""" # Uncomment to load a specific image from the dataset
sample = dataset[sample_id][0][None].to(device)
"""
image = load_image(args.img_url)
sample = dataset.transform(image).unsqueeze(0).to(device)

with torch.autograd.enable_grad():
    test_text = clip_model.tokenizer(args.text_prompt)
    text_out = clip_model.encode_text(test_text.to(device))
    text_out = text_out / text_out.norm(dim=-1, keepdim=True)
    vis_out = vis_model(sample)
    vis_out = vis_out / vis_out.norm(dim=-1, keepdim=True)
    yc_hat = (vis_out @ text_out.T).squeeze()  # scalar similarity
    print(f"yc_hat: {yc_hat.item()}")
    grad = torch.autograd.grad(
        outputs=yc_hat,
        inputs=sae_activations[-1],
        grad_outputs=torch.ones_like(yc_hat),
        retain_graph=True,
    )[0]

attribution = (sae_activations[-1] * grad)[0]
components = attribution.topk(5).indices.detach().cpu().numpy()
path = f"results/global_features_minimal/{dataset_name}/{model_name}"
with safe_open(f"{path}/{pooling_mode}_activations_-1_{dataset_split}.safetensors", framework='pt', device='cpu') as f:
    activations = f.get_tensor(layer_name)

with safe_open(f"{path}/{pooling_mode}_sample_ids_-1_{dataset_split}.safetensors", framework='pt', device='cpu') as f:
    sample_ids = f.get_tensor(layer_name)

n_refimgs = 8
topk = sample_ids[:n_refimgs, components]
topk_activating = activations[:n_refimgs, components]

resize_inp = torchvision.transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC)
dataset.transform = torchvision.transforms.Compose(vis_model.preprocess.transforms[:-1])

fig, axs = plt.subplots(1, 1, dpi=300, figsize=(4, 4))

""" # Uncomment to visualize the image from the dataset
img = np.array(zimage.imgify(dataset[args.sample_id][0].detach().cpu()))
"""
img = np.array(zimage.imgify(dataset.transform(image).detach().cpu()))

axs.imshow(img)
axs.set_xticks([])
axs.set_yticks([])
axs.set_title("input")
plt.tight_layout()
plt.show()

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
    axs[i].set_xlabel(f"activation:\n{sae_activations[-1][:, inds].mean().item():.2f}\nattribution:\n{attribution[inds].mean().item()*100:.1f}%")
    attentions = []
plt.tight_layout()

plt.show()
