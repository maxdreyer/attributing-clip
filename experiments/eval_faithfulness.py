import argparse
import copy

import matplotlib.pyplot as plt

import torch
import torchvision

import numpy as np
from torch.utils.data import DataLoader

from datasets import get_dataset
from model_training.sae import TopKSAE

from models import get_fn_model_loader
from utils.helper import load_config
from tqdm import tqdm
import random

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--config_file', type=str,
        default="configs/train_sae/imagenet/after_train/clip_vit_b32_datacomp_xl_s13b_b90k_imagenet_topk-cls-64-30000_layer--1_lr_0.0002.yaml"
    )
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_steps', type=int, default=5)
    parser.add_argument('--num_classes', type=int, default=2)
    return parser.parse_args()


args = get_args()
config_dict = load_config(args.config_file)

model_name = config_dict["model_name"]
dataset_name = config_dict["dataset_name"]

SPLIT = "test"

device = "cuda" if torch.cuda.is_available() else "cpu"

layer_name = 'sae.hook_hidden_post'

dataset = get_dataset(dataset_name)(data_path=config_dict['data_path'], normalize_data=True, split=SPLIT)
batch_size = args.batch_size
num_steps = args.num_steps
num_classes = args.num_classes

clip_model = get_fn_model_loader(model_name)(return_clip_model=True).to(device)
vis_model = clip_model.visual

sae = TopKSAE.from_config(config=config_dict, model=vis_model)
torch.nn.utils.remove_weight_norm(sae.decoder)
sae.add_residual = False
vis_model.add_sae(sae)
vis_model.eval().to(device)
clip_model.eval().to(device)

dataset.transform = torchvision.transforms.Compose(vis_model.preprocess.transforms)

SETS = ["Act$\\times$Grad (ours)", "Act$\\times$LogitLens", "Integrated Grad", "LogitLens", "Energy", "random"]

all_deletion_local = {k: [] for k in SETS}
all_deletion_global = {k: [] for k in SETS}
all_deletion_random = {k: [] for k in SETS}
all_insertion_local = {k: [] for k in SETS}

W_copy = copy.deepcopy(vis_model.sae.decoder.weight.data)
W = vis_model.sae.decoder.weight.to(device).permute((1, 0))  # (num_components, feature_dim)

sae_activations = [torch.tensor([])]
sae_residual = torch.tensor(0).to(device)
delete_components = []


def hook_activations(module, input, output):
    output = output.detach().requires_grad_()
    sae_activations[0] = output
    if len(delete_components):
        batch_indices = torch.arange(len(delete_components)).unsqueeze(1).expand(-1, delete_components.shape[1])
        output[batch_indices, delete_components] = 0
    return output


hook = sae.hook_hidden_post.register_forward_hook(hook_activations)

sample_ids = np.arange(len(dataset))

for CLASS_ID in tqdm(np.arange(0, 5, 1)):

    object = dataset.class_names[CLASS_ID]
    test_prompt = [f"{object}", ]
    test_text = clip_model.tokenizer(test_prompt)
    text_out = clip_model.encode_text(test_text.to(device))
    text_out /= text_out.norm(dim=-1, keepdim=True)

    sample_ids_class = sample_ids[dataset.targets == CLASS_ID]
    class_dataset = torch.utils.data.Subset(dataset, sample_ids_class)
    class_dl = DataLoader(class_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    sae_activations_class = []
    grads_class = []
    for (x, y) in class_dl:
        with torch.enable_grad():
            vis_out = vis_model(x.to(device).detach())
            image_embedding = vis_out / vis_out.norm(dim=-1, keepdim=True)
            output_score = (image_embedding @ text_out.T.detach()).mean()
            grad = torch.autograd.grad(
                output_score,
                sae_activations[0],
                retain_graph=True)[0]
            sae_activations_class.append(sae_activations[0].detach().cpu())
            grads_class.append((sae_activations[0] * grad).detach().cpu())

    sae_activations_class = torch.cat(sae_activations_class, dim=0)
    grad = torch.cat(grads_class, dim=0)
    integrated_grad = grad

    logitlens = (torch.nn.functional.normalize((vis_model.ln_post(W[:, None])[:, 0]) @ vis_model.proj, dim=-1)
                 @ text_out.T.detach()).mean(-1)
    logitlens = torch.stack([logitlens for _ in range(len(sae_activations_class))], dim=0).detach().cpu()
    act_logitlens = sae_activations_class * logitlens
    energy = sae_activations_class * W.norm(dim=1)[None].detach().cpu()

    random = torch.randn_like(sae_activations_class) * (sae_activations_class > 0)

    to_test = {
        "LogitLens": logitlens,
        "Act$\\times$LogitLens": act_logitlens,
        "Act$\\times$Grad (ours)": grad,
        "Integrated Grad": integrated_grad,
        "Energy": energy,
        "random": random,
    }

    del_scores = {}

    with torch.no_grad():
        for name, vals in (to_test.items()):
            ordering_ = vals.topk(num_steps, dim=-1).indices
            scores = []
            output_scores = []
            for (x, y) in class_dl:
                vis_out = vis_model(x.to(device).detach())
                image_embedding = vis_out / vis_out.norm(dim=-1, keepdim=True)
                output_score = (image_embedding @ text_out.T.detach()).squeeze()
                output_scores.append(output_score.detach().cpu())
            scores.append(torch.cat(output_scores, dim=0))
            for i in range(num_steps):
                perturbed = []
                for j, (x, y) in enumerate(class_dl):
                    delete_components = ordering_[np.arange(len(class_dataset))[j * batch_size:(j + 1) * batch_size], :i + 1]
                    vis_out = vis_model(x.to(device).detach())
                    image_embedding = vis_out / vis_out.norm(dim=-1, keepdim=True)
                    output_score = (image_embedding @ text_out.T.detach()).squeeze()
                    perturbed.append(output_score.detach().cpu())
                scores.append(torch.cat(perturbed, dim=0))
            all_deletion_local[name].append(torch.stack(scores) * 100)
            delete_components = []

all_deletion_local = {k: torch.cat(v, dim=1) for k, v in all_deletion_local.items()}

plt.figure(dpi=300, figsize=(3.9, 2.7))
for name, vals in all_deletion_local.items():
    y_err = np.std([vals.flatten()[i::9].mean() for i in range(9)]) / np.sqrt(9)
    y_err = max(y_err, 0.01)
    plt.plot(vals.mean(1).cpu().flatten(),
             '-o', ms=5,
             label=f"{name}:\n{vals.mean():.2f}$\\pm${y_err:.2f}", )
plt.ylabel("output $y$ (%)")
plt.title("deletion (local)")
plt.xlabel("# of latents deleted", labelpad=0.1)

plt.legend(fontsize="smaller", bbox_to_anchor=(1.05, 0.48), loc='center left', borderaxespad=0.)
plt.tight_layout()
plt.show()
