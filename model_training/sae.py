from abc import abstractmethod

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class TopKSAE(nn.Module):
    def __init__(self,
                 input_dim=768,
                 hidden_dim=30000,
                 k=64,
                 layer_idx=-1,
                 token="cls",
                 *args,
                 **kwargs):
        """
        Top-k Sparse Autoencoder (SAE) with Weight Normalization.

        Args:
        - input_dim (int): Input feature dimension.
        - hidden_dim (int): Latent space dimension.
        - k (int): Number of neurons allowed to be active.
        - layer_idx (int): Index of the layer where SAE is applied. Negative index counts from the end.
        - sae_token (str): Token type for SAE, e.g., "cls" for classification token or 'spatial' for spatial tokens.
        """
        super(TopKSAE, self).__init__()
        print(f"TopKSAE: input_dim={input_dim}, hidden_dim={hidden_dim}, k={k}, layer_idx={layer_idx}, token={token}")
        self.k = k
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_idx = layer_idx
        self.token_type = token
        self.hook_pre_sae = nn.Identity()
        self.hook_hidden_post = nn.Identity()
        self.hook_post_sae = nn.Identity()
        self.add_residual = True  # Whether to add residual connections in SAE
        self.encoder = nn.Sequential(
            weight_norm(nn.Linear(input_dim, hidden_dim)),
        )
        self.decoder = weight_norm(nn.Linear(hidden_dim, input_dim, bias=False), dim=1)  # Apply weight norm
        self._fix_weight_norm()

    def _fix_weight_norm(self):
        """ Set the magnitude parameter (g) to 1 and remove it from train_sae. """
        for layer in [self.decoder]:  # Access weight-normed layers
            layer.weight_g.data.fill_(1.0)  # Set scale to 1
            layer.weight_g.requires_grad = False  # Freeze scale parameter

    def top_k_masking(self, z):
        """ top-k activation masking. """
        values, indices = torch.topk(z, self.k, dim=1)  # Get top-k activations
        mask = torch.zeros_like(z).scatter(1, indices, 1.0)  # Create binary mask
        return z * mask

    def forward(self, x):
        x = self.hook_pre_sae(x)
        if len(x.shape) == 2:
            z = self.encoder(x)
            z_bar = self.top_k_masking(z)
            z_bar = self.hook_hidden_post(z_bar)
            x_recon = self.decoder(z_bar)
            x_recon = self.hook_post_sae(x_recon)
        else:
            batch_size = x.shape[0]
            token_dim = x.shape[1]
            z = self.encoder(x.reshape(batch_size * token_dim, -1))
            z_bar = self.top_k_masking(z)
            z_bar = self.hook_hidden_post(z_bar.reshape(batch_size, token_dim, -1))
            x_recon = self.decoder(z_bar.reshape(batch_size * token_dim, -1))
            x_recon = self.hook_post_sae(x_recon.reshape(batch_size, token_dim, -1))
        if self.add_residual:
            x_recon = x_recon + (x - x_recon).detach()
        return x_recon, z, z_bar

    @classmethod
    def from_config(cls, config: dict, model: nn.Module):
        """Instantiate TopKSAE from a config dictionary."""
        sae_keys = ["sae_input_dim", "sae_hidden_dim", "sae_k", "sae_layer_idx", "sae_token", "sae_ckpt_path"]
        sae_kwargs = {key.replace('sae_', ''): config[key] for key in sae_keys if key in config}
        sae_kwargs['input_dim'] = model.hidden_dim
        sae = cls(**sae_kwargs)
        if "ckpt_path" in sae_kwargs:
            try:
                print(f"Loading SAE weights from {sae_kwargs['ckpt_path']}")
                sae.load_state_dict(torch.load(sae_kwargs["ckpt_path"])["state_dict"])
            except:
                print("Error loading SAE checkpoint, using default weights.")
        return sae
