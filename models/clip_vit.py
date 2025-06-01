from typing import Optional

import open_clip
import torch.hub

from torch.utils.checkpoint import checkpoint


def get_clip_vit(
        name,
        pretrained: str = "datacomp_xl_s13b_b90k",
        return_clip_model: bool = False,
        *args,
        **kwargs):
    clip_model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)

    # unifying the forward pass into one single branch for inference
    for module in clip_model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()

    vision_model = clip_model.visual
    vision_model.preprocess = clip_model.preprocess = preprocess
    vision_model.hidden_dim = {
        "ViT-B-32": 768,
        "ViT-B-16": 768,
        "ViT-L-14": 1024,
        "ViT-L-14-336": 1024,
        "ViT-H-14-quickgelu": 1280,
    }[name]

    def add_sae(self, sae):
        # convert negative layer index to positive
        sae.layer_idx = sae.layer_idx if sae.layer_idx >= 0 else len(vision_model.transformer.resblocks) + sae.layer_idx
        self.sae = sae
        self.transformer.forward = forward_transformer.__get__(self)
        self.use_sae = True

    setattr(vision_model, "add_sae", add_sae.__get__(vision_model))

    if hasattr(vision_model, "transformer"):
        vision_model.transformer.forward = forward_transformer.__get__(vision_model)
    else:
        print("Warning: No transformer found in the vision model. No SAE will be applied.")

    if return_clip_model:
        clip_model.tokenizer = open_clip.get_tokenizer(name)
        return clip_model

    return vision_model


def forward_transformer(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
    if not self.transformer.batch_first:
        x = x.transpose(0, 1).contiguous()
    for i, r in enumerate(self.transformer.resblocks):
        x = checkpoint(r, x, None, None, attn_mask) if self.transformer.grad_checkpointing and not torch.jit.is_scripting() else r(x, attn_mask=attn_mask)
        if i == self.sae.layer_idx:
            # Apply SAE to each token embedding (e.g., shape [B, T, D])
            if self.use_sae:
                if self.sae.token_type == "cls":
                    sae_out = self.sae(x[:, 0])[0]  # Apply SAE to the CLS token
                    x[:, 0] = sae_out + (x[:, 0] - sae_out).detach()
                elif self.sae.token_type == "spatial":
                    sae_out = self.sae(x[:, 1:])  # Apply SAE to the spatial tokens
                    x[:, 1:] = sae_out + (x[:, 1:] - sae_out).detach()

    if not self.transformer.batch_first:
        x = x.transpose(0, 1)
    return x


if __name__ == "__main__":
    vit = get_clip_vit()
