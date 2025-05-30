from typing import Callable
from models.clip_vit import get_clip_vit

MODELS = {
    # CLIP ViT B/32
    "clip_vit_b32_datacomp_m_s128m_b4k": lambda **kwargs: get_clip_vit('ViT-B-32', pretrained='datacomp_m_s128m_b4k', **kwargs),
    "clip_vit_b32_laion400m_e32": lambda **kwargs: get_clip_vit('ViT-B-32', pretrained='laion400m_e32', **kwargs),
    "clip_vit_b32_laion2b_s34b_b79k": lambda **kwargs: get_clip_vit('ViT-B-32', pretrained='laion2b_s34b_b79k', **kwargs),
    "clip_vit_b32_datacomp_xl_s13b_b90k": lambda **kwargs: get_clip_vit('ViT-B-32', pretrained='datacomp_xl_s13b_b90k', **kwargs),

    # CLIP ViT B/16
    "clip_vit_b16_datacomp_xl_s13b_b90k": lambda **kwargs: get_clip_vit('ViT-B-16', pretrained='datacomp_xl_s13b_b90k', **kwargs),

    # CLIP ViT L/14
    "clip_vit_l14_datacomp_xl_s13b_b90k": lambda **kwargs: get_clip_vit('ViT-L-14', pretrained='datacomp_xl_s13b_b90k', **kwargs),
    "clip_vit_l14_336_openai": lambda **kwargs: get_clip_vit('ViT-L-14-336', pretrained='openai', **kwargs),

    # CLIP ViT H/14
    "clip_vit_h14_dfn5b": lambda **kwargs: get_clip_vit('ViT-H-14-quickgelu', pretrained='dfn5b', **kwargs),

    # CLIP ViT Mobile-S2
    "clip_vit_mobiles2_datacompdr": lambda **kwargs: get_clip_vit('MobileCLIP-S2', pretrained='datacompdr', **kwargs),
}


def get_fn_model_loader(model_name: str) -> Callable:
    if model_name in MODELS:
        def model_getter(*args, **kwargs):
            model = MODELS[model_name](*args, **kwargs)
            return model

        fn_model_loader = model_getter
        return fn_model_loader
    else:
        raise KeyError(f"Model {model_name} not available")
