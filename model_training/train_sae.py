import logging
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from datasets import get_dataset
from model_training import get_training_method
from model_training.sae import TopKSAE
from models import get_fn_model_loader


def get_parser():
    parser = ArgumentParser(
        description="Train models.",
    )
    parser.add_argument(
        "--config_file",
        default="configs/train_sae/imagenet/local/clip_vit_b32_datacomp_xl_s13b_b90k_imagenet_topk-spatial-64-30000_lr_0.0001.yaml",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.info(exc)
    config["config_file"] = config_file
    config_name = os.path.basename(config_file)[:-5]
    num_gpu = config.get("num_gpu", 1)
    start_model_correction(config, config_name, num_gpu)


def start_model_correction(config, config_name, num_gpu):
    """Starts model correction for given config file.

    Args:
        config (dict): Dictionary with config parameters for train_sae.
        config_name (str): Name of given config
        num_gpu (int): Number of GPUs
    """

    # setting seeds
    torch.random.manual_seed(config.get("random_seed", 0))
    np.random.seed(config.get("random_seed", 0))
    random.seed(config.get("random_seed", 0))
    seed_everything(config.get("random_seed", 0))

    # Initialize WandB
    if config["wandb_project_name"]:
        wandb_api_key = config["wandb_api_key"]
        wandb_project_name = config["wandb_project_name"]
        os.environ["WANDB_API_KEY"] = wandb_api_key
        wandb_id = f"{config_name}" if config.get("unique_wandb_ids", False) else None
        logger_ = WandbLogger(
            project=wandb_project_name,
            name=f"{config_name}",
            id=wandb_id,
            config=config,
        )
    else:
        logger_ = None

    # Load Dataset
    dataset_name = config["dataset_name"]
    data_path = config["data_path"]
    ds_kwargs = config.get("dataset_kwargs", {})
    dataset = get_dataset(dataset_name)
    dataset_train = dataset(
        data_path=data_path, normalize_data=True, split="train", train_with_augmentation=True, **ds_kwargs
    )
    dataset_val = dataset(
        data_path=data_path, normalize_data=True, split="test", **ds_kwargs
    )

    model = get_fn_model_loader(config["model_name"])()
    preprocess = model.preprocess

    dataset_train.transform = preprocess
    dataset_val.transform = preprocess

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    sae = TopKSAE.from_config(config=config, model=model)
    model.add_sae(sae)

    model = get_training_method(config.get("method", "Vanilla"))(model, sae, config)

    # Define Optimizer and Loss function
    optimizer_name = config.get("optimizer", "adamw")
    lr = config.get("lr", 1e-2)

    model.set_optimizer(
        optimizer_name,
        model.sae.parameters(),
        lr,
        weight_decay=config.get("weight_decay", 0),
        norm_weight_decay=config.get("norm_weight_decay", None),
    )
    model.set_loss(
        config.get("loss_fn", "mse"),
        weights=getattr(dataset_train, "weights", None),
    )

    print(f"Number of samples: {len(dataset_train)} (train) / {len(dataset_val)} (val)")

    batch_size = config["batch_size"]
    print(f"BATCH SIZE: {batch_size}")

    dl_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=12
    )
    dl_val = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, num_workers=12
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Checkpoints
    # Callback to save the model periodically if 'every_n_epochs' is provided otherwise stores only the last model
    periodic_save_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{config_name}",
        filename="model-ckpt-epoch_{epoch:02d}-{"
                 + config.get("monitor_val", "valid_loss")
                 + ":.2f}",
        auto_insert_metric_name=True,
        every_n_epochs=config.get(
            "every_n_epochs", 0
        ),  # if not provided only store the last model
        save_top_k=-1,  # dont overwrite models but keep all of them
        save_weights_only=True,
        save_last=True,  # store the final model
    )

    # Callback to save the best model
    callbacks = [lr_monitor, periodic_save_callback]
    if config.get("store_best_model", True):
        best_model_callback = ModelCheckpoint(
            dirpath=f"checkpoints/{config_name}",
            filename="best_model-{epoch:02d}-{"
                     + config.get("monitor_val", "valid_loss")
                     + ":.2f}",
            auto_insert_metric_name=True,
            monitor=config.get("monitor_val", "valid_loss"),
            mode=config.get("monitor_mode", "max"),
            save_top_k=1,  # overwrite previous best model
            save_weights_only=True,
            every_n_epochs=1,  # check for best model at every epoch
        )
        callbacks.append(best_model_callback)

    trainer = Trainer(
        callbacks=callbacks,
        devices=num_gpu,
        max_epochs=config.get("num_epochs", 20),
        accelerator="gpu",
        precision="16-mixed",
        gradient_clip_val=1.0,
        logger=logger_,
        limit_train_batches=0.1 if "imagenet" in dataset_name else 1.0,  # debug (only 10% of train data)
    )

    trainer.validate(model, dl_val)
    trainer.fit(model, dl_train, dl_val)


if __name__ == "__main__":
    main()
