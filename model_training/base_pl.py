import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import OneCycleLR, ChainedScheduler

from utils.training_utils import get_loss, get_optimizer


class LitClassifier(pl.LightningModule):
    def __init__(self, model, config, **kwargs):
        super().__init__()
        self.loss = None
        self.optim = None
        self.model = model
        self.config = config

    def forward(self, x):
        x = self.model(x)
        return x

    def default_step(self, x, y, stage):
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log_dict(
            {f"{stage}_loss": loss,
             f"{stage}_acc": self.get_accuracy(y_hat, y),
             },
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.default_step(x, y, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.default_step(x, y, stage="valid")

    def test_step(self, batch, batch_idx):
        x, y = batch
        self.default_step(x, y, stage="test")

    def set_optimizer(self, optim_name, params, lr, weight_decay=0.0, norm_weight_decay=None):
        self.lr = lr
        self.optim = get_optimizer(optim_name, params, lr, weight_decay, norm_weight_decay, model=self.model)

    def set_loss(self, loss_name, weights=None):
        self.loss = get_loss(loss_name, weights)

    def configure_optimizers(self, milestones=None):
        milestones = self.config.get("milestones", milestones)
        if milestones is None:
            milestones = [5, 8]
        lr_scheduler = self.config.get("lr_scheduler", "custom")
        if lr_scheduler == "MultiStepLR":
            print(f"Using MultiStepLR with milestones: {milestones}")
            sche = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optim,
                                                        milestones=milestones,
                                                        gamma=0.1)
        elif lr_scheduler == "CosineAnnealingLR":
            sche = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim,
                                                                        T_0=2000,
                                                                        T_mult=1,)
            sche1 = OneCycleLR(self.optim, max_lr=1e-3, total_steps=4000)
            sche = ChainedScheduler([sche1, sche, sche1])
        elif lr_scheduler == "custom":
            warmup_epochs = self.config.get('warm_up_steps', 1000)
            total_epochs = 1000 + warmup_epochs
            cosine_epochs = total_epochs - warmup_epochs

            # Define the warm-up + cosine learning rate schedule
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / (warmup_epochs + 1)  # Linear warm-up
                else:
                    return 0.2 + 0.4 * (1 + torch.cos(
                        torch.tensor((epoch - warmup_epochs) / cosine_epochs * 3.1415926535)))  # Cosine decay

            # Create LR scheduler
            sche = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)


        else:
            raise ValueError(f"Unknown lr_scheduler: {lr_scheduler}")
        scheduler = {
            "scheduler": sche,
            "name": "lr_history",
            "interval": "epoch",
        }

        return [self.optim], [scheduler]

    def state_dict(self, **kwargs):
        return self.model.state_dict()

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return self.model.load_state_dict(state_dict, strict, assign)


class Vanilla(LitClassifier):
    def __init__(self, model, sae, config):
        super().__init__(model, config)
        self.sae = sae
        self.pre_activations = None
        self.post_activations = None
        self.hidden_activations = None
        sae.hook_pre_sae.register_forward_hook(self.hook_pre_activations)
        sae.hook_post_sae.register_forward_hook(self.hook_post_activations)
        sae.hook_hidden_post.register_forward_hook(self.hook_hidden_activations)

        self.activation_counts = torch.zeros(sae.hidden_dim)  # Track activations
        self.total_samples = 0  # Number of samples processed

    def hook_pre_activations(self, module, input, output):
        self.pre_activations = output

    def hook_post_activations(self, module, input, output):
        self.post_activations = output

    def hook_hidden_activations(self, module, input, output):
        self.hidden_activations = output

    def on_train_epoch_start(self) -> None:
        self.model.eval()
        for layer in [self.sae.decoder]:  # Access weight-normed layers
            layer.weight_g.requires_grad = False  # Freeze scale parameter

    def default_step(self, x, y, stage):
        self(x)
        act = self.pre_activations
        rec = self.post_activations
        z_masked = self.hidden_activations

        if len(act.shape) == 3:
            dim_0 = act.shape[0]
            dim_1 = act.shape[1]
            act = act.reshape(dim_0 * dim_1, -1)
            rec = rec.reshape(dim_0 * dim_1, -1)
            z_masked = z_masked.reshape(dim_0 * dim_1, -1)

        self.activation_counts += z_masked.sum(dim=0).detach().cpu()
        self.total_samples += z_masked.size(0)

        dead_neurons = self.activation_counts / self.total_samples < (10 / 50000)
        if self.total_samples > 10000:
            self.activation_counts /= 2
            self.total_samples /= 2

        loss = 100 * self.loss(torch.nn.functional.normalize(rec, dim=-1),
                               torch.nn.functional.normalize(act, dim=-1))
        cosine_loss = (1 - torch.nn.functional.cosine_similarity(act, rec).mean()).abs().item()

        self.log_dict(
            {f"{stage}_loss": loss,
             f"{stage}_cosine": cosine_loss,
             f"{stage}_max_act": z_masked.amax(-1).mean().item(),
             f"{stage}_neg_act": z_masked.amin(-1).mean().item(),
             f"{stage}_dead_neurons": dead_neurons.sum().item()
             },
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def state_dict(self, **kwargs):
        return self.sae.state_dict()

    def set_optimizer(self, optim_name, params, lr, weight_decay=0.0, norm_weight_decay=None):
        self.lr = lr
        self.optim = get_optimizer(optim_name, params, lr, weight_decay, norm_weight_decay, model=self.sae)
