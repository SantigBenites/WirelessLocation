import os, gc, time, warnings
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger, Logger

# Optional: model + wrapper from our previous implementation
try:
    from model_generation import GeneratedModel
except Exception:
    GeneratedModel = None  # If user imports their own model elsewhere


# ---- Speed defaults ----
def enable_speed_mode():
    # Enable TF32 on Ampere+ for matmul/GEMM (huge win for MLPs)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    # Prefer higher precision matmul kernels for better perf/accuracy balance
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


class DummyLogger(Logger):
    def log_metrics(self, metrics, step): pass
    def log_hyperparams(self, params): pass
    @property
    def experiment(self): return None
    @property
    def name(self): return "dummy"
    @property
    def version(self): return "0"


def _make_loss(name: str, smoothing: float = 0.0):
    name = (name or "mse").lower()
    if name == "mae":   return torch.nn.L1Loss()
    if name == "huber": return torch.nn.SmoothL1Loss(beta=1.0)
    return torch.nn.MSELoss()


class LightningWrapper(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate,
        weight_decay,
        loss_name="mse",
        label_smoothing=0.0,
        opt_name="adam",
        sched_cfg=None,
        mc_dropout=False,
        use_torch_compile=False,
        grad_clip_val: float = 0.0,   # <-- NEW: pass through for fused/clip decision
    ):
        super().__init__()
        self.model = model
        self.loss_fn = _make_loss(loss_name, label_smoothing)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.opt_name = (opt_name or "adam").lower()
        self.sched_cfg = sched_cfg or {"name": "plateau", "plateau_patience": 5}
        self.mc_dropout = bool(mc_dropout)
        self.use_torch_compile = bool(use_torch_compile)
        self.grad_clip_val = float(grad_clip_val)

        # Compile after first forward to avoid pickling issues in Lightning
        self._compiled = False

        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        # Compile at first usage to minimize overhead
        if self.use_torch_compile and not self._compiled:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                self._compiled = True
            except Exception:
                self._compiled = True  # Don't retry
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.mc_dropout and hasattr(self.model, "set_mc_dropout"):
            self.model.set_mc_dropout(True)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss}

    def configure_optimizers(self):
        params = self.parameters()

        # 🔧 FIX: disable fused optimizers when gradient clipping is on
        fused_ok = torch.cuda.is_available() and self.grad_clip_val == 0.0
        try_fused = dict(fused=True) if fused_ok else {}

        # Optimizer
        opt = self.opt_name
        if opt == "adamw":
            try:
                optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay, **try_fused)
            except TypeError:
                optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif opt == "sgd":
            optimizer = torch.optim.SGD(params, lr=self.learning_rate, momentum=0.9, nesterov=True, weight_decay=self.weight_decay)
        elif opt == "rmsprop":
            optimizer = torch.optim.RMSprop(params, lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        elif opt == "nadam":
            try:
                optimizer = torch.optim.NAdam(params, lr=self.learning_rate, weight_decay=self.weight_decay, **try_fused)
            except TypeError:
                optimizer = torch.optim.NAdam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            try:
                optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay, **try_fused)
            except TypeError:
                optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        # Scheduler
        sched_name = (self.sched_cfg.get("name") or "plateau").lower()
        if sched_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(self.sched_cfg.get("scheduler_step_size", 10)),
                gamma=float(self.sched_cfg.get("scheduler_gamma", 0.5)),
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        elif sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.trainer.max_epochs) if self.trainer else 10,
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        elif sched_name == "onecycle":
            try:
                total_steps = self.trainer.estimated_stepping_batches
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.learning_rate,
                    total_steps=total_steps,
                )
                return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
            except Exception:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=int(self.trainer.max_epochs) if self.trainer else 10
                )
                return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        elif sched_name == "none":
            return optimizer
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=int(self.sched_cfg.get("plateau_patience", 5)),
                mode="min",
                factor=0.5,
                cooldown=0,
                min_lr=1e-6,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                },
            }


def train_model(config_dict, train_data_ref, val_data_ref, model_index, config, use_wandb=False):

    enable_speed_mode()

    # env noise off
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB_DISABLE_CODE", "true")
    os.environ.setdefault("WANDB_CONSOLE", "off")
    os.environ.setdefault("WANDB_START_METHOD", "thread")

    X_train, y_train = train_data_ref
    X_val, y_val = val_data_ref

    # DataLoader perf knobs
    num_workers = getattr(config, "num_cpu", 4)
    num_workers = max(0, int(num_workers))
    pin_mem = torch.cuda.is_available()
    persistent = num_workers > 0
    prefetch = min(4, max(2, num_workers // 4)) if num_workers > 0 else 2

    arch = config_dict["config"]

    # Build model
    if GeneratedModel is None:
        raise RuntimeError("GeneratedModel not found. Import your own model and adapt train_model accordingly.")
    model = GeneratedModel(input_size=X_train.shape[1], output_size=y_val.shape[1], architecture_config=arch)

    # Training knobs
    lr = float(arch.get("learning_rate", getattr(config, "default_learning_rate", 1e-3)))
    wd = float(arch.get("weight_decay", getattr(config, "default_weight_decay", 0.0)))
    opt_name = arch.get("optimizer", "adam")
    loss_name = arch.get("loss", "mse")
    label_smoothing = float(arch.get("label_smoothing", 0.0))
    mc_dropout = bool(arch.get("uncertainty_estimation", False))
    sched_name = arch.get("lr_scheduler", "plateau")
    sched_cfg = dict(arch); sched_cfg["name"] = sched_name
    # 🔧 Default compile OFF to avoid Triton CUDA headers requirement on cluster nodes
    use_torch_compile = bool(arch.get("compile_model", True))
    grad_clip_val = float(arch.get("grad_clip_val", 0.0))

    lightning_model = LightningWrapper(
        model=model,
        learning_rate=lr,
        weight_decay=wd,
        loss_name=loss_name,
        label_smoothing=label_smoothing,
        opt_name=opt_name,
        sched_cfg=sched_cfg,
        mc_dropout=mc_dropout,
        use_torch_compile=use_torch_compile,
        grad_clip_val=grad_clip_val,   # <-- pass through
    )

    # Logging
    if use_wandb:
        logger = WandbLogger(
            project=getattr(config, "experiment_name", "wifi-rssi-gradient-search"),
            name=config_dict.get("name", f"model_{model_index}"),
            group=getattr(config, "group_name", "MLP"),
            log_model=False,  # avoid artifact overhead
        )
    else:
        logger = DummyLogger()

    # Batch size with adaptive OOM backoff
    base_batch = int(arch.get("batch_size", getattr(config, "default_batch_size", 2048)))
    attempt = 0
    while True:
        batch_size = max(base_batch // (2 ** attempt), 16)
        try:
            train_loader = DataLoader(
                TensorDataset(X_train, y_train),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_mem,
                persistent_workers=persistent,
                prefetch_factor=prefetch if persistent else None,
                drop_last=False,
            )
            val_loader = DataLoader(
                TensorDataset(X_val, y_val),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_mem,
                persistent_workers=persistent,
                prefetch_factor=prefetch if persistent else None,
                drop_last=False,
            )
            break
        except RuntimeError:
            attempt += 1
            if batch_size <= 16:
                raise

    # Precision & validation frequency
    precision = getattr(config, "precision", None)
    if precision is None:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            precision = "bf16-mixed"
        elif torch.cuda.is_available():
            precision = "16-mixed"
        else:
            precision = "32-true"

    check_val_every_n_epoch = int(getattr(config, "check_val_every_n_epoch", 1))
    early_stop_patience = int(getattr(config, "early_stopping_patience", 5))
    early_stop_min_delta = float(getattr(config, "early_stopping_min_delta", 0.0))

    # Trainer
    trainer = Trainer(
        max_epochs=getattr(config, "epochs", 40),
        logger=logger,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[EarlyStopping(monitor="val_loss", patience=early_stop_patience, min_delta=early_stop_min_delta)],
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=1,
        log_every_n_steps=200,
        enable_checkpointing=False,
        gradient_clip_val=grad_clip_val,        # <-- single source of truth
        check_val_every_n_epoch=check_val_every_n_epoch,
        precision=precision,
    )

    trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    val_loss = trainer.callback_metrics.get("val_loss", None)
    result = {
        **config_dict,
        "val_loss": float(val_loss.item()) if val_loss is not None else float("inf"),
        "status": "success",
        "batch_size_used": batch_size,
        "precision": precision,
    }
    return result
