
import os, torch
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
torch.set_num_threads(20)
torch.set_num_interop_threads(20)


import os, torch, warnings, gc, time, sys
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl
from model_generation import GeneratedModel
import pynvml
from pytorch_lightning.loggers import Logger
import wandb

class DummyLogger(Logger):
    def log_metrics(self, metrics, step):
        pass
    def log_hyperparams(self, params):
        pass
    @property
    def experiment(self):
        return None
    @property
    def name(self):
        return "dummy"
    @property
    def version(self):
        return "0"


class LightningWrapper(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, train_data, val_data, learning_rate, weight_decay, optimizer_name='adam'):
        super().__init__()
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.loss_fn = torch.nn.MSELoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss}

    def configure_optimizers(self):
        opt_name = str(self.optimizer_name).lower()
        if opt_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif opt_name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
        elif opt_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }



import os, torch, warnings, gc
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl
from model_generation import GeneratedModel
import wandb
from gpu_fucntion import LightningWrapper


def train_model(config_dict, train_data_ref, val_data_ref, model_index, config, use_wandb=False):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_DISABLE_CODE"] = "true"
    os.environ["WANDB_CONSOLE"] = "off"
    os.environ["WANDB_START_METHOD"] = "thread"

    torch.cuda.empty_cache()
    print(f"🚀 Ray assigned CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    X_train, y_train = train_data_ref
    X_val, y_val = val_data_ref

    try:
        for attempt in range(100):
            try:
                model = GeneratedModel(
                    input_size=X_train.shape[1],
                    output_size=y_val.shape[1],
                    architecture_config=config_dict['config']
                )

                hcfg = config_dict.get('config', {})
                lr = float(hcfg.get('learning_rate', getattr(config, 'default_learning_rate', 0.001)))
                wd = float(hcfg.get('weight_decay', getattr(config, 'default_weight_decay', 0.0)))
                opt = str(hcfg.get('optimizer', 'adam')).lower()

                lightning_model = LightningWrapper(
                    model=model,
                    train_data=(X_train, y_train),
                    val_data=(X_val, y_val),
                    learning_rate=lr,
                    weight_decay=wd,
                    optimizer_name=opt
                )

                logger = None
                if use_wandb:
                    logger = WandbLogger(
                        project="wifi-rssi-gradient-search",
                        name=config_dict["name"],
                        group=config.group_name,
                        log_model=True
                    )
                    logger.watch(model, log="all", log_freq=100)
                    wandb.config.update({
                        "model_index": model_index,
                        "group_name": config.group_name,
                        "learning_rate": config.default_learning_rate,
                        "weight_decay": config.default_weight_decay,
                        "batch_size": config.default_batch_size,
                        "epochs": config.epochs,
                        "architecture": config_dict['config']
                    })
                else:
                    logger = DummyLogger()

                start_bs = int(config_dict.get('config', {}).get('batch_size', getattr(config, 'default_batch_size', 2048)))
                batch_size = max(start_bs // (2 ** attempt), 8)
                # Use far fewer DataLoader workers for small tensors: 0–2 is enough.
                workers = int(getattr(config, "num_dataloader_workers", 0))
                loader_kwargs = {
                    "batch_size": batch_size,
                    "shuffle": True,
                    "pin_memory": True,
                    "num_workers": workers,
                }
                if workers > 0:
                    loader_kwargs["persistent_workers"] = True
                    loader_kwargs["prefetch_factor"] = 2

                train_loader = DataLoader(
                    TensorDataset(X_train, y_train),
                    **loader_kwargs
                )

                # Validation loader (no shuffle)
                val_loader_kwargs = dict(loader_kwargs)
                val_loader_kwargs["shuffle"] = False
                val_loader = DataLoader(
                    TensorDataset(X_val, y_val),
                    **val_loader_kwargs
                )
                trainer = Trainer(
                    max_epochs=config.epochs,
                    logger=logger,
                    enable_progress_bar=False,
                    enable_model_summary=True,
                    callbacks=[EarlyStopping(monitor="val_loss", patience=8)],
                    accelerator="gpu",
                    devices=1,
                    log_every_n_steps=50,
                    enable_checkpointing=False
                )

                trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

                val_loss = trainer.callback_metrics.get("val_loss", None)
                if use_wandb:
                    wandb.log({"final_val_loss": val_loss.item() if val_loss else float('inf')})

                os.makedirs(config.model_save_dir, exist_ok=True)
                model_save_path = os.path.join(config.model_save_dir, f"{config_dict['name']}.ckpt")
                trainer.save_checkpoint(model_save_path)
                print(f"💾 Saved model to {model_save_path}")

                return {
                    **config_dict,
                    "val_loss": val_loss.item() if val_loss else float('inf'),
                    "status": "success"
                }

            except Exception as e:
                import traceback
                print(f"⚠️ Attempt {attempt+1} failed with error: {e}")
                traceback.print_exc()
                torch.cuda.empty_cache()
                gc.collect()
                import time
                time.sleep(min(5 + attempt * 2, 30))

        return {
            **config_dict,
            "val_loss": float("inf"),
            "status": "failed",
            "error": "Training failed after 100 attempts"
        }

    except Exception as e:
        return {
            **config_dict,
            "val_loss": float("inf"),
            "status": "failed",
            "error": str(e)
        }

    finally:
        if use_wandb:
            try:
                wandb.finish()
            except Exception as e:
                print(f"⚠️ wandb.finish() failed: {e}")
        torch.cuda.empty_cache()
        gc.collect()
