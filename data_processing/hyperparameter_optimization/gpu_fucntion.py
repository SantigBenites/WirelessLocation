import os, torch, wandb, warnings, ray, gc, time
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl
from model_generation import GeneratedModel

import pynvml

class LightningWrapper(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, train_data, val_data, learning_rate, weight_decay):
        super().__init__()
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.loss_fn = torch.nn.MSELoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }


def get_least_used_gpu():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    min_used = float("inf")
    best_gpu = 0

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if mem.used < min_used:
            min_used = mem.used
            best_gpu = i

    pynvml.nvmlShutdown()
    return str(best_gpu)


@ray.remote(num_gpus=0.50)
def train_model_ray(config_dict, train_data_ref, val_data_ref, model_index, config, use_wandb=False):
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    import time

    selected_gpu = get_least_used_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore")

    X_train, y_train = train_data_ref
    X_val, y_val = val_data_ref

    try:
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        for attempt in range(100):  # Retry up to 10 times
            try:
                model = GeneratedModel(
                    input_size=X_train.shape[1],
                    output_size=y_val.shape[1],
                    architecture_config=config_dict['config']
                )

                lightning_model = LightningWrapper(
                    model=model,
                    train_data=(X_train, y_train),
                    val_data=(X_val, y_val),
                    learning_rate=config.default_learning_rate,
                    weight_decay=config.default_weight_decay
                )

                wandb_logger = None
                if use_wandb:
                    wandb_logger = WandbLogger(
                        project="wifi-rssi-gradient-search",
                        name=config_dict["name"],
                        group=config.group_name,
                        log_model=True
                    )
                    wandb_logger.watch(model, log="all", log_freq=100)

                    wandb.config.update({
                        "model_index": model_index,
                        "group_name": config.group_name,
                        "learning_rate": config.default_learning_rate,
                        "weight_decay": config.default_weight_decay,
                        "batch_size": config.default_batch_size,
                        "epochs": config.epochs,
                        "architecture": config_dict['config']
                    })

                batch_size = max(config.default_batch_size // (2 ** attempt), 8)
                train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, num_workers=2)
                val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, num_workers=2)

                trainer = Trainer(
                    max_epochs=config.epochs,
                    logger=wandb_logger,
                    enable_progress_bar=False,
                    enable_model_summary=True,
                    callbacks=[EarlyStopping(monitor="val_loss", patience=3)],
                    accelerator="gpu",
                    devices=1,
                    log_every_n_steps=50,
                )

                trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

                val_loss = trainer.callback_metrics.get("val_loss", None)

                if use_wandb:
                    wandb.log({"final_val_loss": val_loss.item() if val_loss else float('inf')})
                    wandb.finish()

                torch.cuda.empty_cache()
                gc.collect()

                print("🏁 Finished training")

                return {
                    **config_dict,
                    "val_loss": val_loss.item() if val_loss else float('inf'),
                    "status": "success"
                }

            except Exception as e:
                import traceback
                traceback_str = traceback.format_exc()
                print(f"⚠️ Error in {config_dict.get('name', 'unknown')} (attempt {attempt + 1}/10):\n{traceback_str}")
                import sys
                sys.stdout.flush()
                if use_wandb:
                    try:
                        wandb.finish()
                    except:
                        pass
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(5)

        # If all retries fail
        return {
            **config_dict,
            "val_loss": float("inf"),
            "status": "failed",
            "error": f"Training failed after 100 attempts"
        }

    except Exception as e:
        if use_wandb:
            try:
                wandb.finish()
            except:
                pass

        torch.cuda.empty_cache()
        gc.collect()

        return {
            **config_dict,
            "val_loss": float("inf"),
            "status": "failed",
            "error": str(e)
        }