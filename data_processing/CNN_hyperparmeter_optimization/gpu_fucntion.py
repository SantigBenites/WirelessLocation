
import os, torch, warnings, gc, time
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, Logger
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl
from model_generation import GeneratedModel
import wandb

warnings.filterwarnings("ignore", category=UserWarning)

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
    def __init__(self, model: torch.nn.Module, train_data, val_data, learning_rate, weight_decay):
        super().__init__()
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.loss_fn = torch.nn.MSELoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }


def train_model(config_dict, train_data_ref, val_data_ref, model_index, config, use_wandb=False):
    # Env for wandb/no-tokenizer parallelism noise
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_DISABLE_CODE"] = "true"
    os.environ["WANDB_CONSOLE"] = "off"
    os.environ["WANDB_START_METHOD"] = "thread"

    torch.cuda.empty_cache()
    torch.cuda.amp.autocast()
    print(f"🚀 Ray assigned CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    X_train, y_train = train_data_ref
    X_val, y_val = val_data_ref

    try:
        for attempt in range(100):
            try:
                model = GeneratedModel(
                    input_shape=(X_train.shape[1], 32, 32),  # channels, height, width
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

                if use_wandb:
                    logger = WandbLogger(
                        project="wifi-rssi-gradient-search",
                        name=config_dict["name"],
                        group=config.group_name,
                        log_model=True
                    )
                    # Watch the underlying nn.Module
                    logger.watch(lightning_model.model, log="all", log_freq=100)
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

                # Backoff batch-size on OOM to push more concurrent jobs
                batch_size = max(config.default_batch_size // (2 ** attempt), 8)
                train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, num_workers=config.num_cpu)
                val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, num_workers=config.num_cpu)

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

