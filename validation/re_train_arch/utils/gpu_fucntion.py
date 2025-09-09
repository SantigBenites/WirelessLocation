
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
import pynvml
from pytorch_lightning.loggers import Logger
import wandb





class LightningWrapper(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, train_data, val_data, learning_rate, weight_decay, optimizer_name='adam'):
        super().__init__()
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.mse_loss = torch.nn.MSELoss(reduction="mean")
        self.l1_loss  = torch.nn.L1Loss(reduction="mean")  # absolute loss (MAE)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        mse = self.mse_loss(y_hat, y)
        mae = self.l1_loss(y_hat, y)

        # optimize w.r.t. MSE
        self.log('train_mse', mse, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True, prog_bar=False)
        return mse

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        mse = self.mse_loss(y_hat, y)
        mae = self.l1_loss(y_hat, y)

        self.log('val_mse', mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=False)
        return {"val_mse": mse, "val_mae": mae}

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

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_mse",  # monitor the MSE you log at validation
                "frequency": 1,
            },
        }
