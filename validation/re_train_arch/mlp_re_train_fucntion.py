import os, torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping


from utils.config import TrainingConfig
from MLP_model.model_generation import GeneratedModel
from utils.gpu_fucntion import LightningWrapper  # your LightningModule wrapper
from utils.data_processing import (
    get_feature_list, get_dataset, combine_arrays, shuffle_array, split_combined_data
)

def _load_xy(collections, db_name):
    feats = get_feature_list(db_name)  # same preset used before
    arrays = [get_dataset(c, db_name, feats) for c in collections]
    arr = shuffle_array(combine_arrays(arrays))
    X, y = split_combined_data(arr, feats)
    return X, y, feats

def mlp_retrain_from_pt(
    pt_path: str,
    out_model_name:str,
    train_collections: list,
    val_collections: list = None,
    db_name: str = None,
    load_weights: bool = False,           # set False to re-init but keep same architecture
    max_epochs: int = None,
    batch_size: int = None,
    lr: float = None,
    weight_decay: float = None,
    optimizer: str = "adamw",
    save_dir: str = "retrained_models",
):
    # ----- Load checkpoint (arch + sizes + weights) -----
    ckpt = torch.load(pt_path, map_location="cpu")
    arch_config = ckpt["arch_config"]
    in_size_ckpt = int(ckpt["input_size"])
    out_size_ckpt = int(ckpt["output_size"])
    state = ckpt["state_dict"]

    cfg = TrainingConfig()
    if db_name is not None:
        cfg.db_name = db_name

    # ----- New data -----
    Xtr, ytr, feats = _load_xy(train_collections, cfg.db_name)
    if val_collections is None:
        val_collections = train_collections
    Xva, yva, _ = _load_xy(val_collections, cfg.db_name)

    if Xtr.shape[1] != in_size_ckpt:
        raise RuntimeError(
            f"Feature count mismatch: checkpoint expects {in_size_ckpt} features, "
            f"but preset '{cfg.db_name}' produced {Xtr.shape[1]}. "
            f"Pick the same preset you trained with or adjust your feature list."
        )

    # ----- Rebuild same architecture -----
    model = GeneratedModel(
        input_size=in_size_ckpt,
        output_size=ytr.shape[1],  # usually 2 (x,y)
        architecture_config=arch_config
    )

    # Load weights (or skip to re-init)
    if load_weights:
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            # if label dims changed, load backbone only
            filtered = {k: v for k, v in state.items() if not k.startswith("head.")}
            model.load_state_dict(filtered, strict=False)

    # ----- Lightning training -----
    lr = lr or cfg.default_learning_rate
    weight_decay = weight_decay or cfg.default_weight_decay
    max_epochs = max_epochs or cfg.epochs
    batch_size = batch_size or cfg.default_batch_size

    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.float32)
    Xva = torch.tensor(Xva, dtype=torch.float32)
    yva = torch.tensor(yva, dtype=torch.float32)

    lit = LightningWrapper(
        model=model,
        train_data=(Xtr, ytr),
        val_data=(Xva, yva),
        learning_rate=lr,
        weight_decay=weight_decay,
        optimizer_name=optimizer,
    )

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_dataloader_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(Xva, yva),
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_dataloader_workers,
        pin_memory=True,
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_checkpointing=False,
        logger=False,
        callbacks=[EarlyStopping(monitor="val_mse", patience=8)],
    )
    trainer.fit(lit, train_loader, val_loader)

    # ----- Save in your plain format (same as original) -----
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(
        save_dir,
        f"{out_model_name}.pt"
    )
    torch.save({
        "state_dict": lit.model.state_dict(),
        "arch_config": arch_config,
        "input_size": in_size_ckpt,
        "output_size": ytr.shape[1],
    }, out_path)
    print(f"Saved to {out_path}")
    return out_path