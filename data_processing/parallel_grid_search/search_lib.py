import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
from monitoring import save_model_performance_immediately
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model_generation import GeneratedModel

MAX_MODELS_PER_GPU = 2 

def train_model_worker(rank, model_index, config, result_queue, X_train, y_train, X_val, y_val, epochs, gpu_slots, slot_lock):
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    with tqdm(total=epochs, desc=f"GPU{rank}: {config['name']}", position=model_index % (torch.cuda.device_count() * MAX_MODELS_PER_GPU)) as pbar:
        result = train_single_model(
            rank=rank,
            model_config=config,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            progress_bar=pbar
        )

    save_model_performance_immediately(result)
    result['gpu_used'] = rank
    result['model_index'] = model_index
    result_queue.put(result)

    with slot_lock:
        gpu_slots[rank] -= 1



# Custom dataset that uses memory-mapped files
class MMapDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return (self.features[idx], self.targets[idx])


def train_single_model(rank, model_config, X_train, y_train, X_val, y_val, epochs, progress_bar=None):
    device = torch.device(f'cuda:{rank}')
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    torch.cuda.empty_cache()  # Clear cache before starting

    # Memory optimization - use memory-mapped files
    train_mmap_path = os.path.join("data_dir", f'train_data_{rank}.npy')
    val_mmap_path = os.path.join("data_dir", f'val_data_{rank}.npy')
    
    # Save data to memory-mapped files if not exists
    if not os.path.exists(train_mmap_path):
        np.save(train_mmap_path, X_train)
        np.save(val_mmap_path, X_val)
    
    # Load memory-mapped arrays
    X_train_mmap = np.load(train_mmap_path, mmap_mode='r')
    X_val_mmap = np.load(val_mmap_path, mmap_mode='r')

    # Create DataLoaders with optimized settings
    batch_size = model_config['params'].get('batch_size', 512)  # Increased batch size
    num_workers = min(8, os.cpu_count() // 2)  # More workers with prefetch
    
    train_dataset = MMapDataset(X_train_mmap, y_train)
    val_dataset = MMapDataset(X_val_mmap, y_val)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,  # Enable pinned memory for faster transfers
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=4  # Prefetch more batches
    )
    
    # Create model with mixed precision
    model = GeneratedModel(
        input_size=X_train.shape[1],
        output_size=y_train.shape[1],
        architecture_config=model_config['config']
    ).to(device)
    
    # Optimizer with gradient accumulation
    optimizer_type = model_config['params'].get('optimizer', 'adam')
    learning_rate = model_config['params'].get('learning_rate', 0.001)
    weight_decay = model_config['params'].get('weight_decay', 0.0)
    accumulation_steps = model_config['params'].get('accumulation_steps', 2)
    
    if optimizer_type == 'adam':
        try:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
                                 weight_decay=weight_decay, fused=True)
        except:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
                                 weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                              weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                            momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, 
                                weight_decay=weight_decay)
    
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler(enabled=True)
    
    # Early stopping setup
    patience = 10
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    
    train_loss_history = []
    val_loss_history = []
    val_interval = max(20, epochs // 5)  # Less frequent validation
    
    for epoch in range(epochs):
            
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)  # Initialize gradients
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            epoch_loss += loss.item() * accumulation_steps
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        
        # Validation - simplified and optimized
        if epoch % val_interval == 0 or epoch == epochs - 1:
            model.eval()
            val_loss = 0.0
            
            # Always use DataLoader for validation to avoid memory issues
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size * 2,
                shuffle=False,
                pin_memory=True,
                num_workers=num_workers
            )
            
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                for val_inputs, val_targets in val_loader:
                    val_inputs = val_inputs.to(device, non_blocking=True)
                    val_targets = val_targets.to(device, non_blocking=True)
                    val_outputs, _ = model(val_inputs)
                    val_loss += criterion(val_outputs, val_targets).item()
                
                val_loss /= len(val_loader)
            
            val_loss_history.append((epoch, val_loss))
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= patience:
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    model.to(device)
                break
        
        if progress_bar:
            progress_bar.set_postfix({
                'loss': f"{avg_train_loss:.4f}",
                'val': f"{val_loss:.4f}" if epoch % val_interval == 0 else "",
                'patience': f"{epochs_without_improvement}/{patience}",
                'mem': f"{torch.cuda.memory_reserved(device)/1e9:.1f}GB"
            })
            progress_bar.update(1)
    
    # Final evaluation with best model weights - optimized
    model.eval()
    train_preds = []
    val_preds = []
    
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        # Process training data
        for inputs, _ in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs, _ = model(inputs)
            train_preds.append(outputs.cpu())
        
        # Process validation data
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers
        )
        for inputs, _ in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs, _ = model(inputs)
            val_preds.append(outputs.cpu())
    
    train_preds = torch.cat(train_preds).numpy()
    val_preds = torch.cat(val_preds).numpy()
    
    # Clean up memory-mapped files
    del X_train_mmap, X_val_mmap
    if os.path.exists(train_mmap_path):
        os.remove(train_mmap_path)
    if os.path.exists(val_mmap_path):
        os.remove(val_mmap_path)
    
    return {
        'model': model.cpu(),
        'model_name': model_config['name'],
        'params': model_config['params'],
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_preds)),
        'val_rmse': np.sqrt(mean_squared_error(y_val, val_preds)),
        'train_mae': mean_absolute_error(y_train, train_preds),
        'val_mae': mean_absolute_error(y_val, val_preds),
        'train_r2': r2_score(y_train, train_preds),
        'val_r2': r2_score(y_val, val_preds),
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'training_time': progress_bar.format_dict['elapsed'] if progress_bar else 0,
        'epochs_trained': epoch + 1,
        'gpu_memory_usage': torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # in GB
    }


