import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
import os
from tqdm import tqdm
import math
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pymongo import MongoClient
import json
import time
from collections import defaultdict
import json
import os
from datetime import datetime
import signal
import sys

STOP_FLAG = False
gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
global MAX_MODELS_PER_GPU
MAX_MODELS_PER_GPU = 2  # ~6GB per mode

def signal_handler(sig, frame):
    global STOP_FLAG
    print("\nReceived interrupt signal. Finishing current model and saving state...")
    STOP_FLAG = True

# Register the signal handler
#signal.signal(signal.SIGINT, signal_handler)
#signal.signal(signal.SIGTERM, signal_handler)

def get_dataset(collection_name, db_name):
    """Optimized data loading with proper type conversion and error handling"""
    client = MongoClient('mongodb://localhost:28910/', 
                       connectTimeoutMS=30000, 
                       socketTimeoutMS=30000,
                       maxPoolSize=20)
    
    db = client[db_name]
    collection = db[collection_name]
    
    # Use aggregation pipeline for server-side processing and type conversion
    pipeline = [
        {'$project': {
            'AP1_rssi': {'$ifNull': ['$AP1_rssi', -100]},
            'AP2_rssi': {'$ifNull': ['$AP2_rssi', -100]},
            'AP3_rssi': {'$ifNull': ['$AP3_rssi', -100]},
            'location_x': 1,
            'location_y': 1
        }},
        {'$match': {
            'location_x': {'$type': 'number'},
            'location_y': {'$type': 'number'},
            'AP1_rssi': {'$type': 'number'},
            'AP2_rssi': {'$type': 'number'},
            'AP3_rssi': {'$type': 'number'}
        }}
    ]
    
    # Get cursor with larger batch size
    cursor = collection.aggregate(pipeline, allowDiskUse=True, batchSize=50000)
    
    data = []
    for doc in cursor:
        try:
            # Convert all values to float explicitly
            row = (
                float(doc['AP1_rssi']),
                float(doc['AP2_rssi']),
                float(doc['AP3_rssi']),
                float(doc['location_x']),
                float(doc['location_y'])
            )
            data.append(row)
        except (ValueError, TypeError, KeyError) as e:
            # Skip malformed documents
            continue
    
    if not data:
        raise ValueError(f"No valid data found in collection {collection_name}")
    
    return np.array(data, dtype=np.float32)

def split_combined_data(combined_array, num_ap=3):
    features = combined_array[:, :num_ap]
    labels = combined_array[:, num_ap:]    
    return features, labels

def combine_arrays(arrays):
    return np.vstack(arrays)

def shuffle_array(arr, random_state=None):
    np.random.seed(random_state)
    shuffled_arr = arr.copy()
    np.random.shuffle(shuffled_arr)
    return shuffled_arr


class GeneratedModel(nn.Module):
    def __init__(self, input_size, output_size, architecture_config):
        super(GeneratedModel, self).__init__()
        self.layers = nn.ModuleList()
        self.architecture_config = architecture_config
        self.activation_map = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
            'elu': nn.ELU(inplace=True),
            'selu': nn.SELU(inplace=True),
            'tanh': nn.Tanh()
        }
        
        prev_size = input_size
        self.residual = architecture_config.get('residual_connections', False)
        
        # Build hidden layers
        for i, layer_spec in enumerate(architecture_config['hidden_layers']):
            layer_size = layer_spec['units']
            
            # Linear layer with optimized initialization
            linear_layer = nn.Linear(prev_size, layer_size)
            init_method = layer_spec.get('initialization', 'default')
            if init_method == 'xavier':
                nn.init.xavier_uniform_(linear_layer.weight)
            elif init_method == 'kaiming':
                nn.init.kaiming_uniform_(linear_layer.weight, mode='fan_in', nonlinearity='relu')
            elif init_method == 'orthogonal':
                nn.init.orthogonal_(linear_layer.weight)
                
            self.layers.append(linear_layer)
            
            # Normalization
            norm_type = layer_spec.get('normalization', 'none')
            if norm_type == 'batch':
                self.layers.append(nn.BatchNorm1d(layer_size))
            elif norm_type == 'layer':
                self.layers.append(nn.LayerNorm(layer_size))
            elif norm_type == 'instance':
                self.layers.append(nn.InstanceNorm1d(layer_size))
            
            # Activation with memory optimization
            activation = layer_spec.get('activation', 'relu')
            self.layers.append(self.activation_map[activation])
            
            # Dropout
            if layer_spec.get('use_dropout', False):
                self.layers.append(nn.Dropout(layer_spec['dropout']))
            
            prev_size = layer_size
        
        # Attention mechanism if enabled
        if architecture_config.get('attention', False):
            self.attention = nn.MultiheadAttention(prev_size, num_heads=4)
            self.attention_norm = nn.LayerNorm(prev_size)
        else:
            self.attention = None
        
        # Output layers
        self.output_layer = nn.Linear(prev_size, output_size)
        
        if architecture_config.get('uncertainty_estimation', False):
            self.uncertainty_layer = nn.Linear(prev_size, output_size)
        else:
            self.uncertainty_layer = None
    
    def forward(self, x):
        residual = x if self.residual else None
        
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = layer(x)
                if self.residual and residual is not None and residual.size(-1) == x.size(-1):
                    x = x + residual
                    residual = x
            else:
                x = layer(x)
        
        if self.attention is not None:
            attn_output, _ = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
            x = self.attention_norm(x + attn_output.squeeze(0))
        
        position = self.output_layer(x)
        uncertainty = torch.sigmoid(self.uncertainty_layer(x)) if self.uncertainty_layer else None
        
        return position, uncertainty

def generate_model_configs(search_space):
    keys, values = zip(*search_space.items())
    configs = []
    
    for combination in product(*values):
        config = dict(zip(keys, combination))
        
        hidden_layers = []
        for i in range(config['num_layers']):
            layer_spec = {
                'units': config['layer_size'],
                'activation': config['activation'],
                'normalization': 'batch' if config['batch_norm'] else 'none',
                'use_dropout': config['use_dropout'],
                'dropout': config['dropout'] if config['use_dropout'] else None,
                'initialization': 'xavier'
            }
            hidden_layers.append(layer_spec)
        
        architecture_config = {
            'hidden_layers': hidden_layers,
            'attention': config['attention'],
            'uncertainty_estimation': config['uncertainty_estimation'],
            'residual_connections': config.get('residual', False),
            'use_checkpointing': config.get('checkpointing', False)
        }
        
        configs.append({
            'name': f"Model_{len(configs)+1}",
            'config': architecture_config,
            'params': config
        })
    
    return configs

def save_model_performance_immediately(result, output_dir="model_performance"):
    """Save error graphs and model descriptions for a single model immediately after training"""
    os.makedirs(output_dir, exist_ok=True)
    model_name = result['model_name']
    
    try:
        # 1. Save loss curve plot
        plt.figure(figsize=(10, 6))
        plt.plot(result['train_loss_history'], label='Training Loss')
        
        # Plot validation points at their actual epochs
        val_epochs, val_losses = zip(*result['val_loss_history'])
        plt.scatter(val_epochs, val_losses, color='red', label='Validation Loss')
        
        plt.title(f"Training Progress: {model_name}")
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(output_dir, f"{model_name}_loss_curve.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Save model description
        desc_path = os.path.join(output_dir, f"{model_name}_description.txt")
        with open(desc_path, 'w') as f:
            f.write(f"Model Name: {model_name}\n")
            f.write(f"Training Time: {result['training_time']:.2f} seconds\n\n")
            
            f.write("=== Performance Metrics ===\n")
            f.write(f"Training RMSE: {result['train_rmse']:.4f}\n")
            f.write(f"Validation RMSE: {result['val_rmse']:.4f}\n")
            f.write(f"Training MAE: {result['train_mae']:.4f}\n")
            f.write(f"Validation MAE: {result['val_mae']:.4f}\n")
            f.write(f"Training R²: {result['train_r2']:.4f}\n")
            f.write(f"Validation R²: {result['val_r2']:.4f}\n\n")
            
            f.write("=== Model Architecture ===\n")
            f.write(f"Number of Layers: {result['params']['num_layers']}\n")
            f.write(f"Layer Size: {result['params']['layer_size']}\n")
            f.write(f"Activation: {result['params']['activation']}\n")
            f.write(f"Batch Normalization: {result['params']['batch_norm']}\n")
            f.write(f"Dropout: {result['params']['dropout'] if result['params']['use_dropout'] else 'None'}\n")
            f.write(f"Attention: {result['params']['attention']}\n")
            f.write(f"Uncertainty Estimation: {result['params']['uncertainty_estimation']}\n")
    
    except Exception as e:
        print(f"Error saving plots for {model_name}: {str(e)}")



def run_model_parallel_grid_search(X_train, y_train, X_val, y_val, search_space, epochs=200):
    # Increase models per GPU based on GPU memory


    model_configs = generate_model_configs(search_space)
    print(f"Number of configs: {len(model_configs)}")

    if not model_configs:
        print("No model configs generated.")
        return []

    world_size = torch.cuda.device_count()
    ctx = mp.get_context('spawn')

    model_queue = ctx.Queue()
    for i, config in enumerate(model_configs):
        model_queue.put((i, config))

    result_queue = ctx.Queue()
    gpu_slots = ctx.Manager().dict({i: 0 for i in range(world_size)})
    slot_lock = ctx.Lock()

    def gpu_manager():
        processes = []
        total_models = len(model_configs)
        completed = 0
        progress = tqdm(total=total_models, desc="Overall Progress")
        results = []

        while completed < total_models:
            for gpu_id in range(world_size):
                with slot_lock:
                    if gpu_slots[gpu_id] < MAX_MODELS_PER_GPU:
                        try:
                            model_index, config = model_queue.get_nowait()
                        except:
                            continue

                        gpu_slots[gpu_id] += 1
                        p = ctx.Process(
                            target=train_model_worker,
                            args=(gpu_id, model_index, config, result_queue, X_train, y_train, X_val, y_val, epochs, gpu_slots, slot_lock)
                        )
                        p.start()
                        processes.append(p)

            try:
                result = result_queue.get(timeout=1)
                if result:
                    completed += 1
                    results.append(result)
                    progress.update(1)
                    progress.set_postfix({
                        'current_gpu': f"GPU{result['gpu_used']}",
                        'model': result['model_name'][:15] + "...",
                        'epochs': result['epochs_trained']
                    })
            except:
                time.sleep(1)

        progress.close()
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join()

        return results

    return gpu_manager()

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
        if STOP_FLAG:
            break
            
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

def generate_comprehensive_summary(results, output_dir="model_performance"):
    """Generate a comprehensive summary of all models"""
    if not results:
        return None
    
    # Sort models by validation RMSE
    results.sort(key=lambda x: x['val_rmse'])
    
    summary = {
        'best_model': results[0]['model_name'],
        'best_val_rmse': results[0]['val_rmse'],
        'worst_model': results[-1]['model_name'],
        'worst_val_rmse': results[-1]['val_rmse'],
        'avg_val_rmse': np.mean([r['val_rmse'] for r in results]),
        'models': []
    }
    
    for result in results:
        model_info = {
            'name': result['model_name'],
            'val_rmse': result['val_rmse'],
            'val_mae': result['val_mae'],
            'val_r2': result['val_r2'],
            'training_time': result['training_time'],
            'params': result['params']
        }
        summary['models'].append(model_info)
    
    # Save JSON summary
    json_path = os.path.join(output_dir, "summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save text summary
    txt_path = os.path.join(output_dir, "summary.txt")
    with open(txt_path, 'w') as f:
        f.write("=== Model Training Summary ===\n\n")
        f.write(f"Total Models Trained: {len(results)}\n")
        f.write(f"Best Model: {summary['best_model']} (RMSE: {summary['best_val_rmse']:.4f})\n")
        f.write(f"Worst Model: {summary['worst_model']} (RMSE: {summary['worst_val_rmse']:.4f})\n")
        f.write(f"Average Validation RMSE: {summary['avg_val_rmse']:.4f}\n\n")
        
        f.write("=== Top Performing Models ===\n")
        for i, model in enumerate(summary['models'][:5], 1):
            f.write(f"{i}. {model['name']}\n")
            f.write(f"   RMSE: {model['val_rmse']:.4f} | MAE: {model['val_mae']:.4f} | R²: {model['val_r2']:.4f}\n")
            f.write(f"   Config: {model['params']['num_layers']}x{model['params']['layer_size']} ")
            f.write(f"{model['params']['activation']} (BN: {model['params']['batch_norm']}, ")
            f.write(f"Dropout: {model['params']['dropout'] if model['params']['use_dropout'] else 'None'})\n\n")
    
    return summary

if __name__ == '__main__':
    try:
        # Get and process data
        print("Loading and processing data...")
        datasets = [
            get_dataset("wifi_data_reto_grande", "wifi_data_db"),
            get_dataset("wifi_data_reto_pequeno", "wifi_data_db"),
            get_dataset("wifi_data_reto_medio", "wifi_data_db")
        ]
        
        combined_data = combine_arrays(datasets)
        shuffled_data = shuffle_array(combined_data)
        global_array_x, global_array_y = split_combined_data(shuffled_data)
        X_train, X_val, y_train, y_val = train_test_split(
            global_array_x, global_array_y, test_size=0.2, random_state=42
        )

        # Define optimized search space
        search_space = {
            'num_layers': [1, 2, 3, 4],  # Deeper networks
            'layer_size': [64, 128, 256, 512],  # Wider networks
            'activation': ['relu', 'leaky_relu', 'elu', 'selu', 'tanh'],
            'batch_norm': [True, False],
            'use_dropout': [True, False],
            'dropout': [0.2, 0.3, 0.4, 0.5],  # Different dropout rates
            'attention': [False, True],  # Option for attention mechanism
            'uncertainty_estimation': [False, True],  # Uncertainty estimation
            'learning_rate': [0.001, 0.0005, 0.0001],  # Learning rate variations
            'weight_decay': [0.0, 0.0001, 0.00001],  # L2 regularization
            'optimizer': ['adam', 'adamw', 'sgd', 'rmsprop'],  # Different optimizers
            'batch_size': [64, 128, 256],  # Different batch sizes
            'normalization': ['none', 'batch', 'layer', 'instance'],  # Different norm types
            'residual_connections': [False, True],  # Skip connections
            'initialization': ['default', 'xavier', 'kaiming', 'orthogonal']
        }


        print(f"\nStarting training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using {torch.cuda.device_count()} GPUs")
        print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

        # Run training
        results = run_model_parallel_grid_search(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            search_space=search_space,
            epochs=200
        )

        if not results:
            raise RuntimeError("No models were successfully trained.")
        
        # Generate final summary
        summary = generate_comprehensive_summary(results)
        
        print(f"\nTraining completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        if summary:
            print(f"Best model: {summary['best_model']} with validation RMSE: {summary['best_val_rmse']:.4f}")
        else:
            print("Warning: No summary was generated.")
            
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise