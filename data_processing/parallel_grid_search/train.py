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
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from pymongo import MongoClient


    
def process_data(data):
    """
    Preprocess the MongoDB documents into a single array with 5 columns.
    Columns: AP1_rssi, AP2_rssi, AP3_rssi, location_x, location_y
    
    Handles NaN values by:
    1. Replacing NaN RSSI values with -100 (standard for missing signal)
    2. Ensuring coordinates are always valid numbers
    """
    combined_data = []
    
    for entry in data:
        # Safely extract RSSI values, handling missing/NaN values
        rssi_values = [
            float(entry.get('AP1_rssi', -100)) if entry.get('AP1_rssi', -100) != None else -100,
            float(entry.get('AP2_rssi', -100)) if entry.get('AP2_rssi', -100) != None else -100,
            float(entry.get('AP3_rssi', -100)) if entry.get('AP3_rssi', -100) != None else -100
        ]
        
        # Validate coordinates
        try:
            x_coord = float(entry['location_x'])
            y_coord = float(entry['location_y'])
            if np.isnan(x_coord) or np.isnan(y_coord):
                continue  # Skip this entry if coordinates are invalid
        except (KeyError, ValueError):
            continue  # Skip this entry if coordinates are missing or invalid
            
        # Combine all values into one row
        combined_row = rssi_values + [x_coord, y_coord]
        combined_data.append(combined_row)
    
    # Convert to numpy array and verify no NaNs remain
    result = np.array(combined_data, dtype=np.float32)
    assert not np.isnan(result).any(), "NaN values detected in final output!"
    
    return result

def get_dataset(collection_name, db_name):
    """
    Args:
        collection_name (str): Name of the MongoDB collection to use
        db_name (str): Name of the MongoDB database
    """
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:28910/')
    db = client[db_name]
    collection = db[collection_name]
    
    # Load all data from the collection
    data = list(collection.find())
    
    # Preprocess the data to extract features and labels
    return process_data(data)


def split_combined_data(combined_array, num_ap=3):

    # Split the array into features (RSSI values) and labels (coordinates)
    features = combined_array[:, :num_ap]  # First num_ap columns are RSSI values
    labels = combined_array[:, num_ap:]    # Last 2 columns are coordinates
    
    return features, labels

def combine_arrays(arrays):
    return np.vstack(arrays)

def shuffle_array(arr, random_state=None):
    np.random.seed(random_state)
    shuffled_arr = arr.copy()
    np.random.shuffle(shuffled_arr)
    return shuffled_arr


def setup(rank, world_size):
    """Initialize the distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up distributed processes"""
    dist.destroy_process_group()

class GeneratedModel(nn.Module):
    def __init__(self, input_size, output_size, architecture_config):
        super(GeneratedModel, self).__init__()
        self.layers = nn.ModuleList()  # Changed from ModuleDict to ModuleList for sequential access
        self.architecture_config = architecture_config
        
        # Build the network dynamically
        prev_size = input_size
        
        for layer_spec in architecture_config['hidden_layers']:
            # Add linear layer
            layer_size = layer_spec['units']
            self.layers.append(nn.Linear(prev_size, layer_size))
            prev_size = layer_size
            
            # Add batch norm if specified
            if layer_spec.get('batch_norm', False):
                self.layers.append(nn.BatchNorm1d(layer_size))
            
            # Add activation
            activation = layer_spec.get('activation', 'relu')
            if activation == 'relu':
                self.layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                self.layers.append(nn.LeakyReLU(0.1))
            
            # Add dropout if specified
            if 'dropout' in layer_spec:
                self.layers.append(nn.Dropout(layer_spec['dropout']))
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)
    
    def forward(self, x):
        # Process through all layers
        for layer in self.layers:
            x = layer(x)
        
        position = self.output_layer(x)
        return position, None  # Returning None for uncertainty

def generate_model_configs(search_space):
    """Generate all possible model configurations from the search space"""
    keys, values = zip(*search_space.items())
    configs = []
    
    for combination in product(*values):
        config = dict(zip(keys, combination))
        
        hidden_layers = []
        for i in range(config['num_layers']):
            layer_spec = {
                'units': config['layer_size'],
                'batch_norm': config['batch_norm'],
                'activation': config['activation'],
                'dropout': config['dropout'] if config['use_dropout'] else None
            }
            hidden_layers.append(layer_spec)
        
        architecture_config = {
            'hidden_layers': hidden_layers,
            'attention': config['attention'],
            'uncertainty_estimation': config['uncertainty_estimation']
        }
        
        configs.append({
            'name': f"Model_{len(configs)+1}",
            'config': architecture_config,
            'params': config
        })
    
    return configs

def train_model(rank, world_size, model_config, X_train, y_train, X_val, y_val, 
               epochs=100, batch_size=32, learning_rate=0.001, progress_callback=None):
    """Train a single model on a specific GPU with optional progress updates"""
    # Set device for this process
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # Get this GPU's portion of the data
    train_size = len(X_train)
    indices = list(range(rank, train_size, world_size))
    X_train_gpu = X_train[indices]
    y_train_gpu = y_train[indices]
    
    # Convert to tensors and move to current GPU
    X_train_tensor = torch.FloatTensor(X_train_gpu).to(device)
    y_train_tensor = torch.FloatTensor(y_train_gpu).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True
    )
    
    # Initialize model
    model = GeneratedModel(
        input_size=X_train.shape[1],
        output_size=y_train.shape[1],
        architecture_config=model_config['config']
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        # Calculate epoch metrics
        train_loss = np.mean(batch_losses)
        train_loss_history.append(train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs, _ = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_loss_history.append(val_loss)
        
        # Update progress if callback provided
        if progress_callback:
            progress_callback(train_loss, val_loss, epoch)
    
    # Gather final metrics
    model.eval()
    with torch.no_grad():
        train_preds, _ = model(X_train_tensor)
        val_preds, _ = model(X_val_tensor)
        
        train_rmse = np.sqrt(mean_squared_error(y_train_gpu, train_preds.cpu().numpy()))
        train_mae = mean_absolute_error(y_train_gpu, train_preds.cpu().numpy())
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds.cpu().numpy()))
        val_mae = mean_absolute_error(y_val, val_preds.cpu().numpy())
    
    return {
        'model': model.cpu(),  # Move model back to CPU for collection
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'model_name': model_config['name'],
        'params': model_config['params']
    }

def run_multi_gpu_grid_search(X_train, y_train, X_val, y_val, search_space, epochs=200):
    """Run grid search using all available GPUs with true model parallelism"""
    # Generate all model configurations
    model_configs = generate_model_configs(search_space)
    print(f"Generated {len(model_configs)} models")
    
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs. Will train {len(model_configs)} models.")
    
    # Distribute models across GPUs (one model per GPU)
    models_per_gpu = (len(model_configs) + world_size - 1) // world_size
    
    # Set start method for multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Create a queue to collect results
    output_queue = mp.Queue()
    
    processes = []
    for rank in range(min(world_size, len(model_configs))):
        # Assign models to this GPU
        start_idx = rank * models_per_gpu
        end_idx = min((rank + 1) * models_per_gpu, len(model_configs))
        gpu_configs = model_configs[start_idx:end_idx]
        
        # Only create process if there are models to train
        if gpu_configs:
            p = mp.Process(
                target=train_models_on_gpu,
                args=(rank, gpu_configs, X_train, y_train, X_val, y_val, epochs, output_queue)
            )
            p.start()
            processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Collect results
    results = []
    while not output_queue.empty():
        results.append(output_queue.get())
    
    return results

def train_models_on_gpu(rank, model_configs, X_train, y_train, X_val, y_val, epochs, output_queue):
    """Train multiple models sequentially on a single GPU with progress bars"""
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    for config in model_configs:
        #print(f"\nTraining {config['name']} on GPU {rank}")
        #print(f"Config: {config['params']}")
        
        # Create progress bar for this model
        with tqdm(total=epochs, desc=f"GPU {rank}: {config['name']}", position=rank) as pbar:
            # Create a closure to update progress bar
            def update_progress(train_loss, val_loss, epoch):
                pbar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}'
                })
                pbar.update(1)
            
            # Train the model with progress updates
            result = train_model(
                rank=rank,
                world_size=1,
                model_config=config,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=epochs,
                progress_callback=update_progress
            )
            output_queue.put(result)

def plot_programmatic_results(results, output_dir="results_plots"):
    """
    Save results plots from grid search to files and provide detailed metrics.
    
    Args:
        results (list): List of training result dictionaries
        output_dir (str): Directory to save plot images (default: "results_plots")
    
    Returns:
        dict: Summary statistics of all models
    """
    if not results:
        print("No results to plot")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize summary statistics
    summary_stats = {
        'best_model': None,
        'best_val_rmse': float('inf'),
        'worst_model': None,
        'worst_val_rmse': -float('inf'),
        'avg_val_rmse': 0,
        'model_details': []
    }
    
    try:
        # Plot and save each model's training curve separately
        for result in results:
            try:
                plt.figure(figsize=(10, 6))
                
                # Plot training and validation curves
                plt.plot(result['train_loss_history'], label='Train Loss', linewidth=2)
                plt.plot(result['val_loss_history'], label='Validation Loss', linewidth=2)
                
                # Add model configuration details
                params = result['params']
                title = (f"{result['model_name']}\n"
                        f"Layers: {params['num_layers']}, Size: {params['layer_size']}\n"
                        f"Act: {params['activation']}, BN: {params['batch_norm']}\n"
                        f"Dropout: {params['dropout'] if params['use_dropout'] else 'No'}\n"
                        f"Val RMSE: {result['val_rmse']:.2f}, Val MAE: {result['val_mae']:.2f}")
                
                plt.title(title, pad=20)
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Loss', fontsize=12)
                plt.legend(fontsize=12)
                plt.grid(True, alpha=0.3)
                
                # Adjust layout and save
                plt.tight_layout()
                filename = os.path.join(output_dir, f"{result['model_name']}_plot.png")
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Update summary statistics
                current_rmse = result['val_rmse']
                summary_stats['avg_val_rmse'] += current_rmse
                
                model_detail = {
                    'name': result['model_name'],
                    'val_rmse': current_rmse,
                    'val_mae': result['val_mae'],
                    'params': params,
                    'plot_file': filename
                }
                summary_stats['model_details'].append(model_detail)
                
                # Track best/worst models
                if current_rmse < summary_stats['best_val_rmse']:
                    summary_stats['best_val_rmse'] = current_rmse
                    summary_stats['best_model'] = model_detail
                
                if current_rmse > summary_stats['worst_val_rmse']:
                    summary_stats['worst_val_rmse'] = current_rmse
                    summary_stats['worst_model'] = model_detail
                    
            except Exception as model_err:
                print(f"\nError processing model {result.get('model_name', 'unknown')}:")
                print(f"Error type: {type(model_err).__name__}")
                print(f"Error message: {str(model_err)}")
                print(f"Model params: {result.get('params', 'N/A')}")
                print("Skipping this model...\n")
                continue
        
        # Calculate average RMSE
        summary_stats['avg_val_rmse'] /= len(results)
        
        # Generate and save summary report
        summary_report = generate_summary_report(summary_stats, output_dir)
        print(f"\nSaved all plots to directory: {output_dir}")
        print(f"Summary report saved to: {summary_report}")
        
        return summary_stats
        
    except Exception as main_err:
        print("\nCritical error in plot_programmatic_results():")
        print(f"Error type: {type(main_err).__name__}")
        print(f"Error message: {str(main_err)}")
        print("Partial results may have been saved.")
        return None

def generate_summary_report(summary_stats, output_dir):
    """Generate a text summary report of all model performances"""
    report_path = os.path.join(output_dir, "model_summary.txt")
    
    with open(report_path, 'w') as f:
        # Write header
        f.write("Model Training Results Summary\n")
        f.write("="*50 + "\n\n")
        
        # Write best/worst performers
        f.write(f"Best Model: {summary_stats['best_model']['name']}\n")
        f.write(f"- Val RMSE: {summary_stats['best_val_rmse']:.4f}\n")
        f.write(f"- Configuration: {summary_stats['best_model']['params']}\n\n")
        
        f.write(f"Worst Model: {summary_stats['worst_model']['name']}\n")
        f.write(f"- Val RMSE: {summary_stats['worst_val_rmse']:.4f}\n")
        f.write(f"- Configuration: {summary_stats['worst_model']['params']}\n\n")
        
        f.write(f"Average Val RMSE across all models: {summary_stats['avg_val_rmse']:.4f}\n\n")
        
        # Write detailed table
        f.write("Detailed Model Performance:\n")
        f.write("-"*50 + "\n")
        f.write("{:<15} {:<10} {:<10} {:<15} {:<15} {:<15}\n".format(
            "Model", "Val RMSE", "Val MAE", "Layers", "Size", "Activation"))
        
        for model in sorted(summary_stats['model_details'], key=lambda x: x['val_rmse']):
            params = model['params']
            f.write("{:<15} {:<10.4f} {:<10.4f} {:<15} {:<15} {:<15}\n".format(
                model['name'],
                model['val_rmse'],
                model['val_mae'],
                params['num_layers'],
                params['layer_size'],
                params['activation']))
    
    return report_path
        

if __name__ == '__main__':
    # Get datasets from all collections
    datasets = [
        get_dataset("wifi_data_reto_grande", "wifi_data_db"),
        get_dataset("wifi_data_reto_pequeno", "wifi_data_db"),
        get_dataset("wifi_data_reto_medio", "wifi_data_db")
    ]

    # Combine all datasets into one array
    combined_data = combine_arrays(datasets)

    # Shuffle the combined data
    shuffled_data = shuffle_array(combined_data)

    # Split into features and labels
    global_array_x, global_array_y = split_combined_data(shuffled_data)

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(global_array_x, global_array_y, test_size=0.2, random_state=42)

    search_space = {
        'num_layers': [3, 4],
        'layer_size': [128, 256],
        'activation': ['relu', 'leaky_relu'],
        'batch_norm': [True, False],
        'use_dropout': [True],
        'dropout': [0.3],
        'attention': [False],
        'uncertainty_estimation': [False]
    }

    # Run the multi-GPU grid search
    results = run_multi_gpu_grid_search(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        search_space=search_space,
        epochs=200
    )

    results = run_multi_gpu_grid_search(...)
    summary = plot_programmatic_results(results, "experiment_1_results")

    if summary:
        print(f"\nBest model was {summary['best_model']['name']} with RMSE {summary['best_val_rmse']:.4f}")
        print(f"Complete results saved in 'experiment_1_results' directory")