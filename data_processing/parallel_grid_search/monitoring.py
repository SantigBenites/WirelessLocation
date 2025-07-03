import matplotlib.pyplot as plt
import os, json
import numpy as np


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
