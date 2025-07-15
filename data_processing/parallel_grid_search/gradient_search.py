from model_generation import generate_random_model_configs, generate_similar_model_configs
from search_lib import train_model_worker , MAX_MODELS_PER_GPU
import time
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
import numpy as np
import queue

NUMBER_OF_MODELS_PER_DEPTH = 4

def run_model_parallel_gradient_search(X_train, y_train, X_val, y_val, epochs=200, training_depth=10):
    # Initial random model generation
    model_configs = generate_random_model_configs(number_of_models=NUMBER_OF_MODELS_PER_DEPTH)
    print(f"Number of original configs: {len(model_configs)}")

    if not model_configs:
        print("No model configs generated.")
        return []

    world_size = torch.cuda.device_count()
    ctx = mp.get_context('spawn')
    all_results = []

    for depth in range(training_depth):
        print(f"\n=== Starting training depth {depth + 1}/{training_depth} ===")
        
        with ctx.Manager() as manager:

            # Prepare queues for this depth
            model_queue = manager.Queue()
            for i, config in enumerate(model_configs):
                model_queue.put((i, config))
    
            result_queue = manager.Queue()
            
            
            gpu_slots = manager.dict({i: 0 for i in range(world_size)})
            slot_lock = manager.Lock()

            # Track results for this depth
            depth_results = []
            total_models = len(model_configs)
            completed = 0
            
            progress = tqdm(total=total_models, desc=f"Depth {depth + 1} Progress")
            
            processes = []
            try:
                while completed < total_models:
                    # Assign models to available GPUs
                    for gpu_id in range(world_size):
                        with slot_lock:
                            if gpu_slots[gpu_id] < MAX_MODELS_PER_GPU:
                                try:
                                    model_index, config = model_queue.get_nowait()
                                except queue.Empty:
                                    continue

                                gpu_slots[gpu_id] += 1
                                p = ctx.Process(
                                    target=train_model_worker,
                                    args=(gpu_id, model_index, config, result_queue, 
                                         X_train, y_train, X_val, y_val, 
                                         epochs, gpu_slots, slot_lock)
                                )
                                p.start()
                                processes.append(p)

                    # Collect results
                    try:
                        result = result_queue.get(timeout=5)
                        if result:
                            completed += 1
                            depth_results.append(result)
                            all_results.append(result)
                            progress.update(1)
                            progress.set_postfix({
                                'current_gpu': f"GPU{result['gpu_used']}",
                                'model': result['model_name'][:15] + "...",
                                'val_score': f"{result['val_score']:.4f}",
                                'epochs': result['epochs_trained']
                            })
                    except queue.Empty:
                        # Check if any processes died
                        time.sleep(1)

                progress.close()
                
                # Find best model from this depth to generate next configs
                if depth_results:
                    best_model = max(depth_results, key=lambda x: x['val_score'])
                    print(f"\nBest model at depth {depth + 1}: {best_model['model_name']} with val_score {best_model['val_score']:.4f}")
                    
                    # Prepare base model config for next depth
                    base_model = {
                        'name': f"Depth{depth}_Best",
                        'config': best_model['config'],
                        'params': best_model['params']
                    }
                    
                    # Generate new similar configs for next depth
                    model_configs = generate_similar_model_configs(
                        base_model=base_model,
                        number_of_models=NUMBER_OF_MODELS_PER_DEPTH,
                        variation_factor=0.3 - (0.02 * depth)
                    )
                else:
                    print("No results in this depth, using random configs for next depth")
                    model_configs = generate_random_model_configs(number_of_models=NUMBER_OF_MODELS_PER_DEPTH)

            finally:
                # Clean up processes
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                    p.join()

    return all_results