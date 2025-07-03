from model_generation import generate_model_configs
from data_processing.parallel_grid_search.search_lib import train_model_worker , MAX_MODELS_PER_GPU
import time
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
import numpy as np



def run_model_parallel_grid_search(X_train, y_train, X_val, y_val, epochs=200):
    # Increase models per GPU based on GPU memory


    model_configs = generate_model_configs()
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


