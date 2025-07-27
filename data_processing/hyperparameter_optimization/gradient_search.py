import multiprocessing, time, os
from model_generation import generate_random_model_configs, generate_similar_model_configs
from config import TrainingConfig

def _train_model_wrapper(cfg, train_data_ref, val_data_ref,
                         model_index, config, gpu_id,
                         return_dict, gpu_slots):
    """
    Runs inside a **fresh child process**.
    We must decide which GPU it sees *before* importing torch.
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # expose only this GPU

    # Delay heavy imports until after masking
    from gpu_fucntion import train_model

    try:
        print(f"🎯 Starting model {cfg['name']} on physical GPU {gpu_id}")
        result = train_model(
            config_dict=cfg,
            train_data_ref=train_data_ref,
            val_data_ref=val_data_ref,
            model_index=model_index,
            config=config,
            use_wandb=False,
            gpu_id=gpu_id,
        )
        return_dict[model_index] = result
    except Exception as e:
        print(f"❌ Model {cfg['name']} failed on GPU {gpu_id}: {e}")
        return_dict[model_index] = {
            **cfg,
            "val_loss": float("inf"),
            "status": "failed",
            "error": str(e)
        }



def run_model_parallel_gradient_search(X_train, y_train, X_val, y_val, config: TrainingConfig):

    import torch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    train_data_ref = (X_train, y_train)
    val_data_ref = (X_val, y_val)

    top_models = []
    gpu_count = torch.cuda.device_count()

    for depth in range(config.training_depth):
        print(f"\n🔍 Search Depth {depth + 1}/{config.training_depth}")

        if depth == 0:
            current_configs = generate_random_model_configs(number_of_models=config.models_per_depth)
        else:
            current_configs = []
            for base_model in top_models:
                current_configs.extend(generate_similar_model_configs(
                    base_model,
                    number_of_models=config.models_per_depth,
                    variation_factor=config.initial_variation_factor * (1 - config.variation_decay_rate * depth)
                ))

        for i, cfg in enumerate(current_configs):
            cfg['name'] = f"{config.group_name}_depth{depth}_model{i}"

        print(f"\n⏳ Scheduling {len(current_configs)} models over {gpu_count} GPUs (1 at a time per GPU)")

        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        gpu_slots = manager.dict({str(i): False for i in range(gpu_count)})  # GPU id -> is_busy
        model_queue = current_configs.copy()
        active_processes = []
        model_index = 0

        while model_queue or any(p.is_alive() for p in active_processes):
            # Remove finished processes and mark GPUs as free
            new_active = []
            for p in active_processes:
                if p.is_alive():
                    new_active.append(p)
                else:
                    if hasattr(p, "gpu_id"):
                        gpu_slots[str(p.gpu_id)] = False
            active_processes = new_active

            # Try to schedule new models only if GPUs are free
            for gpu_id, is_busy in list(gpu_slots.items()):
                if not is_busy and model_queue:
                    cfg = model_queue.pop(0)
                    gpu_slots[gpu_id] = True

                    p = multiprocessing.Process(
                        target=_train_model_wrapper,
                        args=(cfg, train_data_ref, val_data_ref, model_index, config, gpu_id, return_dict, gpu_slots)
                    )
                    p.gpu_id = int(gpu_id)  # for tracking in cleanup
                    p.start()
                    active_processes.append(p)
                    model_index += 1

            time.sleep(0.5)  # avoid busy-wait

        # End of depth: collect and sort results
        completed_results = [return_dict[i] for i in sorted(return_dict.keys())]
        successes = [r for r in completed_results if r["status"] == "success"]
        successes.sort(key=lambda x: x['val_loss'])
        top_models = successes[:config.models_per_depth]

        failures = [r for r in completed_results if r["status"] == "failed"]
        if failures:
            print(f"❌ {len(failures)} models failed:")
            for r in failures:
                print(f"    {r.get('name', '?')} failed due to: {r.get('error', 'Unknown error')}")

        if top_models:
            best_model_config = top_models[0]
            best_model_config["name"] = f"{config.group_name}_best_depth{depth}"
            print(f"\n📊 Logging best model {best_model_config['name']} to Weights & Biases...")

            # Find free GPU again
            for gpu_id, is_busy in gpu_slots.items():
                if not is_busy:
                    gpu_slots[gpu_id] = True
                    _train_model_wrapper(
                        best_model_config,
                        train_data_ref,
                        val_data_ref,
                        -1,
                        config,
                        gpu_id,
                        return_dict={},
                        gpu_slots=gpu_slots
                    )
                    break

    return top_models
