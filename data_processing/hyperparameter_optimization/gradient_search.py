import ray, torch
from model_generation import generate_random_model_configs, generate_similar_model_configs
from config import TrainingConfig
from gpu_fucntion import train_model_ray
import ray.exceptions, time 

def run_model_parallel_gradient_search(X_train, y_train, X_val, y_val, config):
    if not ray.is_initialized():
        import torch
        ray.init(num_cpus=24, num_gpus=torch.cuda.device_count(), log_to_driver=False)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    train_ref = ray.put((X_train, y_train))
    val_ref = ray.put((X_val, y_val))
    top_models = []

    for depth in range(config.training_depth):
        print(f"\n🔍 Search Depth {depth + 1}/{config.training_depth}")

        if depth == 0:
            current_configs = generate_random_model_configs(number_of_models=config.models_per_depth)
        else:
            current_configs = []
            for base_model in top_models:
                variations = generate_similar_model_configs(
                    base_model,
                    number_of_models=config.models_per_depth,
                    variation_factor=config.initial_variation_factor * (1 - config.variation_decay_rate * depth)
                )
                current_configs.extend(variations)

        for i, cfg in enumerate(current_configs):
            cfg['name'] = f"{config.group_name}_depth{depth}_model{i}"

        completed_results = []
        total_models = len(current_configs)
        gpu_capacity = int(torch.cuda.device_count() * 2)  # assuming 0.5 GPU per model

        print("⏳ Launching model batches...")
        for i in range(0, total_models, gpu_capacity):
            batch = current_configs[i:i + gpu_capacity]
            futures = [
                train_model_ray.remote(cfg, train_ref, val_ref, i + j, config, use_wandb=False)
                for j, cfg in enumerate(batch)
            ]

            start_time = time.time()
            pending = futures.copy()
            timeout_seconds = 1800
            poll_interval = 10

            while pending and (time.time() - start_time) < timeout_seconds:
                done, pending = ray.wait(pending, timeout=poll_interval, num_returns=1)
                for ref in done:
                    try:
                        result = ray.get(ref)
                        completed_results.append(result)
                    except Exception as e:
                        print(f"⚠️ A model failed with error: {str(e)}")
                print(f"✅ {len(completed_results)} models finished so far. ⏳ {len(pending)} still running...")

            for ref in pending:
                try:
                    result = ray.get(ref, timeout=30)
                    completed_results.append(result)
                except Exception as e:
                    print(f"❌ Timed out or failed to retrieve result: {str(e)}")

        results = completed_results

        successes = [r for r in results if r["status"] == "success"]
        results = sorted(results, key=lambda x: x['val_loss'])
        successes = sorted(successes, key=lambda x: x['val_loss'])
        top_models = successes[:config.models_per_depth]

        failures = [r for r in results if r["status"] == "failed"]
        if failures:
            print(f"❌ {len(failures)} models failed:")
            for r in failures:
                print(f"    {r.get('name', '?')} failed due to: {r.get('error', 'Unknown error')}")

        if top_models:
            best_model_config = top_models[0]
            best_model_config["name"] = f"{config.group_name}_best_depth{depth}"
            print(f"\n📊 Logging best model {best_model_config['name']} to Weights & Biases...")
            _ = ray.get(train_model_ray.remote(
                best_model_config,
                train_ref,
                val_ref,
                -1,
                config,
                use_wandb=True
            ))

    return top_models
