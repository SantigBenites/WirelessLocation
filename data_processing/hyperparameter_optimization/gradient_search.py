# gradient_search.py
import ray
from model_generation import generate_random_model_configs, generate_similar_model_configs
from config import TrainingConfig
from gpu_fucntion import train_model_ray



def run_model_parallel_gradient_search(X_train, y_train, X_val, y_val, config):

    if not ray.is_initialized():
        ray.init(log_to_driver=False)

    # Put data into Ray object store once
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

        # Ensure unique names for each model
        for i, cfg in enumerate(current_configs):
            cfg['name'] = f"depth{depth}_model{i}"

        futures = [
            train_model_ray.remote(
                model_config,
                train_ref,
                val_ref,
                i,
                config,
                use_wandb=False
            )
            for i, model_config in enumerate(current_configs)
        ]

        results = ray.get(futures)
        successes = [r for r in results if r["status"] == "success"]
        results = sorted(results, key=lambda x: x['val_loss'])
        successes = sorted(successes, key=lambda x: x['val_loss'])
        top_models = successes[:config.models_per_depth]

        # Logging
        failures = [r for r in results if r["status"] == "failed"]
        if failures:
            print(f"❌ {len(failures)} models failed:")
            for r in failures:
                print(f"    {r['name']} failed due to: {r['error']}")

        best_model_config = top_models[0]
        best_model_config["name"] += "_wandb"
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