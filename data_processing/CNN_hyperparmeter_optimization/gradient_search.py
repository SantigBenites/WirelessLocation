
import ray
import torch
import time
import random
from model_generation import generate_random_model_configs, generate_similar_model_configs
from config import TrainingConfig
from gpu_fucntion import train_model

@ray.remote(num_gpus=0.50)
def ray_train_model(cfg, train_data_ref, val_data_ref, model_index, config, use_wandb):
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            result = train_model(
                config_dict=cfg,
                train_data_ref=train_data_ref,
                val_data_ref=val_data_ref,
                model_index=model_index,
                config=config,
                use_wandb=use_wandb,
            )
            return result
        except Exception as e:
            print(f"❌ Attempt {attempt+1} failed for {cfg['name']}: {e}")
            time.sleep(5 + attempt * 2)
    return {
        **cfg,
        "val_loss": float("inf"),
        "status": "failed",
        "error": f"Failed after {max_attempts} attempts"
    }

def run_model_parallel_gradient_search(X_train, y_train, X_val, y_val, config: TrainingConfig):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    train_data_ref = (X_train, y_train)
    val_data_ref = (X_val, y_val)

    top_models = []
    final_best_model = None

    for depth in range(config.training_depth):
        print(f"\n🔍 Search Depth {depth + 1}/{config.training_depth}")

        if depth == 0:
            current_configs = generate_random_model_configs(
                number_of_models=config.models_per_depth
            )
        else:
            exploration_chance = 0.2
            base_model = top_models[0]

            if len(top_models) > 1 and random.random() < exploration_chance:
                base_model = random.choice(top_models[1:])
                print(f"🧭 Exploring variation of a non-best model: {base_model['name']}")
            else:
                print(f"🎯 Exploiting best model: {base_model['name']}")

            current_configs = generate_similar_model_configs(
                base_model=base_model,
                number_of_models=config.models_per_depth,
                variation_factor=config.initial_variation_factor * (1 - config.variation_decay_rate * depth)
            )

        for i, cfg in enumerate(current_configs):
            cfg['name'] = f"{config.experiment_name}_run{config.run_index}_depth{depth}_model{i}"

        print(f"⏳ Dispatching {len(current_configs)} models via Ray...")

        # Run all models without wandb
        futures = [
            ray_train_model.remote(cfg, train_data_ref, val_data_ref, i, config, False)
            for i, cfg in enumerate(current_configs)
        ]

        results = ray.get(futures)
        successes = [r for r in results if r["status"] == "success"]
        successes.sort(key=lambda x: x['val_loss'])

        top_models = successes[:config.models_per_depth]
        if final_best_model is None or top_models[0]["val_loss"] < final_best_model["val_loss"]:
            final_best_model = top_models[0] 

        failures = [r for r in results if r["status"] == "failed"]
        if failures:
            print(f"❌ {len(failures)} models failed:")
            for r in failures:
                print(f"    {r.get('name', '?')} failed due to: {r.get('error', 'Unknown error')}")

    if final_best_model:
        print(f"📊 Logging best overall model: {final_best_model['name']}")
        ray.get(ray_train_model.remote(final_best_model, train_data_ref, val_data_ref, 0, config, True))



    ray.shutdown()
    return top_models
