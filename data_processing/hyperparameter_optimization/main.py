from gradient_search import run_model_parallel_gradient_search
from gpu_fucntion import train_model_ray
from data_processing import get_dataset, combine_arrays, shuffle_array, split_combined_data
from sklearn.model_selection import train_test_split
import torch, time, pickle, os
from config import TrainingConfig
import logging, warnings
import ray

# Configure environment
torch.set_float32_matmul_precision('high')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_DISABLE_CODE"] = "true"
os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_START_METHOD"] = "thread"

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("wandb").setLevel(logging.CRITICAL)
from pytorch_lightning.utilities import rank_zero
rank_zero._get_rank = lambda: 1

# List of all collections
all_collections = [
    "equilatero_grande_garage",
    "equilatero_grande_outdoor",
    "equilatero_medio_garage",
    "equilatero_medio_outdoor",
    "isosceles_grande_indoor",
    "isosceles_grande_outdoor",
    "isosceles_medio_outdoor",
    "obtusangulo_grande_outdoor",
    "obtusangulo_pequeno_outdoor",
    "reto_grande_garage",
    "reto_grande_indoor",
    "reto_grande_outdoor",
    "reto_medio_garage",
    "reto_medio_outdoor",
    "reto_n_quadrado_grande_indoor",
    "reto_n_quadrado_grande_outdoor",
    "reto_n_quadrado_pequeno_outdoor",
    "reto_pequeno_garage",
    "reto_pequeno_outdoor",
]

def group_by_location(collections, locations):
    result = []
    for name in collections:
        if any(loc in name for loc in locations):
            result.append(name)
    return result

def load_and_process_data(train_collections, db_name="wifi_fingerprinting_data"):
    print(f"📡 Loading training datasets: {train_collections}")
    train_datasets = [get_dataset(name, db_name) for name in train_collections]
    combined_train = combine_arrays(train_datasets)
    shuffled_train = shuffle_array(combined_train)
    X_train, y_train = split_combined_data(shuffled_train)

    print("📡 Loading validation datasets: all collections")
    val_datasets = [get_dataset(name, db_name) for name in all_collections]
    combined_val = combine_arrays(val_datasets)
    shuffled_val = shuffle_array(combined_val)
    X_val, y_val = split_combined_data(shuffled_val)

    return X_train, y_train, X_val, y_val

if __name__ == '__main__':
    try:
        config = TrainingConfig()

        experiments = {
            "outdoor_only": group_by_location(all_collections, ["outdoor"]),
            "indoor_only": group_by_location(all_collections, ["indoor"]),
            "garage_only": group_by_location(all_collections, ["garage"]),
            "outdoor_and_indoor": group_by_location(all_collections, ["outdoor", "indoor"]),
            "outdoor_and_garage": group_by_location(all_collections, ["outdoor", "garage"]),
            "outdoor_indoor_and_garage": group_by_location(all_collections, ["indoor", "outdoor", "garage"]),
            "all_data": all_collections,
        }

        if not ray.is_initialized():
            ray.init(num_cpus=24, log_to_driver=False)

        for experiment_name, train_collections in experiments.items():
            print(f"\n🔬 Starting experiment: {experiment_name}")
            X_train, y_train, X_val, y_val = load_and_process_data(train_collections)

            all_best_models = []
            print(f"\n🚀 Running {config.num_gradient_runs} independent gradient searches...")

            for run_index in range(config.num_gradient_runs):
                print(f"\n🔁 Run {run_index + 1}/{config.num_gradient_runs}")
                run_config = TrainingConfig(**vars(config))
                run_config.group_name = f"{experiment_name}_run{run_index}"

                top_models = run_model_parallel_gradient_search(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    config=run_config
                )

                best_model = top_models[0]
                print(f"✅ Best model val_loss: {best_model['val_loss']:.4f} - {best_model['name']}")
                all_best_models.append(best_model)

            result_path = os.path.join(os.getcwd(), f"best_models_{experiment_name}.pkl")
            with open(result_path, 'wb') as f:
                pickle.dump(all_best_models, f)

            print(f"\n📝 Logging {len(all_best_models)} best models to W&B under group: {experiment_name}_final_log")

            train_ref = ray.put((X_train, y_train))
            val_ref = ray.put((X_val, y_val))

            wandb_log_futures = [
                train_model_ray.remote(
                    {**model, "name": f"{model['name']}_final_log"},
                    train_ref,
                    val_ref,
                    idx,
                    TrainingConfig(group_name=f"{experiment_name}_final_log"),
                    use_wandb=True
                )
                for idx, model in enumerate(all_best_models)
            ]

            ray.get(wandb_log_futures)
            print(f"\n🏁 Finished logging experiment: {experiment_name}")

    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        raise
