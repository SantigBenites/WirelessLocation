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

def load_and_process_data():
    print("📡 Loading and processing data...")
    datasets = [
        get_dataset("wifi_data_equilatero_grande", "wifi_data_db"),
        get_dataset("wifi_data_equilatero_medio", "wifi_data_db"),
        get_dataset("wifi_data_reto_grande", "wifi_data_db"),
        get_dataset("wifi_data_reto_pequeno", "wifi_data_db"),
        get_dataset("wifi_data_reto_medio", "wifi_data_db")
    ]
    combined_data = combine_arrays(datasets)
    shuffled_data = shuffle_array(combined_data)
    global_array_x, global_array_y = split_combined_data(shuffled_data)
    return global_array_x, global_array_y

if __name__ == '__main__':
    try:
        config = TrainingConfig()
        X, y = load_and_process_data()
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state
        )

        if not ray.is_initialized():
            ray.init(
                num_cpus = 48,
                log_to_driver=False
                )

        all_best_models = []
        print(f"\n🚀 Running {config.num_gradient_runs} independent gradient searches...")

        for run_index in range(config.num_gradient_runs):
            print(f"\n🔁 Run {run_index + 1}/{config.num_gradient_runs}")
            run_config = TrainingConfig(**vars(config))
            run_config.group_name = f"{config.group_name}_run{run_index}"

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

        result_path = os.path.join(os.getcwd(), f"best_models_{config.group_name}.pkl")
        with open(result_path, 'wb') as f:
            pickle.dump(all_best_models, f)

        print(f"\n📝 Logging {len(all_best_models)} best models to W&B under group: {config.log_best_group}")
        train_ref = ray.put((X_train, y_train))
        val_ref = ray.put((X_val, y_val))

        wandb_log_futures = [
            train_model_ray.remote(
                {**model, "name": f"{model['name']}_final_log"},
                train_ref,
                val_ref,
                idx,
                TrainingConfig(group_name=config.log_best_group),
                use_wandb=True
            )
            for idx, model in enumerate(all_best_models)
        ]

        ray.get(wandb_log_futures)
        print(f"\n🏁 All best models logged and saved to: {result_path}")

    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        raise
