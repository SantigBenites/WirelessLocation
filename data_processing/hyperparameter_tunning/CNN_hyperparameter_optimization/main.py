
from gradient_search import run_model_parallel_gradient_search
from data_processing import get_dataset, combine_arrays, shuffle_array, split_combined_data, get_feature_list
from sklearn.model_selection import train_test_split
import torch, time, pickle, os
from config import TrainingConfig
import logging, warnings
import multiprocessing
import logging
import ray
from typing import Dict
from experiments import group_by_location,all_collections,experiments

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


def load_and_process_data(train_collections,val_collections, db_name):
    # Resolve which features to use for this DB (preset name or explicit list)
    feature_list = get_feature_list(db_name)

    print(f"🧰 Database in use: {db_name}")
    # Uncomment to see the exact feature order:
    print("Features:", feature_list)

    # ---- Training data
    print(f"📡 Loading training datasets: {train_collections}")
    train_datasets = [get_dataset(name, db_name, feature_list) for name in train_collections]
    combined_train = combine_arrays(train_datasets)
    shuffled_train = shuffle_array(combined_train)
    X_train, y_train = split_combined_data(shuffled_train, feature_list)

    # ---- Validation data
    print("📡 Loading validation datasets: all collections")
    val_datasets = [get_dataset(name, db_name, feature_list) for name in val_collections]
    combined_val = combine_arrays(val_datasets)
    shuffled_val = shuffle_array(combined_val)
    X_val, y_val = split_combined_data(shuffled_val, feature_list)

    print(f"📊 Final shapes -> X_train: {X_train.shape}, y_train: {y_train.shape}, "
          f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    return X_train, y_train, X_val, y_val



def singular_run(config:TrainingConfig, experiments):
    try:

        for experiment_name, collections in experiments.items():
            print(f"🔬 Starting experiment: {experiment_name}")
            train_collections = collections[0]
            val_collections = collections[1]
            X_train, y_train, X_val, y_val = load_and_process_data(train_collections, val_collections, config.db_name)

            all_best_models = []
            print(f"🚀 Running {config.num_gradient_runs} independent gradient searches...")

            for run_index in range(config.num_gradient_runs):
                print(f"🔁 Run {run_index + 1}/{config.num_gradient_runs}")
                run_config = TrainingConfig(**vars(config))
                run_config.experiment_name = experiment_name
                run_config.run_index = run_index

                top_models = run_model_parallel_gradient_search(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    config=run_config
                )

                best_model = top_models[0]
                print(f"✅ Best model val_mse: {best_model['val_mse']:.4f} - {best_model['name']}")
                all_best_models.append(best_model)

            result_path = os.path.join(os.getcwd(), f"best_models_{experiment_name}.pkl")
            with open(result_path, 'wb') as f:
                pickle.dump(all_best_models, f)

            print(f"📦 Saved {len(all_best_models)} best models for experiment: {experiment_name}")

    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        raise


def run_pipeline():
    collections = ["indoor","garage","outdoor"]
    databases = ["wifi_fingerprinting_data","wifi_fingerprinting_data_exponential","wifi_fingerprinting_data_extra_features_no_leak"]
    database_name = {
        "wifi_fingerprinting_data": "XY_norm_FINAL",
        "wifi_fingerprinting_data_exponential": "XY_RSSI_norm_FINAL",
        "wifi_fingerprinting_data_extra_features_no_leak": "EXTRA_FEAT_FINAL",
    }


    for current_database in databases:

        for current_collection in collections: 

            print(f"Current Database {current_database} with collection {current_collection}")

            current_config = TrainingConfig()
            current_config.db_name = current_database
            current_config.group_name = f"CNN_{database_name[current_database]}_{current_collection}"
            current_config.model_save_dir = f"model_storage_{database_name[current_database]}_{current_collection}"
            print(f"🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩 RUN START")
            singular_run(current_config)
            print(f"🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥RUN END")


def one_run_pipeline():
    current_config = TrainingConfig()
    current_config.db_name = "wifi_fingerprinting_data_extra_features_no_leak"
    current_config.group_name = f"CNN_DELTA_FINAL_outdoor"
    current_config.model_save_dir = f"model_storage_DELTA_FINAL_outdoor"
    print(f"🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩 RUN START")
    singular_run(current_config)
    print(f"🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥RUN END")


def missing_experiment_pipeline():


    for experiment in experiments.keys():

        config = experiments[experiment]
        current_config = TrainingConfig()
        current_config.db_name = config["db_name"]
        current_config.group_name = config["group_name"]
        current_config.model_save_dir = config["model_save_dir"]
        current_experiment = config["experiments"]
        print(f"🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩 RUN START")
        #print(f"Running experimetn with {current_config.db_name}, {current_config.group_name}, {current_config.model_save_dir}, {current_experiment}")
        singular_run(current_config,current_experiment)
        print(f"🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥RUN END")

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=True)
    # Suppress Ray internal logs
    logging.getLogger("ray").setLevel(logging.ERROR)
    missing_experiment_pipeline()


#experiments = {
#    #"outdoor_only": group_by_location(all_collections, ["outdoor"]),
#    #"indoor_only": group_by_location(all_collections, ["indoor"]),
#    #"garage_only": group_by_location(all_collections, ["garage"]),
#    #"outdoor_and_indoor": group_by_location(all_collections, ["outdoor", "indoor"]),
#    #"outdoor_and_garage": group_by_location(all_collections, ["outdoor", "garage"]),
#    #"garage_and_indoor": group_by_location(all_collections, ["garage", "indoor"])
#    #"outdoor_indoor_and_garage": group_by_location(all_collections, ["indoor", "outdoor", "garage"]),
#    #"all_data": all_collections,
#}