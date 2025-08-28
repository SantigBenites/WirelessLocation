from grid_search_main import run_model_parallel_grid_search
from gradient_search import run_model_parallel_gradient_search
from monitoring import generate_comprehensive_summary
from data_processing import get_dataset,combine_arrays,shuffle_array,split_combined_data
from sklearn.model_selection import train_test_split
import torch, time, pickle, os


if __name__ == '__main__':
    try:
        # Get and process data
        print("Loading and processing data...")
        datasets = [
            get_dataset("wifi_data_reto_grande", "wifi_data_db"),
            get_dataset("wifi_data_reto_pequeno", "wifi_data_db"),
            get_dataset("wifi_data_reto_medio", "wifi_data_db")
        ]
        
        combined_data = combine_arrays(datasets)
        shuffled_data = shuffle_array(combined_data)
        global_array_x, global_array_y = split_combined_data(shuffled_data)
        X_train, X_val, y_train, y_val = train_test_split(
            global_array_x, global_array_y, test_size=0.2, random_state=42
        )



        print(f"\nStarting training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using {torch.cuda.device_count()} GPUs")
        print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

        SEARCH = "GRADIENT"


        if SEARCH == 'GRID':
            # Run training
            results = run_model_parallel_grid_search(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=50
            )
        elif SEARCH == 'GRADIENT':
            results = run_model_parallel_gradient_search(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=20,
            training_depth=5
            )

            # The best model results are in results['final_result']
            best_model_result = results['final_result']
        else:
            raise RuntimeError("No valid search")
        
        d = { "abc" : [1, 2, 3], "qwerty" : [4,5,6] }
        with open(f"{os.getcwd()}/results", 'wb') as f:
            pickle.dump(d, f)
        
        if not results:
            raise RuntimeError("No models were successfully trained.")
        
        # Generate final summary
        summary = generate_comprehensive_summary(results)
        
        print(f"\nTraining completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        if summary:
            print(f"Best model: {summary['best_model']} with validation RMSE: {summary['best_val_rmse']:.4f}")
        else:
            print("Warning: No summary was generated.")
            
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise