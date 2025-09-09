from cnn_re_train_fucntion import cnn_retrain_from_pt
from nn_re_train_fucntion import nn_retrain_from_pt
from mlp_re_train_fucntion import mlp_retrain_from_pt

CNN_ROOT = "/home/admindi/sbenites/WirelessLocation/data_processing/hyperparameter_tunning/CNN_hyperparameter_optimization/"
NN_ROOT = "/home/admindi/sbenites/WirelessLocation/data_processing/hyperparameter_tunning/NN_hyperparameter_optimization/"
MLP_ROOT = "/home/admindi/sbenites/WirelessLocation/data_processing/hyperparameter_tunning/MLP_hyperparameter_optimization/"

re_trains = {

    "CNN":{
        "garage" : f"{CNN_ROOT}/model_storage_RAW_FINAL_garage/all_data_run4_depth9_model10.pt",
        "outdoor" : f"{CNN_ROOT}/model_storage_RAW_FINAL_outdoor/all_data_run2_depth4_model6.pt",
        "indoor" : f"{CNN_ROOT}/model_storage_RAW_FINAL_indoor/all_data_run2_depth3_model0.pt"
    },

    "NN":{
        "garage" : f"{NN_ROOT}/model_storage_RAW_FINAL_garage/all_data_run4_depth0_model8.pt",
        "outdoor" : f"{NN_ROOT}/model_storage_RAW_FINAL_outdoor/all_data_run1_depth8_model3.pt",
        "indoor" : f"{NN_ROOT}/model_storage_RAW_FINAL_indoor/all_data_run4_depth0_model5.pt"
    },


    "MLP":{
        "garage" : f"{MLP_ROOT}/model_storage_RAW_FINAL_garage/all_data_run4_depth6_model11.pt",
        "outdoor" : f"{MLP_ROOT}/model_storage_RAW_FINAL_outdoor/all_data_run4_depth7_model1.pt",
        "indoor" : f"{MLP_ROOT}/model_storage_RAW_FINAL_indoor/all_data_run4_depth2_model5.pt"
    },

}

data_sets = ["wifi_fingerprinting_data_raw","wifi_fingerprinting_data_exponential","wifi_fingerprinting_data","wifi_fingerprinting_data_extra_features_no_leak"]
BATCHSIZE = 2048
MAX_EPOCHS = 50
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0

for current_type in re_trains.keys():

    models_area_dict = re_trains[current_type]

    for current_area in models_area_dict.keys():

        model_path = models_area_dict[current_area]
        collection = f"reto_grande_{current_area}"

        for current_dataset in data_sets:
            
            model_file_name = f"{current_type}_{current_area}_{current_dataset}"

            if current_type == "CNN":
                cnn_retrain_from_pt(model_path,model_file_name,[collection],[collection],current_dataset,False,MAX_EPOCHS,BATCHSIZE,LEARNING_RATE,WEIGHT_DECAY)
            if current_type == "NN":
                nn_retrain_from_pt(model_path,model_file_name,[collection],[collection],current_dataset,False,MAX_EPOCHS,BATCHSIZE,LEARNING_RATE,WEIGHT_DECAY)
            if current_type == "MLP":
                mlp_retrain_from_pt(model_path,model_file_name,[collection],[collection],current_dataset,False,MAX_EPOCHS,BATCHSIZE,LEARNING_RATE,WEIGHT_DECAY)

