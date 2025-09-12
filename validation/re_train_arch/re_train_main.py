from CNN_model.cnn_re_train_fucntion import cnn_retrain_from_pt
from NN_model.nn_re_train_fucntion import nn_retrain_from_pt
from MLP_model.mlp_re_train_fucntion import mlp_retrain_from_pt
from consts import CNN_ROOT, NN_ROOT, MLP_ROOT

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

collections = ["indoor","garage","outdoor"]
databases = ["wifi_fingerprinting_data","wifi_fingerprinting_data_exponential","wifi_fingerprinting_data_extra_features_no_leak"]
database_name = {
    "wifi_fingerprinting_data": "XY_norm_FINAL",
    "wifi_fingerprinting_data_exponential": "XY_RSSI_norm_FINAL",
    "wifi_fingerprinting_data_extra_features_no_leak": "EXTRA_FEAT_FINAL",
}

data_sets = ["wifi_fingerprinting_data_raw","wifi_fingerprinting_data_exponential","wifi_fingerprinting_data","wifi_fingerprinting_data_extra_features_no_leak"]

#for current_type in re_trains.keys():
#
#    models_area_dict = re_trains[current_type]
#
#    for current_area in models_area_dict.keys():
#
#        model_path = models_area_dict[current_area]
#        collection = f"reto_grande_{current_area}"
#
#        for current_dataset in data_sets:
#            
#            model_file_name = f"{current_type}_{current_area}_{current_dataset}"
#
#            if current_type == "CNN":
#                cnn_retrain_from_pt(model_path,model_file_name,[collection],[collection],current_dataset,False,MAX_EPOCHS,BATCHSIZE,LEARNING_RATE,WEIGHT_DECAY)
#            if current_type == "NN":
#                nn_retrain_from_pt(model_path,model_file_name,[collection],[collection],current_dataset,False,MAX_EPOCHS,BATCHSIZE,LEARNING_RATE,WEIGHT_DECAY)
#            if current_type == "MLP":
#                mlp_retrain_from_pt(model_path,model_file_name,[collection],[collection],current_dataset,False,MAX_EPOCHS,BATCHSIZE,LEARNING_RATE,WEIGHT_DECAY)


# from CNN_dict import best_model_dictionary
# 
# for current_database in databases:
# 
#     for current_collection in collections: 
# 
#         database_designation = database_name[current_database]
#         best_models = best_model_dictionary[f"CNN_{database_designation}_{current_collection}"]
#         for idx, bm in enumerate(best_models):
# 
#             model_storage_dir_name = f"{CNN_ROOT}model_storage_{database_designation}_{current_collection}"
# 
#             model_path = f"{model_storage_dir_name}/{bm}.pt"
#             collection = f"reto_grande_{current_collection}"
#             model_file_name = f"CNN_{database_designation}_{current_collection}_{current_database}_{idx}"
# 
#             cnn_retrain_from_pt(model_path,model_file_name,[collection],[collection],current_database,False,MAX_EPOCHS,BATCHSIZE,LEARNING_RATE,WEIGHT_DECAY,"adamw","retrained_cnn_all_models")