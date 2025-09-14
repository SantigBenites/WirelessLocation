best_model_dictionary = {
    "CNN_EXTRA_FEAT_FINAL_outdoor" : [
        "all_data_run4_depth8_model1",
        "all_data_run3_depth6_model8",
        "all_data_run2_depth7_model4",
        "all_data_run1_depth1_model1",
        "all_data_run0_depth9_model2",
    ],

    "CNN_EXTRA_FEAT_FINAL_garage" : [
        "all_data_run4_depth3_model10",
        "all_data_run3_depth6_model8",
        "all_data_run2_depth3_model9",
        "all_data_run1_depth9_model4",
        "all_data_run0_depth6_model11",
    ],

    "CNN_EXTRA_FEAT_FINAL_indoor" : [
        "all_data_run4_depth2_model11",
        "all_data_run3_depth9_model2",
        "all_data_run2_depth9_model10",
        "all_data_run1_depth9_model2",
        "all_data_run0_depth5_model6",
    ],

    "CNN_XY_RSSI_norm_FINAL_outdoor" : [
        "all_data_run4_depth1_model5",
        "all_data_run3_depth6_model4",
        "all_data_run2_depth4_model6",
        "all_data_run1_depth5_model9",
        "all_data_run0_depth6_model1",
    ],

    "CNN_XY_RSSI_norm_FINAL_garage" : [
        "all_data_run4_depth9_model11",
        "all_data_run3_depth7_model9",
        "all_data_run2_depth7_model4",
        "all_data_run1_depth9_model0",
        "all_data_run0_depth5_model4",
    ],

    "CNN_XY_RSSI_norm_FINAL_indoor" : [
        "all_data_run4_depth8_model10",
        "all_data_run3_depth5_model11",
        "all_data_run2_depth8_model6",
        "all_data_run1_depth5_model7",
        "all_data_run0_depth5_model7",
    ],

    "CNN_XY_norm_FINAL_outdoor" : [
        "all_data_run4_depth7_model7",
        "all_data_run3_depth9_model3",
        "all_data_run2_depth3_model9",
        "all_data_run1_depth1_model10",
        "all_data_run0_depth6_model5",
    ],

    "CNN_XY_norm_FINAL_garage" : [
        "all_data_run4_depth7_model7",
        "all_data_run3_depth3_model9",
        "all_data_run2_depth8_model1",
        "all_data_run1_depth9_model8",
        "all_data_run0_depth7_model3",
    ],

    "CNN_XY_norm_FINAL_indoor" : [
        "all_data_run4_depth8_model0",
        "all_data_run3_depth6_model6",
        "all_data_run2_depth6_model10",
        "all_data_run1_depth7_model7",
        "all_data_run0_depth6_model10",
    ],

}

CNN_ROOT = "/home/admindi/sbenites/WirelessLocation/data_processing/hyperparameter_tunning/CNN_hyperparameter_optimization/"
NN_ROOT = "/home/admindi/sbenites/WirelessLocation/data_processing/hyperparameter_tunning/NN_hyperparameter_optimization/"
MLP_ROOT = "/home/admindi/sbenites/WirelessLocation/data_processing/hyperparameter_tunning/MLP_hyperparameter_optimization/"

BATCHSIZE = 2048
MAX_EPOCHS = 50
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0

group_data = {
"CNN_only_relative_xy" : 
    {"model_storage" : f"{CNN_ROOT}model_storage_only_relative_xy",
     "database" : "wifi_fingerprinting_data",
     "relative_coeficient" : 32},
"CNN_extra_features_no_leak_xy_free" : 
    {"model_storage" : f"{CNN_ROOT}model_storage_extra_features_run_no_leak_xy_free",
     "database" : "wifi_fingerprinting_data_extra_features_no_leak",
     "relative_coeficient" : 32},
"CNN_extra_features_no_leaking" : 
    {"model_storage" : f"{CNN_ROOT}model_storage_extra_features_run_no_leaking",
     "database" : "wifi_fingerprinting_data_extra_features",
     "relative_coeficient" : 32},
"CNN_final_raw" : 
    {"model_storage" : f"{CNN_ROOT}model_storage_final_raw",
     "database" : "wifi_fingerprinting_data_raw",
     "relative_coeficient" : 3.2},
"CNN_final_meters" : 
    {"model_storage" : f"{CNN_ROOT}model_storage_final_meters",
     "database" : "wifi_fingerprinting_data_meters",
     "relative_coeficient" : 1},
"CNN_final_exponential" : 
    {"model_storage" : f"{CNN_ROOT}model_storage_final_exponential",
     "database" : "wifi_fingerprinting_data_exponential",
     "relative_coeficient" : 32},
}

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
