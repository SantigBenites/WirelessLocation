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
    return [name for name in collections if any(loc in name for loc in locations)]

CNN_ROOT = "/home/admindi/sbenites/WirelessLocation/data_processing/hyperparameter_tunning/CNN_hyperparameter_optimization/"
NN_ROOT = "/home/admindi/sbenites/WirelessLocation/data_processing/hyperparameter_tunning/NN_hyperparameter_optimization/"
MLP_ROOT = "/home/admindi/sbenites/WirelessLocation/data_processing/hyperparameter_tunning/MLP_hyperparameter_optimization/"
CNN_DATA_ROOT = "/home/admindi/sbenites/WirelessLocation/data_processing/hyperparameter_tunning/CNN_hyperparameter_optimization/CNN_DATA_ANALYSIS/"

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
"CNN_final_delta" :
    {"model_storage" : f"{CNN_ROOT}model_storage_final_delta",
     "database" : "wifi_fingerprinting_data_extra_features_no_leak",
     "relative_coeficient" : 32},
}
# ANALYSIS

CNN_validation_collections = {  "reto_grande_outdoor"   : ["reto_grande_outdoor"],
                                "reto_grande_indoor"    : ["reto_grande_indoor"],
                                "reto_grande_garage"    : ["reto_grande_garage"],
                                "all_reto_grande"       : ["reto_grande_outdoor","reto_grande_indoor","reto_grande_garage"]}


CNN_space_combinations = {"all_collections_garage"                  : group_by_location(all_collections, ["garage"]),
                          "all_collections_outdoor"                 : group_by_location(all_collections, ["outdoor"]),
                          "all_collections_indoor"                  : group_by_location(all_collections, ["indoor"]),
                          "all_collections_garage_outdoor"          : group_by_location(all_collections, ["garage", "outdoor"]),
                          "all_collections_garage_indoor"           : group_by_location(all_collections, ["garage", "indoor"]),
                          "all_collections_indoor_outdoor"          : group_by_location(all_collections, ["indoor", "outdoor"]),
                          "all_collections_garage_outdoor_indoor"   : group_by_location(all_collections, ["garage", "outdoor","indoor"])}


cnn_group_data = {
"CNN_DATA_ANALYSIS_reto_grande_garage_exponential":
    {"model_storage" : f"{CNN_DATA_ROOT}model_storage_reto_grande_garage_exponential",
     "database" : "wifi_fingerprinting_data_exponential",
     "relative_coeficient" : 32,
     "validation_collections" : CNN_validation_collections},

"CNN_DATA_ANALYSIS_reto_grande_indoor_exponential":
    {"model_storage" : f"{CNN_DATA_ROOT}model_storage_reto_grande_indoor_exponential",
     "database" : "wifi_fingerprinting_data_exponential",
     "relative_coeficient" : 32,
     "validation_collections" : CNN_validation_collections},

"CNN_DATA_ANALYSIS_reto_grande_outdoor_exponential":
    {"model_storage" : f"{CNN_DATA_ROOT}model_storage_reto_grande_outdoor_exponential",
     "database" : "wifi_fingerprinting_data_exponential",
     "relative_coeficient" : 32,
     "validation_collections" : CNN_validation_collections},

"CNN_DATA_ANALYSIS_reto_grande_garage_raw":
    {"model_storage" : f"{CNN_DATA_ROOT}model_storage_reto_grande_garage_raw",
     "database" : "wifi_fingerprinting_data_raw",
     "relative_coeficient" : 3.2,
     "validation_collections" : CNN_validation_collections},

"CNN_DATA_ANALYSIS_reto_grande_indoor_raw":
    {"model_storage" : f"{CNN_DATA_ROOT}model_storage_reto_grande_indoor_raw",
     "database" : "wifi_fingerprinting_data_raw",
     "relative_coeficient" : 3.2,
     "validation_collections" : CNN_validation_collections},

"CNN_DATA_ANALYSIS_reto_grande_outdoor_raw":
    {"model_storage" : f"{CNN_DATA_ROOT}model_storage_reto_grande_outdoor_raw",
     "database" : "wifi_fingerprinting_data_raw",
     "relative_coeficient" : 3.2,
     "validation_collections" : CNN_validation_collections},

"CNN_DATA_ANALYSIS_reto_grande_garage_meters":
    {"model_storage" : f"{CNN_DATA_ROOT}model_storage_reto_grande_garage_meters",
     "database" : "wifi_fingerprinting_data_meters",
     "relative_coeficient" : 1,
     "validation_collections" : CNN_validation_collections},

"CNN_DATA_ANALYSIS_reto_grande_indoor_meters":
    {"model_storage" : f"{CNN_DATA_ROOT}model_storage_reto_grande_indoor_meters",
     "database" : "wifi_fingerprinting_data_meters",
     "relative_coeficient" : 1,
     "validation_collections" : CNN_validation_collections},

"CNN_DATA_ANALYSIS_reto_grande_outdoor_meters":
    {"model_storage" : f"{CNN_DATA_ROOT}model_storage_reto_grande_outdoor_meters",
     "database" : "wifi_fingerprinting_data_meters",
     "relative_coeficient" : 1,
     "validation_collections" : CNN_validation_collections},

"CNN_DATA_ANALYSIS_all_collections_garage_outdoor":
    {"model_storage" : f"{CNN_DATA_ROOT}model_storage_all_collections_garage_outdoor",
     "database" : "wifi_fingerprinting_data_exponential",
     "relative_coeficient" : 32,
     "validation_collections" : CNN_space_combinations},
    
"CNN_DATA_ANALYSIS_all_collections_indoor_garage":
    {"model_storage" : f"{CNN_DATA_ROOT}model_storage_all_collections_indoor_garage",
     "database" : "wifi_fingerprinting_data_exponential",
     "relative_coeficient" : 32,
     "validation_collections" : CNN_space_combinations},
    
"CNN_DATA_ANALYSIS_all_collections_indoor_outdoor":
    {"model_storage" : f"{CNN_DATA_ROOT}model_storage_all_collections_indoor_outdoor",
     "database" : "wifi_fingerprinting_data_exponential",
     "relative_coeficient" : 32,
     "validation_collections" : CNN_space_combinations},

"CNN_DATA_ANALYSIS_all_collections_garage_outdoor_indoor":
    {"model_storage" : f"{CNN_DATA_ROOT}model_storage_all_collections_garage_outdoor_indoor",
     "database" : "wifi_fingerprinting_data_exponential",
     "relative_coeficient" : 32,
     "validation_collections" : CNN_space_combinations},

"CNN_DATA_ANALYSIS_all_spaces_reto_grande":
    {"model_storage" : f"{CNN_DATA_ROOT}model_storage_all_spaces_reto_grande",
     "database" : "wifi_fingerprinting_data_exponential",
     "relative_coeficient" : 32,
     "validation_collections" : CNN_validation_collections},
}

