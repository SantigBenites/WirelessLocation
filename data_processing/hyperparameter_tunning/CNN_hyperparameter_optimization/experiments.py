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

experiments={

    ## METERS EXPERIMENT
    "CNN_reto_grande_outdoor_meters" : {
        "db_name" : "wifi_fingerprinting_data_meters",
        "group_name" : "CNN_DATA_ANALYSIS_reto_grande_outdoor_meters",
        "model_save_dir" : "CNN_DATA_ANALYSIS/model_storage_reto_grande_outdoor_meters",
        "experiments" : {
            "reto_grande_outdoor_meters": (["reto_grande_outdoor"],["reto_grande_outdoor"]),
        }
    },
    "CNN_reto_grande_indoor_meters" : {
        "db_name" : "wifi_fingerprinting_data_meters",
        "group_name" : "CNN_DATA_ANALYSIS_reto_grande_indoor_meters",
        "model_save_dir" : "CNN_DATA_ANALYSIS/model_storage_reto_grande_indoor_meters",
        "experiments" : {
            "reto_grande_indoor_meters": (["reto_grande_indoor"],["reto_grande_indoor"]),
        }
    },
    "CNN_reto_grande_garage_meters" : {
        "db_name" : "wifi_fingerprinting_data_meters",
        "group_name" : "CNN_DATA_ANALYSIS_reto_grande_garage_meters",
        "model_save_dir" : "CNN_DATA_ANALYSIS/model_storage_reto_grande_garage_meters",
        "experiments" : {
            "reto_grande_garage_meters": (["reto_grande_garage"],["reto_grande_garage"]),
        }
    },

    # RAW EXPERIMENT
    "CNN_reto_grande_outdoor_raw" : {
        "db_name" : "wifi_fingerprinting_data_raw",
        "group_name" : "CNN_DATA_ANALYSIS_reto_grande_outdoor_raw",
        "model_save_dir" : "CNN_DATA_ANALYSIS/model_storage_reto_grande_outdoor_raw",
        "experiments" : {
            "reto_grande_outdoor_raw": (["reto_grande_outdoor"],["reto_grande_outdoor"]),
        }
    },
    "CNN_reto_grande_indoor_raw" : {
        "db_name" : "wifi_fingerprinting_data_raw",
        "group_name" : "CNN_DATA_ANALYSIS_reto_grande_indoor_raw",
        "model_save_dir" : "CNN_DATA_ANALYSIS/model_storage_reto_grande_indoor_raw",
        "experiments" : {
            "reto_grande_indoor_raw": (["reto_grande_indoor"],["reto_grande_indoor"]),
        }
    },
    "CNN_reto_grande_garage_raw" : {
        "db_name" : "wifi_fingerprinting_data_raw",
        "group_name" : "CNN_DATA_ANALYSIS_reto_grande_garage_raw",
        "model_save_dir" : "CNN_DATA_ANALYSIS/model_storage_reto_grande_garage_raw",
        "experiments" : {
            "reto_grande_garage_raw": (["reto_grande_garage"],["reto_grande_garage"]),
        }
    },

    # Exponential
    "CNN_reto_grande_outdoor_exponential" : {
        "db_name" : "wifi_fingerprinting_data_exponential",
        "group_name" : "CNN_DATA_ANALYSIS_reto_grande_outdoor_exponential",
        "model_save_dir" : "CNN_DATA_ANALYSIS/model_storage_reto_grande_outdoor_exponential",
        "experiments" : {
            "reto_grande_outdoor_exponential": (["reto_grande_outdoor"],["reto_grande_outdoor"]),
        }
    },
    "CNN_reto_grande_indoor_exponential" : {
        "db_name" : "wifi_fingerprinting_data_exponential",
        "group_name" : "CNN_DATA_ANALYSIS_reto_grande_indoor_exponential",
        "model_save_dir" : "CNN_DATA_ANALYSIS/model_storage_reto_grande_indoor_exponential",
        "experiments" : {
            "reto_grande_indoor_exponential": (["reto_grande_indoor"],["reto_grande_indoor"]),
        }
    },
    "CNN_reto_grande_garage_exponential" : {
        "db_name" : "wifi_fingerprinting_data_exponential",
        "group_name" : "CNN_DATA_ANALYSIS_reto_grande_garage_exponential",
        "model_save_dir" : "CNN_DATA_ANALYSIS/model_storage_reto_grande_garage_exponential",
        "experiments" : {
            "reto_grande_garage_exponential": (["reto_grande_garage"],["reto_grande_garage"]),
        }
    },

    # Exponential spaces

    "CNN_all_spaces_reto_grande" : {
        "db_name" : "wifi_fingerprinting_data_exponential",
        "group_name" : "CNN_DATA_ANALYSIS_all_spaces_reto_grande",
        "model_save_dir" : "CNN_DATA_ANALYSIS/model_storage_all_spaces_reto_grande",
        "experiments" : {
            "all_spaces_reto_grande": (["reto_grande_outdoor","reto_grande_garage","reto_grande_indoor"],["reto_grande_outdoor","reto_grande_garage","reto_grande_indoor"]),
        }
    },

    # Exponential combinations

    "CNN_all_collections_indoor_outdoor" : {
        "db_name" : "wifi_fingerprinting_data_exponential",
        "group_name" : "CNN_DATA_ANALYSIS_all_collections_indoor_outdoor",
        "model_save_dir" : "CNN_DATA_ANALYSIS/model_storage_all_collections_indoor_outdoor",
        "experiments" : {
            "all_collections_indoor_outdoor": (group_by_location(all_collections, ["indoor", "outdoor"]),group_by_location(all_collections, ["indoor", "outdoor"])),
        }
    },
    
    "CNN_all_collections_indoor_garage" : {
        "db_name" : "wifi_fingerprinting_data_exponential",
        "group_name" : "CNN_DATA_ANALYSIS_all_collections_indoor_garage",
        "model_save_dir" : "CNN_DATA_ANALYSIS/model_storage_all_collections_indoor_garage",
        "experiments" : {
            "all_collections_indoor_garage": (group_by_location(all_collections, ["indoor", "garage"]),group_by_location(all_collections, ["indoor", "garage"])),
        }
    },

    "CNN_all_collections_garage_outdoor" : {
        "db_name" : "wifi_fingerprinting_data_exponential",
        "group_name" : "CNN_DATA_ANALYSIS_all_collections_garage_outdoor",
        "model_save_dir" : "CNN_DATA_ANALYSIS/model_storage_all_collections_garage_outdoor",
        "experiments" : {
            "all_collections_garage_outdoor": (group_by_location(all_collections, ["garage", "outdoor"]),group_by_location(all_collections, ["garage", "outdoor"])),
        }
    },
}