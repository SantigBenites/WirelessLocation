from typing import Dict, List, Any, Tuple


runs: List[Dict[str, Any]] = [
    {
        "collections": ["reto_pequeno_garage"],
        "database": "wifi_fingerprinting_data_meters",
        "model_name" : "1triangle_1space_garage_small",
        "scale" : 1
    },
    {
        "collections": ["reto_pequeno_outdoor"],
        "database": "wifi_fingerprinting_data_meters",
        "model_name" : "1triangle_1space_garage_small",
        "scale" : 1
    },
    {
        "collections": ["reto_pequeno_outdoor","reto_pequeno_garage"],
        "database": "wifi_fingerprinting_data_meters",
        "model_name" : "1triangle_2space_garage_outdoor",
        "scale" : 1
    },
    {
        "collections": ["reto_grande_garage","reto_grande_indoor","reto_grande_outdoor"],
        "database": "wifi_fingerprinting_data_meters",
        "model_name" : "1triangle_all_spaces_meters",
        "scale" : 1
    },
]