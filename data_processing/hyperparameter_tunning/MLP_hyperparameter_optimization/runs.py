from typing import Dict, List, Any, Tuple


runs: List[Dict[str, Any]] = [
    {
        "collections": ["reto_grande_garage"],
        "database": "wifi_fingerprinting_data_meters",
        "model_name" : "1triangle_1space_garage"
    },
    {
        "collections": ["reto_grande_indoor"],
        "database": "wifi_fingerprinting_data_meters",
        "model_name" : "1triangle_1space_indoor"
    },
    {
        "collections": ["reto_grande_outdoor"],
        "database": "wifi_fingerprinting_data_meters",
        "model_name" : "1triangle_1space_outdoor"
    },
    {
        "collections": ["reto_grande_garage","reto_grande_indoor","reto_grande_outdoor"],
        "database": "wifi_fingerprinting_data_exponential",
        "model_name" : "1triangle_all_spaces"
    },
    {
        "collections": [
            "equilatero_grande_outdoor",
            "equilatero_medio_outdoor",
            "isosceles_grande_outdoor",
            "isosceles_medio_outdoor",
            "obtusangulo_grande_outdoor",
            "obtusangulo_pequeno_outdoor",
            "reto_grande_outdoor",
            "reto_medio_outdoor",
            "reto_n_quadrado_grande_outdoor",
            "reto_n_quadrado_pequeno_outdoor",
            "reto_pequeno_outdoor",
        ],
        "database": "wifi_fingerprinting_data_exponential",
        "model_name" : "all_triangles_1space_outdoor"
    },
    {
        "collections": [
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
                    ],
        "database": "wifi_fingerprinting_data_meters",
        "model_name" : "all_triangles_all_spaces"
    },
]