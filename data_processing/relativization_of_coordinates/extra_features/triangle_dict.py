from datetime import datetime


ap_mapping = {
    "freind1": ["ec:01:d5:2b:5f:e0", "ec:01:d5:2b:5f:e1"],
    "freind2": ["ec:01:d5:27:1d:00", "ec:01:d5:27:1d:01"],
    "freind3": ["ec:01:d5:28:fa:c0", "ec:01:d5:28:fa:c1"]
}

triangle_dictionary = {
    "reto_grande_outdoor": {
        "start":datetime(2025, 5, 13, 20, 10),
        "end":datetime(2025, 5, 13, 21, 42),
        "db": "wifi_data_db",
        "collection": "wifi_client_data_global",
        "ap_positions":{
            "freind1" : (0,0),
            "freind2" : (4,0),
            "freind3" : (0,4)
        }
    },
    "reto_medio_outdoor": {
        "start":datetime(2025, 5, 13, 21, 46),
        "end":datetime(2025, 5, 13, 22, 49),
        "db": "wifi_data_db",
        "collection": "wifi_client_data_global",
        "ap_positions":{
            "freind1" : (1,1),
            "freind2" : (3,1),
            "freind3" : (1,3)
        }
    },
    "reto_pequeno_outdoor": {
        "start":datetime(2025, 5, 13, 22, 51),
        "end":datetime(2025, 5, 13, 23, 53),
        "db": "wifi_data_db",
        "collection": "wifi_client_data_global",
        "ap_positions":{
            "freind1" : (1,1),
            "freind2" : (2,1),
            "freind3" : (1,2)
        }
    },
    "equilatero_grande_outdoor": {
        "start":datetime(2025, 6, 28, 19, 45),
        "end":datetime(2025, 6, 28, 21, 15),
        "db": "wifi_data_db",
        "collection": "wifi_client_data_global",
        "ap_positions":{
            "freind1" : (0,0),
            "freind2" : (4,0),
            "freind3" : (2,4)
        }
    },
    "equilatero_medio_outdoor": {
        "start":datetime(2025, 6, 28, 22, 5),
        "end":datetime(2025, 6, 28, 23, 30),
        "db": "wifi_data_db",
        "collection": "wifi_client_data_global",
        "ap_positions":{
            "freind1" : (1,1),
            "freind2" : (2,1),
            "freind3" : (2,3)
        }
    },
    "isosceles_grande_outdoor": {
        "start":datetime(2025, 7, 5, 12, 20),
        "end":datetime(2025, 7, 5, 13, 10),
        "db": "wifi_data_db",
        "collection": "wifi_client_data_global",
        "ap_positions":{
            "freind1" : (0,1),
            "freind2" : (4,1),
            "freind3" : (3,2)
        }
    },
    "isosceles_medio_outdoor": {
        "start":datetime(2025, 7, 5, 13, 24),
        "end":datetime(2025, 7, 5, 14, 30),
        "db": "wifi_data_db",
        "collection": "wifi_client_data_global",
        "ap_positions":{
            "freind1" : (1,1),
            "freind2" : (3,1),
            "freind3" : (2,2)
        }
    },
    "reto_n_quadrado_grande_outdoor": {
        "start":datetime(2025, 7, 5, 15, 4),
        "end":datetime(2025, 7, 5, 15, 54),
        "db": "wifi_data_db",
        "collection": "wifi_client_data_global",
        "ap_positions":{
            "freind1" : (0,1),
            "freind2" : (4,1),
            "freind3" : (4,3)
        }
    },
    "reto_n_quadrado_pequeno_outdoor": {
        "start":datetime(2025, 7, 5, 15, 55),
        "end":datetime(2025, 7, 5, 16, 42),
        "db": "wifi_data_db",
        "collection": "wifi_client_data_global",
        "ap_positions":{
            "freind1" : (1,1),
            "freind2" : (3,1),
            "freind3" : (3,2)
        }
    },
    "obtusangulo_grande_outdoor": {
        "start":datetime(2025, 7, 5, 16, 43),
        "end":datetime(2025, 7, 5, 17, 29),
        "db": "wifi_data_db",
        "collection": "wifi_client_data_global",
        "ap_positions":{
            "freind1" : (0,1),
            "freind2" : (2,1),
            "freind3" : (4,3)
        }
    },
    "obtusangulo_pequeno_outdoor": {
        "start":datetime(2025, 7, 5, 17, 30),
        "end":datetime(2025, 7, 5, 19, 00),
        "db": "wifi_data_db",
        "collection": "wifi_client_data_global",
        "ap_positions":{
            "freind1" : (1,1),
            "freind2" : (2,1),
            "freind3" : (3,2)
        }
    },
    "reto_grande_garage": {
        "start": datetime(2025, 7, 19, 11, 27),
        "end": datetime(2025, 7, 19, 12, 29),
        "db": "wifi_data_db_garage",
        "collection": "wifi_client_data_garage",
        "ap_positions":{
            "freind1" : (0,0),
            "freind2" : (4,0),
            "freind3" : (0,4)
        }
    },
    "reto_medio_garage": {
        "start": datetime(2025, 7, 19, 12, 31),
        "end": datetime(2025, 7, 19, 14, 7),
        "db": "wifi_data_db_garage",
        "collection": "wifi_client_data_garage",
        "ap_positions":{
            "freind1" : (1,1),
            "freind2" : (3,1),
            "freind3" : (1,3)
        }
    },
    "reto_pequeno_garage": {
        "start": datetime(2025, 7, 19, 14, 11),
        "end": datetime(2025, 7, 19, 15, 4),
        "db": "wifi_data_db_garage",
        "collection": "wifi_client_data_garage",
        "ap_positions":{
            "freind1" : (1,1),
            "freind2" : (2,1),
            "freind3" : (1,2)
        }
    },
    "equilatero_grande_garage": {
        "start": datetime(2025, 7, 19, 15, 5),
        "end": datetime(2025, 7, 19, 15, 50),
        "db": "wifi_data_db_garage",
        "collection": "wifi_client_data_garage",
        "ap_positions":{
            "freind1" : (0,0),
            "freind2" : (4,0),
            "freind3" : (2,4)
        }
    },
    "equilatero_medio_garage": {
        "start": datetime(2025, 7, 19, 15, 58),
        "end": datetime(2025, 7, 19, 18, 0),
        "db": "wifi_data_db_garage",
        "collection": "wifi_client_data_garage",
        "ap_positions":{
            "freind1" : (1,1),
            "freind2" : (2,1),
            "freind3" : (2,3)
        }
    },
}


triangle_dictionary_indoor = {
    "reto_grande_indoor": {
        "triangle_name" : "reto_grande",
        "db": "wifi_data_db_indoor",
        "collection": "wifi_data_indoor_global",
        "ap_positions":{
            "freind1" : (0,0),
            "freind2" : (5,0),
            "freind3" : (0,5)
        }
    },
    "isosceles_grande_indoor": {
        "triangle_name" : "isosceles_grande",
        "db": "wifi_data_db_indoor",
        "collection": "wifi_data_indoor_global",
        "ap_positions":{
            "freind1" : (0,0),
            "freind2" : (2.5,5),
            "freind3" : (0,5)
        }
    },
    "reto_n_quadrado_grande_indoor": {
        "triangle_name" : "reto_n_quadrado_grande",
        "db": "wifi_data_db_indoor",
        "collection": "wifi_data_indoor_global",
        "ap_positions":{
            "freind1" : (2.5,0),
            "freind2" : (2.5,5),
            "freind3" : (0,5)
        }
    },
}