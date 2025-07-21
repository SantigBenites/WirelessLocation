import dash
from dash import dcc, html, Output, Input
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime

# Connect to MongoDB
client = MongoClient("mongodb://localhost:28910/")
db = client["wifi_data_db"]
collection = db["wifi_client_data_global"]

# Triangle metadata
# Your triangle dictionary
triangle_dictionary = {
    "reto_grande_wifi_client_data_global": {
        "start":datetime(2025, 5, 13, 20, 10),
        "end":datetime(2025, 5, 13, 21, 42),
        "db": "wifi_data_db",
        "collection": "wifi_client_data_global"
    },
    "reto_medio_wifi_client_data_global": {
        "start":datetime(2025, 5, 13, 21, 46),
        "end":datetime(2025, 5, 13, 22, 49),
        "db": "wifi_data_db",
        "collection": "wifi_client_data_global"
    },
    "reto_pequeno_wifi_client_data_global": {
        "start":datetime(2025, 5, 13, 22, 51),
        "end":datetime(2025, 5, 13, 23, 53),
        "db": "wifi_data_db",
        "collection": "wifi_client_data_global"
    },
    "equilatero_grande_wifi_client_data_global": {
        "start":datetime(2025, 6, 28, 19, 45),
        "end":datetime(2025, 6, 28, 21, 15),
        "db": "wifi_data_db",
        "collection": "wifi_client_data_global"
    },
    "equilatero_medio_wifi_client_data_global": {
        "start":datetime(2025, 6, 28, 22, 5),
        "end":datetime(2025, 6, 28, 23, 30),
        "db": "wifi_data_db",
        "collection": "wifi_client_data_global"
    },
    "isosceles_grande_wifi_client_data_global": {
        "start":datetime(2025, 7, 5, 12, 20),
        "end":datetime(2025, 7, 5, 13, 10),
        "db": "wifi_data_db",
        "collection": "wifi_client_data_global"
    },
    "isosceles_medio_wifi_client_data_global": {
        "start":datetime(2025, 7, 5, 13, 24),
        "end":datetime(2025, 7, 5, 14, 30),
        "db": "wifi_data_db",
        "collection": "wifi_client_data_global"
    },
    "reto_n_quadrado_grande_wifi_client_data_global": {
        "start":datetime(2025, 7, 5, 15, 4),
        "end":datetime(2025, 7, 5, 15, 54),
        "db": "wifi_data_db",
        "collection": "wifi_client_data_global"
    },
    "reto_n_quadrado_pequeno_wifi_client_data_global": {
        "start":datetime(2025, 7, 5, 15, 55),
        "end":datetime(2025, 7, 5, 16, 42),
        "db": "wifi_data_db",
        "collection": "wifi_client_data_global"
    },
    "obtusangulo_grande_wifi_client_data_global": {
        "start":datetime(2025, 7, 5, 16, 43),
        "end":datetime(2025, 7, 5, 17, 29),
        "db": "wifi_data_db",
        "collection": "wifi_client_data_global"
    },
    "obtusangulo_pequeno_wifi_client_data_global": {
        "start":datetime(2025, 7, 5, 17, 30),
        "end":datetime(2025, 7, 5, 19, 00),
        "db": "wifi_data_db",
        "collection": "wifi_client_data_global"
    },
    "reto_grande_wifi_client_data_garage": {
        "start": datetime(2025, 7, 19, 11, 27),
        "end": datetime(2025, 7, 19, 12, 29),
        "db": "wifi_data_db_garage",
        "collection": "wifi_client_data_garage"
    },
    "reto_medio_wifi_client_data_garage": {
        "start": datetime(2025, 7, 19, 12, 31),
        "end": datetime(2025, 7, 19, 14, 7),
        "db": "wifi_data_db_garage",
        "collection": "wifi_client_data_garage"
    },
    "reto_pequeno_wifi_client_data_garage": {
        "start": datetime(2025, 7, 19, 14, 11),
        "end": datetime(2025, 7, 19, 15, 4),
        "db": "wifi_data_db_garage",
        "collection": "wifi_client_data_garage"
    },
    "equilatero_grande_wifi_client_data_garage": {
        "start": datetime(2025, 7, 19, 15, 5),
        "end": datetime(2025, 7, 19, 15, 50),
        "db": "wifi_data_db_other",
        "collection": "wifi_client_data_garage"
    },
    "equilatero_medio_wifi_client_data_garage": {
        "start": datetime(2025, 7, 19, 15, 58),
        "end": datetime(2025, 7, 19, 18, 0),
        "db": "wifi_data_db_other",
        "collection": "wifi_client_data_garage"
    },
}


custom_pico_order = [31, 32, 33, 34, 35, 36, 37, 38, 39, 30]
pico_mapping = {val: idx for idx, val in enumerate(custom_pico_order)}

def get_collection(info):
    return client[info["db"]][info["collection"]]


# Cache the BSSIDs used per experiment
def get_bssids_for_experiment(exp_name):
    info = triangle_dictionary[exp_name]
    collection = get_collection(info)

    pipeline = [
        {
            "$match": {
                "timestamp": {
                    "$gte": info["start"].timestamp(),
                    "$lte": info["end"].timestamp()
                },
                "data.SSID": "fun_network"
            }
        },
        {"$unwind": "$data"},
        {"$match": {"data.SSID": "fun_network"}},
        {"$group": {"_id": "$data.BSSID"}}
    ]
    return [x["_id"] for x in collection.aggregate(pipeline)]


experiment_bssids = {
    name: get_bssids_for_experiment(name)
    for name in triangle_dictionary
}

def build_surface(name, bssid):
    info = triangle_dictionary[name]
    collection = get_collection(info)
    start_time = info["start"].timestamp()
    end_time = info["end"].timestamp()

    pipeline = [
        {
            "$match": {
                "timestamp": {"$gte": start_time, "$lte": end_time},
                "data": {
                    "$elemMatch": {
                        "SSID": "fun_network",
                        "BSSID": bssid
                    }
                }
            }
        },
        {
            "$addFields": {
                "y_coordinate": {
                    "$toInt": {
                        "$arrayElemAt": [
                            {"$split": ["$metadata.pico_ip", "."]},
                            3
                        ]
                    }
                },
                "filtered_data": {
                    "$filter": {
                        "input": "$data",
                        "as": "item",
                        "cond": {
                            "$and": [
                                {"$eq": ["$$item.SSID", "fun_network"]},
                                {"$eq": ["$$item.BSSID", bssid]}
                            ]
                        }
                    }
                }
            }
        },
        {"$unwind": "$filtered_data"},
        {
            "$project": {
                "x": "$metadata.button_id",
                "y": "$y_coordinate",
                "z": "$filtered_data.RSSI",
                "_id": 0
            }
        }
    ]

    data = list(collection.aggregate(pipeline))

    if not data:
        return None

    df = pd.DataFrame(data)
    df['y_mapped'] = df['y'].map(pico_mapping)

    pivot = df.pivot_table(index='y_mapped', columns='x', values='z', aggfunc='mean')
    pivot = pivot.sort_index()
    pivot = pivot.reindex(index=range(len(custom_pico_order)))
    pivot = pivot.sort_index(axis=1)

    x_vals = pivot.columns.values
    y_vals = list(range(len(custom_pico_order)))  # 0 to 9
    Z = pd.DataFrame(pivot.values).interpolate(axis=1, limit_direction='both').interpolate(axis=0, limit_direction='both').values

    # Create meshgrid for hovertemplate formatting
    X, Y = np.meshgrid(x_vals, y_vals)

    fig = go.Figure(
        data=[go.Surface(
            z=Z,
            x=x_vals,
            y=y_vals,
            #surfacecolor=Z,
            colorscale='Viridis',
            opacity=0.9,
            showscale=True,
            colorbar=dict(title='RSSI (dBm)'),
            hovertemplate=
                'Button: %{x}<br>' +
                'Pico IP: %{y}<br>' +
                'RSSI: %{z:.2f} dBm<br>' +
                '<extra></extra>'
        )]
    )

    fig.update_layout(
        title=f"Experiment: {name} | BSSID: {bssid[:8]}...",
        scene=dict(
            xaxis_title='Button ID',
            yaxis_title='Pico IP',
            zaxis_title='RSSI (dBm)',
            yaxis=dict(
                tickvals=list(range(len(custom_pico_order))),
                ticktext=[str(ip) for ip in custom_pico_order]
            )
        ),
        width=500,
        height=500,
        margin=dict(r=20, l=20, b=20, t=60)
    )

    return fig


# Dash App
app = dash.Dash(__name__)
app.title = "RSSI Mesh Per Experiment"

app.layout = html.Div([
    html.H2("RSSI Mesh Visualizer (Per Triangle & Per AP)"),
    html.Div([
        html.Label("Primary Experiment"),
        dcc.Dropdown(
            id='experiment-a',
            options=[{"label": k, "value": k} for k in triangle_dictionary],
            value=list(triangle_dictionary.keys())[0],
            clearable=False,
            style={'width': '45%'}
        ),
        html.Label("Secondary Experiment (optional)"),
        dcc.Dropdown(
            id='experiment-b',
            options=[{"label": k, "value": k} for k in triangle_dictionary],
            placeholder="None",
            style={'width': '45%'}
        )
    ], style={"display": "flex", "justify-content": "space-between", "margin-bottom": "20px"}),

    html.Div(id="plots-container", style={"display": "flex", "gap": "40px"})
])

@app.callback(
    Output('plots-container', 'children'),
    Input('experiment-a', 'value'),
    Input('experiment-b', 'value')
)
def update_experiment_comparison(exp_a, exp_b):
    def get_column(experiment_name):
        column = []
        for bssid in experiment_bssids.get(experiment_name, []):
            fig = build_surface(experiment_name, bssid)
            if fig:
                column.append(dcc.Graph(figure=fig))
        return html.Div(column, style={"flex": "1"})

    layout = [get_column(exp_a)]
    if exp_b:
        layout.append(get_column(exp_b))
    return layout

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
