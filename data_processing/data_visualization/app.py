import dash
from dash import dcc, html, Output, Input
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime
from triangle_dict import triangle_dictionary, ap_mapping

# Connect to MongoDB
client = MongoClient("mongodb://localhost:28910/")
db = client["wifi_data_db"]
collection = db["wifi_client_data_global"]


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

    # Step 1: Identify which AP corresponds to the current BSSID
    ap_positions = info["ap_positions"]
    ap_name = None
    for name_, bssids in ap_mapping.items():
        if bssid in bssids:
            ap_name = name_
            break

    if not ap_name or ap_name not in ap_positions:
        print(f"AP not found for BSSID: {bssid}")
        return None

    ap_origin = ap_positions[ap_name]
    ap_x, ap_y = ap_origin

    # MongoDB pipeline
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
    y_vals = list(range(len(custom_pico_order)))
    Z_measured = pd.DataFrame(pivot.values).interpolate(axis=1, limit_direction='both').interpolate(axis=0, limit_direction='both').values

    # Distance parameters
    grid_spacing = 3.55  # meters
    d0 = 1.0
    rssi0 = -40
    path_loss_exp = 2.0

    # Step 2: Offset grid by AP origin for theoretical model
    X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
    X_meters = (X_mesh - ap_x) * grid_spacing
    Y_meters = (Y_mesh - ap_y) * grid_spacing
    distances = np.sqrt(X_meters ** 2 + Y_meters ** 2)
    distances[distances < d0] = d0
    Z_theoretical = rssi0 - 10 * path_loss_exp * np.log10(distances / d0)

    # Plotting (same as before)
    fig = go.Figure()

    fig.add_trace(go.Surface(
        z=Z_measured,
        x=x_vals,
        y=y_vals,
        colorscale='Viridis',
        opacity=0.9,
        name="Measured",
        showscale=True,
        colorbar=dict(title='RSSI (dBm)'),
        hovertemplate='Button: %{x}<br>Pico IP: %{y}<br>RSSI (Measured): %{z:.2f} dBm<br><extra></extra>'
    ))


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
        width=700,
        height=600,
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
