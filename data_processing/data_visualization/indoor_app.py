import dash
from dash import dcc, html, Output, Input
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:28910/")
db = client["wifi_data_db_indoor"]
collection = db["wifi_data_indoor_global"]

# Constants
TARGET_SSID = "fun_network"

# Fetch unique triangle shapes
def get_available_triangles():
    return sorted(collection.distinct("metadata.triangle_shape"))

# Get BSSIDs for a triangle
def get_bssids_for_triangle(triangle):
    pipeline = [
        {"$match": {"metadata.triangle_shape": triangle}},
        {"$unwind": "$data"},
        {"$match": {"data.SSID": TARGET_SSID}},
        {"$group": {"_id": "$data.BSSID"}}
    ]
    return [doc["_id"] for doc in collection.aggregate(pipeline)]

# Build surface plot
def build_surface(triangle, bssid):
    # Step 1: Fetch all (x, y) positions for this triangle
    all_positions_cursor = collection.find(
        {"metadata.triangle_shape": triangle},
        {"metadata.x": 1, "metadata.y": 1, "_id": 0}
    )
    all_positions = {(doc["metadata"]["x"], doc["metadata"]["y"]) for doc in all_positions_cursor}

    # Step 2: Fetch RSSI data for the target BSSID
    pipeline = [
        {
            "$match": {
                "metadata.triangle_shape": triangle,
                "data": {
                    "$elemMatch": {
                        "SSID": TARGET_SSID,
                        "BSSID": bssid
                    }
                }
            }
        },
        {"$addFields": {
            "filtered_data": {
                "$filter": {
                    "input": "$data",
                    "as": "item",
                    "cond": {
                        "$and": [
                            {"$eq": ["$$item.SSID", TARGET_SSID]},
                            {"$eq": ["$$item.BSSID", bssid]}
                        ]
                    }
                }
            }
        }},
        {"$unwind": "$filtered_data"},
        {"$project": {
            "x": "$metadata.x",
            "y": "$metadata.y",
            "z": "$filtered_data.RSSI",
            "_id": 0
        }}
    ]

    docs = list(collection.aggregate(pipeline))
    data = {(d["x"], d["y"]): d["z"] for d in docs}

    # Step 3: Fill missing positions with placeholder RSSI
    complete_data = []
    for (x, y) in all_positions:
        rssi = data.get((x, y), -100)
        complete_data.append({"x": x, "y": y, "z": rssi})

    df = pd.DataFrame(complete_data)

    # Step 4: Create pivot table and surface
    pivot = df.pivot_table(index='y', columns='x', values='z', aggfunc='mean')

    x_vals = pivot.columns.values
    y_vals = pivot.index.values
    Z = pivot.values

    X, Y = np.meshgrid(x_vals, y_vals)

    fig = go.Figure(
        data=[go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale='Viridis',
            opacity=0.9,
            colorbar=dict(title='RSSI (dBm)'),
            hovertemplate='x: %{x}<br>y: %{y}<br>RSSI: %{z:.2f} dBm<extra></extra>'
        )]
    )

    fig.update_layout(
        title=f"Triangle: {triangle} | BSSID: {bssid[:8]}...",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='RSSI (dBm)'
        ),
        width=500,
        height=500,
        margin=dict(r=20, l=20, b=20, t=60)
    )

    return fig


# Dash App
app = dash.Dash(__name__)
app.title = "RSSI Mesh Per Triangle"

app.layout = html.Div([
    html.H2("RSSI Mesh Visualizer (Per Triangle & AP)"),

    html.Div([
        html.Label("Primary Triangle"),
        dcc.Dropdown(
            id='triangle-a',
            options=[{"label": k, "value": k} for k in get_available_triangles()],
            value=None,
            clearable=False,
            style={'width': '45%'}
        ),
        html.Label("Secondary Triangle (optional)"),
        dcc.Dropdown(
            id='triangle-b',
            options=[{"label": k, "value": k} for k in get_available_triangles()],
            placeholder="None",
            style={'width': '45%'}
        )
    ], style={"display": "flex", "justify-content": "space-between", "margin-bottom": "20px"}),

    html.Div(id="plots-container", style={"display": "flex", "gap": "40px"})
])

@app.callback(
    Output('plots-container', 'children'),
    Input('triangle-a', 'value'),
    Input('triangle-b', 'value')
)
def update_comparison(tri_a, tri_b):
    def build_column(triangle):
        if not triangle:
            return html.Div()
        column = []
        for bssid in get_bssids_for_triangle(triangle):
            fig = build_surface(triangle, bssid)
            if fig:
                column.append(dcc.Graph(figure=fig))
        return html.Div(column, style={"flex": "1"})

    layout = [build_column(tri_a)]
    if tri_b:
        layout.append(build_column(tri_b))
    return layout

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
