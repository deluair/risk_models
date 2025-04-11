"""
Simple Dash Dashboard for Risk Model Visualization
"""
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# Create paths for data
DATA_DIR = Path("data/raw")

def load_scenario_data(scenario_name):
    """Load data for a specific scenario
    
    Args:
        scenario_name: Name of the scenario
        
    Returns:
        Dictionary of DataFrames
    """
    scenario_dir = DATA_DIR / f"scenario_{scenario_name}"
    data = {}
    
    # Load market indices
    indices_file = scenario_dir / "market_indices.csv"
    if indices_file.exists():
        data['indices'] = pd.read_csv(indices_file, index_col=0, parse_dates=True)
    
    # Load volatility
    vol_file = scenario_dir / "market_volatility.csv"
    if vol_file.exists():
        data['volatility'] = pd.read_csv(vol_file, index_col=0, parse_dates=True)
    
    # Load credit spreads
    spreads_file = scenario_dir / "market_credit_spreads.csv"
    if spreads_file.exists():
        data['spreads'] = pd.read_csv(spreads_file, index_col=0, parse_dates=True)
    
    # Load rates
    rates_file = scenario_dir / "market_rates.csv"
    if rates_file.exists():
        data['rates'] = pd.read_csv(rates_file, index_col=0, parse_dates=True)
    
    # Load credit assets
    assets_file = scenario_dir / "credit_assets.csv"
    if assets_file.exists():
        data['assets'] = pd.read_csv(assets_file)
    
    return data

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Available scenarios
scenarios = ["base", "market_crash", "credit_deterioration", "combined_stress"]

# Create app layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.H1("Financial Risk Model Dashboard", className="text-center my-4"), width=12)
    ]),
    
    # Controls
    dbc.Row([
        dbc.Col([
            html.H4("Scenario Selection"),
            dcc.Dropdown(
                id='scenario-dropdown',
                options=[{'label': s.replace('_', ' ').title(), 'value': s} for s in scenarios],
                value=['base', 'market_crash'],
                multi=True
            ),
        ], width=6),
        
        dbc.Col([
            html.H4("Chart Type"),
            dcc.RadioItems(
                id='chart-type-radio',
                options=[
                    {'label': 'Market Indices', 'value': 'indices'},
                    {'label': 'Volatility', 'value': 'volatility'},
                    {'label': 'Credit Spreads', 'value': 'spreads'},
                    {'label': 'Interest Rates', 'value': 'rates'}
                ],
                value='indices',
                labelStyle={'display': 'inline-block', 'margin-right': '10px'}
            )
        ], width=6)
    ], className="mb-4"),
    
    # Main chart
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4(id="chart-title", children="Market Indices")),
                dbc.CardBody([
                    dcc.Graph(id='main-chart')
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Statistics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Scenario Statistics")),
                dbc.CardBody([
                    html.Div(id='stats-container')
                ])
            ])
        ], width=12)
    ])
], fluid=True)

# Callback for updating main chart
@app.callback(
    [Output('main-chart', 'figure'),
     Output('chart-title', 'children')],
    [Input('scenario-dropdown', 'value'),
     Input('chart-type-radio', 'value')]
)
def update_main_chart(selected_scenarios, chart_type):
    """Update the main chart based on selected scenarios and chart type"""
    # Load data for each selected scenario
    data = {}
    for scenario in selected_scenarios:
        data[scenario] = load_scenario_data(scenario)
    
    # Create figure
    fig = go.Figure()
    
    # Map chart type to data key and title
    chart_mapping = {
        'indices': {'key': 'indices', 'col': 'sp500', 'title': 'Market Indices (S&P 500)'},
        'volatility': {'key': 'volatility', 'col': 'vix', 'title': 'Market Volatility (VIX)'},
        'spreads': {'key': 'spreads', 'col': 'us_a', 'title': 'Credit Spreads (US A-Rated)'},
        'rates': {'key': 'rates', 'col': 'treasury_10y', 'title': '10-Year Treasury Rate'}
    }
    
    data_key = chart_mapping[chart_type]['key']
    col_name = chart_mapping[chart_type]['col']
    chart_title = chart_mapping[chart_type]['title']
    
    for scenario, scenario_data in data.items():
        if data_key in scenario_data and col_name in scenario_data[data_key].columns:
            df = scenario_data[data_key]
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col_name],
                mode='lines',
                name=scenario.replace('_', ' ').title()
            ))
    
    # Update layout
    fig.update_layout(
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_title="Date",
        hovermode="x unified"
    )
    
    return fig, chart_title

# Callback for updating statistics
@app.callback(
    Output('stats-container', 'children'),
    [Input('scenario-dropdown', 'value'),
     Input('chart-type-radio', 'value')]
)
def update_statistics(selected_scenarios, chart_type):
    """Update statistics based on selected scenarios and chart type"""
    # Load data for each selected scenario
    data = {}
    for scenario in selected_scenarios:
        data[scenario] = load_scenario_data(scenario)
    
    # Map chart type to data key and column
    chart_mapping = {
        'indices': {'key': 'indices', 'col': 'sp500'},
        'volatility': {'key': 'volatility', 'col': 'vix'},
        'spreads': {'key': 'spreads', 'col': 'us_a'},
        'rates': {'key': 'rates', 'col': 'treasury_10y'}
    }
    
    data_key = chart_mapping[chart_type]['key']
    col_name = chart_mapping[chart_type]['col']
    
    # Calculate statistics
    stats = []
    for scenario, scenario_data in data.items():
        if data_key in scenario_data and col_name in scenario_data[data_key].columns:
            series = scenario_data[data_key][col_name]
            
            stats.append(
                dbc.Row([
                    dbc.Col(html.H5(scenario.replace('_', ' ').title()), width=12),
                    dbc.Col([
                        html.P(f"Mean: {series.mean():.2f}"),
                        html.P(f"Std Dev: {series.std():.2f}"),
                    ], width=4),
                    dbc.Col([
                        html.P(f"Min: {series.min():.2f}"),
                        html.P(f"Max: {series.max():.2f}"),
                    ], width=4),
                    dbc.Col([
                        html.P(f"Start: {series.iloc[0]:.2f}"),
                        html.P(f"End: {series.iloc[-1]:.2f}"),
                    ], width=4),
                ], className="mb-3")
            )
    
    return stats

# Run the server
if __name__ == '__main__':
    app.run(debug=True, port=8050) 