"""
Interactive dashboard for the Financial Risk Analysis System
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from src.core.config import settings
from src.analytics.analysis_engine import AnalysisEngine


class Dashboard:
    """Interactive dashboard for visualizing risk analysis results"""
    
    def __init__(self, analysis_results: Dict[str, Any]):
        """Initialize the dashboard
        
        Args:
            analysis_results: Dictionary containing results from AnalysisEngine
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Dashboard")
        
        # Store analysis results directly
        self.data = analysis_results
        
        # Extract scenario names (assuming stress_tests key exists)
        self.scenarios = list(self.data.get('stress_tests', {}).keys())
        if not self.scenarios:
            self.logger.warning("No stress test scenarios found in analysis results.")
            self.scenarios = ['baseline'] # Default if none found

        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            title="Financial Risk Analysis System"
        )
        
        # Create layout
        self._create_layout()
        
        # Setup callbacks
        self._setup_callbacks()
        
        self.logger.info("Dashboard initialized successfully")
    
    def run(self, host="127.0.0.1", port=8050, debug=False):
        """Run the dashboard server
        
        Args:
            host: Host to run on
            port: Port to run on
            debug: Whether to run in debug mode
        """
        self.app.run(host=host, port=port, debug=debug)
    
    def _create_layout(self):
        """Create the dashboard layout"""
        self.app.layout = dbc.Container(
            [
                # Header
                dbc.Row(
                    dbc.Col(
                        html.H1("Financial Risk Analysis System", className="text-center p-3"),
                        width=12
                    ),
                    className="mb-4"
                ),
                
                # Controls Row
                dbc.Row(
                    [
                        # Risk Category Selection
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Select Risk Category"),
                                    dbc.CardBody(
                                        dcc.Dropdown(
                                            id="risk-category-dropdown",
                                            options=[
                                                # Use risk categories from settings or registry if available
                                                # For now, keeping the hardcoded list, adjust if needed
                                                {"label": "Market Risk", "value": "market_risk"},
                                                {"label": "Credit Risk", "value": "credit_risk"},
                                                {"label": "Liquidity Risk", "value": "liquidity_risk"},
                                                {"label": "Operational Risk", "value": "operational_risk"},
                                                {"label": "Climate Risk", "value": "climate_risk"},
                                                {"label": "Cyber Risk", "value": "cyber_risk"},
                                                {"label": "AI Risk", "value": "ai_risk"},
                                                {"label": "Digitalization", "value": "digitalization"},
                                                {"label": "Nonbank Intermediation", "value": "nonbank_intermediation"},
                                                {"label": "Global Architecture", "value": "global_architecture"},
                                                {"label": "Network Analysis", "value": "network"},
                                                {"label": "Systemic Risk", "value": "systemic_risk"} # Added systemic
                                            ],
                                            value="market_risk"
                                        )
                                    )
                                ]
                            ),
                            width=6 # Adjusted width
                        ),
                        # Scenario Selection
                        dbc.Col(
                             dbc.Card(
                                [
                                    dbc.CardHeader("Select Scenario"),
                                    dbc.CardBody(
                                        dcc.Dropdown(
                                            id="scenario-dropdown",
                                            options=[{'label': s.replace('_', ' ').title(), 'value': s} for s in self.scenarios],
                                            value=self.scenarios[0] if self.scenarios else None # Default to first scenario
                                        )
                                    )
                                ]
                            ),
                            width=6 # Adjusted width
                        )
                    ],
                    className="mb-4"
                ),

                # Risk Heatmap and Summary (Combined Row)
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Risk Summary"),
                                    dbc.CardBody(
                                        dcc.Graph(id="risk-summary-chart")
                                    )
                                ]
                            ),
                            width=6
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Risk Heatmap"),
                                    dbc.CardBody(
                                        dcc.Graph(id="risk-heatmap")
                                    )
                                ]
                            ),
                            width=6
                        )
                    ],
                    className="mb-4"
                ),
                
                # Detailed risk metrics and Additional Analysis (Combined Row)
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.Div(id="detailed-header")),
                                    dbc.CardBody(
                                        dcc.Graph(id="detailed-metrics")
                                    )
                                ]
                            ),
                            width=6
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Additional Analysis"),
                                    dbc.CardBody(
                                        dcc.Graph(id="additional-metrics")
                                    )
                                ]
                            ),
                            width=6
                        )
                    ],
                    className="mb-4"
                ),
                
                # Network visualization and Stress test results (Combined Row)
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Network Analysis"),
                                    dbc.CardBody(
                                        dcc.Graph(id="network-graph", style={"height": "600px"})
                                    )
                                ]
                            ),
                            width=6 # Adjusted width
                        ),
                         dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Stress Test Results"),
                                    dbc.CardBody(
                                        dcc.Graph(id="stress-test-chart")
                                    )
                                ]
                            ),
                            width=6 # Adjusted width
                        )
                    ],
                    className="mb-4"
                ),
                
                # Time controls
                dbc.Row(
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Time Period"),
                                dbc.CardBody(
                                    dcc.RangeSlider(
                                        id="time-slider",
                                        min=0,
                                        max=10,
                                        step=1,
                                        marks={i: f'Q{i+1} 2021' if i < 4 else f'Q{i-3} 2022' for i in range(11)},
                                        value=[0, 10]
                                    )
                                )
                            ]
                        ),
                        width=12
                    ),
                    className="mb-4"
                ),
                
                # Footer
                dbc.Row(
                    dbc.Col(
                        html.P(
                            "Financial Risk Analysis System - v1.0.0",
                            className="text-center text-muted"
                        ),
                        width=12
                    )
                )
            ],
            fluid=True,
            style={"margin-top": "20px", "margin-bottom": "20px"}
        )
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            Output("risk-summary-chart", "figure"),
            Input("time-slider", "value")
        )
        def update_risk_summary(time_range):
            """Update risk summary chart using actual data"""
            fig = go.Figure()
            self.logger.debug("Updating risk summary chart.")

            try:
                # Assuming results structure like: self.data['systemic_risk']['overall_risk_by_category']
                risk_data = self.data.get('systemic_risk', {}).get('overall_risk_by_category', {})
                
                if not risk_data:
                    self.logger.warning("Systemic risk summary data not found.")
                    # Return empty fig or placeholder text
                    fig.update_layout(title="Risk Summary Data Not Available", xaxis={'visible': False}, yaxis={'visible': False})
                    return fig

                categories = list(risk_data.keys())
                # Assuming values are risk scores or levels that can be mapped numerically
                risk_scores = [] 
                risk_levels = [] # Store original level strings
                level_to_score = {'low': 25, 'medium': 50, 'high': 75, 'critical': 100} # Example mapping

                for cat in categories:
                    level = risk_data[cat].get('level', 'low') # Default to low if missing
                    risk_levels.append(level)
                    risk_scores.append(level_to_score.get(level, 0))
            
                # Create figure
                fig = go.Figure(data=[
                    go.Bar(
                        x=categories,
                        y=risk_scores,
                        text=risk_levels, # Show level name on hover/text
                        textposition='auto',
                        marker_color=[
                            'green' if x < 40 else 'orange' if x < 70 else 'red'
                            for x in risk_scores
                        ]
                    )
                ])
                
                fig.update_layout(
                    title="Overall Risk Level by Category",
                    xaxis_title="Risk Category",
                    yaxis_title="Risk Score (0-100)",
                    yaxis=dict(range=[0, 100]), # Fixed scale
                    template=settings.DEFAULT_CHART_THEME
                )
            except Exception as e:
                self.logger.error(f"Error updating risk summary chart: {e}", exc_info=True)
                fig.update_layout(title=f"Error: {e}", xaxis={'visible': False}, yaxis={'visible': False})

            return fig
        
        @self.app.callback(
            Output("risk-heatmap", "figure"),
            Input("time-slider", "value")
        )
        def update_risk_heatmap(time_range):
            """Update risk heatmap using actual data"""
            fig = go.Figure()
            self.logger.debug("Updating risk heatmap.")

            try:
                # --- This requires defining which metrics form the heatmap ---
                # Option 1: Use predefined key metrics across categories
                # Option 2: Let user select metrics?
                # For now, try to map placeholder metrics to potential real data
                
                heatmap_data = {}
                categories_in_heatmap = settings.RISK_CATEGORIES # Use categories from config
                metrics_in_heatmap = [
                    'value_at_risk', 'expected_shortfall', 'volatility', # Market
                    'default_rate', 'credit_spread', # Credit (Needs Credit Analysis impl)
                    'liquidity_ratio', 'funding_cost' # Liquidity (Needs Liquidity Analysis impl)
                ] 
                # Note: Many risk categories don't have obvious numeric metrics yet based on AnalysisEngine placeholders

                risk_values = []
                metric_labels = []

                for cat in categories_in_heatmap:
                    cat_data = self.data.get(cat, {})
                    row_values = []
                    valid_metrics_in_row = []
                    for metric in metrics_in_heatmap:
                        # Need to access specific value within the metric dict, e.g., var_95, score, level
                        # This logic needs refinement based on actual data structure
                        value = None
                        if metric == 'value_at_risk' and 'sp500' in cat_data.get(metric, {}):
                            value = abs(cat_data[metric]['sp500'].get('var_95', 0)) * 100 # Example: Use SP500 VaR 95%
                        elif metric == 'volatility' and 'sp500' in cat_data.get(metric, {}):
                            value = cat_data[metric]['sp500'].get('annualized_vol', 0) * 100 # Example: Use SP500 Vol
                        # Add similar logic for other metrics/categories when implemented
                        
                        # Placeholder for unimplemented metrics/categories
                        if value is None:
                            value = np.random.uniform(0, 100) if cat_data else 0 # Replace with 0 or NaN
                        
                        row_values.append(value)
                        if cat == categories_in_heatmap[0]: # Add metric label only once
                            metric_labels.append(metric.replace('_', ' ').title())
                        
                    risk_values.append(row_values)
                
                if not risk_values or not metric_labels:
                    self.logger.warning("Could not assemble data for risk heatmap.")
                    fig.update_layout(title="Risk Heatmap Data Not Available", xaxis={'visible': False}, yaxis={'visible': False})
                    return fig

                # Create heatmap
                fig = px.imshow(
                    np.array(risk_values), # Convert to numpy array
                    x=metric_labels,
                    y=[c.replace('_', ' ').title() for c in categories_in_heatmap],
                    color_continuous_scale="RdYlGn_r", 
                    zmin=0,
                    zmax=100, # Assuming risk scores are 0-100
                    aspect="auto"
                )
                
                fig.update_layout(
                    title="Risk Metric Heatmap (Illustrative)", # Mark as illustrative
                    xaxis_title="Risk Metric",
                    yaxis_title="Risk Category",
                    coloraxis_colorbar_title="Risk Score/Value",
                    template=settings.DEFAULT_CHART_THEME
                )
            except Exception as e:
                self.logger.error(f"Error updating risk heatmap: {e}", exc_info=True)
                fig.update_layout(title=f"Error: {e}", xaxis={'visible': False}, yaxis={'visible': False})
               
            return fig
        
        @self.app.callback(
            [Output("detailed-header", "children"), 
             Output("detailed-metrics", "figure")],
            Input("risk-category-dropdown", "value")
        )
        def update_detailed_metrics(category):
            """Update detailed metrics for the selected category using actual data"""
            fig = go.Figure()
            category_display = category.replace("_", " ").title() if category else "No Category"
            header = f"{category_display} Detailed Analysis"
            self.logger.debug(f"Updating detailed metrics for category: {category}")

            if not category:
                return "No Category Selected", go.Figure()
            
            try:
                category_data = self.data.get(category)

                if not category_data or not isinstance(category_data, dict):
                    self.logger.warning(f"Data for risk category '{category}' not found or not a dict.")
                    fig.update_layout(title=f"Data for {category_display} Not Available", xaxis={'visible': False}, yaxis={'visible': False})
                    return header, fig

                # --- Logic to extract and plot metrics based on category ---
                metrics = []
                values = []
                sub_metrics = [] # For nested metrics like VaR per index

                if category == "market_risk":
                    # Example: Plot VaR 95% for different indices
                    var_data = category_data.get('value_at_risk', {})
                    if var_data:
                        metrics = list(var_data.keys())
                        values = [abs(var_data[m].get('var_95', 0)) * 100 for m in metrics] # Plot abs VaR %
                        chart_title = "Value at Risk (95%)" 
                    else:
                        chart_title = "Market Risk Data Missing"
                elif category == "network":
                    # Example: Plot centrality measures
                    network_metrics = category_data.get('centrality', {})
                    if network_metrics:
                        metrics = list(network_metrics.keys())
                        # Need to decide which centrality measure to plot, e.g., degree
                        # This depends heavily on the actual structure of network_metrics
                        # Placeholder: Assume it's a dict of node -> centrality score
                        # values = list(network_metrics.values()) # Simplistic if node->score
                        # Assume structure is measure -> {node: score}
                        if 'degree_centrality' in network_metrics:
                            nodes = list(network_metrics['degree_centrality'].keys())
                            metrics = nodes # Plot nodes on x-axis
                            values = list(network_metrics['degree_centrality'].values())
                            chart_title = "Node Degree Centrality"
                        else:
                            chart_title = "Network Centrality Data Missing"
                            metrics = []
                            values = []
                    else:
                        chart_title = "Network Data Missing"
                # Add elif blocks for other categories (credit, liquidity, etc.) as their analysis is implemented
                # and their data structure is known. Example:
                # elif category == "credit_risk":
                #     default_rates = category_data.get('default_rates') # Assuming this exists
                #     if default_rates is not None and not default_rates.empty:
                #         # Example: Plot latest default rates by sector
                #         latest_rates = default_rates.iloc[-1]
                #         metrics = list(latest_rates.index)
                #         values = list(latest_rates.values * 100) # Percentage
                #         chart_title = "Latest Default Rates by Sector (%)"
                #     else:
                #         chart_title = "Credit Default Rate Data Missing"

                else:
                    # Generic fallback - try to find simple key-value pairs
                    simple_metrics = {k: v for k, v in category_data.items() if isinstance(v, (int, float))}
                    if simple_metrics:
                        metrics = list(simple_metrics.keys())
                        values = list(simple_metrics.values())
                        chart_title = f"Key Metrics for {category_display}"
                    else:
                        # If no simple metrics, maybe plot overall risk level?
                        overall = category_data.get('overall_risk')
                        if overall and isinstance(overall, dict):
                            metrics = ['Overall Risk Score']
                            values = [level_to_score.get(overall.get('level', 'low'), 0)]
                            chart_title = f"Overall Risk Assessment for {category_display}"
                        else:
                            chart_title = f"No specific metrics available for {category_display}"

                if metrics and values:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[m.replace('_', ' ').title() for m in metrics],
                            y=values,
                            marker_color='rgba(50, 171, 96, 0.6)' # Example color
                        )
                    ])
                    fig.update_layout(
                        title=chart_title,
                        xaxis_title="Metric / Item",
                        yaxis_title="Value / Score",
                        template=settings.DEFAULT_CHART_THEME
                    )
                else:
                    # If no metrics/values found after checks, display message
                    fig.update_layout(title=chart_title if 'chart_title' in locals() else f"No Data Available for {category_display}", 
                                      xaxis={'visible': False}, yaxis={'visible': False})

            except Exception as e:
                self.logger.error(f"Error updating detailed metrics for {category}: {e}", exc_info=True)
                fig.update_layout(title=f"Error: {e}", xaxis={'visible': False}, yaxis={'visible': False})
            return header, fig
        
        @self.app.callback(
            Output("additional-metrics", "figure"),
            Input("risk-category-dropdown", "value")
        )
        def update_additional_metrics(category):
            """Update additional metrics (time series) for the selected category using actual data"""
            fig = go.Figure()
            category_display = category.replace("_", " ").title() if category else "No Category"
            self.logger.debug(f"Updating additional metrics for category: {category}")

            if not category:
                return go.Figure()
            
            try:
                # --- Logic to find and plot relevant time series data ---
                # This needs specific logic based on available data for each category
                ts_data = None
                ts_title = f"Time Series Data for {category_display}"

                if category == "market_risk":
                    market_indices = self.data_manager.market_data.get('indices') # Use raw data or processed? Check DataManager
                    if market_indices is not None and 'sp500' in market_indices.columns:
                        ts_data = market_indices['sp500']
                        ts_title = "Market Risk: S&P 500 Index" 
                elif category == "credit_risk":
                    default_rates = self.data_manager.credit_data.get('default_rates')
                    if default_rates is not None and 'financial' in default_rates.columns:
                        ts_data = default_rates['financial']
                        ts_title = "Credit Risk: Financial Sector Default Rate"
                # Add elif for other categories as data becomes available

                if ts_data is not None and isinstance(ts_data, pd.Series):
                    fig = go.Figure(data=[
                        go.Scatter(
                            x=ts_data.index,
                            y=ts_data.values,
                            mode='lines',
                            line=dict(color='rgba(0, 123, 255, 0.8)') # Example color
                        )
                    ])
                    fig.update_layout(
                        title=ts_title,
                        xaxis_title="Date",
                        yaxis_title="Value / Rate",
                        template=settings.DEFAULT_CHART_THEME
                    )
                else:
                    self.logger.warning(f"No suitable time series data found for category '{category}'.")
                    fig.update_layout(title=f"No Time Series Available for {category_display}", xaxis={'visible': False}, yaxis={'visible': False})
            except Exception as e:
                self.logger.error(f"Error updating additional metrics for {category}: {e}", exc_info=True)
                fig.update_layout(title=f"Error: {e}", xaxis={'visible': False}, yaxis={'visible': False})
            return fig
        
        @self.app.callback(
            Output("network-graph", "figure"),
            [Input("risk-category-dropdown", "value")]
        )
        def update_network_graph(category):
            """Update network graph visualization using actual data"""
            fig = go.Figure()
            self.logger.debug("Updating network graph.")

            try:
                network_data = self.data.get('network')
                
                if not network_data or not isinstance(network_data, dict):
                    self.logger.warning("Network analysis data not found or not a dict.")
                    fig.update_layout(title="Network Data Not Available", xaxis={'visible': False}, yaxis={'visible': False})
                    return fig
                
                # Check for necessary keys like 'graph', 'pos' (positions)
                G = network_data.get('graph')
                pos = network_data.get('pos')
                
                if G is None or pos is None or not isinstance(G, nx.Graph) or not isinstance(pos, dict):
                    self.logger.warning("Network graph or positions missing/invalid in network data.")
                    fig.update_layout(title="Network Graph Data Incomplete", xaxis={'visible': False}, yaxis={'visible': False})
                    return fig

                # --- Generate Plotly traces from NetworkX graph --- 
                edge_x, edge_y = [], []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )

                node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
                # Use centrality or other attributes for size/color if available
                centrality_data = network_data.get('centrality', {}).get('degree_centrality', {})
                max_centrality = max(centrality_data.values()) if centrality_data else 1

                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(f"{node}\nDegree: {G.degree(node)}\nCentrality: {centrality_data.get(node, 'N/A'):.2f}")
                    # Size/color based on centrality or degree
                    size = 10 + 20 * (centrality_data.get(node, 0) / max_centrality) if max_centrality > 0 else 10
                    node_size.append(size)
                    node_color.append(size) # Color by size for example

                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    hoverinfo='text',
                    text=node_text,
                    marker=dict(
                        showscale=True,
                        colorscale='YlGnBu',
                        size=node_size,
                        color=node_color,
                        colorbar=dict(
                            thickness=15,
                            title='Node Centrality/Size',
                            xanchor='left',
                            titleside='right'
                        ),
                        line=dict(width=1)
                    )
                )
                # Create figure
                fig = go.Figure(data=[edge_trace, node_trace],
                                layout=go.Layout(
                                    title='Financial System Network Interconnectedness',
                                    titlefont=dict(size=16),
                                    showlegend=False,
                                    hovermode='closest',
                                    margin=dict(b=20, l=5, r=5, t=40),
                                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    template=settings.DEFAULT_CHART_THEME
                                ))
            except Exception as e:
                self.logger.error(f"Error updating network graph: {e}", exc_info=True)
                fig.update_layout(title=f"Error: {e}", xaxis={'visible': False}, yaxis={'visible': False})
            return fig
        
        @self.app.callback(
            Output("stress-test-chart", "figure"),
            [Input("scenario-dropdown", "value"), 
             Input("risk-category-dropdown", "value")]
        )
        def update_stress_test_chart(selected_scenario, selected_category):
            """Update stress test results chart using actual data for selected scenario"""
            fig = go.Figure()
            scenario_display = selected_scenario.replace('_', ' ').title() if selected_scenario else "No Scenario"
            self.logger.debug(f"Updating stress test chart for scenario: {selected_scenario}")

            if not selected_scenario:
                return go.Figure()

            try:
                stress_test_results = self.data.get('stress_tests')
                if not stress_test_results or not isinstance(stress_test_results, dict):
                    self.logger.warning("Stress test results data not found or not a dict.")
                    fig.update_layout(title="Stress Test Data Not Available", xaxis={'visible': False}, yaxis={'visible': False})
                    return fig
                
                scenario_data = stress_test_results.get(selected_scenario)
                if not scenario_data or not isinstance(scenario_data, dict):
                    self.logger.warning(f"Data for stress scenario '{selected_scenario}' not found.")
                    fig.update_layout(title=f"Data for Scenario '{scenario_display}' Not Available", xaxis={'visible': False}, yaxis={'visible': False})
                    return fig

                # --- Plotting logic for stress test results --- 
                # Assuming scenario_data is a dictionary like: {'risk_category': {'metric': value, 'impact_score': score}, ...}
                # Or maybe {'metric_name': impact_score, ...}
                # Needs clarification based on _run_scenario output in AnalysisEngine

                # Example: Plot impact scores for different risk categories under the selected scenario
                risk_categories = list(scenario_data.keys()) # Assuming keys are risk categories
                impact_scores = []
                category_labels = []

                for cat in risk_categories:
                    # Try to find an overall impact score for the category
                    impact = scenario_data[cat].get('impact_score') # Hypothetical structure
                    if impact is None:
                        # Maybe look for a primary metric's value?
                        # This part is highly dependent on the actual structure
                        impact = scenario_data[cat].get('overall_risk', {}).get('score', 0) # Guessing

                    if isinstance(impact, (int, float)): # Only plot if we found a number
                        impact_scores.append(impact)
                        category_labels.append(cat.replace('_', ' ').title())

                if impact_scores and category_labels:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=category_labels,
                            y=impact_scores,
                            text=[f'{s:.1f}' for s in impact_scores], # Show score on bar
                            textposition='auto'
                            # Add colors based on score? 
                        )
                    ])
                    fig.update_layout(
                        title=f"Stress Test Impacts ({scenario_display} Scenario)",
                        xaxis_title="Risk Category",
                        yaxis_title="Impact Score / Change",
                        template=settings.DEFAULT_CHART_THEME
                    )
                else:
                    self.logger.warning(f"Could not extract impact scores for scenario '{selected_scenario}'. Data structure: {scenario_data}")
                    fig.update_layout(title=f"Impact Scores Not Available for '{scenario_display}'", xaxis={'visible': False}, yaxis={'visible': False})
            except Exception as e:
                self.logger.error(f"Error updating stress test chart for scenario {selected_scenario}: {e}", exc_info=True)
                fig.update_layout(title=f"Error: {e}", xaxis={'visible': False}, yaxis={'visible': False})
            return fig 