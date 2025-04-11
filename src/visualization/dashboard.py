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
import networkx as nx

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
    
    def update_data(self, data: Dict[str, Any]):
        """Update dashboard data with new analysis results
        
        Args:
            data: New analysis results for the dashboard
        """
        self.data = data
        # Update scenarios based on new data
        self.scenarios = list(self.data.get('stress_tests', {}).keys())
        if not self.scenarios:
            self.logger.warning("No stress test scenarios found in updated analysis results.")
            self.scenarios = ['baseline']
        # Potentially re-trigger layout update if scenarios changed? For now, assume Dropdown handles options update.
        self.logger.info("Dashboard data updated")

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
                # Assuming results structure: self.data['systemic_risk']['overall_risk_by_category']
                # This key comes from calculate_systemic_risk_metrics in AnalysisEngine
                risk_data = self.data.get('systemic_risk', {}).get('overall_risk_by_category', {})
                
                if not risk_data:
                     self.logger.warning("Systemic risk summary data ('systemic_risk.overall_risk_by_category') not found in self.data.")
                     fig.update_layout(title="Risk Summary Data Not Available", xaxis={'visible': False}, yaxis={'visible': False})
                     return fig

                categories = list(risk_data.keys())
                risk_scores = [] 
                risk_levels = [] # Store original level strings
                level_to_score = {'low': 25, 'medium': 50, 'high': 75, 'critical': 100} # Example mapping

                for cat in categories:
                    # Expecting structure like {'level': 'high', 'score': 2.7}
                    level = risk_data[cat].get('level', 'low') # Default to low if missing
                    score = risk_data[cat].get('score') 
                    risk_levels.append(level.title()) # e.g., "High"
                    # Use score if available and map, otherwise use level mapping
                    if score is not None:
                         # Normalize score (assuming it's 1-4 scale from AnalysisEngine) to 0-100
                         risk_scores.append(min(100, max(0, (score - 1) / 3 * 100))) 
                    else:
                         risk_scores.append(level_to_score.get(level, 0))
            
                # Create figure
                fig = go.Figure(data=[
                    go.Bar(
                        x=[c.replace('_',' ').title() for c in categories],
                        y=risk_scores,
                        text=risk_levels, # Show level name 
                        hoverinfo='x+text+y', # Show category, level, and score
                        textposition='none', # Avoid cluttering bars directly
                        marker=dict(
                            color=risk_scores, # Color bars based on score
                            colorscale=[[0, 'green'], [0.33, 'yellow'], [0.66, 'orange'], [1, 'red']], # Low->Medium->High->Critical
                            cmin=0,
                            cmax=100,
                            colorbar=dict(title='Risk Score')
                        )
                        # marker_color=[ # Old color logic
                        #     'green' if x < 40 else 'orange' if x < 70 else 'red'
                        #     for x in risk_scores
                        # ] 
                    )
                ])
                
                fig.update_layout(
                    title="Overall Risk Score by Category",
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
            """Update risk heatmap using baseline overall risk scores"""
            fig = go.Figure()
            self.logger.debug("Updating risk heatmap using baseline overall scores.")
            level_to_score = {'low': 25, 'medium': 50, 'high': 75, 'critical': 100}

            try:
                baseline_stress_data = self.data.get('stress_tests', {}).get('baseline', {})
                
                if not baseline_stress_data:
                     self.logger.warning("Baseline stress test data not found for heatmap.")
                     fig.update_layout(title="Risk Heatmap Data Not Available (Missing Baseline)", xaxis={'visible': False}, yaxis={'visible': False})
                     return fig

                categories_in_heatmap = list(baseline_stress_data.keys())
                # Remove non-category keys if present
                categories_in_heatmap = [c for c in categories_in_heatmap if c not in ['scenario_info']]
                
                risk_scores = []
                valid_categories = []

                for cat in categories_in_heatmap:
                    overall_baseline = baseline_stress_data[cat].get('overall_risk')
                    if overall_baseline and isinstance(overall_baseline, dict):
                        score = overall_baseline.get('score')
                        level = overall_baseline.get('level', 'low')
                        value = 0
                        if score is not None:
                             # Normalize score (assuming 1-4) to 0-100
                             value = min(100, max(0, (score - 1) / 3 * 100))
                        else:
                             value = level_to_score.get(level, 0)
                        risk_scores.append(value)
                        valid_categories.append(cat.replace('_',' ').title())
                    else:
                         self.logger.warning(f"Could not find valid 'overall_risk' data for category '{cat}' in baseline stress data for heatmap.")
                
                if not risk_scores or not valid_categories:
                    self.logger.warning("Could not assemble any baseline overall risk scores for heatmap.")
                    fig.update_layout(title="Risk Heatmap: No Baseline Scores Found", xaxis={'visible': False}, yaxis={'visible': False})
                    return fig

                # Create heatmap (Categories x 1 Metric)
                # Reshape scores for px.imshow which expects a 2D array
                heatmap_values = np.array(risk_scores).reshape(-1, 1)
                metric_label = ["Baseline Overall Risk Score"]
                
                fig = px.imshow(
                    heatmap_values, 
                    x=metric_label, # Single column
                    y=valid_categories,
                    color_continuous_scale="RdYlGn_r", 
                    zmin=0,
                    zmax=100, # Scores are 0-100
                    aspect="auto",
                    labels=dict(x="Metric", y="Risk Category", color="Risk Score")
                )
                
                fig.update_layout(
                    title="Baseline Overall Risk Score Heatmap", 
                    xaxis_title=None, # Only one metric, title is clear
                    yaxis_title="Risk Category",
                    coloraxis_colorbar_title="Risk Score",
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
            level_to_score = {'low': 25, 'medium': 50, 'high': 75, 'critical': 100} 

            if not category:
                return "No Category Selected", go.Figure()
            
            try:
                category_data = self.data.get(category)

                if not category_data or not isinstance(category_data, dict):
                    self.logger.warning(f"Data for risk category '{category}' not found or not a dict in self.data.")
                    fig.update_layout(title=f"Data for {category_display} Not Available", xaxis={'visible': False}, yaxis={'visible': False})
                    return header, fig

                metrics = []
                values = []
                chart_title = f"Detailed Metrics for {category_display}" # Default title

                if category == "market_risk":
                    # Example: Plot VaR 95% for different indices
                    var_data = category_data.get('value_at_risk', {})
                    if var_data:
                        metrics = list(var_data.keys())
                        values = [abs(var_data[m].get('var_95', 0)) * 100 for m in metrics] # Plot abs VaR %
                        chart_title = "Market Risk: Value at Risk (95%, % of Portfolio)" 
                    else:
                         chart_title = "Market Risk: Value at Risk Data Missing"
                
                elif category == "network":
                     # Example: Plot degree centrality for nodes
                     network_metrics = category_data.get('centrality', {})
                     if network_metrics:
                          # Use the correct key 'degree' based on analyze_network output
                          degree_values = network_metrics.get('degree') 
                          if degree_values and isinstance(degree_values, dict):
                               # Sort nodes by degree for better visualization
                               sorted_nodes = sorted(degree_values.items(), key=lambda item: item[1], reverse=True)
                               metrics = [item[0] for item in sorted_nodes[:20]] # Show top 20 nodes
                               values = [item[1] for item in sorted_nodes[:20]]
                               chart_title = "Network Analysis: Top 20 Nodes by Degree"
                          else:
                               chart_title = "Network Analysis: Degree Data Missing/Invalid"
                     else:
                          chart_title = "Network Analysis: Centrality Data Missing"
                
                # Add elif blocks for other *implemented* categories here
                # elif category == "credit_risk": ... etc ...

                else: # Fallback for unimplemented/placeholder categories
                    chart_title = f"No Detailed Metrics Implemented for {category_display}" # Default title
                    self.logger.info(f"Attempting to display baseline overall risk for unimplemented category: {category}")
                    # Try to plot the simulated overall risk score from the baseline scenario as a proxy
                    try:
                        baseline_stress_data = self.data.get('stress_tests', {}).get('baseline', {})
                        category_baseline_data = baseline_stress_data.get(category, {})
                        overall_baseline = category_baseline_data.get('overall_risk') # <-- This is the crucial dict

                        if overall_baseline and isinstance(overall_baseline, dict) and 'level' in overall_baseline:
                             metrics = ['Baseline Overall Risk Score'] # Clarify it's baseline
                             values = [level_to_score.get(overall_baseline.get('level', 'low'), 0)]
                             chart_title = f"{category_display}: Baseline Overall Risk is '{overall_baseline.get('level','N/A').title()}'"
                             self.logger.info(f"Successfully found and plotted baseline overall risk for {category}.")
                        else:
                             # Keep metrics/values empty if baseline overall risk wasn't found as expected
                             self.logger.warning(f"Could not find valid 'overall_risk' dict with 'level' key in baseline data for category: {category}. Found: {overall_baseline}")
                    except Exception as fallback_ex:
                        self.logger.error(f"Error occurred while trying to fetch baseline overall risk for {category}: {fallback_ex}", exc_info=True)
                        # Keep metrics/values empty on error

                # --- Figure Generation & Layout Update ---
                if metrics and values: # Check if metrics/values were successfully populated above
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[str(m).replace('_',' ').title() for m in metrics], # Ensure x-axis labels are strings
                            y=values,
                            marker_color='rgba(50, 171, 96, 0.6)' # Example color
                        )
                    ])
                # else: keep fig = go.Figure() (it's initialized at the start)

                # Update layout OUTSIDE the if/else for metrics/values, using the determined chart_title
                fig.update_layout(
                    title=chart_title, # Use the title set in the logic above
                    xaxis_title="Metric / Item",
                    yaxis_title="Value / Score",
                    template=settings.DEFAULT_CHART_THEME,
                    xaxis={'visible': bool(metrics and values)}, # Hide axis if no data
                    yaxis={'visible': bool(metrics and values)}
                )

            except Exception as e:
                self.logger.error(f"Error updating detailed metrics for {category}: {e}", exc_info=True)
                fig.update_layout(title=f"Error Displaying Metrics: {e}", xaxis={'visible': False}, yaxis={'visible': False})
            
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
                # PROBLEM: AnalysisEngine results (self.data) typically contain *metrics* calculated *from* time series,
                # but not necessarily the raw time series themselves (like daily index levels).
                # We need to adjust AnalysisEngine or AppManager to pass necessary raw data if we want these plots.
                
                ts_data = None
                ts_title = f"Time Series Data for {category_display}"
                
                # Example: If market_risk analysis *did* return the sp500 series:
                # if category == "market_risk":
                #     market_data = self.data.get('market_risk', {})
                #     sp500_series = market_data.get('sp500_time_series') # Hypothetical key
                #     if sp500_series is not None and isinstance(sp500_series, pd.Series):
                #         ts_data = sp500_series
                #         ts_title = "Market Risk: S&P 500 Index"

                # For now, display a message indicating limitation
                fig.update_layout(title=f"Time Series Plotting Not Implemented for {category_display}", 
                                  xaxis={'visible': False}, yaxis={'visible': False})
                self.logger.warning(f"Time series plotting for {category} skipped: Raw time series data not found in analysis results structure.")

                # If ts_data was populated:
                # if ts_data is not None and isinstance(ts_data, pd.Series):
                #     fig = go.Figure(data=[ ... scatter plot ... ])
                #     fig.update_layout(...)
                # else:
                #     fig.update_layout(title=f"No Time Series Available for {category_display}", xaxis={'visible': False}, yaxis={'visible': False})
            
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
                    self.logger.warning("Network analysis data ('network' key) not found or not a dict in self.data.")
                    fig.update_layout(title="Network Data Not Available", xaxis={'visible': False}, yaxis={'visible': False})
                    return fig
                    
                G = network_data.get('graph')
                pos = network_data.get('pos')
                
                if G is None or pos is None or not isinstance(G, nx.Graph) or not isinstance(pos, dict):
                     self.logger.warning("Network graph or positions missing/invalid in network data.")
                     fig.update_layout(title="Network Graph Data Incomplete (Missing Graph or Positions)", xaxis={'visible': False}, yaxis={'visible': False})
                     return fig

                # --- Generate Plotly traces from NetworkX graph --- 
                edge_x, edge_y = [], []
                if G.number_of_edges() > 0: # Check if edges exist
                    for edge in G.edges():
                        # Check if nodes exist in pos dictionary
                        if edge[0] not in pos or edge[1] not in pos:
                           self.logger.warning(f"Position missing for node in edge: {edge}")
                           continue # Skip edge if node position is missing
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                else: self.logger.info("Network graph has no edges.")


                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )

                node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
                
                # Use centrality for size/color if available
                # Use the correct key 'degree' based on analyze_network output
                centrality_data = network_data.get('centrality', {}).get('degree', {}) 
                max_centrality = 1 # Avoid division by zero
                # Check if centrality_data is a dict and not empty before finding max
                if centrality_data and isinstance(centrality_data, dict):
                   valid_centralities = [v for v in centrality_data.values() if isinstance(v, (int, float))]
                   if valid_centralities:
                      max_centrality = max(valid_centralities) if max(valid_centralities) > 0 else 1
                
                if G.number_of_nodes() > 0:
                    for node in G.nodes():
                         if node not in pos:
                             self.logger.warning(f"Position missing for node: {node}")
                             continue # Skip node if position is missing
                         x, y = pos[node]
                         node_x.append(x)
                         node_y.append(y)
                         degree = G.degree(node) if G.has_node(node) else 0
                         centrality_val = centrality_data.get(node, 0) if isinstance(centrality_data, dict) else 0
                         node_text.append(f"{node}<br>Degree: {degree}<br>Centrality: {centrality_val:.2f}")
                         
                         # Size/color based on centrality or degree
                         size = 10 + 20 * (centrality_val / max_centrality) if max_centrality > 0 else 10
                         node_size.append(size)
                         node_color.append(size) # Color by size for example
                else: self.logger.info("Network graph has no nodes.")

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
                fig.update_layout(title=f"Error Displaying Network: {e}", xaxis={'visible': False}, yaxis={'visible': False})
            
            return fig
        
        @self.app.callback(
            Output("stress-test-chart", "figure"),
            Input("scenario-dropdown", "value")
        )
        def update_stress_test_chart(selected_scenario):
            """Update stress test results chart using actual data for selected scenario"""
            fig = go.Figure()
            scenario_display = selected_scenario.replace('_', ' ').title() if selected_scenario else "No Scenario"
            self.logger.debug(f"Updating stress test chart for scenario: {selected_scenario}")
            level_to_score = {'low': 25, 'medium': 50, 'high': 75, 'critical': 100}

            if not selected_scenario:
                # Maybe default to baseline or show a message?
                selected_scenario = 'baseline' 
                scenario_display = 'Baseline'
                # return go.Figure() # Or show baseline by default

            try:
                stress_test_results = self.data.get('stress_tests')
                if not stress_test_results or not isinstance(stress_test_results, dict):
                    self.logger.warning("Stress test results data ('stress_tests' key) not found or not a dict in self.data.")
                    fig.update_layout(title="Stress Test Data Not Available", xaxis={'visible': False}, yaxis={'visible': False})
                    return fig
                
                scenario_data = stress_test_results.get(selected_scenario)
                if not scenario_data or not isinstance(scenario_data, dict):
                    self.logger.warning(f"Data for stress scenario '{selected_scenario}' not found in stress_tests dict.")
                    fig.update_layout(title=f"Data for Scenario '{scenario_display}' Not Available", xaxis={'visible': False}, yaxis={'visible': False})
                    return fig

                # Plot impact scores (overall risk score) for different risk categories under the selected scenario
                # Assumes scenario_data structure: {'market_risk': {'overall_risk': {'score': 2.5, 'level': 'high'}}, ...}
                risk_categories = list(scenario_data.keys()) 
                impact_scores = []
                category_labels = []
                level_labels = []

                for cat in risk_categories:
                    # Find the overall risk assessment for this category within this scenario
                    overall_risk_data = scenario_data[cat].get('overall_risk')
                    if overall_risk_data and isinstance(overall_risk_data, dict):
                        score = overall_risk_data.get('score')
                        level = overall_risk_data.get('level', 'low')
                        
                        # Use score if available, otherwise map level
                        if score is not None:
                            # Normalize score (assuming 1-4) to 0-100
                            impact_scores.append(min(100, max(0, (score - 1) / 3 * 100)))
                        else:
                            impact_scores.append(level_to_score.get(level, 0))
                            
                        category_labels.append(cat.replace('_',' ').title())
                        level_labels.append(level.title())
                    else:
                         self.logger.warning(f"Could not find 'overall_risk' dict for category '{cat}' in scenario '{selected_scenario}'")


                if impact_scores and category_labels:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=category_labels,
                            y=impact_scores,
                            text=level_labels, # Show risk level
                            hoverinfo='x+text+y',
                            textposition='none',
                            marker=dict(
                                color=impact_scores, # Color bars based on score
                                colorscale=[[0, 'green'], [0.33, 'yellow'], [0.66, 'orange'], [1, 'red']], 
                                cmin=0,
                                cmax=100,
                                colorbar=dict(title='Risk Score')
                            )
                        )
                    ])
                    fig.update_layout(
                        title=f"Overall Risk Scores by Category ({scenario_display} Scenario)",
                        xaxis_title="Risk Category",
                        yaxis_title="Impact Score (0-100)",
                        yaxis=dict(range=[0, 100]), # Fixed scale
                        template=settings.DEFAULT_CHART_THEME
                    )
                else:
                    self.logger.warning(f"Could not extract impact scores for scenario '{selected_scenario}'. Scenario data structure: {scenario_data}")
                    fig.update_layout(title=f"Impact Scores Not Available for '{scenario_display}'", xaxis={'visible': False}, yaxis={'visible': False})
            
            except Exception as e:
                self.logger.error(f"Error updating stress test chart for scenario {selected_scenario}: {e}", exc_info=True)
                fig.update_layout(title=f"Error Displaying Stress Test: {e}", xaxis={'visible': False}, yaxis={'visible': False})
                
            return fig 