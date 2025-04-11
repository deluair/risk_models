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
    
    def __init__(self, analysis_engine: AnalysisEngine):
        """Initialize the dashboard
        
        Args:
            analysis_engine: Analysis engine with risk results
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Dashboard")
        
        self.analysis_engine = analysis_engine
        self.data = {}
        
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
        """Update dashboard data
        
        Args:
            data: New data for the dashboard
        """
        self.data = data
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
                
                # Risk Heatmap and Summary
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
                
                # Category selection for detailed view
                dbc.Row(
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Select Risk Category"),
                                dbc.CardBody(
                                    dcc.Dropdown(
                                        id="risk-category-dropdown",
                                        options=[
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
                                            {"label": "Systemic Risk", "value": "systemic"}
                                        ],
                                        value="market_risk"
                                    )
                                )
                            ]
                        ),
                        width=12
                    ),
                    className="mb-4"
                ),
                
                # Detailed risk metrics
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
                
                # Network visualization
                dbc.Row(
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Network Analysis"),
                                dbc.CardBody(
                                    dcc.Graph(id="network-graph", style={"height": "600px"})
                                )
                            ]
                        ),
                        width=12
                    ),
                    className="mb-4"
                ),
                
                # Stress test results
                dbc.Row(
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Stress Test Results"),
                                dbc.CardBody(
                                    dcc.Graph(id="stress-test-chart")
                                )
                            ]
                        ),
                        width=12
                    ),
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
            """Update risk summary chart"""
            # Create a placeholder for the risk summary
            categories = [
                "Market Risk", "Credit Risk", "Liquidity Risk", "Operational Risk",
                "Climate Risk", "Cyber Risk", "AI Risk", "Digitalization",
                "Nonbank", "Global Architecture"
            ]
            
            # Generate random risk levels for demonstration
            np.random.seed(42)  # For reproducibility
            risk_levels = np.random.uniform(0, 100, len(categories))
            
            # Use data if available
            if self.data:
                # This would extract actual risk levels from analysis results
                pass
            
            # Create figure
            fig = go.Figure(data=[
                go.Bar(
                    x=categories,
                    y=risk_levels,
                    marker_color=[
                        'green' if x < 40 else 'orange' if x < 70 else 'red'
                        for x in risk_levels
                    ]
                )
            ])
            
            fig.update_layout(
                title="Risk Level by Category",
                xaxis_title="Risk Category",
                yaxis_title="Risk Score",
                yaxis=dict(range=[0, 100]),
                template=settings.DEFAULT_CHART_THEME
            )
            
            return fig
        
        @self.app.callback(
            Output("risk-heatmap", "figure"),
            Input("time-slider", "value")
        )
        def update_risk_heatmap(time_range):
            """Update risk heatmap"""
            # Create placeholder data for the heatmap
            categories = [
                "Market", "Credit", "Liquidity", "Operational",
                "Climate", "Cyber", "AI", "Digital",
                "Nonbank", "Global"
            ]
            
            metrics = [
                "Value at Risk", "Expected Shortfall", "Volatility",
                "Default Rate", "Liquidity Ratio", "Recovery Time"
            ]
            
            # Generate random risk values for demonstration
            np.random.seed(42)  # For reproducibility
            risk_values = np.random.uniform(0, 100, (len(categories), len(metrics)))
            
            # Create heatmap
            fig = px.imshow(
                risk_values,
                x=metrics,
                y=categories,
                color_continuous_scale="RdYlGn_r",
                zmin=0,
                zmax=100
            )
            
            fig.update_layout(
                title="Risk Metric Heatmap",
                xaxis_title="Risk Metric",
                yaxis_title="Risk Category",
                coloraxis_colorbar_title="Risk Score",
                template=settings.DEFAULT_CHART_THEME
            )
            
            return fig
        
        @self.app.callback(
            [Output("detailed-header", "children"), 
             Output("detailed-metrics", "figure")],
            Input("risk-category-dropdown", "value")
        )
        def update_detailed_metrics(category):
            """Update detailed metrics for the selected category"""
            if not category:
                return "No Category Selected", go.Figure()
            
            # Format category name for display
            category_display = category.replace("_", " ").title()
            
            # Generate placeholder data based on the selected category
            if category == "market_risk":
                # Example for market risk
                metrics = ["VaR", "ES", "Volatility", "Beta", "Correlation"]
                values = np.random.uniform(0, 100, len(metrics))
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=metrics,
                        y=values,
                        marker_color='rgba(50, 171, 96, 0.6)'
                    )
                ])
                
            elif category == "network":
                # Example for network analysis
                entities = ["Bank1", "Bank2", "Bank3", "Insurer1", "AM1", "HF1"]
                centrality = np.random.uniform(0, 1, len(entities))
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=entities,
                        y=centrality,
                        marker_color='rgba(71, 58, 131, 0.6)'
                    )
                ])
                
                metrics = entities
                values = centrality
                
            else:
                # Generic example for other risk types
                metrics = ["Metric1", "Metric2", "Metric3", "Metric4", "Metric5"]
                values = np.random.uniform(0, 100, len(metrics))
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=metrics,
                        y=values,
                        marker_color='rgba(246, 78, 139, 0.6)'
                    )
                ])
            
            fig.update_layout(
                title=f"Detailed {category_display} Metrics",
                xaxis_title="Metric",
                yaxis_title="Value",
                template=settings.DEFAULT_CHART_THEME
            )
            
            return f"{category_display} Detailed Analysis", fig
        
        @self.app.callback(
            Output("additional-metrics", "figure"),
            Input("risk-category-dropdown", "value")
        )
        def update_additional_metrics(category):
            """Update additional metrics for the selected category"""
            if not category:
                return go.Figure()
            
            # Generate placeholder time series data
            dates = pd.date_range(start='2021-01-01', periods=100, freq='D')
            values = np.cumsum(np.random.normal(0, 1, 100))
            
            # Create figure
            fig = go.Figure(data=[
                go.Scatter(
                    x=dates,
                    y=values,
                    mode='lines',
                    line=dict(color='rgba(0, 123, 255, 0.8)')
                )
            ])
            
            fig.update_layout(
                title=f"{category.replace('_', ' ').title()} Time Series",
                xaxis_title="Date",
                yaxis_title="Value",
                template=settings.DEFAULT_CHART_THEME
            )
            
            return fig
        
        @self.app.callback(
            Output("network-graph", "figure"),
            [Input("risk-category-dropdown", "value")]
        )
        def update_network_graph(category):
            """Update network graph visualization"""
            # Create a placeholder network graph
            # In a real application, this would use actual network data
            
            # Define nodes
            nodes = [
                "Bank1", "Bank2", "Bank3", "Bank4", "Bank5",
                "Insurer1", "Insurer2", "Insurer3",
                "AM1", "AM2", "AM3",
                "HF1", "HF2",
                "NBFI1", "NBFI2", "NBFI3"
            ]
            
            # Generate random positions
            np.random.seed(42)  # For reproducibility
            pos = {node: [np.random.uniform(-10, 10), np.random.uniform(-10, 10)] for node in nodes}
            
            # Generate random edges
            edges = []
            for i, source in enumerate(nodes):
                for j, target in enumerate(nodes):
                    if i != j and np.random.random() < 0.2:  # 20% chance of an edge
                        weight = np.random.uniform(0.1, 2.0)
                        edges.append((source, target, weight))
            
            # Create node traces
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            
            for node in nodes:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                
                # Nodes sizes - banks are larger
                if node.startswith("Bank"):
                    node_size.append(20)
                elif node.startswith("Insurer"):
                    node_size.append(15)
                else:
                    node_size.append(10)
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    size=node_size,
                    color=node_size,
                    colorbar=dict(
                        thickness=15,
                        title='Node Size',
                        xanchor='left',
                        titleside='right'
                    ),
                    line=dict(width=2)
                )
            )
            
            # Create edge traces
            edge_x = []
            edge_y = []
            edge_width = []
            
            for source, target, weight in edges:
                x0, y0 = pos[source]
                x1, y1 = pos[target]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_width.append(weight)
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                title='Financial System Network',
                                titlefont=dict(size=16),
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20, l=5, r=5, t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                template=settings.DEFAULT_CHART_THEME
                            ))
            
            return fig
        
        @self.app.callback(
            Output("stress-test-chart", "figure"),
            [Input("risk-category-dropdown", "value")]
        )
        def update_stress_test_chart(category):
            """Update stress test results chart"""
            # Create placeholder data for stress tests
            scenarios = [
                "Baseline", "Adverse", "Severely Adverse", 
                "Climate Transition", "Cyber Attack", "Liquidity Freeze"
            ]
            
            risk_types = ["Market", "Credit", "Liquidity", "Operational"]
            
            # Generate random impact values
            np.random.seed(42)  # For reproducibility
            impacts = np.random.uniform(0, 100, (len(scenarios), len(risk_types)))
            
            # Create figure with subplots
            fig = make_subplots(
                rows=1, cols=1,
                subplot_titles=["Stress Test Impacts by Risk Type"]
            )
            
            # Add bars for each risk type
            for i, risk in enumerate(risk_types):
                fig.add_trace(
                    go.Bar(
                        name=risk,
                        x=scenarios,
                        y=impacts[:, i],
                        text=impacts[:, i].round(1),
                        textposition='auto'
                    )
                )
            
            # Update layout
            fig.update_layout(
                title="Stress Test Results",
                barmode='group',
                yaxis_title="Impact Score",
                legend_title="Risk Type",
                template=settings.DEFAULT_CHART_THEME
            )
            
            return fig 