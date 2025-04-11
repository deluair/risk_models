"""
Visualization Engine for the Financial Risk Analysis System
Provides plotting capabilities for risk metrics visualization
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from typing import Dict, List, Any, Optional
import networkx as nx

class VisualizationEngine:
    """Engine for visualizing risk analysis results"""
    
    def __init__(self):
        """Initialize the visualization engine"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing VisualizationEngine")
    
    def plot_market_risk(self, market_risk_results: Dict[str, Any]) -> Dict[str, plt.Figure]:
        """Create visualizations for market risk results
        
        Args:
            market_risk_results: Dictionary with market risk analysis results
            
        Returns:
            Dictionary of Matplotlib figures
        """
        self.logger.info("Creating market risk visualizations")
        figures = {}
        
        # Value at Risk (VaR) plot
        if 'var' in market_risk_results:
            fig, ax = plt.subplots(figsize=(10, 6))
            var_values = [
                market_risk_results['var']['var_95'],
                market_risk_results['var']['var_99']
            ]
            labels = ['VaR (95%)', 'VaR (99%)']
            ax.bar(labels, var_values, color=['blue', 'red'])
            ax.set_title('Value at Risk (VaR)')
            ax.set_ylabel('Percentage (%)')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            for i, v in enumerate(var_values):
                ax.text(i, v + 0.1, f"{v:.2f}%", ha='center')
                
            figures['var'] = fig
        
        # Volatility plot
        if 'volatility' in market_risk_results and isinstance(market_risk_results['volatility'], pd.Series):
            fig, ax = plt.subplots(figsize=(12, 6))
            market_risk_results['volatility'].plot(ax=ax)
            ax.set_title('Market Volatility Over Time')
            ax.set_ylabel('Volatility (%)')
            ax.set_xlabel('Date')
            ax.grid(True, alpha=0.3)
            figures['volatility'] = fig
        
        # Market Index plot
        if 'market_indices' in market_risk_results and isinstance(market_risk_results['market_indices'], pd.DataFrame):
            fig, ax = plt.subplots(figsize=(12, 6))
            market_risk_results['market_indices'].plot(ax=ax)
            ax.set_title('Market Indices')
            ax.set_ylabel('Index Value')
            ax.set_xlabel('Date')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            figures['market_indices'] = fig
        
        return figures
    
    def plot_credit_risk(self, credit_risk_results: Dict[str, Any]) -> Dict[str, plt.Figure]:
        """Create visualizations for credit risk results
        
        Args:
            credit_risk_results: Dictionary with credit risk analysis results
            
        Returns:
            Dictionary of Matplotlib figures
        """
        self.logger.info("Creating credit risk visualizations")
        figures = {}
        
        # Expected and Unexpected Loss
        if all(k in credit_risk_results for k in ['expected_loss', 'unexpected_loss']):
            fig, ax = plt.subplots(figsize=(8, 6))
            labels = ['Expected Loss', 'Unexpected Loss']
            values = [
                credit_risk_results['expected_loss'],
                credit_risk_results['unexpected_loss']
            ]
            ax.bar(labels, values, color=['green', 'orange'])
            ax.set_title('Credit Risk Losses')
            ax.set_ylabel('Amount ($)')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            for i, v in enumerate(values):
                ax.text(i, v + 0.1, f"${v:,.2f}", ha='center')
                
            figures['losses'] = fig
        
        # Rating distribution
        if 'rating_distribution' in credit_risk_results and isinstance(credit_risk_results['rating_distribution'], dict):
            fig, ax = plt.subplots(figsize=(10, 6))
            ratings = list(credit_risk_results['rating_distribution'].keys())
            values = list(credit_risk_results['rating_distribution'].values())
            
            ax.bar(ratings, values)
            ax.set_title('Portfolio Rating Distribution')
            ax.set_ylabel('Percentage (%)')
            ax.set_xlabel('Rating')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            figures['rating_distribution'] = fig
        
        return figures
    
    def plot_network_risk(self, network_risk_results: Dict[str, Any]) -> Dict[str, plt.Figure]:
        """Create visualizations for network risk results
        
        Args:
            network_risk_results: Dictionary with network risk analysis results
            
        Returns:
            Dictionary of Matplotlib figures
        """
        self.logger.info("Creating network risk visualizations")
        figures = {}
        
        # Create network graph visualization
        if 'network' in network_risk_results and isinstance(network_risk_results['network'], nx.Graph):
            fig, ax = plt.subplots(figsize=(12, 10))
            graph = network_risk_results['network']
            
            # Use centrality for node size if available
            node_size = 300
            if 'centrality' in network_risk_results:
                centrality = network_risk_results['centrality']
                node_size = [centrality.get(node, 0.1) * 3000 for node in graph.nodes()]
            
            # Use community detection for node color if available
            node_color = 'skyblue'
            if 'communities' in network_risk_results:
                communities = network_risk_results['communities']
                node_color = [communities.get(node, 0) for node in graph.nodes()]
            
            # Draw network
            pos = nx.spring_layout(graph, seed=42)
            nx.draw_networkx(
                graph, pos=pos, with_labels=True, 
                node_size=node_size, node_color=node_color,
                font_size=8, alpha=0.8, ax=ax
            )
            
            ax.set_title('Financial Network')
            ax.axis('off')
            figures['network'] = fig
        
        # Centrality measures
        if 'centrality' in network_risk_results and isinstance(network_risk_results['centrality'], dict):
            # Take top 10 entities by centrality
            centrality = network_risk_results['centrality']
            top_entities = dict(sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(top_entities.keys(), top_entities.values())
            ax.set_title('Top 10 Entities by Centrality')
            ax.set_ylabel('Centrality Score')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            fig.tight_layout()
            figures['centrality'] = fig
        
        return figures
    
    def plot_systemic_risk(self, systemic_risk_metrics: Dict[str, Any]) -> Dict[str, plt.Figure]:
        """Create visualizations for systemic risk metrics
        
        Args:
            systemic_risk_metrics: Dictionary with systemic risk metrics
            
        Returns:
            Dictionary of Matplotlib figures
        """
        self.logger.info("Creating systemic risk visualizations")
        figures = {}
        
        # Systemic risk indicators
        indicators = {k: v for k, v in systemic_risk_metrics.items() 
                     if isinstance(v, (int, float)) and k != 'timestamp'}
        
        if indicators:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(indicators.keys(), indicators.values())
            ax.set_title('Systemic Risk Indicators')
            ax.set_ylabel('Risk Score')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            fig.tight_layout()
            figures['indicators'] = fig
        
        # Time series data
        time_series_data = {k: v for k, v in systemic_risk_metrics.items() 
                           if isinstance(v, (pd.Series, pd.DataFrame))}
        
        for name, data in time_series_data.items():
            fig, ax = plt.subplots(figsize=(12, 6))
            
            if isinstance(data, pd.Series):
                data.plot(ax=ax)
            else:  # DataFrame
                data.plot(ax=ax)
                
            ax.set_title(f'{name.replace("_", " ").title()} Over Time')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            figures[name] = fig
        
        return figures 