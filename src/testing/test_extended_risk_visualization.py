"""
Test script for extended risk visualization module

This script demonstrates the functionality of the ExtendedRiskVisualizer
by generating sample risk data and creating various visualizations.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.testing.extended_risk_simulation import ExtendedRiskSimulator, save_extended_risk_data
from src.visualization.extended_risk_visualizer import ExtendedRiskVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Create output directory
    output_dir = Path("output/extended_risk_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating sample risk data...")
    
    # Generate data for all scenarios
    scenarios = ["base", "market_crash", "credit_deterioration", "combined_stress", "custom"]
    
    for scenario in scenarios:
        logger.info(f"Generating data for {scenario} scenario...")
        save_extended_risk_data(scenario=scenario, sim_days=252, seed=42)
    
    # Initialize visualizer
    visualizer = ExtendedRiskVisualizer()
    
    # Create visualizations for each risk type
    risk_types = [
        'operational_risk',
        'climate_risk',
        'cyber_risk',
        'ai_risk',
        'digitalization_risk'
    ]
    
    # Generate individual visualizations
    logger.info("Creating individual visualizations...")
    for risk_type in risk_types:
        for scenario in scenarios:
            # Time series plot
            fig = visualizer.plot_time_series(risk_type, scenario)
            fig.write_html(output_dir / f"{risk_type}_{scenario}_time_series.html")
            
            # Radar chart
            fig = visualizer.plot_radar_chart(risk_type, scenario)
            fig.write_html(output_dir / f"{risk_type}_{scenario}_radar.html")
            
            # Heatmap
            fig = visualizer.plot_heatmap(risk_type, scenario)
            fig.write_html(output_dir / f"{risk_type}_{scenario}_heatmap.html")
    
    # Generate comparison visualizations
    logger.info("Creating comparison visualizations...")
    for scenario in scenarios:
        # Risk comparison radar chart
        fig = visualizer.plot_risk_comparison(scenario)
        fig.write_html(output_dir / f"risk_comparison_{scenario}.html")
    
    for risk_type in risk_types:
        # Scenario comparison
        fig = visualizer.plot_scenario_comparison(risk_type)
        fig.write_html(output_dir / f"{risk_type}_scenario_comparison.html")
        
        # Dashboard
        fig = visualizer.create_risk_dashboard(risk_type, 'base')
        fig.write_html(output_dir / f"{risk_type}_dashboard.html")
    
    logger.info(f"All visualizations saved to {output_dir}")
    
    # Generate summary report
    create_summary_report(output_dir, risk_types, scenarios)

def create_summary_report(output_dir, risk_types, scenarios):
    """Create a summary HTML report with links to all visualizations"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Extended Risk Visualizations</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2, h3 { color: #333; }
            .container { margin-bottom: 30px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            a { color: #0066cc; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>Extended Risk Visualizations</h1>
        
        <div class="container">
            <h2>Risk Dashboards</h2>
            <ul>
    """
    
    # Add dashboard links
    for risk_type in risk_types:
        display_name = risk_type.replace('_', ' ').title()
        html_content += f'<li><a href="{risk_type}_dashboard.html" target="_blank">{display_name} Dashboard</a></li>\n'
    
    html_content += """
            </ul>
        </div>
        
        <div class="container">
            <h2>Risk Comparison by Scenario</h2>
            <ul>
    """
    
    # Add risk comparison links
    for scenario in scenarios:
        display_name = scenario.replace('_', ' ').title()
        html_content += f'<li><a href="risk_comparison_{scenario}.html" target="_blank">{display_name} Scenario</a></li>\n'
    
    html_content += """
            </ul>
        </div>
        
        <div class="container">
            <h2>Scenario Comparison by Risk Type</h2>
            <ul>
    """
    
    # Add scenario comparison links
    for risk_type in risk_types:
        display_name = risk_type.replace('_', ' ').title()
        html_content += f'<li><a href="{risk_type}_scenario_comparison.html" target="_blank">{display_name}</a></li>\n'
    
    html_content += """
            </ul>
        </div>
        
        <div class="container">
            <h2>Individual Visualizations</h2>
            <table>
                <tr>
                    <th>Risk Type</th>
                    <th>Scenario</th>
                    <th>Time Series</th>
                    <th>Radar Chart</th>
                    <th>Heatmap</th>
                </tr>
    """
    
    # Add individual visualization links
    for risk_type in risk_types:
        display_risk = risk_type.replace('_', ' ').title()
        
        for scenario in scenarios:
            display_scenario = scenario.replace('_', ' ').title()
            
            html_content += f"""
                <tr>
                    <td>{display_risk}</td>
                    <td>{display_scenario}</td>
                    <td><a href="{risk_type}_{scenario}_time_series.html" target="_blank">View</a></td>
                    <td><a href="{risk_type}_{scenario}_radar.html" target="_blank">View</a></td>
                    <td><a href="{risk_type}_{scenario}_heatmap.html" target="_blank">View</a></td>
                </tr>
            """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    # Write the HTML file
    with open(output_dir / "index.html", "w") as f:
        f.write(html_content)
    
    logger.info(f"Summary report created at {output_dir}/index.html")

if __name__ == "__main__":
    main() 