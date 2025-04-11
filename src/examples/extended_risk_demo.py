"""
Extended Risk Simulation and Visualization Demo

This module demonstrates how to use the extended risk simulator and visualizer together
to generate synthetic risk data and create visualizations for different risk types.
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

# Add parent directory to path to enable imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testing.extended_risk_simulation import ExtendedRiskSimulator, save_extended_risk_data
from visualization.extended_risk_visualizer import ExtendedRiskVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_risk_analysis_demo(data_dir="data/simulated", 
                          output_dir="reports/visualizations",
                          n_days=252,
                          scenarios=None):
    """
    Run a complete risk analysis demonstration, generating data and visualizations.
    
    Args:
        data_dir (str): Directory to save/load simulated data
        output_dir (str): Directory to save visualization outputs
        n_days (int): Number of days to simulate
        scenarios (list): List of scenarios to simulate
        
    Returns:
        str: Path to the generated report
    """
    if scenarios is None:
        scenarios = ['base', 'market_crash', 'credit_deterioration', 'combined_stress']
    
    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Generate risk data for all scenarios
    logger.info("Generating extended risk data...")
    
    for scenario in scenarios:
        save_extended_risk_data(
            output_dir=data_dir,
            scenario=scenario,
            n_days=n_days,
            seed=42  # Fixed seed for reproducibility
        )
    
    # Step 2: Initialize visualizer
    logger.info("Initializing visualizer...")
    visualizer = ExtendedRiskVisualizer(output_dir=output_dir)
    
    # Step 3: Load data
    logger.info("Loading data for visualization...")
    risk_data = {}
    risk_types = ['operational_risk', 'climate_risk', 'cyber_risk', 'ai_risk', 'digitalization_risk']
    
    for scenario in scenarios:
        risk_data[scenario] = {}
        for risk_type in risk_types:
            data_file = os.path.join(data_dir, f"{risk_type}_{scenario}.csv")
            risk_data[scenario][risk_type] = pd.read_csv(data_file)
    
    # Step 4: Create visualizations for each risk type and scenario
    logger.info("Creating visualizations...")
    
    # Time series plots
    for scenario in scenarios:
        for risk_type in risk_types:
            visualizer.plot_time_series(
                risk_data[scenario][risk_type],
                risk_type=risk_type,
                scenario=scenario
            )
    
    # Radar charts
    for scenario in scenarios:
        for risk_type in risk_types:
            visualizer.plot_radar(
                risk_data[scenario][risk_type],
                risk_type=risk_type,
                scenario=scenario
            )
    
    # Heatmaps
    for scenario in scenarios:
        for risk_type in risk_types:
            visualizer.plot_heatmap(
                risk_data[scenario][risk_type],
                risk_type=risk_type,
                scenario=scenario,
                n_periods=10  # Show last 10 days
            )
    
    # Step 5: Create comparison visualizations
    logger.info("Creating comparison visualizations...")
    
    # Compare scenarios for each risk type
    for risk_type in risk_types:
        scenario_data_dict = {scenario: risk_data[scenario][risk_type] for scenario in scenarios}
        visualizer.create_scenario_comparison(scenario_data_dict, risk_type)
    
    # Compare risk types for each scenario
    for scenario in scenarios:
        risk_type_data_dict = {risk_type: risk_data[scenario][risk_type] for risk_type in risk_types}
        visualizer.create_risk_comparison(risk_type_data_dict, scenario)
    
    # Step 6: Generate report
    logger.info("Generating final report...")
    report_path = visualizer.generate_report()
    
    logger.info(f"Demo completed! Report available at: {report_path}")
    return report_path

def custom_risk_event_demo():
    """
    Demonstrate how to create and visualize custom risk events.
    """
    # Define custom risk events
    custom_events = {
        'operational_risk': [
            {'day': 50, 'effect': {'process_failure': 0.8, 'human_error': 0.6, 'overall': 0.4}, 'duration': 10},
            {'day': 150, 'effect': {'regulatory_compliance': 0.9, 'overall': 0.5}, 'duration': 15}
        ],
        'cyber_risk': [
            {'day': 80, 'effect': {'data_breach': 0.95, 'overall': 0.7}, 'duration': 20},
            {'day': 180, 'effect': {'system_outage': 0.85, 'ddos': 0.75, 'overall': 0.6}, 'duration': 12}
        ]
    }
    
    # Create simulator with custom events
    simulator = ExtendedRiskSimulator(seed=42)
    
    # Generate risk data with custom events
    data_dir = "data/simulated/custom"
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate data for affected risk types
    for risk_type in custom_events:
        df = simulator.generate_risk_data(
            risk_type=risk_type,
            n_days=252,
            scenario='custom',
            custom_events=custom_events[risk_type]
        )
        output_file = os.path.join(data_dir, f"{risk_type}_custom.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Generated custom {risk_type} data: {output_file}")
    
    # Generate data for unaffected risk types
    for risk_type in ['climate_risk', 'ai_risk', 'digitalization_risk']:
        if risk_type not in custom_events:
            df = simulator.generate_risk_data(
                risk_type=risk_type,
                n_days=252,
                scenario='base'  # Use base scenario for unaffected types
            )
            output_file = os.path.join(data_dir, f"{risk_type}_custom.csv")
            df.to_csv(output_file, index=False)
            logger.info(f"Generated base {risk_type} data: {output_file}")
    
    # Initialize visualizer and create visualizations
    visualizer = ExtendedRiskVisualizer(output_dir="reports/visualizations/custom")
    
    # Load and visualize the data
    risk_data = {}
    for risk_type in ['operational_risk', 'climate_risk', 'cyber_risk', 'ai_risk', 'digitalization_risk']:
        data_file = os.path.join(data_dir, f"{risk_type}_custom.csv")
        risk_data[risk_type] = pd.read_csv(data_file)
        
        # Create visualizations
        visualizer.plot_time_series(risk_data[risk_type], risk_type, 'custom')
        visualizer.plot_radar(risk_data[risk_type], risk_type, 'custom')
        visualizer.plot_heatmap(risk_data[risk_type], risk_type, 'custom')
    
    # Create risk comparison
    visualizer.create_risk_comparison(risk_data, 'custom')
    
    # Generate report
    report_path = visualizer.generate_report()
    logger.info(f"Custom event demo completed! Report available at: {report_path}")
    return report_path

if __name__ == "__main__":
    # Run the standard demo
    standard_report = run_risk_analysis_demo()
    
    # Run custom event demo
    custom_report = custom_risk_event_demo()
    
    print(f"Standard demo report: {standard_report}")
    print(f"Custom event demo report: {custom_report}") 