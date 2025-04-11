"""
Custom Risk Simulation Scenario Script
Demonstrates using both simulation and visualization modules
"""
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from src.testing.simulation import FinancialDataSimulator
from src.visualization.visualize_data import load_market_data, load_credit_data, visualize_market_data, visualize_credit_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_custom_scenario():
    """Run a custom scenario with specific parameters"""
    logger.info("Starting custom scenario simulation")
    
    # Create output directory
    output_dir = Path("results/custom_scenario")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize simulator
    simulator = FinancialDataSimulator()
    
    # Run a custom scenario - a late 2023 market crash
    scenario_name = "custom_scenario"
    scenario_data = simulator.run_simulation_scenario(
        "market_crash", 
        crash_date="2023-10-15",
        crash_duration_days=60
    )
    
    # Save scenario data to a custom location
    data_dir = Path("data/raw/scenario_custom")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save market data
    for key, df in scenario_data["market"].items():
        file_path = data_dir / f"market_{key}.csv"
        df.to_csv(file_path)
        logger.info(f"Saved {file_path}")
    
    # Save credit data
    for key, data in scenario_data["credit"].items():
        if isinstance(data, pd.DataFrame):
            file_path = data_dir / f"credit_{key}.csv"
            data.to_csv(file_path)
            logger.info(f"Saved {file_path}")
    
    # Load and visualize the generated data
    market_data = load_market_data("custom")
    credit_data = load_credit_data("custom")
    
    visualize_market_data(market_data, save_dir=output_dir)
    visualize_credit_data(credit_data, save_dir=output_dir)
    
    # Create a special comparison plot for volatility in different market crash scenarios
    plt.figure(figsize=(12, 6))
    
    # Load volatility from standard market crash
    base_crash_data = load_market_data("market_crash")
    if 'volatility' in base_crash_data and 'vix' in base_crash_data['volatility']:
        plt.plot(
            base_crash_data['volatility'].index, 
            base_crash_data['volatility']['vix'],
            label='June 2023 Market Crash'
        )
    
    # Load volatility from custom scenario
    if 'volatility' in market_data and 'vix' in market_data['volatility']:
        plt.plot(
            market_data['volatility'].index, 
            market_data['volatility']['vix'],
            label='October 2023 Market Crash'
        )
    
    plt.title('Volatility Comparison - Different Market Crash Scenarios')
    plt.xlabel('Date')
    plt.ylabel('VIX')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_dir / "volatility_comparison.png")
    plt.close()
    
    logger.info(f"Custom scenario visualizations saved to {output_dir}")
    return output_dir

if __name__ == "__main__":
    run_custom_scenario() 