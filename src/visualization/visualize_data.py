"""
Script to visualize the simulated financial data
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_market_data(scenario_name="base"):
    """Load market data from CSV files
    
    Args:
        scenario_name: Name of the scenario to load
        
    Returns:
        Dictionary of DataFrames with market data
    """
    data_dir = Path(f"data/raw/scenario_{scenario_name}")
    market_data = {}
    
    # Load indices data
    indices_file = data_dir / "market_indices.csv"
    if indices_file.exists():
        market_data['indices'] = pd.read_csv(indices_file, index_col=0, parse_dates=True)
        
    # Load rates data
    rates_file = data_dir / "market_rates.csv"
    if rates_file.exists():
        market_data['rates'] = pd.read_csv(rates_file, index_col=0, parse_dates=True)
        
    # Load volatility data
    volatility_file = data_dir / "market_volatility.csv"
    if volatility_file.exists():
        market_data['volatility'] = pd.read_csv(volatility_file, index_col=0, parse_dates=True)
        
    # Load credit spreads data
    spreads_file = data_dir / "market_credit_spreads.csv"
    if spreads_file.exists():
        market_data['credit_spreads'] = pd.read_csv(spreads_file, index_col=0, parse_dates=True)
    
    return market_data

def load_credit_data(scenario_name="base"):
    """Load credit data from CSV files
    
    Args:
        scenario_name: Name of the scenario to load
        
    Returns:
        Dictionary of DataFrames with credit data
    """
    data_dir = Path(f"data/raw/scenario_{scenario_name}")
    credit_data = {}
    
    # Load assets data
    assets_file = data_dir / "credit_assets.csv"
    if assets_file.exists():
        credit_data['assets'] = pd.read_csv(assets_file, index_col=0)
        
    # Load issuers data
    issuers_file = data_dir / "credit_issuers.csv"
    if issuers_file.exists():
        credit_data['issuers'] = pd.read_csv(issuers_file, index_col=0)
        
    # Load transition matrix
    transition_file = data_dir / "credit_transition_matrix.csv"
    if transition_file.exists():
        credit_data['transition_matrix'] = pd.read_csv(transition_file, index_col=0)
    
    return credit_data

def visualize_market_data(market_data, save_dir=None):
    """Create visualizations for market data
    
    Args:
        market_data: Dictionary of market data DataFrames
        save_dir: Directory to save plots (if None, just display)
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot market indices
    if 'indices' in market_data:
        plt.figure(figsize=(12, 6))
        market_data['indices'].plot()
        plt.title('Market Indices Over Time')
        plt.xlabel('Date')
        plt.ylabel('Index Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        if save_dir:
            plt.savefig(save_dir / "market_indices.png")
            plt.close()
        else:
            plt.show()
    
    # Plot interest rates
    if 'rates' in market_data:
        plt.figure(figsize=(12, 6))
        market_data['rates'].plot()
        plt.title('Interest Rates Over Time')
        plt.xlabel('Date')
        plt.ylabel('Rate (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        if save_dir:
            plt.savefig(save_dir / "interest_rates.png")
            plt.close()
        else:
            plt.show()
    
    # Plot volatility
    if 'volatility' in market_data:
        plt.figure(figsize=(12, 6))
        market_data['volatility'].plot()
        plt.title('Market Volatility Over Time')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.grid(True, alpha=0.3)
        plt.legend()
        if save_dir:
            plt.savefig(save_dir / "market_volatility.png")
            plt.close()
        else:
            plt.show()
    
    # Plot credit spreads
    if 'credit_spreads' in market_data:
        plt.figure(figsize=(12, 6))
        market_data['credit_spreads'].plot()
        plt.title('Credit Spreads Over Time')
        plt.xlabel('Date')
        plt.ylabel('Spread (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        if save_dir:
            plt.savefig(save_dir / "credit_spreads.png")
            plt.close()
        else:
            plt.show()

def visualize_credit_data(credit_data, save_dir=None):
    """Create visualizations for credit data
    
    Args:
        credit_data: Dictionary of credit data DataFrames
        save_dir: Directory to save plots (if None, just display)
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot rating distribution
    if 'assets' in credit_data:
        rating_counts = credit_data['assets']['rating'].value_counts().sort_index()
        
        plt.figure(figsize=(10, 6))
        plt.bar(rating_counts.index, rating_counts.values)
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        if save_dir:
            plt.savefig(save_dir / "rating_distribution.png")
            plt.close()
        else:
            plt.show()
    
    # Plot industry distribution
    if 'assets' in credit_data:
        industry_counts = credit_data['assets']['industry'].value_counts()
        
        plt.figure(figsize=(12, 6))
        plt.bar(industry_counts.index, industry_counts.values)
        plt.title('Industry Distribution')
        plt.xlabel('Industry')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / "industry_distribution.png")
            plt.close()
        else:
            plt.show()
    
    # Plot expected loss vs PD
    if 'assets' in credit_data:
        plt.figure(figsize=(10, 6))
        plt.scatter(credit_data['assets']['pd_1y'], credit_data['assets']['expected_loss'], 
                   alpha=0.7)
        plt.title('Expected Loss vs Probability of Default')
        plt.xlabel('Probability of Default (1Y)')
        plt.ylabel('Expected Loss ($)')
        plt.grid(True, alpha=0.3)
        if save_dir:
            plt.savefig(save_dir / "expected_loss_vs_pd.png")
            plt.close()
        else:
            plt.show()

def compare_scenarios(scenarios=None):
    """Compare market data across different scenarios
    
    Args:
        scenarios: List of scenarios to compare (default: all available scenarios)
    """
    if scenarios is None:
        scenarios = ["base", "market_crash", "credit_deterioration", "combined_stress"]
    
    # Create output directory
    output_dir = Path("results/scenario_comparison_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load indices data for each scenario
    indices_data = {}
    vix_data = {}
    
    for scenario in scenarios:
        data_dir = Path(f"data/raw/scenario_{scenario}")
        
        # Load market indices 
        indices_file = data_dir / "market_indices.csv"
        if indices_file.exists():
            df = pd.read_csv(indices_file, index_col=0, parse_dates=True)
            # Use S&P 500 as the benchmark index
            if 'sp500' in df.columns:
                indices_data[scenario] = df['sp500']
        
        # Load volatility (VIX)
        vol_file = data_dir / "market_volatility.csv"
        if vol_file.exists():
            df = pd.read_csv(vol_file, index_col=0, parse_dates=True)
            if 'vix' in df.columns:
                vix_data[scenario] = df['vix']
    
    # Plot S&P 500 comparison
    if indices_data:
        plt.figure(figsize=(12, 6))
        for scenario, series in indices_data.items():
            plt.plot(series.index, series.values, label=scenario)
        
        plt.title('S&P 500 Index Comparison Across Scenarios')
        plt.xlabel('Date')
        plt.ylabel('Index Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(output_dir / "sp500_comparison.png")
        plt.close()
    
    # Plot VIX comparison
    if vix_data:
        plt.figure(figsize=(12, 6))
        for scenario, series in vix_data.items():
            plt.plot(series.index, series.values, label=scenario)
        
        plt.title('Market Volatility (VIX) Comparison Across Scenarios')
        plt.xlabel('Date')
        plt.ylabel('VIX')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(output_dir / "vix_comparison.png")
        plt.close()
    
    # Compare credit data
    pd_means = {}
    for scenario in scenarios:
        data_dir = Path(f"data/raw/scenario_{scenario}")
        assets_file = data_dir / "credit_assets.csv"
        if assets_file.exists():
            df = pd.read_csv(assets_file)
            pd_means[scenario] = df['pd_1y'].mean()
    
    if pd_means:
        plt.figure(figsize=(10, 6))
        plt.bar(pd_means.keys(), pd_means.values())
        plt.title('Average Probability of Default Comparison')
        plt.xlabel('Scenario')
        plt.ylabel('Average PD (1Y)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(output_dir / "pd_comparison.png")
        plt.close()

def main():
    """Main function to run all visualizations"""
    # Create base directory for all visualizations
    output_base_dir = Path("results/visualizations")
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize market data for each scenario
    scenarios = ["base", "market_crash", "credit_deterioration", "combined_stress"]
    
    for scenario in scenarios:
        print(f"Visualizing {scenario} scenario data...")
        market_data = load_market_data(scenario)
        credit_data = load_credit_data(scenario)
        
        scenario_dir = output_base_dir / scenario
        scenario_dir.mkdir(exist_ok=True)
        
        visualize_market_data(market_data, save_dir=scenario_dir)
        visualize_credit_data(credit_data, save_dir=scenario_dir)
    
    # Compare scenarios
    print("Comparing scenarios...")
    compare_scenarios(scenarios)
    
    print(f"All visualizations saved to {output_base_dir}")

if __name__ == "__main__":
    main() 