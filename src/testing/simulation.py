"""
Simulation module for generating synthetic financial data
Used for testing the Financial Risk Analysis System
"""
import numpy as np
import pandas as pd
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta
import random

from src.core.config import settings


class FinancialDataSimulator:
    """Generates synthetic financial data for testing the risk analysis system"""
    
    def __init__(self, seed=42):
        """Initialize the simulator
        
        Args:
            seed: Random seed for reproducibility
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing FinancialDataSimulator")
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        # Create output directories if they don't exist
        settings.RAW_DATA_DIR.mkdir(exist_ok=True)
        settings.PROCESSED_DATA_DIR.mkdir(exist_ok=True)
        
        self.logger.info("FinancialDataSimulator initialized successfully")
    
    def generate_market_data(self, 
                            start_date='2018-01-01', 
                            end_date='2023-12-31', 
                            include_crash=True,
                            crash_date='2020-03-15',
                            crash_duration_days=30):
        """Generate synthetic market data
        
        Args:
            start_date: Start date for the time series
            end_date: End date for the time series
            include_crash: Whether to include a market crash in the simulation
            crash_date: Approximate date for the market crash
            crash_duration_days: Duration of the market crash in days
            
        Returns:
            Dictionary containing various market data DataFrames
        """
        self.logger.info("Generating synthetic market data")
        
        # Generate date range for business days
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Create crash effect if requested
        crash_effect = np.ones(len(dates))
        if include_crash:
            crash_start_idx = max(0, np.where(dates >= pd.Timestamp(crash_date))[0][0] - 5)
            crash_end_idx = min(len(dates) - 1, crash_start_idx + crash_duration_days)
            
            # Pre-crash buildup (slight increase)
            pre_crash_period = 30
            pre_crash_start = max(0, crash_start_idx - pre_crash_period)
            for i in range(pre_crash_start, crash_start_idx):
                factor = (i - pre_crash_start) / (crash_start_idx - pre_crash_start) * 0.05  # 5% increase
                crash_effect[i] = 1.0 + factor
            
            # Crash period
            crash_magnitude = -0.35  # 35% drop
            for i in range(crash_start_idx, crash_end_idx):
                progress = (i - crash_start_idx) / (crash_end_idx - crash_start_idx)
                # Mostly crash at the beginning, then start recovery
                if progress < 0.6:
                    factor = progress / 0.6 * crash_magnitude
                else:
                    factor = crash_magnitude * (1 - (progress - 0.6) / 0.4 * 0.5)  # Recover 50% of losses
                crash_effect[i] = 1.0 + factor
            
            # Recovery period
            recovery_period = 120
            recovery_end_idx = min(len(dates) - 1, crash_end_idx + recovery_period)
            for i in range(crash_end_idx, recovery_end_idx):
                progress = (i - crash_end_idx) / (recovery_end_idx - crash_end_idx)
                factor = crash_magnitude * (1 - (progress * 0.8 + 0.5 * 0.5))  # Continue recovery
                crash_effect[i] = 1.0 + factor
        
        # Market indices
        indices_data = {}
        # Base parameters for different indices
        index_params = {
            'sp500': {'drift': 0.0005, 'vol': 0.01, 'start': 3000},
            'nasdaq': {'drift': 0.0006, 'vol': 0.012, 'start': 8000},
            'russell2000': {'drift': 0.0004, 'vol': 0.011, 'start': 1500},
            'eurostoxx': {'drift': 0.0003, 'vol': 0.01, 'start': 400},
            'nikkei': {'drift': 0.0004, 'vol': 0.0095, 'start': 23000},
            'ftse': {'drift': 0.0003, 'vol': 0.009, 'start': 7500},
        }
        
        # Generate prices for each index
        for index, params in index_params.items():
            drift = params['drift']
            vol = params['vol']
            start_price = params['start']
            
            # Generate returns with specified drift and volatility
            returns = np.random.normal(drift, vol, len(dates))
            
            # Apply crash effect
            modified_returns = returns * crash_effect
            
            # Convert returns to price series
            prices = start_price * np.cumprod(1 + modified_returns)
            
            indices_data[index] = prices
        
        # Store as DataFrame
        market_data = {}
        market_data['indices'] = pd.DataFrame(indices_data, index=dates)
        
        # Generate volatility indices
        volatility_data = {}
        vix_base = 15 + 5 * np.sin(np.linspace(0, 20, len(dates))) + np.random.normal(0, 3, len(dates))
        vstoxx_base = 17 + 6 * np.sin(np.linspace(0.5, 20.5, len(dates))) + np.random.normal(0, 3.2, len(dates))
        vxn_base = 16 + 5.5 * np.sin(np.linspace(0.2, 20.2, len(dates))) + np.random.normal(0, 2.8, len(dates))
        
        # Apply inverse crash effect to volatility (when market crashes, volatility spikes)
        if include_crash:
            vix_crash = np.ones(len(dates))
            vstoxx_crash = np.ones(len(dates))
            vxn_crash = np.ones(len(dates))
            
            # Pre-crash: slight decrease in volatility
            for i in range(pre_crash_start, crash_start_idx):
                factor = (i - pre_crash_start) / (crash_start_idx - pre_crash_start) * 0.1
                vix_crash[i] = 1.0 - factor
                vstoxx_crash[i] = 1.0 - factor * 0.8
                vxn_crash[i] = 1.0 - factor * 0.9
            
            # During crash: volatility spike
            for i in range(crash_start_idx, crash_end_idx):
                progress = (i - crash_start_idx) / (crash_end_idx - crash_start_idx)
                if progress < 0.3:
                    factor = progress / 0.3 * 2.5  # Spike up to 250%
                else:
                    factor = 2.5 * (1 - (progress - 0.3) / 0.7 * 0.6)  # Slowly decrease
                vix_crash[i] = 1.0 + factor
                vstoxx_crash[i] = 1.0 + factor * 0.9
                vxn_crash[i] = 1.0 + factor * 0.95
            
            # Recovery period: volatility normalizes
            for i in range(crash_end_idx, recovery_end_idx):
                progress = (i - crash_end_idx) / (recovery_end_idx - crash_end_idx)
                factor = 2.5 * 0.4 * (1 - progress)  # Continue decreasing
                vix_crash[i] = 1.0 + factor
                vstoxx_crash[i] = 1.0 + factor * 0.85
                vxn_crash[i] = 1.0 + factor * 0.9
            
            # Apply the volatility crash effect
            vix_base = vix_base * vix_crash
            vstoxx_base = vstoxx_base * vstoxx_crash
            vxn_base = vxn_base * vxn_crash
        
        volatility_data['vix'] = vix_base
        volatility_data['vstoxx'] = vstoxx_base
        volatility_data['vxn'] = vxn_base
        
        market_data['volatility'] = pd.DataFrame(volatility_data, index=dates)
        
        # Generate interest rates
        rates_data = {}
        # Starting values for rates
        rate_params = {
            'us_1y': {'start': 1.5, 'vol': 0.01},
            'us_5y': {'start': 2.0, 'vol': 0.009},
            'us_10y': {'start': 2.5, 'vol': 0.008},
            'euro_1y': {'start': 0.5, 'vol': 0.007},
            'euro_5y': {'start': 1.0, 'vol': 0.006},
            'euro_10y': {'start': 1.5, 'vol': 0.005},
        }
        
        # Generate rates for each tenor
        for rate, params in rate_params.items():
            start_rate = params['start']
            vol = params['vol']
            
            # Generate random moves with lower volatility during the crash (flight to safety)
            rate_changes = np.random.normal(0, vol, len(dates))
            
            # Apply crash effect - interest rates tend to fall during market crashes
            if include_crash:
                for i in range(crash_start_idx, crash_end_idx + 60):  # Extended period of low rates
                    if i < len(rate_changes):
                        # More negative drift during crash
                        rate_changes[i] = rate_changes[i] - 0.002
            
            # Convert changes to rate series
            rates = start_rate + np.cumsum(rate_changes)
            # Ensure rates don't go too negative
            rates = np.maximum(rates, -0.5)
            
            rates_data[rate] = rates
        
        # Store as DataFrame
        market_data['rates'] = pd.DataFrame(rates_data, index=dates)
        
        # Generate credit spreads
        spread_data = {}
        spread_params = {
            'us_aa': {'start': 0.5, 'vol': 0.002},
            'us_a': {'start': 1.0, 'vol': 0.003},
            'us_bbb': {'start': 2.0, 'vol': 0.004},
            'us_bb': {'start': 3.5, 'vol': 0.006},
            'us_b': {'start': 5.0, 'vol': 0.01},
        }
        
        # Generate spreads for each credit rating
        for spread, params in spread_params.items():
            start_spread = params['start']
            vol = params['vol']
            
            # Generate random moves with higher volatility during the crash
            spread_changes = np.random.normal(0, vol, len(dates))
            
            # Apply crash effect - credit spreads widen during market crashes
            if include_crash:
                spread_crash = np.ones(len(dates))
                
                # During crash: spread widening
                for i in range(crash_start_idx, crash_end_idx):
                    progress = (i - crash_start_idx) / (crash_end_idx - crash_start_idx)
                    if progress < 0.4:
                        factor = progress / 0.4 * 0.005  # Increase daily changes
                    else:
                        factor = 0.005 * (1 - (progress - 0.4) / 0.6 * 0.7)  # Slowly decrease
                    spread_changes[i] = spread_changes[i] + factor
                
                # Recovery period: spreads normalize gradually
                for i in range(crash_end_idx, recovery_end_idx):
                    progress = (i - crash_end_idx) / (recovery_end_idx - crash_end_idx)
                    factor = 0.005 * 0.3 * (1 - progress)  # Continue decreasing
                    spread_changes[i] = spread_changes[i] + factor
            
            # Convert changes to spread series
            spreads = start_spread + np.cumsum(spread_changes)
            # Ensure spreads don't go negative
            spreads = np.maximum(spreads, 0.1)
            
            spread_data[spread] = spreads
        
        # Store as DataFrame
        market_data['credit_spreads'] = pd.DataFrame(spread_data, index=dates)
        
        self.logger.info("Synthetic market data generated successfully")
        return market_data
    
    def generate_credit_data(self, num_assets=100, num_issuers=30):
        """Generate synthetic credit portfolio data
        
        Args:
            num_assets: Number of assets in the portfolio
            num_issuers: Number of distinct issuers
            
        Returns:
            Dictionary containing credit data DataFrames
        """
        self.logger.info(f"Generating synthetic credit data with {num_assets} assets")
        
        # Create issuers
        industries = ["Financial", "Technology", "Healthcare", "Energy", 
                    "Utilities", "Consumer", "Industrial", "Materials", 
                    "Real Estate", "Telecommunications"]
        
        ratings = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
        rating_probs = [0.02, 0.08, 0.15, 0.30, 0.25, 0.15, 0.05]
        
        countries = ["US", "UK", "EU", "JP", "CN", "CA", "AU", "BR", "IN", "RU", "KR"]
        country_probs = [0.4, 0.1, 0.2, 0.08, 0.07, 0.05, 0.03, 0.02, 0.02, 0.02, 0.01]
        
        # Generate issuer data
        issuers = []
        for i in range(num_issuers):
            issuer = {
                'issuer_id': f"ISS{i+1:03d}",
                'issuer_name': f"Company {i+1}",
                'industry': random.choice(industries),
                'rating': random.choices(ratings, weights=rating_probs)[0],
                'country': random.choices(countries, weights=country_probs)[0],
                'size': random.choice(["Large", "Medium", "Small"]),
                'pd_1y': np.random.beta(1, 20) * 0.1,  # Probability of default (1 year)
                'lgd': np.random.uniform(0.3, 0.7),  # Loss given default
            }
            issuers.append(issuer)
        
        # Bond characteristics
        maturities = [1, 2, 3, 5, 7, 10, 15, 20, 30]
        maturity_probs = [0.05, 0.1, 0.15, 0.25, 0.15, 0.15, 0.05, 0.05, 0.05]
        
        # Generate asset data
        assets = []
        for i in range(num_assets):
            issuer = random.choice(issuers)
            maturity = random.choices(maturities, weights=maturity_probs)[0]
            issue_date = datetime.now() - timedelta(days=random.randint(0, 365*5))
            asset = {
                'asset_id': f"BND{i+1:04d}",
                'issuer_id': issuer['issuer_id'],
                'issuer_name': issuer['issuer_name'],
                'industry': issuer['industry'],
                'rating': issuer['rating'],
                'country': issuer['country'],
                'asset_type': random.choice(["Corporate Bond", "Sovereign Bond", "Municipal Bond"]),
                'issue_date': issue_date,
                'maturity_date': issue_date + timedelta(days=365*maturity),
                'maturity_years': maturity,
                'coupon': max(0.5, np.random.normal(3.0, 1.5)),
                'face_value': random.choice([1000, 5000, 10000]),
                'market_value': random.uniform(0.85, 1.15) * 1000,
                'quantity': random.randint(10, 1000),
                'pd_1y': issuer['pd_1y'] * random.uniform(0.8, 1.2),
                'lgd': issuer['lgd'],
                'ead': random.uniform(0.9, 1.0),  # Exposure at default
                'risk_weight': random.choice([0.5, 1.0, 1.5, 2.0]),
            }
            
            # Calculate expected loss
            asset['expected_loss'] = asset['pd_1y'] * asset['lgd'] * asset['ead'] * asset['market_value'] * asset['quantity']
            
            assets.append(asset)
        
        # Create portfolio summary
        portfolio_value = sum(asset['market_value'] * asset['quantity'] for asset in assets)
        total_expected_loss = sum(asset['expected_loss'] for asset in assets)
        
        portfolio_summary = {
            'total_assets': num_assets,
            'total_issuers': num_issuers,
            'portfolio_value': portfolio_value,
            'expected_loss': total_expected_loss,
            'expected_loss_ratio': total_expected_loss / portfolio_value,
            'rating_distribution': {rating: sum(1 for asset in assets if asset['rating'] == rating) / num_assets for rating in ratings},
            'industry_distribution': {industry: sum(1 for asset in assets if asset['industry'] == industry) / num_assets for industry in industries},
            'country_distribution': {country: sum(1 for asset in assets if asset['country'] == country) / num_assets for country in countries},
        }
        
        # Create transition matrix
        transition_probs = {
            'AAA': {'AAA': 0.9, 'AA': 0.07, 'A': 0.02, 'BBB': 0.007, 'BB': 0.002, 'B': 0.001, 'CCC': 0.0, 'D': 0.0},
            'AA': {'AAA': 0.01, 'AA': 0.91, 'A': 0.06, 'BBB': 0.01, 'BB': 0.005, 'B': 0.003, 'CCC': 0.002, 'D': 0.0},
            'A': {'AAA': 0.005, 'AA': 0.02, 'A': 0.92, 'BBB': 0.04, 'BB': 0.01, 'B': 0.004, 'CCC': 0.001, 'D': 0.0},
            'BBB': {'AAA': 0.001, 'AA': 0.005, 'A': 0.03, 'BBB': 0.9, 'BB': 0.05, 'B': 0.01, 'CCC': 0.003, 'D': 0.001},
            'BB': {'AAA': 0.0, 'AA': 0.001, 'A': 0.005, 'BBB': 0.04, 'BB': 0.86, 'B': 0.08, 'CCC': 0.01, 'D': 0.004},
            'B': {'AAA': 0.0, 'AA': 0.0, 'A': 0.001, 'BBB': 0.005, 'BB': 0.05, 'B': 0.85, 'CCC': 0.08, 'D': 0.014},
            'CCC': {'AAA': 0.0, 'AA': 0.0, 'A': 0.0, 'BBB': 0.001, 'BB': 0.01, 'B': 0.05, 'CCC': 0.8, 'D': 0.139},
        }
        
        # Compile results
        credit_data = {
            'assets': pd.DataFrame(assets),
            'issuers': pd.DataFrame(issuers),
            'portfolio_summary': portfolio_summary,
            'transition_matrix': pd.DataFrame(transition_probs)
        }
        
        self.logger.info("Synthetic credit data generated successfully")
        return credit_data
    
    def save_simulated_data(self, output_dir=None):
        """Generate and save all simulated data
        
        Args:
            output_dir: Directory to save data (defaults to settings.RAW_DATA_DIR)
        """
        if output_dir is None:
            output_dir = settings.RAW_DATA_DIR
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Saving simulated data to {output_dir}")
        
        # Generate all data types
        market_data = self.generate_market_data(include_crash=True)
        credit_data = self.generate_credit_data(num_assets=100)
        
        # Save market data
        for key, df in market_data.items():
            file_path = output_dir / f"market_{key}.csv"
            df.to_csv(file_path)
            self.logger.info(f"Saved {file_path}")
        
        # Save credit data
        for key, data in credit_data.items():
            if isinstance(data, pd.DataFrame):
                file_path = output_dir / f"credit_{key}.csv"
                data.to_csv(file_path)
                self.logger.info(f"Saved {file_path}")
        
        # Save portfolio summary as JSON
        import json
        with open(output_dir / "credit_portfolio_summary.json", "w") as f:
            json.dump(credit_data["portfolio_summary"], f, indent=4, default=str)
            
        self.logger.info("All simulated data saved successfully")
        
    def run_simulation_scenario(self, scenario_name="base", **kwargs):
        """Run a specific simulation scenario
        
        Args:
            scenario_name: Name of the scenario
            **kwargs: Parameters for the scenario
        
        Returns:
            Dictionary with generated data for the scenario
        """
        self.logger.info(f"Running simulation scenario: {scenario_name}")
        
        scenario_data = {}
        
        if scenario_name == "base":
            # Base scenario - no changes
            scenario_data["market"] = self.generate_market_data(include_crash=False)
            scenario_data["credit"] = self.generate_credit_data()
            
        elif scenario_name == "market_crash":
            # Market crash scenario
            crash_date = kwargs.get("crash_date", "2023-06-15")
            crash_duration = kwargs.get("crash_duration_days", 45)
            scenario_data["market"] = self.generate_market_data(
                include_crash=True,
                crash_date=crash_date,
                crash_duration_days=crash_duration
            )
            scenario_data["credit"] = self.generate_credit_data()
            
        elif scenario_name == "credit_deterioration":
            # Credit quality deterioration
            scenario_data["market"] = self.generate_market_data(include_crash=False)
            
            # Modify credit data with higher default probabilities
            credit_data = self.generate_credit_data()
            if isinstance(credit_data["assets"], pd.DataFrame):
                # Increase PDs by factor of 2-3x
                credit_data["assets"]["pd_1y"] = credit_data["assets"]["pd_1y"] * np.random.uniform(2, 3, len(credit_data["assets"]))
                # Recalculate expected loss
                credit_data["assets"]["expected_loss"] = (
                    credit_data["assets"]["pd_1y"] * 
                    credit_data["assets"]["lgd"] * 
                    credit_data["assets"]["ead"] * 
                    credit_data["assets"]["market_value"] * 
                    credit_data["assets"]["quantity"]
                )
            scenario_data["credit"] = credit_data
            
        elif scenario_name == "liquidity_stress":
            # Liquidity stress scenario
            scenario_data["market"] = self.generate_market_data(include_crash=False)
            # In a real implementation, would generate stressed liquidity metrics
            
        elif scenario_name == "combined_stress":
            # Combined stress: market crash + credit deterioration
            crash_date = kwargs.get("crash_date", "2023-06-15")
            scenario_data["market"] = self.generate_market_data(
                include_crash=True,
                crash_date=crash_date
            )
            
            # Modify credit data with higher default probabilities
            credit_data = self.generate_credit_data()
            if isinstance(credit_data["assets"], pd.DataFrame):
                credit_data["assets"]["pd_1y"] = credit_data["assets"]["pd_1y"] * np.random.uniform(2.5, 4, len(credit_data["assets"]))
                credit_data["assets"]["expected_loss"] = (
                    credit_data["assets"]["pd_1y"] * 
                    credit_data["assets"]["lgd"] * 
                    credit_data["assets"]["ead"] * 
                    credit_data["assets"]["market_value"] * 
                    credit_data["assets"]["quantity"]
                )
            scenario_data["credit"] = credit_data
            
        else:
            self.logger.warning(f"Unknown scenario: {scenario_name}")
            
        # Create scenario directory and save data
        scenario_dir = settings.RAW_DATA_DIR / f"scenario_{scenario_name}"
        scenario_dir.mkdir(exist_ok=True)
        
        # Save market data
        if "market" in scenario_data:
            for key, df in scenario_data["market"].items():
                file_path = scenario_dir / f"market_{key}.csv"
                df.to_csv(file_path)
                
        # Save credit data
        if "credit" in scenario_data:
            for key, data in scenario_data["credit"].items():
                if isinstance(data, pd.DataFrame):
                    file_path = scenario_dir / f"credit_{key}.csv"
                    data.to_csv(file_path)
                    
        self.logger.info(f"Scenario {scenario_name} data saved to {scenario_dir}")
        return scenario_data


def run_simulation(scenarios=None):
    """Run simulation with specified scenarios
    
    Args:
        scenarios: List of scenarios to run
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting financial data simulation")
    
    simulator = FinancialDataSimulator()
    
    # Save base simulated data
    simulator.save_simulated_data()
    
    # Run specific scenarios if requested
    if scenarios:
        for scenario in scenarios:
            if isinstance(scenario, dict):
                scenario_name = scenario.get("name", "unnamed")
                simulator.run_simulation_scenario(scenario_name, **scenario)
            else:
                simulator.run_simulation_scenario(scenario)
                
    logger.info("Financial data simulation completed")
    

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run simulation with specific scenarios
    scenarios = [
        "base",
        "market_crash",
        "credit_deterioration",
        "combined_stress",
        {"name": "custom_crash", "crash_date": "2023-10-01", "crash_duration_days": 60}
    ]
    
    run_simulation(scenarios) 