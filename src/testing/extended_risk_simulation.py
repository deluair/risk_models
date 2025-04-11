"""
Extended Risk Simulation Module

This module provides functionality to generate synthetic data for extended risk types:
- Operational Risk
- Climate Risk
- Cyber Risk
- AI Risk
- Digitalization Risk

These simulations can be used for testing and development of risk visualization
and analysis systems.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ExtendedRiskSimulator:
    """
    Class for simulating extended risk data for various risk categories.
    """
    
    def __init__(self, seed=None):
        """
        Initialize the simulator with an optional random seed for reproducibility.
        
        Args:
            seed (int, optional): Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Define risk type metrics
        self.risk_metrics = {
            'operational_risk': [
                'process_failure', 'human_error', 'system_failure', 
                'legal_risk', 'regulatory_compliance', 'fraud_risk', 'overall'
            ],
            'climate_risk': [
                'transition_risk', 'physical_risk', 'regulatory_risk',
                'market_risk', 'technology_risk', 'reputation_risk', 'overall'
            ],
            'cyber_risk': [
                'data_breach', 'system_outage', 'ddos', 'ransomware',
                'phishing', 'insider_threat', 'overall'
            ],
            'ai_risk': [
                'model_risk', 'data_quality', 'bias', 'explainability',
                'stability', 'regulatory_compliance', 'overall'
            ],
            'digitalization_risk': [
                'legacy_systems', 'digital_transformation', 'tech_debt',
                'innovation_gap', 'digital_competence', 'data_management', 'overall'
            ]
        }
        
        # Initialize base parameters for each risk type
        self.base_params = {
            'operational_risk': {
                'mean': 0.3,
                'volatility': 0.05,
                'risk_coefficient': 1.2
            },
            'climate_risk': {
                'mean': 0.25,
                'volatility': 0.03,
                'risk_coefficient': 1.0
            },
            'cyber_risk': {
                'mean': 0.4,
                'volatility': 0.07,
                'risk_coefficient': 1.5
            },
            'ai_risk': {
                'mean': 0.35,
                'volatility': 0.06,
                'risk_coefficient': 1.3
            },
            'digitalization_risk': {
                'mean': 0.3,
                'volatility': 0.04,
                'risk_coefficient': 1.1
            }
        }
        
        # Scenario parameter modifiers
        self.scenario_modifiers = {
            'base': {
                'mean_mod': 1.0,
                'vol_mod': 1.0,
                'event_probability': 0.01
            },
            'market_crash': {
                'mean_mod': 1.2,
                'vol_mod': 1.8,
                'event_probability': 0.05
            },
            'credit_deterioration': {
                'mean_mod': 1.3,
                'vol_mod': 1.5,
                'event_probability': 0.03
            },
            'combined_stress': {
                'mean_mod': 1.5,
                'vol_mod': 2.0,
                'event_probability': 0.07
            }
        }
    
    def generate_risk_data(self, risk_type, n_days=252, start_date=None, scenario='base', custom_events=None):
        """
        Generate synthetic risk data for a specified risk type.
        
        Args:
            risk_type (str): Type of risk to simulate
            n_days (int): Number of days to simulate
            start_date (datetime, optional): Start date for the simulation
            scenario (str): Scenario to simulate (base, market_crash, credit_deterioration, combined_stress)
            custom_events (list, optional): List of custom risk events to apply
            
        Returns:
            pd.DataFrame: DataFrame containing simulated risk data
        """
        if risk_type not in self.risk_metrics:
            raise ValueError(f"Unknown risk type: {risk_type}. Available types: {list(self.risk_metrics.keys())}")
        
        if scenario not in self.scenario_modifiers:
            logger.warning(f"Unknown scenario: {scenario}. Using 'base' scenario.")
            scenario = 'base'
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=n_days)
        
        # Get base parameters and apply scenario modifiers
        base_params = self.base_params[risk_type].copy()
        scenario_mod = self.scenario_modifiers[scenario]
        
        mean = base_params['mean'] * scenario_mod['mean_mod']
        volatility = base_params['volatility'] * scenario_mod['vol_mod']
        risk_coefficient = base_params['risk_coefficient']
        event_probability = scenario_mod['event_probability']
        
        # Generate dates
        dates = [start_date + timedelta(days=i) for i in range(n_days)]
        
        # Generate base risk values for each metric
        metrics = self.risk_metrics[risk_type]
        risk_data = {metric: [] for metric in metrics}
        risk_data['date'] = dates
        
        # Generate initial values for each metric (0-1 scale)
        initial_values = {}
        for metric in metrics:
            if metric == 'overall':
                # Overall will be calculated as a weighted average later
                initial_values[metric] = 0.0
            else:
                # Random starting value between 0.1 and 0.5
                initial_values[metric] = 0.1 + self.rng.random() * 0.4
        
        # Generate time series with mean-reversion and occasional jumps for each metric
        for metric in metrics:
            if metric == 'overall':
                continue  # Skip overall for now
                
            current_value = initial_values[metric]
            values = []
            
            for i in range(n_days):
                # Add random noise with mean reversion
                noise = self.rng.normal(0, volatility)
                mean_reversion = (mean - current_value) * 0.05  # 5% reversion to mean
                
                # Check for random risk events
                if self.rng.random() < event_probability:
                    # Add a jump (up or down, but more likely up for risk)
                    jump_direction = 1 if self.rng.random() < 0.7 else -1
                    jump_size = self.rng.random() * 0.2 * risk_coefficient  # Max 20% jump, scaled by risk coefficient
                    current_value += jump_direction * jump_size
                
                # Update value with mean reversion and noise
                current_value += mean_reversion + noise
                
                # Apply constraints (0-1 scale)
                current_value = max(0.01, min(0.99, current_value))
                values.append(current_value)
            
            risk_data[metric] = values
        
        # Apply custom events if specified
        if custom_events:
            self._apply_custom_events(risk_data, custom_events, n_days)
        
        # Calculate overall risk as weighted average of other metrics
        overall_values = []
        for i in range(n_days):
            metric_values = [risk_data[m][i] for m in metrics if m != 'overall']
            # Random weights that sum to 1
            weights = self.rng.dirichlet(np.ones(len(metric_values)))
            overall_value = np.sum(np.array(metric_values) * weights)
            overall_values.append(overall_value)
        
        risk_data['overall'] = overall_values
        
        # Convert to DataFrame
        df = pd.DataFrame(risk_data)
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def _apply_custom_events(self, risk_data, custom_events, n_days):
        """
        Apply custom risk events to the generated data.
        
        Args:
            risk_data (dict): Dictionary of risk data
            custom_events (list): List of custom events
            n_days (int): Number of days in the simulation
        """
        for event in custom_events:
            day = event.get('day', 0)
            if day >= n_days:
                logger.warning(f"Event day {day} is beyond simulation range {n_days}. Skipping.")
                continue
                
            effect = event.get('effect', {})
            duration = event.get('duration', 1)
            
            for metric, impact in effect.items():
                if metric in risk_data:
                    # Apply impact over the duration
                    for d in range(duration):
                        if day + d < n_days:
                            # Apply the impact (ensure it stays within 0-1 range)
                            risk_data[metric][day + d] = min(0.99, risk_data[metric][day + d] * (1 + impact))
                else:
                    logger.warning(f"Metric {metric} not found in risk data. Skipping.")

def save_extended_risk_data(output_dir="data/simulated", risk_type=None, scenario='base', n_days=252, seed=None):
    """
    Generate and save extended risk data to CSV files.
    
    Args:
        output_dir (str): Directory to save output files
        risk_type (str, optional): Risk type to simulate. If None, all types are simulated.
        scenario (str): Scenario to simulate
        n_days (int): Number of days to simulate
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        dict: Dictionary of generated DataFrames
    """
    simulator = ExtendedRiskSimulator(seed=seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    risk_types = [risk_type] if risk_type else simulator.risk_metrics.keys()
    dataframes = {}
    
    for risk_type in risk_types:
        df = simulator.generate_risk_data(
            risk_type=risk_type,
            n_days=n_days,
            scenario=scenario
        )
        
        output_file = os.path.join(output_dir, f"{risk_type}_{scenario}.csv")
        df.to_csv(output_file, index=False)
        
        dataframes[risk_type] = df
        logger.info(f"Generated and saved {risk_type} data for {scenario} scenario to {output_file}")
    
    return dataframes

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    save_extended_risk_data(scenario='base', n_days=252)
    save_extended_risk_data(scenario='market_crash', n_days=252)
    
    # Example with custom events
    simulator = ExtendedRiskSimulator(seed=42)
    custom_events = [
        {'day': 50, 'effect': {'data_breach': 0.5, 'overall': 0.3}, 'duration': 10},
        {'day': 150, 'effect': {'system_outage': 0.7, 'overall': 0.4}, 'duration': 5}
    ]
    
    cyber_df = simulator.generate_risk_data(
        risk_type='cyber_risk',
        n_days=252,
        scenario='base',
        custom_events=custom_events
    )
    
    os.makedirs("data/simulated", exist_ok=True)
    cyber_df.to_csv("data/simulated/cyber_risk_custom_events.csv", index=False)
    logger.info("Generated cyber risk data with custom events") 