"""
Extended Risk Modules for the Financial Risk Analysis System
Implements additional risk types: operational, climate, cyber, AI, and digitalization
"""
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import random

from src.core.config import settings

class ExtendedRiskGenerator:
    """
    Generator for extended risk metrics including:
    - Operational risk
    - Climate risk
    - Cyber risk
    - AI risk
    - Digitalization risk
    """
    
    def __init__(self, seed=None):
        """Initialize the risk generator
        
        Args:
            seed: Random seed for reproducibility
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing ExtendedRiskGenerator")
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Risk parameters
        self.risk_params = {
            'operational_risk': {
                'baseline': 0.3,
                'volatility': 0.05,
                'trend': 0.001,  # Slight upward trend
            },
            'climate_risk': {
                'baseline': 0.4,
                'volatility': 0.08,
                'seasonal_factor': 0.1,  # Seasonal variations
                'trend': 0.002,  # Increasing trend due to climate change
            },
            'cyber_risk': {
                'baseline': 0.35,
                'volatility': 0.15,
                'shock_probability': 0.05,  # Probability of cyber attack
                'shock_magnitude': 0.3,  # Impact of cyber attack
            },
            'ai_risk': {
                'baseline': 0.25,
                'volatility': 0.12,
                'trend': 0.003,  # Increasing with AI adoption
            },
            'digitalization': {
                'baseline': 0.3,
                'volatility': 0.07,
                'trend': 0.002,  # Increasing with digital transformation
            },
        }
        
        self.logger.info("ExtendedRiskGenerator initialized successfully")
    
    def generate_operational_risk(self, start_date, periods=365, include_events=True):
        """Generate operational risk metrics
        
        Args:
            start_date: Start date for the time series
            periods: Number of periods (days)
            include_events: Whether to include operational risk events
            
        Returns:
            DataFrame with operational risk metrics
        """
        self.logger.info("Generating operational risk metrics")
        
        # Create date range
        dates = pd.date_range(start=start_date, periods=periods)
        
        # Base operational risk score
        params = self.risk_params['operational_risk']
        base_risk = params['baseline']
        
        # Generate daily operational risk scores with random noise
        risk_scores = np.random.normal(
            base_risk + np.linspace(0, params['trend'] * periods, periods),
            params['volatility'], 
            periods
        )
        
        # Add operational risk events
        if include_events:
            # Simulate random operational risk events
            num_events = np.random.poisson(3)  # Average 3 events per year
            event_days = np.random.choice(range(periods), size=num_events, replace=False)
            event_magnitudes = np.random.uniform(0.05, 0.25, size=num_events)
            
            for day, magnitude in zip(event_days, event_magnitudes):
                # Spike on event day
                risk_scores[day] += magnitude
                
                # Decay over next 10 days
                decay_period = min(10, periods - day - 1)
                decay_factors = np.linspace(magnitude, 0, decay_period + 2)[1:-1]
                risk_scores[day+1:day+1+decay_period] += decay_factors
        
        # Ensure values are between 0 and 1
        risk_scores = np.clip(risk_scores, 0, 1)
        
        # Create DataFrame
        df = pd.DataFrame({
            'operational_risk_score': risk_scores,
        }, index=dates)
        
        # Add operational risk components
        df['process_risk'] = np.random.normal(risk_scores * 0.8, 0.05)
        df['people_risk'] = np.random.normal(risk_scores * 1.2, 0.07)
        df['systems_risk'] = np.random.normal(risk_scores * 0.9, 0.06)
        df['external_events_risk'] = np.random.normal(risk_scores * 1.1, 0.08)
        
        # Clip values again
        df = df.clip(0, 1)
        
        return df
    
    def generate_climate_risk(self, start_date, periods=365):
        """Generate climate risk metrics
        
        Args:
            start_date: Start date for the time series
            periods: Number of periods (days)
            
        Returns:
            DataFrame with climate risk metrics
        """
        self.logger.info("Generating climate risk metrics")
        
        # Create date range
        dates = pd.date_range(start=start_date, periods=periods)
        
        # Base climate risk score
        params = self.risk_params['climate_risk']
        base_risk = params['baseline']
        
        # Generate daily climate risk scores with trend and seasonality
        trend = np.linspace(0, params['trend'] * periods, periods)
        
        # Add seasonality (higher in summer/winter)
        t = np.arange(periods)
        seasonality = params['seasonal_factor'] * np.sin(2 * np.pi * t / 365 * 2)
        
        # Combine trend, seasonality and random noise
        risk_scores = base_risk + trend + seasonality + np.random.normal(0, params['volatility'], periods)
        
        # Ensure values are between 0 and 1
        risk_scores = np.clip(risk_scores, 0, 1)
        
        # Create DataFrame
        df = pd.DataFrame({
            'climate_risk_score': risk_scores,
        }, index=dates)
        
        # Add climate risk components
        df['transition_risk'] = np.random.normal(risk_scores * 0.9, 0.06)
        df['physical_risk'] = np.random.normal(risk_scores * 1.1, 0.08)
        df['regulatory_risk'] = np.random.normal(risk_scores * 0.85, 0.05)
        
        # Clip values again
        df = df.clip(0, 1)
        
        return df
    
    def generate_cyber_risk(self, start_date, periods=365, include_attacks=True):
        """Generate cyber risk metrics
        
        Args:
            start_date: Start date for the time series
            periods: Number of periods (days)
            include_attacks: Whether to include cyber attacks
            
        Returns:
            DataFrame with cyber risk metrics
        """
        self.logger.info("Generating cyber risk metrics")
        
        # Create date range
        dates = pd.date_range(start=start_date, periods=periods)
        
        # Base cyber risk score
        params = self.risk_params['cyber_risk']
        base_risk = params['baseline']
        
        # Generate daily cyber risk scores with random noise
        risk_scores = np.random.normal(base_risk, params['volatility'], periods)
        
        # Add cyber attacks
        attack_days = []
        if include_attacks:
            # Simulate random cyber attacks
            for i in range(periods):
                if np.random.random() < params['shock_probability']:
                    # Record attack day
                    attack_days.append(dates[i])
                    
                    # Spike on attack day
                    risk_scores[i] += params['shock_magnitude']
                    
                    # Decay over next 20 days
                    decay_period = min(20, periods - i - 1)
                    decay_factors = np.linspace(params['shock_magnitude'], 0, decay_period + 2)[1:-1]
                    risk_scores[i+1:i+1+decay_period] += decay_factors
        
        # Ensure values are between 0 and 1
        risk_scores = np.clip(risk_scores, 0, 1)
        
        # Create DataFrame
        df = pd.DataFrame({
            'cyber_risk_score': risk_scores,
        }, index=dates)
        
        # Add cyber risk components
        df['data_breach_risk'] = np.random.normal(risk_scores * 1.1, 0.08)
        df['ransomware_risk'] = np.random.normal(risk_scores * 1.2, 0.1)
        df['infrastructure_risk'] = np.random.normal(risk_scores * 0.9, 0.06)
        
        # Mark attack days
        df['attack_day'] = [1 if date in attack_days else 0 for date in dates]
        
        # Clip values again (except attack_day)
        cols_to_clip = [col for col in df.columns if col != 'attack_day']
        df[cols_to_clip] = df[cols_to_clip].clip(0, 1)
        
        return df
    
    def generate_ai_risk(self, start_date, periods=365):
        """Generate AI risk metrics
        
        Args:
            start_date: Start date for the time series
            periods: Number of periods (days)
            
        Returns:
            DataFrame with AI risk metrics
        """
        self.logger.info("Generating AI risk metrics")
        
        # Create date range
        dates = pd.date_range(start=start_date, periods=periods)
        
        # Base AI risk score
        params = self.risk_params['ai_risk']
        base_risk = params['baseline']
        
        # Generate daily AI risk scores with trend and random noise
        trend = np.linspace(0, params['trend'] * periods, periods)
        risk_scores = base_risk + trend + np.random.normal(0, params['volatility'], periods)
        
        # Ensure values are between 0 and 1
        risk_scores = np.clip(risk_scores, 0, 1)
        
        # Create DataFrame
        df = pd.DataFrame({
            'ai_risk_score': risk_scores,
        }, index=dates)
        
        # Add AI risk components
        df['model_risk'] = np.random.normal(risk_scores * 0.95, 0.07)
        df['bias_risk'] = np.random.normal(risk_scores * 0.85, 0.05)
        df['decision_risk'] = np.random.normal(risk_scores * 1.1, 0.08)
        
        # Clip values again
        df = df.clip(0, 1)
        
        return df
    
    def generate_digitalization_risk(self, start_date, periods=365):
        """Generate digitalization risk metrics
        
        Args:
            start_date: Start date for the time series
            periods: Number of periods (days)
            
        Returns:
            DataFrame with digitalization risk metrics
        """
        self.logger.info("Generating digitalization risk metrics")
        
        # Create date range
        dates = pd.date_range(start=start_date, periods=periods)
        
        # Base digitalization risk score
        params = self.risk_params['digitalization']
        base_risk = params['baseline']
        
        # Generate daily digitalization risk scores with trend and random noise
        trend = np.linspace(0, params['trend'] * periods, periods)
        risk_scores = base_risk + trend + np.random.normal(0, params['volatility'], periods)
        
        # Ensure values are between 0 and 1
        risk_scores = np.clip(risk_scores, 0, 1)
        
        # Create DataFrame
        df = pd.DataFrame({
            'digitalization_risk_score': risk_scores,
        }, index=dates)
        
        # Add digitalization risk components
        df['digital_transformation_risk'] = np.random.normal(risk_scores * 0.9, 0.06)
        df['tech_adoption_risk'] = np.random.normal(risk_scores * 1.05, 0.07)
        df['legacy_system_risk'] = np.random.normal(risk_scores * 1.15, 0.09)
        
        # Clip values again
        df = df.clip(0, 1)
        
        return df
    
    def generate_all_extended_risks(self, start_date=None, periods=365, scenario="base"):
        """Generate all extended risk metrics
        
        Args:
            start_date: Start date for the time series (default: today - 1 year)
            periods: Number of periods (days)
            scenario: Scenario name for risk adjustments
            
        Returns:
            Dictionary with DataFrames for each risk type
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=periods)
        
        self.logger.info(f"Generating all extended risks for scenario: {scenario}")
        
        # Apply scenario adjustments
        self._apply_scenario_adjustments(scenario)
        
        # Generate all risk metrics
        extended_risks = {
            'operational_risk': self.generate_operational_risk(start_date, periods),
            'climate_risk': self.generate_climate_risk(start_date, periods),
            'cyber_risk': self.generate_cyber_risk(start_date, periods),
            'ai_risk': self.generate_ai_risk(start_date, periods),
            'digitalization_risk': self.generate_digitalization_risk(start_date, periods)
        }
        
        # Reset scenario adjustments
        self._reset_risk_params()
        
        return extended_risks
    
    def save_extended_risks(self, output_dir, extended_risks=None, start_date=None, periods=365, scenario="base"):
        """Save all extended risk metrics to CSV files
        
        Args:
            output_dir: Directory to save risk data
            extended_risks: Pre-generated extended risks (if None, will generate)
            start_date: Start date for the time series
            periods: Number of periods (days)
            scenario: Scenario name for risk adjustments
            
        Returns:
            Dictionary with file paths of saved data
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving extended risks to {output_dir}")
        
        # Generate risks if not provided
        if extended_risks is None:
            extended_risks = self.generate_all_extended_risks(start_date, periods, scenario)
        
        # Save each risk type
        saved_files = {}
        for risk_type, risk_data in extended_risks.items():
            file_path = output_dir / f"{risk_type}.csv"
            risk_data.to_csv(file_path)
            saved_files[risk_type] = file_path
            self.logger.info(f"Saved {file_path}")
        
        return saved_files
    
    def _apply_scenario_adjustments(self, scenario):
        """Apply risk adjustments based on scenario
        
        Args:
            scenario: Scenario name
        """
        # Store original params
        self._original_params = self.risk_params.copy()
        
        # Apply scenario-specific adjustments
        if scenario == "market_crash":
            # During market crash, operational and cyber risks increase
            self.risk_params['operational_risk']['baseline'] *= 1.3
            self.risk_params['operational_risk']['volatility'] *= 1.5
            self.risk_params['cyber_risk']['baseline'] *= 1.2
            self.risk_params['cyber_risk']['shock_probability'] *= 2
            
        elif scenario == "credit_deterioration":
            # During credit deterioration, operational and digitalization risks increase
            self.risk_params['operational_risk']['baseline'] *= 1.2
            self.risk_params['digitalization_risk']['baseline'] *= 1.15
            
        elif scenario == "combined_stress":
            # Combined effect of market crash and credit deterioration
            self.risk_params['operational_risk']['baseline'] *= 1.4
            self.risk_params['operational_risk']['volatility'] *= 1.5
            self.risk_params['cyber_risk']['baseline'] *= 1.3
            self.risk_params['cyber_risk']['shock_probability'] *= 2.5
            self.risk_params['digitalization_risk']['baseline'] *= 1.25
            self.risk_params['climate_risk']['baseline'] *= 1.1
            
        elif scenario == "climate_transition":
            # Climate transition risk scenario
            self.risk_params['climate_risk']['baseline'] *= 1.5
            self.risk_params['climate_risk']['volatility'] *= 1.7
            self.risk_params['climate_risk']['trend'] *= 2
            
        elif scenario == "cyber_attack":
            # Major cyber attack scenario
            self.risk_params['cyber_risk']['baseline'] *= 1.8
            self.risk_params['cyber_risk']['volatility'] *= 2
            self.risk_params['cyber_risk']['shock_probability'] *= 4
            self.risk_params['cyber_risk']['shock_magnitude'] *= 1.5
            
        elif scenario == "ai_disruption":
            # AI disruption scenario
            self.risk_params['ai_risk']['baseline'] *= 1.6
            self.risk_params['ai_risk']['volatility'] *= 1.8
            self.risk_params['ai_risk']['trend'] *= 3
            
        elif scenario == "digital_transformation":
            # Accelerated digital transformation scenario
            self.risk_params['digitalization_risk']['baseline'] *= 1.4
            self.risk_params['digitalization_risk']['trend'] *= 2.5
    
    def _reset_risk_params(self):
        """Reset risk parameters to original values"""
        if hasattr(self, '_original_params'):
            self.risk_params = self._original_params.copy()


# Function to run from command line
def generate_extended_risks(output_dir=None, scenario="base"):
    """Generate and save extended risk metrics
    
    Args:
        output_dir: Directory to save risk data (default: data/raw/scenario_{scenario})
        scenario: Scenario name
    
    Returns:
        Dictionary with file paths of saved data
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    if output_dir is None:
        output_dir = Path(f"data/raw/scenario_{scenario}")
    
    logger.info(f"Generating extended risks for scenario: {scenario}")
    
    # Generate extended risks
    generator = ExtendedRiskGenerator(seed=42)
    extended_risks = generator.generate_all_extended_risks(scenario=scenario)
    
    # Save extended risks
    saved_files = generator.save_extended_risks(output_dir, extended_risks, scenario=scenario)
    
    logger.info(f"Extended risks saved to {output_dir}")
    return saved_files


if __name__ == "__main__":
    # Generate extended risks for all scenarios
    scenarios = ["base", "market_crash", "credit_deterioration", "combined_stress", 
                "climate_transition", "cyber_attack", "ai_disruption", "digital_transformation"]
    
    for scenario in scenarios:
        generate_extended_risks(scenario=scenario) 