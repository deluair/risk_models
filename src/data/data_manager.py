"""
Data management for the Financial Risk Analysis System
"""
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import datetime as dt

from src.core.config import settings


class DataManager:
    """Data loading and processing manager"""
    
    def __init__(self):
        """Initialize the data manager"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing DataManager")
        
        # Create data directories if they don't exist
        settings.RAW_DATA_DIR.mkdir(exist_ok=True)
        settings.PROCESSED_DATA_DIR.mkdir(exist_ok=True)
        
        # Initialize data containers
        self.market_data = {}
        self.credit_data = {}
        self.liquidity_data = {}
        self.operational_data = {}
        self.climate_data = {}
        self.cyber_data = {}
        self.network_data = {}
        
        # Processed data
        self.processed_data = {}
        
        self.logger.info("DataManager initialized successfully")
    
    def load_market_data(self):
        """Load market risk related data"""
        self.logger.info("Loading market risk data")
        
        try:
            # This is a placeholder - in a real system, you would load actual data
            # from files or APIs (Bloomberg, Refinitiv, etc.)
            
            # For demonstration, we'll create some synthetic data
            dates = pd.date_range(start='2018-01-01', end='2023-12-31', freq='B')
            
            # Market indices
            self.market_data['indices'] = pd.DataFrame({
                'date': dates,
                'sp500': np.random.normal(0.0005, 0.01, len(dates)).cumsum(),
                'nasdaq': np.random.normal(0.0006, 0.012, len(dates)).cumsum(),
                'russell2000': np.random.normal(0.0004, 0.011, len(dates)).cumsum(),
                'eurostoxx': np.random.normal(0.0003, 0.01, len(dates)).cumsum(),
                'nikkei': np.random.normal(0.0004, 0.0095, len(dates)).cumsum(),
                'ftse': np.random.normal(0.0003, 0.009, len(dates)).cumsum(),
            }).set_index('date')
            
            # Volatility indices
            self.market_data['volatility'] = pd.DataFrame({
                'date': dates,
                'vix': 15 + 5 * np.sin(np.linspace(0, 20, len(dates))) + np.random.normal(0, 3, len(dates)),
                'vstoxx': 17 + 6 * np.sin(np.linspace(0.5, 20.5, len(dates))) + np.random.normal(0, 3.2, len(dates)),
                'vxn': 16 + 5.5 * np.sin(np.linspace(0.2, 20.2, len(dates))) + np.random.normal(0, 2.8, len(dates)),
            }).set_index('date')
            
            # Interest rates
            self.market_data['rates'] = pd.DataFrame({
                'date': dates,
                'us_1y': 1.5 + np.cumsum(np.random.normal(0, 0.01, len(dates))),
                'us_5y': 2.0 + np.cumsum(np.random.normal(0, 0.009, len(dates))),
                'us_10y': 2.5 + np.cumsum(np.random.normal(0, 0.008, len(dates))),
                'euro_1y': 0.5 + np.cumsum(np.random.normal(0, 0.007, len(dates))),
                'euro_5y': 1.0 + np.cumsum(np.random.normal(0, 0.006, len(dates))),
                'euro_10y': 1.5 + np.cumsum(np.random.normal(0, 0.005, len(dates))),
            }).set_index('date')
            
            # Credit spreads
            self.market_data['credit_spreads'] = pd.DataFrame({
                'date': dates,
                'us_aa': 0.5 + np.cumsum(np.random.normal(0, 0.002, len(dates))),
                'us_a': 1.0 + np.cumsum(np.random.normal(0, 0.003, len(dates))),
                'us_bbb': 2.0 + np.cumsum(np.random.normal(0, 0.004, len(dates))),
                'us_bb': 3.5 + np.cumsum(np.random.normal(0, 0.006, len(dates))),
                'us_b': 5.0 + np.cumsum(np.random.normal(0, 0.01, len(dates))),
            }).set_index('date')
            
            self.logger.info("Market risk data loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading market risk data: {e}")
            raise
    
    def load_credit_data(self):
        """Load credit risk related data"""
        self.logger.info("Loading credit risk data")
        
        try:
            # Placeholder for synthetic credit risk data
            dates = pd.date_range(start='2018-01-01', end='2023-12-31', freq='M')
            
            # Default rates by sector
            self.credit_data['default_rates'] = pd.DataFrame({
                'date': dates,
                'financial': 0.02 + 0.01 * np.sin(np.linspace(0, 10, len(dates))) + np.random.normal(0, 0.002, len(dates)),
                'technology': 0.015 + 0.008 * np.sin(np.linspace(0.5, 10.5, len(dates))) + np.random.normal(0, 0.0015, len(dates)),
                'healthcare': 0.01 + 0.005 * np.sin(np.linspace(1, 11, len(dates))) + np.random.normal(0, 0.001, len(dates)),
                'consumer': 0.025 + 0.012 * np.sin(np.linspace(1.5, 11.5, len(dates))) + np.random.normal(0, 0.002, len(dates)),
                'industrial': 0.02 + 0.01 * np.sin(np.linspace(2, 12, len(dates))) + np.random.normal(0, 0.0018, len(dates)),
                'energy': 0.03 + 0.015 * np.sin(np.linspace(2.5, 12.5, len(dates))) + np.random.normal(0, 0.0025, len(dates)),
            }).set_index('date')
            
            # Credit ratings migration
            # This would normally be a transition matrix, but we'll simplify for the example
            ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']
            migration_data = {}
            
            for i, rating in enumerate(ratings[:-1]):  # Exclude 'D' as source
                # Probability of staying in the same rating
                migration_data[f'{rating}_to_{rating}'] = 0.7 + 0.05 * i + np.random.normal(0, 0.01, len(dates))
                
                # Probability of moving one notch down
                if i < len(ratings) - 2:
                    migration_data[f'{rating}_to_{ratings[i+1]}'] = 0.15 + 0.03 * i + np.random.normal(0, 0.005, len(dates))
                
                # Probability of moving two notches down
                if i < len(ratings) - 3:
                    migration_data[f'{rating}_to_{ratings[i+2]}'] = 0.05 + 0.01 * i + np.random.normal(0, 0.002, len(dates))
                
                # Probability of default
                migration_data[f'{rating}_to_D'] = 0.01 + 0.005 * i + np.random.normal(0, 0.001, len(dates))
            
            migration_data['date'] = dates
            self.credit_data['ratings_migration'] = pd.DataFrame(migration_data).set_index('date')
            
            # Corporate leverage ratios
            self.credit_data['leverage'] = pd.DataFrame({
                'date': dates,
                'financial': 10 + np.cumsum(np.random.normal(0, 0.1, len(dates))),
                'technology': 5 + np.cumsum(np.random.normal(0, 0.08, len(dates))),
                'healthcare': 6 + np.cumsum(np.random.normal(0, 0.07, len(dates))),
                'consumer': 7 + np.cumsum(np.random.normal(0, 0.09, len(dates))),
                'industrial': 8 + np.cumsum(np.random.normal(0, 0.1, len(dates))),
                'energy': 9 + np.cumsum(np.random.normal(0, 0.11, len(dates))),
            }).set_index('date')
            
            self.logger.info("Credit risk data loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading credit risk data: {e}")
            raise
    
    def load_liquidity_data(self):
        """Load liquidity risk related data"""
        self.logger.info("Loading liquidity risk data")
        
        try:
            # Placeholder for synthetic liquidity risk data
            dates = pd.date_range(start='2018-01-01', end='2023-12-31', freq='B')
            
            # Market liquidity metrics
            self.liquidity_data['market_liquidity'] = pd.DataFrame({
                'date': dates,
                'bid_ask_treasury': 0.01 + 0.005 * np.sin(np.linspace(0, 15, len(dates))) + np.random.normal(0, 0.001, len(dates)),
                'bid_ask_corporate': 0.03 + 0.01 * np.sin(np.linspace(0.5, 15.5, len(dates))) + np.random.normal(0, 0.003, len(dates)),
                'bid_ask_mbs': 0.02 + 0.008 * np.sin(np.linspace(1, 16, len(dates))) + np.random.normal(0, 0.002, len(dates)),
                'market_depth': 100 + 20 * np.sin(np.linspace(1.5, 16.5, len(dates))) + np.random.normal(0, 5, len(dates)),
                'price_impact': 0.05 + 0.02 * np.sin(np.linspace(2, 17, len(dates))) + np.random.normal(0, 0.005, len(dates)),
            }).set_index('date')
            
            # Funding liquidity metrics
            self.liquidity_data['funding_liquidity'] = pd.DataFrame({
                'date': dates,
                'libor_ois': 0.2 + 0.1 * np.sin(np.linspace(0, 12, len(dates))) + np.random.normal(0, 0.02, len(dates)),
                'repo_treasury': 0.1 + 0.05 * np.sin(np.linspace(0.5, 12.5, len(dates))) + np.random.normal(0, 0.01, len(dates)),
                'cp_treasury': 0.15 + 0.08 * np.sin(np.linspace(1, 13, len(dates))) + np.random.normal(0, 0.015, len(dates)),
                'cross_currency_basis': -20 + 10 * np.sin(np.linspace(1.5, 13.5, len(dates))) + np.random.normal(0, 2, len(dates)),
            }).set_index('date')
            
            # Non-bank intermediation metrics
            monthly_dates = pd.date_range(start='2018-01-01', end='2023-12-31', freq='M')
            self.liquidity_data['nonbank'] = pd.DataFrame({
                'date': monthly_dates,
                'money_market_assets': 3000 + np.cumsum(np.random.normal(0, 20, len(monthly_dates))),
                'hedge_fund_leverage': 2.5 + 0.5 * np.sin(np.linspace(0, 10, len(monthly_dates))) + np.random.normal(0, 0.1, len(monthly_dates)),
                'repo_volume': 2000 + np.cumsum(np.random.normal(0, 15, len(monthly_dates))),
                'securitization_issuance': 500 + 100 * np.sin(np.linspace(0, 8, len(monthly_dates))) + np.random.normal(0, 20, len(monthly_dates)),
            }).set_index('date')
            
            self.logger.info("Liquidity risk data loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading liquidity risk data: {e}")
            raise
    
    def load_operational_data(self):
        """Load operational risk related data"""
        self.logger.info("Loading operational risk data")
        
        try:
            # Placeholder for synthetic operational risk data
            monthly_dates = pd.date_range(start='2018-01-01', end='2023-12-31', freq='M')
            
            # Operational risk events
            self.operational_data['events'] = pd.DataFrame({
                'date': monthly_dates,
                'internal_fraud': np.random.poisson(2, len(monthly_dates)),
                'external_fraud': np.random.poisson(5, len(monthly_dates)),
                'employment_practices': np.random.poisson(1, len(monthly_dates)),
                'clients_products': np.random.poisson(3, len(monthly_dates)),
                'damage_to_assets': np.random.poisson(0.5, len(monthly_dates)),
                'business_disruption': np.random.poisson(2, len(monthly_dates)),
                'execution_delivery': np.random.poisson(4, len(monthly_dates)),
            }).set_index('date')
            
            # Operational risk losses
            loss_data = {}
            for event_type in self.operational_data['events'].columns:
                # Losses follow a lognormal distribution
                losses = []
                for count in self.operational_data['events'][event_type]:
                    if count > 0:
                        # Generate loss amounts based on type
                        if event_type in ['internal_fraud', 'external_fraud']:
                            multiplier = 2.0
                        elif event_type in ['employment_practices', 'damage_to_assets']:
                            multiplier = 1.0
                        else:
                            multiplier = 1.5
                            
                        loss = np.random.lognormal(mean=multiplier, sigma=0.8, size=count).sum()
                        losses.append(loss)
                    else:
                        losses.append(0)
                        
                loss_data[f'{event_type}_loss'] = losses
                
            loss_data['date'] = monthly_dates
            self.operational_data['losses'] = pd.DataFrame(loss_data).set_index('date')
            
            # Key Risk Indicators
            self.operational_data['kri'] = pd.DataFrame({
                'date': monthly_dates,
                'staff_turnover': 5 + 2 * np.sin(np.linspace(0, 8, len(monthly_dates))) + np.random.normal(0, 0.5, len(monthly_dates)),
                'system_availability': 99.5 + np.random.normal(0, 0.2, len(monthly_dates)),
                'processing_errors': 10 + 5 * np.sin(np.linspace(0, 10, len(monthly_dates))) + np.random.poisson(2, len(monthly_dates)),
                'regulatory_breaches': np.random.poisson(0.5, len(monthly_dates)),
                'customer_complaints': 50 + 20 * np.sin(np.linspace(0, 12, len(monthly_dates))) + np.random.poisson(5, len(monthly_dates)),
            }).set_index('date')
            
            self.logger.info("Operational risk data loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading operational risk data: {e}")
            raise
    
    def load_climate_data(self):
        """Load climate risk related data"""
        self.logger.info("Loading climate risk data")
        
        try:
            # Placeholder for synthetic climate risk data
            annual_dates = pd.date_range(start='2018-01-01', end='2023-12-31', freq='A')
            quarterly_dates = pd.date_range(start='2018-01-01', end='2023-12-31', freq='Q')
            
            # Physical risk indicators
            self.climate_data['physical_risk'] = pd.DataFrame({
                'date': quarterly_dates,
                'extreme_weather_events': np.random.poisson(5, len(quarterly_dates)),
                'agricultural_losses': 100 + 50 * np.sin(np.linspace(0, 8, len(quarterly_dates))) + np.random.normal(0, 10, len(quarterly_dates)),
                'coastal_flooding_risk': 20 + 10 * np.sin(np.linspace(0, 6, len(quarterly_dates))) + np.random.normal(0, 2, len(quarterly_dates)),
                'water_stress_index': 30 + 15 * np.sin(np.linspace(0, 7, len(quarterly_dates))) + np.random.normal(0, 3, len(quarterly_dates)),
                'wildfire_risk': 15 + 8 * np.sin(np.linspace(0, 9, len(quarterly_dates))) + np.random.normal(0, 2, len(quarterly_dates)),
            }).set_index('date')
            
            # Transition risk indicators
            self.climate_data['transition_risk'] = pd.DataFrame({
                'date': annual_dates,
                'carbon_price': 25 + np.cumsum(np.random.normal(2, 1, len(annual_dates))),
                'renewable_investment': 200 + np.cumsum(np.random.normal(20, 5, len(annual_dates))),
                'stranded_assets': 500 - np.cumsum(np.random.normal(30, 10, len(annual_dates))),
                'green_bonds_issuance': 100 + np.cumsum(np.random.normal(15, 5, len(annual_dates))),
                'policy_stringency': 40 + np.cumsum(np.random.normal(5, 2, len(annual_dates))),
            }).set_index('date')
            
            # Sector exposure
            self.climate_data['sector_exposure'] = pd.DataFrame({
                'sector': ['Energy', 'Utilities', 'Materials', 'Industrials', 'Consumer', 'Healthcare', 'Financial', 'Technology'],
                'carbon_intensity': [85, 70, 65, 40, 25, 15, 10, 8],
                'transition_readiness': [30, 45, 40, 60, 70, 80, 75, 90],
                'physical_risk_exposure': [65, 60, 55, 45, 40, 30, 25, 20],
                'vulnerability_score': [75, 65, 60, 45, 35, 25, 30, 15],
            })
            
            self.logger.info("Climate risk data loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading climate risk data: {e}")
            raise
    
    def load_cyber_data(self):
        """Load cyber risk related data"""
        self.logger.info("Loading cyber risk data")
        
        try:
            # Placeholder for synthetic cyber risk data
            monthly_dates = pd.date_range(start='2018-01-01', end='2023-12-31', freq='M')
            
            # Cyber incidents
            self.cyber_data['incidents'] = pd.DataFrame({
                'date': monthly_dates,
                'ddos_attacks': np.random.poisson(3, len(monthly_dates)),
                'data_breaches': np.random.poisson(1, len(monthly_dates)),
                'malware': np.random.poisson(5, len(monthly_dates)),
                'insider_threats': np.random.poisson(0.5, len(monthly_dates)),
                'phishing': np.random.poisson(8, len(monthly_dates)),
                'ransomware': np.random.poisson(2, len(monthly_dates)),
            }).set_index('date')
            
            # Financial impact
            impact_data = {}
            for incident_type in self.cyber_data['incidents'].columns:
                # Financial impacts follow a lognormal distribution
                impacts = []
                for count in self.cyber_data['incidents'][incident_type]:
                    if count > 0:
                        # Generate impact amounts based on type
                        if incident_type in ['data_breaches', 'ransomware']:
                            multiplier = 2.5
                        elif incident_type in ['ddos_attacks', 'malware']:
                            multiplier = 1.5
                        else:
                            multiplier = 1.0
                            
                        impact = np.random.lognormal(mean=multiplier, sigma=1.0, size=count).sum()
                        impacts.append(impact)
                    else:
                        impacts.append(0)
                        
                impact_data[f'{incident_type}_impact'] = impacts
                
            impact_data['date'] = monthly_dates
            self.cyber_data['financial_impact'] = pd.DataFrame(impact_data).set_index('date')
            
            # Cyber security metrics
            self.cyber_data['security_metrics'] = pd.DataFrame({
                'date': monthly_dates,
                'vulnerability_count': 100 + np.random.poisson(20, len(monthly_dates)) - np.linspace(0, 40, len(monthly_dates)),
                'patch_time_days': 30 + np.random.normal(0, 5, len(monthly_dates)) - np.linspace(0, 15, len(monthly_dates)),
                'security_spending': 1000 + np.cumsum(np.random.normal(10, 3, len(monthly_dates))),
                'employee_training': 70 + np.cumsum(np.random.normal(0.5, 0.2, len(monthly_dates))),
                'third_party_vendors': 50 + np.cumsum(np.random.normal(0.3, 0.1, len(monthly_dates))),
            }).set_index('date')
            
            self.logger.info("Cyber risk data loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading cyber risk data: {e}")
            raise
    
    def load_network_data(self):
        """Load network data for interconnected risk analysis"""
        self.logger.info("Loading network data")
        
        try:
            # Placeholder for synthetic network data
            
            # Define financial institutions
            institutions = [
                'Bank1', 'Bank2', 'Bank3', 'Bank4', 'Bank5',
                'Insurer1', 'Insurer2', 'Insurer3',
                'AssetManager1', 'AssetManager2', 'AssetManager3',
                'Hedge1', 'Hedge2',
                'NBFI1', 'NBFI2', 'NBFI3'
            ]
            n_institutions = len(institutions)
            
            # Create interbank exposure matrix (asymmetric)
            np.random.seed(42)  # For reproducibility
            exposure_matrix = np.zeros((n_institutions, n_institutions))
            
            # Fill the matrix with random exposures
            for i in range(n_institutions):
                for j in range(n_institutions):
                    if i != j:  # No exposure to self
                        # Banks have higher mutual exposures
                        if i < 5 and j < 5:
                            exposure_matrix[i, j] = np.random.exponential(100)
                        # Banks and insurers have medium exposures
                        elif (i < 5 and 5 <= j < 8) or (5 <= i < 8 and j < 5):
                            exposure_matrix[i, j] = np.random.exponential(70)
                        # All other relationships have lower exposures
                        else:
                            exposure_matrix[i, j] = np.random.exponential(40)
            
            # Store as DataFrame
            self.network_data['exposures'] = pd.DataFrame(
                exposure_matrix,
                index=institutions,
                columns=institutions
            )
            
            # Common asset holdings
            assets = ['TreasuryA', 'TreasuryB', 'CorporateBondA', 'CorporateBondB', 
                     'MBS', 'Equities', 'RealEstate', 'Commodities', 'EmergingMarket']
            
            holdings_matrix = np.zeros((n_institutions, len(assets)))
            
            # Fill with random holdings
            for i in range(n_institutions):
                for j in range(len(assets)):
                    # Banks have more treasuries and MBS
                    if i < 5 and j in [0, 1, 4]:
                        holdings_matrix[i, j] = np.random.uniform(50, 200)
                    # Insurers have more corporate bonds
                    elif 5 <= i < 8 and j in [2, 3]:
                        holdings_matrix[i, j] = np.random.uniform(30, 150)
                    # Asset managers have diverse holdings
                    elif 8 <= i < 11:
                        holdings_matrix[i, j] = np.random.uniform(20, 100)
                    # Hedge funds have more complex assets
                    elif 11 <= i < 13 and j in [5, 6, 7, 8]:
                        holdings_matrix[i, j] = np.random.uniform(40, 180)
                    # NBFIs have a mix
                    else:
                        holdings_matrix[i, j] = np.random.uniform(10, 50)
            
            # Store as DataFrame
            self.network_data['holdings'] = pd.DataFrame(
                holdings_matrix,
                index=institutions,
                columns=assets
            )
            
            # Institution characteristics
            characteristics = {
                'size': np.random.lognormal(mean=5, sigma=1, size=n_institutions),
                'leverage': np.random.uniform(5, 25, n_institutions),
                'liquidity_ratio': np.random.uniform(0.05, 0.3, n_institutions),
                'interconnectedness': np.random.uniform(0.1, 0.9, n_institutions),
                'complexity': np.random.uniform(0.2, 0.8, n_institutions),
            }
            
            # Store as DataFrame
            self.network_data['characteristics'] = pd.DataFrame(
                characteristics,
                index=institutions
            )
            
            self.logger.info("Network data loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading network data: {e}")
            raise
    
    def process_all_data(self):
        """Process and prepare all loaded data for analysis"""
        self.logger.info("Processing all data")
        
        try:
            # Process market data
            self.process_market_data()
            
            # Process credit data
            self.process_credit_data()
            
            # Process liquidity data
            self.process_liquidity_data()
            
            # Process operational data
            self.process_operational_data()
            
            # Process climate data
            self.process_climate_data()
            
            # Process cyber data
            self.process_cyber_data()
            
            # Process network data
            self.process_network_data()
            
            # Integrate all processed data
            self.integrate_data()
            
            self.logger.info("All data processed successfully")
            
        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
            raise
    
    def process_market_data(self):
        """Process market risk data"""
        # Placeholder for data processing logic
        # This would include calculating returns, volatilities, correlations, etc.
        
        if not self.market_data:
            self.logger.warning("No market data to process")
            return
        
        # Calculate returns
        if 'indices' in self.market_data:
            self.processed_data['market_returns'] = self.market_data['indices'].pct_change().dropna()
            self.processed_data['market_volatility'] = self.processed_data['market_returns'].rolling(21).std() * np.sqrt(252)
            self.processed_data['market_correlation'] = self.processed_data['market_returns'].rolling(63).corr()
    
    def process_credit_data(self):
        """Process credit risk data"""
        # Placeholder for credit data processing
        pass
    
    def process_liquidity_data(self):
        """Process liquidity risk data"""
        # Placeholder for liquidity data processing
        pass
    
    def process_operational_data(self):
        """Process operational risk data"""
        # Placeholder for operational data processing
        pass
    
    def process_climate_data(self):
        """Process climate risk data"""
        # Placeholder for climate data processing
        pass
    
    def process_cyber_data(self):
        """Process cyber risk data"""
        # Placeholder for cyber data processing
        pass
    
    def process_network_data(self):
        """Process network data for interconnectedness analysis"""
        # Placeholder for network data processing
        pass
    
    def integrate_data(self):
        """Integrate data from different risk categories"""
        # Placeholder for data integration logic
        pass 