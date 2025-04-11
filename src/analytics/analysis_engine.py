"""
Analysis engine for the Financial Risk Analysis System
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import networkx as nx
from datetime import datetime

from src.core.config import settings
from src.data.data_manager import DataManager
from src.risk_modules.risk_registry import RiskRegistry
from src.models.model_registry import ModelRegistry


class AnalysisEngine:
    """Core analysis engine for risk calculations"""
    
    def __init__(self, data_manager: DataManager, risk_registry: RiskRegistry, model_registry: ModelRegistry):
        """Initialize the analysis engine
        
        Args:
            data_manager: Data manager instance with loaded data
            risk_registry: Risk registry with risk definitions
            model_registry: Model registry with trained models
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing AnalysisEngine")
        
        self.data_manager = data_manager
        self.risk_registry = risk_registry
        self.model_registry = model_registry
        
        # Analysis outputs
        self.results = {}
        
        self.logger.info("AnalysisEngine initialized successfully")
    
    def analyze_market_risk(self) -> Dict[str, Any]:
        """Analyze market risk
        
        Returns:
            Dictionary of market risk metrics and analysis results
        """
        self.logger.info("Analyzing market risk")
        
        try:
            results = {}
            
            # Get relevant data
            if not self.data_manager.processed_data.get('market_returns') is not None:
                self.logger.warning("Market returns data not found. Processing market data.")
                self.data_manager.process_market_data()
                
            returns = self.data_manager.processed_data.get('market_returns')
            volatility = self.data_manager.processed_data.get('market_volatility')
            
            if returns is None:
                raise ValueError("Market returns data not available")
            
            # Calculate VaR (Value at Risk)
            # Using historical simulation method at 95% confidence
            risk_category = self.risk_registry.get_category("market_risk")
            
            results['value_at_risk'] = {}
            results['expected_shortfall'] = {}
            results['volatility'] = {}
            results['tail_risk'] = {}
            
            for col in returns.columns:
                # VaR calculation
                var_95 = returns[col].quantile(0.05)
                var_99 = returns[col].quantile(0.01)
                
                # Expected Shortfall (Conditional VaR)
                es_95 = returns[col][returns[col] <= var_95].mean()
                es_99 = returns[col][returns[col] <= var_99].mean()
                
                # Volatility
                vol = volatility[col].iloc[-1] if volatility is not None else returns[col].std() * np.sqrt(252)
                
                # Tail risk (using excess kurtosis as a simple proxy)
                tail_risk = returns[col].kurt()
                
                # Store results
                results['value_at_risk'][col] = {
                    'var_95': var_95,
                    'var_99': var_99,
                    'dollar_var_95': var_95 * 100000000,  # Assuming $100M portfolio
                    'dollar_var_99': var_99 * 100000000,
                    'risk_level': risk_category.get_metric('value_at_risk').assess_risk_level(abs(var_95 * 100000000))
                }
                
                results['expected_shortfall'][col] = {
                    'es_95': es_95,
                    'es_99': es_99,
                    'dollar_es_95': es_95 * 100000000,
                    'dollar_es_99': es_99 * 100000000,
                    'risk_level': risk_category.get_metric('expected_shortfall').assess_risk_level(abs(es_95 * 100000000))
                }
                
                results['volatility'][col] = {
                    'annualized_vol': vol,
                    'risk_level': risk_category.get_metric('volatility').assess_risk_level(vol * 100)  # Convert to percentage
                }
                
                results['tail_risk'][col] = {
                    'excess_kurtosis': tail_risk,
                    'risk_level': risk_category.get_metric('tail_risk').assess_risk_level(
                        50 + (tail_risk / 10) * 50  # Scale kurtosis to risk score
                    )
                }
            
            # Calculate correlation matrix and connectedness
            results['correlation'] = returns.corr().to_dict()
            
            # Overall market risk assessment
            risk_levels = [
                results['value_at_risk'][col]['risk_level'] for col in returns.columns
            ] + [
                results['volatility'][col]['risk_level'] for col in returns.columns
            ]
            
            risk_score = {
                'low': 1,
                'medium': 2,
                'high': 3,
                'critical': 4
            }
            
            risk_scores = [risk_score[level] for level in risk_levels]
            avg_risk_score = sum(risk_scores) / len(risk_scores)
            
            if avg_risk_score >= 3.5:
                overall_risk = "critical"
            elif avg_risk_score >= 2.5:
                overall_risk = "high"
            elif avg_risk_score >= 1.5:
                overall_risk = "medium"
            else:
                overall_risk = "low"
                
            results['overall_risk'] = {
                'score': avg_risk_score,
                'level': overall_risk
            }
            
            self.logger.info(f"Market risk analysis completed with overall risk level: {overall_risk}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing market risk: {e}")
            raise
    
    def analyze_credit_risk(self) -> Dict[str, Any]:
        """Analyze credit risk
        
        Returns:
            Dictionary of credit risk metrics and analysis results
        """
        self.logger.info("Analyzing credit risk")
        # Placeholder for credit risk analysis
        return {}
    
    def analyze_liquidity_risk(self) -> Dict[str, Any]:
        """Analyze liquidity risk
        
        Returns:
            Dictionary of liquidity risk metrics and analysis results
        """
        self.logger.info("Analyzing liquidity risk")
        # Placeholder for liquidity risk analysis
        return {}
    
    def analyze_operational_risk(self) -> Dict[str, Any]:
        """Analyze operational risk
        
        Returns:
            Dictionary of operational risk metrics and analysis results
        """
        self.logger.info("Analyzing operational risk")
        # Placeholder for operational risk analysis
        return {}
    
    def analyze_climate_risk(self) -> Dict[str, Any]:
        """Analyze climate risk
        
        Returns:
            Dictionary of climate risk metrics and analysis results
        """
        self.logger.info("Analyzing climate risk")
        # Placeholder for climate risk analysis
        return {}
    
    def analyze_cyber_risk(self) -> Dict[str, Any]:
        """Analyze cyber risk
        
        Returns:
            Dictionary of cyber risk metrics and analysis results
        """
        self.logger.info("Analyzing cyber risk")
        # Placeholder for cyber risk analysis
        return {}
    
    def analyze_ai_risk(self) -> Dict[str, Any]:
        """Analyze AI risk
        
        Returns:
            Dictionary of AI risk metrics and analysis results
        """
        self.logger.info("Analyzing AI risk")
        # Placeholder for AI risk analysis
        return {}
    
    def analyze_digitalization(self) -> Dict[str, Any]:
        """Analyze digitalization risks
        
        Returns:
            Dictionary of digitalization risk metrics and analysis results
        """
        self.logger.info("Analyzing digitalization risks")
        # Placeholder for digitalization risk analysis
        return {}
    
    def analyze_nonbank_intermediation(self) -> Dict[str, Any]:
        """Analyze nonbank intermediation risks
        
        Returns:
            Dictionary of nonbank intermediation risk metrics and analysis results
        """
        self.logger.info("Analyzing nonbank intermediation risks")
        # Placeholder for nonbank intermediation risk analysis
        return {}
    
    def analyze_global_architecture(self) -> Dict[str, Any]:
        """Analyze global financial architecture risks
        
        Returns:
            Dictionary of global architecture risk metrics and analysis results
        """
        self.logger.info("Analyzing global financial architecture risks")
        # Placeholder for global architecture risk analysis
        return {}
    
    def analyze_network(self) -> Dict[str, Any]:
        """Analyze network interconnectedness and systemic risk
        
        Returns:
            Dictionary of network analysis results
        """
        self.logger.info("Analyzing network interconnectedness")
        
        try:
            results = {}
            
            # Get exposures from network data
            exposures = self.data_manager.network_data.get('exposures')
            
            if exposures is None:
                self.logger.warning("Network exposure data not found")
                return {"error": "Network exposure data not found"}
            
            # Create directed graph from exposures
            G = nx.DiGraph()
            
            # Add nodes
            for institution in exposures.index:
                G.add_node(institution)
            
            # Add edges with weights
            for source in exposures.index:
                for target in exposures.columns:
                    weight = exposures.loc[source, target]
                    if weight > 0:
                        G.add_edge(source, target, weight=weight)
            
            # Calculate network metrics
            results['centrality'] = {}
            
            # Degree centrality
            results['centrality']['degree'] = nx.degree_centrality(G)
            
            # In-degree centrality (vulnerability)
            results['centrality']['in_degree'] = nx.in_degree_centrality(G)
            
            # Out-degree centrality (impact on others)
            results['centrality']['out_degree'] = nx.out_degree_centrality(G)
            
            # Betweenness centrality
            results['centrality']['betweenness'] = nx.betweenness_centrality(G, weight='weight')
            
            # Eigenvector centrality
            try:
                results['centrality']['eigenvector'] = nx.eigenvector_centrality_numpy(G, weight='weight')
            except Exception as e:
                self.logger.warning(f"Could not calculate eigenvector centrality: {e}")
                results['centrality']['eigenvector'] = {}
            
            # PageRank (alternative influence measure)
            results['centrality']['pagerank'] = nx.pagerank(G, weight='weight')
            
            # Community detection
            try:
                communities = nx.community.greedy_modularity_communities(G.to_undirected())
                results['communities'] = []
                for i, community in enumerate(communities):
                    results['communities'].append({
                        'id': i,
                        'nodes': list(community)
                    })
            except Exception as e:
                self.logger.warning(f"Could not detect communities: {e}")
                results['communities'] = []
            
            # Calculate systemic importance score
            # Simple weighted average of centrality measures
            results['systemic_importance'] = {}
            for node in G.nodes():
                score = (
                    0.25 * results['centrality']['degree'].get(node, 0) +
                    0.25 * results['centrality']['betweenness'].get(node, 0) +
                    0.25 * results['centrality']['eigenvector'].get(node, 0) +
                    0.25 * results['centrality']['pagerank'].get(node, 0)
                )
                results['systemic_importance'][node] = score
            
            # Sort by systemic importance
            sorted_importance = sorted(
                results['systemic_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            results['sorted_importance'] = sorted_importance
            
            # Network-level metrics
            results['network_metrics'] = {
                'density': nx.density(G),
                'transitivity': nx.transitivity(G),
                'reciprocity': nx.reciprocity(G),
                'diameter': nx.diameter(G.to_undirected()) if nx.is_connected(G.to_undirected()) else float('inf'),
                'average_shortest_path': nx.average_shortest_path_length(G) if nx.is_strongly_connected(G) else None,
                'assortativity': nx.degree_assortativity_coefficient(G)
            }
            
            self.logger.info("Network analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing network: {e}")
            raise
    
    def run_stress_tests(self) -> Dict[str, Any]:
        """Run stress tests on different risk types
        
        Returns:
            Dictionary of stress test results
        """
        self.logger.info("Running stress tests")
        
        try:
            results = {}
            
            # Define stress scenarios from configuration
            scenarios = settings.STRESS_TEST_SCENARIOS
            
            for scenario in scenarios:
                results[scenario] = self._run_scenario(scenario)
            
            self.logger.info(f"Completed {len(scenarios)} stress test scenarios")
            return results
            
        except Exception as e:
            self.logger.error(f"Error running stress tests: {e}")
            raise
    
    def _run_scenario(self, scenario: str) -> Dict[str, Any]:
        """Run a specific stress test scenario
        
        Args:
            scenario: Name of the scenario to run
            
        Returns:
            Dictionary of scenario results
        """
        self.logger.info(f"Running stress scenario: {scenario}")
        
        # Placeholder function that would typically:
        # 1. Apply shocks to market factors based on scenario
        # 2. Propagate impacts across the financial network
        # 3. Calculate resulting risk metrics
        
        # For demonstration we'll return placeholder results
        return {
            'name': scenario,
            'description': f"Stress test scenario: {scenario}",
            'timestamp': datetime.now().isoformat(),
            'impact': {
                'market_risk': np.random.uniform(0.5, 5.0),
                'credit_risk': np.random.uniform(0.5, 5.0),
                'liquidity_risk': np.random.uniform(0.5, 5.0),
                'operational_risk': np.random.uniform(0.5, 5.0)
            },
            'systemic_risk_change': np.random.uniform(0.3, 3.0)
        }
    
    def calculate_systemic_risk_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate systemic risk metrics
        
        Returns:
            Dictionary of systemic risk metrics
        """
        self.logger.info("Calculating systemic risk metrics")
        
        # Placeholder function that would:
        # 1. Combine network analysis with individual risk metrics
        # 2. Calculate system-wide vulnerability measures
        # 3. Identify key transmission channels
        
        # For demonstration, return placeholder metrics
        results = {
            'absorption_ratio': np.random.uniform(0.5, 0.9),
            'systemic_expected_shortfall': np.random.uniform(5000000, 50000000),
            'conditional_value_at_risk': np.random.uniform(0.02, 0.1),
            'network_vulnerability_index': np.random.uniform(0.3, 0.8),
            'financial_stress_index': np.random.uniform(0.2, 0.7)
        }
        
        # Risk level assessment
        if results['financial_stress_index'] > 0.6:
            results['risk_level'] = 'critical'
        elif results['financial_stress_index'] > 0.4:
            results['risk_level'] = 'high'
        elif results['financial_stress_index'] > 0.25:
            results['risk_level'] = 'medium'
        else:
            results['risk_level'] = 'low'
        
        self.logger.info(f"Systemic risk assessment completed with risk level: {results['risk_level']}")
        return results 