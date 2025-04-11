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
    
    def _get_placeholder_result(self, category_name: str) -> Dict[str, Any]:
        """Generates a structured placeholder result for unimplemented analysis functions."""
        self.logger.debug(f"Generating placeholder result for {category_name}")
        # Simulate overall risk
        risk_level_map = {1: 'low', 2: 'medium', 3: 'high', 4: 'critical'}
        score_to_level = lambda s: risk_level_map.get(min(4, max(1, int(np.ceil(s)))), 'low')
        simulated_score = np.random.uniform(1.0, 4.0)
        simulated_level = score_to_level(simulated_score)
        
        # Simulate some dummy metrics
        dummy_metrics = {
            f'metric_{category_name}_1': np.random.rand() * 100,
            f'metric_{category_name}_2': np.random.rand() * 50
        }
        
        # Simulate dummy time series (optional, might be large)
        # dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        # dummy_ts = pd.Series(np.random.randn(10).cumsum(), index=dates)

        return {
            'overall_risk': {
                'score': simulated_score,
                'level': simulated_level,
                'status': 'placeholder' # Mark as placeholder
            },
            'key_metrics': dummy_metrics,
            # 'sample_timeseries': dummy_ts # Uncomment to add dummy time series
            'status': 'placeholder'
        }

    def analyze_credit_risk(self) -> Dict[str, Any]:
        """Analyze credit risk"""
        self.logger.info("Analyzing credit risk (Placeholder)")
        # Placeholder for credit risk analysis
        return self._get_placeholder_result("credit_risk")
    
    def analyze_liquidity_risk(self) -> Dict[str, Any]:
        """Analyze liquidity risk"""
        self.logger.info("Analyzing liquidity risk (Placeholder)")
        # Placeholder for liquidity risk analysis
        return self._get_placeholder_result("liquidity_risk")
    
    def analyze_operational_risk(self) -> Dict[str, Any]:
        """Analyze operational risk"""
        self.logger.info("Analyzing operational risk (Placeholder)")
        # Placeholder for operational risk analysis
        return self._get_placeholder_result("operational_risk")
    
    def analyze_climate_risk(self) -> Dict[str, Any]:
        """Analyze climate risk"""
        self.logger.info("Analyzing climate risk (Placeholder)")
        # Placeholder for climate risk analysis
        return self._get_placeholder_result("climate_risk")
    
    def analyze_cyber_risk(self) -> Dict[str, Any]:
        """Analyze cyber risk"""
        self.logger.info("Analyzing cyber risk (Placeholder)")
        # Placeholder for cyber risk analysis
        return self._get_placeholder_result("cyber_risk")
    
    def analyze_ai_risk(self) -> Dict[str, Any]:
        """Analyze AI risk"""
        self.logger.info("Analyzing AI risk (Placeholder)")
        # Placeholder for AI risk analysis
        return self._get_placeholder_result("ai_risk")
    
    def analyze_digitalization(self) -> Dict[str, Any]:
        """Analyze digitalization risks"""
        self.logger.info("Analyzing digitalization risks (Placeholder)")
        # Placeholder for digitalization risk analysis
        return self._get_placeholder_result("digitalization")
    
    def analyze_nonbank_intermediation(self) -> Dict[str, Any]:
        """Analyze nonbank intermediation risks"""
        self.logger.info("Analyzing nonbank intermediation risks (Placeholder)")
        # Placeholder for nonbank intermediation risk analysis
        return self._get_placeholder_result("nonbank_intermediation")
    
    def analyze_global_architecture(self) -> Dict[str, Any]:
        """Analyze global financial architecture risks"""
        self.logger.info("Analyzing global financial architecture risks (Placeholder)")
        # Placeholder for global architecture risk analysis
        return self._get_placeholder_result("global_architecture")
    
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
            
            # Degree centrality (Using degree view directly is often sufficient)
            # results['centrality']['degree'] = nx.degree_centrality(G)
            results['centrality']['degree'] = {node: val for node, val in G.degree()} # Store raw degree
            
            # In-degree centrality (vulnerability)
            # results['centrality']['in_degree'] = nx.in_degree_centrality(G)
            results['centrality']['in_degree'] = {node: val for node, val in G.in_degree()}
            
            # Out-degree centrality (impact on others)
            # results['centrality']['out_degree'] = nx.out_degree_centrality(G)
            results['centrality']['out_degree'] = {node: val for node, val in G.out_degree()}
            
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
            # Ensure graph has nodes/edges before calculating metrics that require them
            undirected_G = G.to_undirected()
            is_connected = nx.is_connected(undirected_G) if len(G) > 0 else False
            # is_strongly_connected = nx.is_strongly_connected(G) if len(G) > 0 else False # expensive
            
            results['network_metrics'] = {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'density': nx.density(G),
                'transitivity': nx.transitivity(G) if len(G) > 0 else 0,
                'reciprocity': nx.reciprocity(G) if len(G) > 0 else 0,
                # 'diameter': nx.diameter(undirected_G) if is_connected else float('inf'), # Can be slow
                # 'average_shortest_path': nx.average_shortest_path_length(G) if is_strongly_connected else None, # Slow & needs strong connection
                'assortativity': nx.degree_assortativity_coefficient(G, weight='weight') if G.number_of_edges() > 0 else 0
            }
            
            # --- Add Graph and Positions for Visualization ---
            if G.number_of_nodes() > 0:
                try:
                    # Calculate positions (can be computationally intensive for large graphs)
                    # Use a fixed seed for reproducible layouts
                    results['pos'] = nx.spring_layout(G, seed=42, k=0.5) # Adjust k for spacing if needed
                except Exception as layout_error:
                    self.logger.warning(f"Could not calculate graph layout: {layout_error}")
                    results['pos'] = None # Set pos to None if layout fails
            else:
                 results['pos'] = {}
                 
            # Add the graph object itself (ensure it's serializable if needed elsewhere, but fine for in-memory Dash)
            results['graph'] = G 
            results['status'] = 'implemented' # Mark as implemented
            # -------------------------------------------------
            
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
        """Run a specific stress test scenario by simulating impacts on baseline analysis.
        
        Args:
            scenario: Name of the scenario to run
            
        Returns:
            Dictionary of scenario results, structured similarly to main analysis results.
        """ 
        self.logger.info(f"Running stress scenario simulation: {scenario}")
        
        # NOTE: This is still a simulation. Proper implementation would involve
        # applying shocks to input data and re-running models.
        
        scenario_results = {}
        
        # Define risk categories to analyze (could come from settings)
        risk_categories_to_analyze = [
            'market_risk', 'credit_risk', 'liquidity_risk', 'operational_risk',
            'climate_risk', 'cyber_risk', 'ai_risk', 'digitalization',
            'nonbank_intermediation', 'global_architecture'
        ]
        
        # Define scenario descriptions and impact multipliers
        scenario_impacts = {
            'baseline': {
                'description': 'Baseline scenario with current market conditions',
                'multiplier': 1.0,
                'category_multipliers': {}  # No specific category effects
            },
            'adverse': {
                'description': 'Adverse scenario with moderate economic deterioration',
                'multiplier': 1.5,
                'category_multipliers': {
                    'market_risk': 1.7,
                    'credit_risk': 1.8,
                    'liquidity_risk': 1.6
                }
            },
            'severely_adverse': {
                'description': 'Severely adverse scenario with significant economic deterioration',
                'multiplier': 2.0,
                'category_multipliers': {
                    'market_risk': 2.2,
                    'credit_risk': 2.3,
                    'liquidity_risk': 2.1,
                    'operational_risk': 1.8
                }
            },
            'climate_transition': {
                'description': 'Rapid climate transition with carbon pricing and stranded assets',
                'multiplier': 1.3,
                'category_multipliers': {
                    'climate_risk': 2.5,
                    'market_risk': 1.5,
                    'credit_risk': 1.4
                }
            },
            'cyber_attack': {
                'description': 'Major cyber attack on financial infrastructure',
                'multiplier': 1.4,
                'category_multipliers': {
                    'cyber_risk': 2.8,
                    'operational_risk': 2.2,
                    'digitalization': 1.8
                }
            },
            'liquidity_freeze': {
                'description': 'Systemic liquidity freeze in markets',
                'multiplier': 1.6,
                'category_multipliers': {
                    'liquidity_risk': 2.9,
                    'market_risk': 2.0,
                    'credit_risk': 1.7
                }
            },
            'rate_hike': {
                'description': 'Sudden central bank rate hike scenario',
                'multiplier': 1.4,
                'category_multipliers': {
                    'market_risk': 1.8,
                    'credit_risk': 1.6,
                    'liquidity_risk': 1.5
                }
            },
            'rate_cut': {
                'description': 'Emergency central bank rate cut scenario',
                'multiplier': 1.2,
                'category_multipliers': {
                    'market_risk': 1.5,
                    'liquidity_risk': 1.3
                }
            },
            'market_crash': {
                'description': 'Severe market crash with 30%+ equity decline',
                'multiplier': 1.9,
                'category_multipliers': {
                    'market_risk': 2.7,
                    'liquidity_risk': 2.3,
                    'credit_risk': 1.9
                }
            },
            'banking_crisis': {
                'description': 'Systemic banking crisis with multiple failures',
                'multiplier': 2.1,
                'category_multipliers': {
                    'credit_risk': 2.7,
                    'liquidity_risk': 2.5,
                    'market_risk': 2.2,
                    'nonbank_intermediation': 2.0
                }
            },
            'sovereign_debt_crisis': {
                'description': 'Sovereign debt crisis with potential defaults',
                'multiplier': 1.9,
                'category_multipliers': {
                    'credit_risk': 2.5,
                    'market_risk': 2.1,
                    'global_architecture': 2.4
                }
            },
            'currency_crisis': {
                'description': 'Major currency devaluation and capital flight',
                'multiplier': 1.7,
                'category_multipliers': {
                    'market_risk': 2.3,
                    'liquidity_risk': 2.0,
                    'global_architecture': 1.9
                }
            },
            'global_recession': {
                'description': 'Severe global recession with widespread impacts',
                'multiplier': 2.2,
                'category_multipliers': {
                    'market_risk': 2.4,
                    'credit_risk': 2.3,
                    'liquidity_risk': 2.2,
                    'operational_risk': 1.8,
                    'nonbank_intermediation': 2.0
                }
            },
            'tech_bubble_burst': {
                'description': 'Technology sector bubble burst with contagion',
                'multiplier': 1.6,
                'category_multipliers': {
                    'market_risk': 2.3,
                    'credit_risk': 1.8,
                    'ai_risk': 2.1,
                    'digitalization': 2.0
                }
            },
            'supply_chain_disruption': {
                'description': 'Major global supply chain disruption',
                'multiplier': 1.5,
                'category_multipliers': {
                    'operational_risk': 2.0,
                    'market_risk': 1.7,
                    'liquidity_risk': 1.6
                }
            },
            'pandemic': {
                'description': 'Global pandemic with economic shutdowns',
                'multiplier': 1.8,
                'category_multipliers': {
                    'operational_risk': 2.4,
                    'market_risk': 2.1,
                    'credit_risk': 2.0,
                    'liquidity_risk': 1.9
                }
            },
            'inflation_shock': {
                'description': 'Sudden inflation shock with price instability',
                'multiplier': 1.7,
                'category_multipliers': {
                    'market_risk': 2.2,
                    'credit_risk': 1.8,
                    'liquidity_risk': 1.7
                }
            },
            'combined_market_cyber': {
                'description': 'Combined market crash and cyber attack scenario',
                'multiplier': 2.3,
                'category_multipliers': {
                    'market_risk': 2.7,
                    'cyber_risk': 2.8,
                    'operational_risk': 2.4,
                    'liquidity_risk': 2.2
                }
            },
            'combined_climate_sovereign': {
                'description': 'Combined climate transition and sovereign debt crisis',
                'multiplier': 2.2,
                'category_multipliers': {
                    'climate_risk': 2.6,
                    'credit_risk': 2.5,
                    'market_risk': 2.3,
                    'global_architecture': 2.4
                }
            }
        }
        
        # Get scenario impact data or use default values
        scenario_info = scenario_impacts.get(scenario, {
            'description': f"Simulated stress test scenario: {scenario}",
            'multiplier': 1.5,
            'category_multipliers': {}
        })
        
        impact_multiplier = scenario_info['multiplier']
        category_multipliers = scenario_info['category_multipliers']
        scenario_description = scenario_info['description']

        risk_level_map = {1: 'low', 2: 'medium', 3: 'high', 4: 'critical'}
        score_to_level = lambda s: risk_level_map.get(min(4, max(1, int(np.ceil(s)))), 'low')

        # Run baseline analysis for each category and simulate stress impact
        for category in risk_categories_to_analyze:
            try:
                # Dynamically get the analysis method for the category
                analysis_method_name = f"analyze_{category}"
                if hasattr(self, analysis_method_name) and callable(getattr(self, analysis_method_name)):
                    # Get baseline results for this category
                    baseline_result = getattr(self, analysis_method_name)()
                    
                    # Simulate stressed result based on baseline overall risk
                    stressed_result = baseline_result.copy() # Start with baseline structure
                    baseline_overall = baseline_result.get('overall_risk', {'score': 1.5, 'level': 'low'}) # Default if missing
                    
                    # Apply category-specific multiplier if available, otherwise use general multiplier
                    category_multiplier = category_multipliers.get(category, impact_multiplier)
                    
                    # Simulate increased risk score
                    stressed_score = min(4.0, baseline_overall.get('score', 1.5) * category_multiplier) 
                    stressed_level = score_to_level(stressed_score)
                    
                    # Add impact relative to baseline for visualization
                    impact_percent = ((stressed_score - baseline_overall.get('score', 1.5)) / baseline_overall.get('score', 1.5)) * 100
                    
                    stressed_result['overall_risk'] = {
                        'score': stressed_score,
                        'level': stressed_level,
                        'baseline_level': baseline_overall.get('level', 'low'),
                        'baseline_score': baseline_overall.get('score', 1.5),
                        'impact_percent': impact_percent,
                        # Add status from baseline if available
                        'status': baseline_overall.get('status', 'placeholder') 
                    }
                    scenario_results[category] = stressed_result
                else:
                    self.logger.warning(f"Analysis method {analysis_method_name} not found for scenario {scenario}.")
                    # Add placeholder if method doesn't exist
                    category_multiplier = category_multipliers.get(category, impact_multiplier)
                    stressed_score = min(4.0, 1.5 * category_multiplier)
                    stressed_level = score_to_level(stressed_score)
                    scenario_results[category] = {
                         'overall_risk': {
                             'score': stressed_score, 
                             'level': stressed_level,
                             'baseline_score': 1.5,
                             'baseline_level': 'low',
                             'impact_percent': ((stressed_score - 1.5) / 1.5) * 100
                         } 
                    }
            except Exception as cat_ex:
                 self.logger.error(f"Error analyzing category '{category}' during scenario '{scenario}': {cat_ex}", exc_info=True)
                 scenario_results[category] = {
                     'error': str(cat_ex), 
                     'overall_risk': {
                         'score': 4.0, 
                         'level': 'critical',
                         'baseline_score': 1.5,
                         'baseline_level': 'low',
                         'impact_percent': 166.7  # (4.0 - 1.5) / 1.5 * 100
                     }
                 }
                 
        # Add general scenario info
        scenario_results['scenario_info'] = {
            'name': scenario,
            'description': scenario_description,
            'timestamp': datetime.now().isoformat(),
            'general_impact_multiplier': impact_multiplier,
            'category_specific_multipliers': category_multipliers
        }

        return scenario_results
    
    def calculate_systemic_risk_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate systemic risk metrics
        
        Returns:
            Dictionary of systemic risk metrics
        """
        self.logger.info("Calculating systemic risk metrics")
        
        # Placeholder function - NEEDS PROPER IMPLEMENTATION
        # For now, provide dummy data in the structure expected by the dashboard's Risk Summary
        results = {
            'financial_stress_index': np.random.uniform(0.2, 0.7),
            # Add other top-level systemic metrics here if needed
        }

        # --- Dummy Overall Risk by Category --- 
        # Create placeholder risk levels for categories based on FSI
        overall_risk_by_category = {}
        fsi = results['financial_stress_index']
        base_score = 1.0 + fsi * 3.0 # Map FSI (0.2-0.7) roughly to score (1.6-3.1)
        risk_level_map = {1: 'low', 2: 'medium', 3: 'high', 4: 'critical'}
        score_to_level = lambda s: risk_level_map.get(min(4, max(1, int(np.ceil(s)))), 'low')

        for category in settings.RISK_CATEGORIES:
            # Slightly randomize score around base FSI mapping
            cat_score = min(4.0, max(1.0, base_score + np.random.uniform(-0.5, 0.5)))
            cat_level = score_to_level(cat_score)
            overall_risk_by_category[category] = {
                'score': cat_score,
                'level': cat_level,
                'status': 'placeholder' # Mark as placeholder
            }
        results['overall_risk_by_category'] = overall_risk_by_category
        # -----------------------------------------

        # Risk level assessment (based on FSI)
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