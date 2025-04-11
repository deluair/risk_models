"""
Risk Simulation Runner
Tests the full Financial Risk Analysis System with simulated data
"""
import os
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.core.config import settings
from src.data.data_manager import DataManager
from src.analytics.analysis_engine import AnalysisEngine
from src.visualization.visualization import VisualizationEngine
from src.testing.simulation import FinancialDataSimulator, run_simulation


def setup_logging():
    """Configure logging for the simulation run"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"risk_simulation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def run_risk_system_test(scenario_name="base", clean_data=False):
    """Run a test of the risk system using simulated data
    
    Args:
        scenario_name: Name of the scenario to test
        clean_data: Whether to regenerate data even if it exists
    """
    logger = logging.getLogger(__name__)
    logger.info(f"=== Starting risk system test for scenario: {scenario_name} ===")
    
    # Create results directory
    results_dir = Path(f"results/simulation_{scenario_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate simulation data or use existing
    data_dir = settings.RAW_DATA_DIR / f"scenario_{scenario_name}"
    
    if not data_dir.exists() or clean_data:
        logger.info(f"Generating new simulation data for scenario: {scenario_name}")
        simulator = FinancialDataSimulator()
        
        if scenario_name == "base":
            simulator.run_simulation_scenario("base")
        elif scenario_name == "market_crash":
            simulator.run_simulation_scenario("market_crash", crash_date="2023-06-15")
        elif scenario_name == "credit_deterioration":
            simulator.run_simulation_scenario("credit_deterioration")
        elif scenario_name == "combined_stress":
            simulator.run_simulation_scenario("combined_stress", crash_date="2023-06-15")
        else:
            logger.warning(f"Unknown scenario: {scenario_name}, using base scenario")
            simulator.run_simulation_scenario("base")
    else:
        logger.info(f"Using existing simulation data for scenario: {scenario_name}")
    
    # Step 2: Initialize data manager and load data
    logger.info("Initializing data manager and loading data")
    data_manager = DataManager()
    
    try:
        # Point data manager to the scenario data
        data_manager.raw_data_dir = data_dir
        data_manager.processed_data_dir = settings.PROCESSED_DATA_DIR / f"scenario_{scenario_name}"
        data_manager.processed_data_dir.mkdir(exist_ok=True)
        
        # Load market data
        market_data = data_manager.load_market_data()
        logger.info(f"Loaded market data with shapes: {[df.shape for df in market_data.values()]}")
        
        # Load credit data
        credit_data = data_manager.load_credit_data()
        logger.info(f"Loaded credit data with shapes: {[df.shape if isinstance(df, pd.DataFrame) else 'dict' for df in credit_data.values()]}")
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise
    
    # Step 3: Initialize analysis engine
    logger.info("Initializing analysis engine")
    analysis_engine = AnalysisEngine(data_manager)
    
    # Step 4: Run risk analysis
    logger.info("Running risk analysis")
    try:
        # Market risk analysis
        market_risk_results = analysis_engine.analyze_market_risk()
        logger.info(f"Market risk analysis completed: {list(market_risk_results.keys())}")
        
        # Credit risk analysis
        credit_risk_results = analysis_engine.analyze_credit_risk()
        logger.info(f"Credit risk analysis completed: {list(credit_risk_results.keys())}")
        
        # Liquidity risk analysis
        liquidity_risk_results = analysis_engine.analyze_liquidity_risk()
        logger.info(f"Liquidity risk analysis completed: {list(liquidity_risk_results.keys())}")
        
        # Operational risk analysis
        operational_risk_results = analysis_engine.analyze_operational_risk()
        logger.info(f"Operational risk analysis completed: {list(operational_risk_results.keys())}")
        
        # Network risk analysis
        network_risk_results = analysis_engine.analyze_network_risk()
        logger.info(f"Network risk analysis completed: {list(network_risk_results.keys())}")
        
        # Systemic risk metrics
        systemic_risk_metrics = analysis_engine.calculate_systemic_risk_metrics()
        logger.info(f"Systemic risk metrics calculated: {list(systemic_risk_metrics.keys())}")
        
    except Exception as e:
        logger.error(f"Error in risk analysis: {str(e)}")
        raise
    
    # Step 5: Visualize results
    logger.info("Creating visualizations")
    try:
        visualization = VisualizationEngine()
        
        # Market risk visualizations
        market_risk_plots = visualization.plot_market_risk(market_risk_results)
        for name, fig in market_risk_plots.items():
            fig_path = results_dir / f"market_risk_{name}.png"
            fig.savefig(fig_path)
            plt.close(fig)
            logger.info(f"Saved {fig_path}")
        
        # Credit risk visualizations
        credit_risk_plots = visualization.plot_credit_risk(credit_risk_results)
        for name, fig in credit_risk_plots.items():
            fig_path = results_dir / f"credit_risk_{name}.png"
            fig.savefig(fig_path)
            plt.close(fig)
            logger.info(f"Saved {fig_path}")
        
        # Network risk visualizations
        network_risk_plots = visualization.plot_network_risk(network_risk_results)
        for name, fig in network_risk_plots.items():
            fig_path = results_dir / f"network_risk_{name}.png"
            fig.savefig(fig_path)
            plt.close(fig)
            logger.info(f"Saved {fig_path}")
        
        # Systemic risk visualizations
        systemic_risk_plots = visualization.plot_systemic_risk(systemic_risk_metrics)
        for name, fig in systemic_risk_plots.items():
            fig_path = results_dir / f"systemic_risk_{name}.png"
            fig.savefig(fig_path)
            plt.close(fig)
            logger.info(f"Saved {fig_path}")
        
    except Exception as e:
        logger.error(f"Error in visualization: {str(e)}")
    
    # Step 6: Save results to JSON
    logger.info("Saving results")
    try:
        # Helper function to make results JSON serializable
        def make_serializable(obj):
            if isinstance(obj, (np.ndarray, pd.Series)):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(i) for i in obj]
            else:
                return str(obj)
        
        # Save market risk results
        with open(results_dir / "market_risk_results.json", "w") as f:
            json.dump(make_serializable(market_risk_results), f, indent=2)
        
        # Save credit risk results
        with open(results_dir / "credit_risk_results.json", "w") as f:
            json.dump(make_serializable(credit_risk_results), f, indent=2)
        
        # Save network risk results
        with open(results_dir / "network_risk_results.json", "w") as f:
            json.dump(make_serializable(network_risk_results), f, indent=2)
        
        # Save systemic risk metrics
        with open(results_dir / "systemic_risk_metrics.json", "w") as f:
            json.dump(make_serializable(systemic_risk_metrics), f, indent=2)
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
    
    # Step 7: Generate summary report
    logger.info("Generating summary report")
    try:
        report_content = []
        report_content.append(f"# Risk Analysis Report - {scenario_name.upper()} Scenario")
        report_content.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Market Risk Summary
        report_content.append("\n## Market Risk Summary")
        if 'var' in market_risk_results:
            report_content.append(f"\nValue at Risk (95%): {market_risk_results['var']['var_95']:.2f}%")
            report_content.append(f"Value at Risk (99%): {market_risk_results['var']['var_99']:.2f}%")
        if 'expected_shortfall' in market_risk_results:
            report_content.append(f"Expected Shortfall (95%): {market_risk_results['expected_shortfall']['es_95']:.2f}%")
        if 'volatility' in market_risk_results:
            report_content.append(f"Average Volatility: {market_risk_results['volatility'].mean():.2f}%")
        
        # Credit Risk Summary
        report_content.append("\n## Credit Risk Summary")
        if 'expected_loss' in credit_risk_results:
            report_content.append(f"\nTotal Expected Loss: ${credit_risk_results['expected_loss']:,.2f}")
        if 'unexpected_loss' in credit_risk_results:
            report_content.append(f"Unexpected Loss: ${credit_risk_results['unexpected_loss']:,.2f}")
        if 'credit_var' in credit_risk_results:
            report_content.append(f"Credit VaR (99%): ${credit_risk_results['credit_var']:,.2f}")
        
        # Network Risk Summary
        report_content.append("\n## Network Risk Summary")
        if 'centrality' in network_risk_results:
            top_entities = pd.Series(network_risk_results['centrality']).sort_values(ascending=False).head(5)
            report_content.append("\nTop 5 most central entities:")
            for entity, centrality in top_entities.items():
                report_content.append(f"- {entity}: {centrality:.4f}")
        if 'interconnectedness' in network_risk_results:
            report_content.append(f"\nSystem Interconnectedness: {network_risk_results['interconnectedness']:.4f}")
        
        # Systemic Risk Summary
        report_content.append("\n## Systemic Risk Summary")
        if 'srisk' in systemic_risk_metrics:
            report_content.append(f"\nSystemic Risk Index: {systemic_risk_metrics['srisk']:.4f}")
        if 'contagion_risk' in systemic_risk_metrics:
            report_content.append(f"Contagion Risk: {systemic_risk_metrics['contagion_risk']:.4f}")
        if 'vulnerability_index' in systemic_risk_metrics:
            report_content.append(f"Vulnerability Index: {systemic_risk_metrics['vulnerability_index']:.4f}")
        
        # Save report
        with open(results_dir / "risk_analysis_report.md", "w") as f:
            f.write("\n".join(report_content))
        
        logger.info(f"Report saved to {results_dir / 'risk_analysis_report.md'}")
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
    
    logger.info(f"=== Risk system test completed for scenario: {scenario_name} ===")
    return results_dir


def run_scenario_comparison(scenarios=None):
    """Run multiple scenarios and compare results
    
    Args:
        scenarios: List of scenarios to run
    """
    if scenarios is None:
        scenarios = ["base", "market_crash", "credit_deterioration", "combined_stress"]
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting comparison of {len(scenarios)} scenarios")
    
    # Create results directory
    comparison_dir = Path("results/scenario_comparison")
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Run all scenarios
    scenario_results = {}
    for scenario in scenarios:
        try:
            results_dir = run_risk_system_test(scenario)
            
            # Load results from JSON files
            scenario_results[scenario] = {
                'market_risk': json.loads((results_dir / "market_risk_results.json").read_text()),
                'credit_risk': json.loads((results_dir / "credit_risk_results.json").read_text()),
                'network_risk': json.loads((results_dir / "network_risk_results.json").read_text()),
                'systemic_risk': json.loads((results_dir / "systemic_risk_metrics.json").read_text()),
            }
        except Exception as e:
            logger.error(f"Error running scenario {scenario}: {str(e)}")
    
    # Compare key metrics across scenarios
    logger.info("Comparing key metrics across scenarios")
    
    try:
        # Market risk comparison
        var_comparison = {
            scenario: results['market_risk']['var']['var_99'] 
            for scenario, results in scenario_results.items() 
            if 'var' in results['market_risk']
        }
        
        es_comparison = {
            scenario: results['market_risk']['expected_shortfall']['es_95'] 
            for scenario, results in scenario_results.items() 
            if 'expected_shortfall' in results['market_risk']
        }
        
        # Credit risk comparison
        el_comparison = {
            scenario: results['credit_risk']['expected_loss'] 
            for scenario, results in scenario_results.items() 
            if 'expected_loss' in results['credit_risk']
        }
        
        # Systemic risk comparison
        srisk_comparison = {
            scenario: results['systemic_risk']['srisk'] 
            for scenario, results in scenario_results.items() 
            if 'srisk' in results['systemic_risk']
        }
        
        # Plot comparisons
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.bar(var_comparison.keys(), var_comparison.values())
        plt.title('Value at Risk (99%) Comparison')
        plt.ylabel('VaR (%)')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        plt.bar(es_comparison.keys(), es_comparison.values())
        plt.title('Expected Shortfall (95%) Comparison')
        plt.ylabel('ES (%)')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        plt.bar(el_comparison.keys(), el_comparison.values())
        plt.title('Expected Loss Comparison')
        plt.ylabel('Expected Loss ($)')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        plt.bar(srisk_comparison.keys(), srisk_comparison.values())
        plt.title('Systemic Risk Index Comparison')
        plt.ylabel('SRISK')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(comparison_dir / "scenario_comparison.png")
        plt.close()
        
        # Create comparison table
        comparison_table = []
        comparison_table.append("# Scenario Comparison")
        comparison_table.append("\nGenerated on: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        comparison_table.append("\n## Key Risk Metrics")
        
        # Create markdown table
        headers = ["Metric"] + list(scenario_results.keys())
        comparison_table.append("\n| " + " | ".join(headers) + " |")
        comparison_table.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        # Market risk metrics
        comparison_table.append("| VaR (99%) | " + " | ".join([f"{var_comparison.get(s, 'N/A'):.2f}%" for s in scenario_results.keys()]) + " |")
        comparison_table.append("| ES (95%) | " + " | ".join([f"{es_comparison.get(s, 'N/A'):.2f}%" for s in scenario_results.keys()]) + " |")
        
        # Credit risk metrics
        comparison_table.append("| Expected Loss | " + " | ".join([f"${el_comparison.get(s, 'N/A'):,.2f}" for s in scenario_results.keys()]) + " |")
        
        # Systemic risk metrics
        comparison_table.append("| Systemic Risk | " + " | ".join([f"{srisk_comparison.get(s, 'N/A'):.4f}" for s in scenario_results.keys()]) + " |")
        
        # Save comparison table
        with open(comparison_dir / "scenario_comparison.md", "w") as f:
            f.write("\n".join(comparison_table))
        
        logger.info(f"Comparison results saved to {comparison_dir}")
        
    except Exception as e:
        logger.error(f"Error comparing scenarios: {str(e)}")
    
    logger.info("Scenario comparison completed")


if __name__ == "__main__":
    # Setup logging
    logger = setup_logging()
    logger.info("Starting risk simulation process")
    
    try:
        # Define scenarios to test
        scenarios = [
            "base",  # Baseline scenario
            "market_crash",  # Market crash scenario
            "credit_deterioration",  # Credit quality deterioration
            "combined_stress"  # Combined stress scenario
        ]
        
        # Run comparison of all scenarios
        run_scenario_comparison(scenarios)
        
    except Exception as e:
        logger.error(f"Error in simulation process: {str(e)}")
    
    logger.info("Risk simulation process completed") 