"""
Application manager for the Financial Risk Analysis System
"""
import logging
from pathlib import Path
import importlib
from typing import Dict, List, Any, Optional

from src.core.config import settings
from src.data.data_manager import DataManager
from src.risk_modules.risk_registry import RiskRegistry
from src.analytics.analysis_engine import AnalysisEngine
from src.models.model_registry import ModelRegistry
from src.visualization.dashboard import Dashboard


class RiskAnalysisSystem:
    """Main application class that orchestrates all components of the system"""
    
    def __init__(self):
        """Initialize the Risk Analysis System"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Risk Analysis System")
        
        # Initialize components
        self.data_manager = DataManager()
        self.risk_registry = RiskRegistry()
        self.model_registry = ModelRegistry()
        self.analysis_engine = AnalysisEngine(
            self.data_manager, 
            self.risk_registry,
            self.model_registry
        )
        # Initialize analysis_results BEFORE initializing Dashboard
        self.analysis_results = {}
        self.dashboard = Dashboard(self.analysis_results)
        
        # Track execution state
        self.is_data_loaded = False
        
        self.logger.info("Risk Analysis System initialized successfully")
    
    def load_data(self):
        """Load and prepare all required data"""
        self.logger.info("Loading data")
        try:
            self.data_manager.load_market_data()
            self.data_manager.load_credit_data()
            self.data_manager.load_liquidity_data()
            self.data_manager.load_operational_data()
            self.data_manager.load_climate_data()
            self.data_manager.load_cyber_data()
            self.data_manager.load_network_data()
            
            # Process and prepare data
            self.data_manager.process_all_data()
            
            self.is_data_loaded = True
            self.logger.info("Data loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def run_analysis(self):
        """Run all analysis modules"""
        self.logger.info("Running risk analysis")
        
        if not self.is_data_loaded:
            self.logger.warning("Data not loaded. Loading data first.")
            self.load_data()
        
        try:
            # Run traditional risk analysis
            self.analysis_results["market_risk"] = self.analysis_engine.analyze_market_risk()
            self.analysis_results["credit_risk"] = self.analysis_engine.analyze_credit_risk()
            self.analysis_results["liquidity_risk"] = self.analysis_engine.analyze_liquidity_risk()
            self.analysis_results["operational_risk"] = self.analysis_engine.analyze_operational_risk()
            
            # Run emerging risks analysis
            self.analysis_results["climate_risk"] = self.analysis_engine.analyze_climate_risk()
            self.analysis_results["cyber_risk"] = self.analysis_engine.analyze_cyber_risk()
            self.analysis_results["ai_risk"] = self.analysis_engine.analyze_ai_risk()
            
            # Run structural shift analysis
            self.analysis_results["digitalization"] = self.analysis_engine.analyze_digitalization()
            self.analysis_results["nonbank_intermediation"] = self.analysis_engine.analyze_nonbank_intermediation()
            self.analysis_results["global_architecture"] = self.analysis_engine.analyze_global_architecture()
            
            # Run network analysis
            self.analysis_results["network"] = self.analysis_engine.analyze_network()
            
            # Run stress tests
            self.analysis_results["stress_tests"] = self.analysis_engine.run_stress_tests()
            
            # Calculate aggregate systemic risk metrics
            self.analysis_results["systemic"] = self.analysis_engine.calculate_systemic_risk_metrics()
            
            self.logger.info("Risk analysis completed successfully")
            return self.analysis_results
            
        except Exception as e:
            self.logger.error(f"Error running analysis: {e}")
            raise
    
    def start_dashboard(self):
        """Start the interactive dashboard"""
        self.logger.info("Starting dashboard")
        
        try:
            if not self.analysis_results:
                self.logger.warning("No analysis results found. Running analysis first.")
                self.run_analysis()
            
            # No need to call update_data anymore
            # self.dashboard.update_data(self.analysis_results)
            self.dashboard.run(
                host=settings.DASHBOARD_HOST,
                port=settings.DASHBOARD_PORT,
                debug=settings.DASHBOARD_DEBUG
            )
            
            self.logger.info("Dashboard started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting dashboard: {e}")
            raise
    
    def generate_report(self, report_type="comprehensive", output_format="html"):
        """Generate risk analysis report"""
        self.logger.info(f"Generating {report_type} report in {output_format} format")
        
        try:
            from src.visualization.report_generator import ReportGenerator
            
            report_gen = ReportGenerator(self.analysis_results)
            report_path = report_gen.generate_report(report_type, output_format)
            
            self.logger.info(f"Report generated successfully: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            raise 