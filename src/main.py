#!/usr/bin/env python
"""
Main application entry point for Financial Risk Analysis System
"""
import sys
import os
import logging
from pathlib import Path

# Add parent directory to path to allow imports from other modules
sys.path.append(str(Path(__file__).parent.parent))

from src.core.config import settings
from src.core.logging_config import setup_logging
from src.core.app_manager import RiskAnalysisSystem


def main():
    """Main function to initialize and run the Risk Analysis System"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Financial Risk Analysis System")
    
    try:
        # Initialize the risk analysis system
        risk_system = RiskAnalysisSystem()
        
        # Load and prepare data
        logger.info("Loading data...")
        risk_system.load_data()
        
        # Run analysis modules
        logger.info("Running risk analysis...")
        results = risk_system.run_analysis()
        
        # Log scenario information
        stress_tests = results.get("stress_tests", {})
        if stress_tests:
            available_scenarios = list(stress_tests.keys())
            logger.info(f"Analysis complete with {len(available_scenarios)} stress test scenarios available")
            logger.info(f"Scenarios: {', '.join(available_scenarios)}")
        else:
            logger.warning("No stress test scenarios were generated during analysis")
        
        # Start dashboard (if in dashboard mode)
        if settings.RUN_DASHBOARD:
            logger.info("Starting dashboard...")
            risk_system.start_dashboard()
            
        logger.info("Risk Analysis System completed successfully")
        
    except Exception as e:
        logger.exception(f"Error running Risk Analysis System: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 