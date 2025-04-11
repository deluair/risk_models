"""
Configuration settings for the Financial Risk Analysis System
"""
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


# Define base directory
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


@dataclass
class Settings:
    """Application settings container"""
    # General settings
    APP_NAME: str = "Financial Risk Analysis System"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    
    # Data settings
    DATA_DIR: Path = DATA_DIR
    RAW_DATA_DIR: Path = RAW_DATA_DIR
    PROCESSED_DATA_DIR: Path = PROCESSED_DATA_DIR
    
    # Analysis settings
    RISK_CATEGORIES: List[str] = field(default_factory=lambda: [
        "market_risk", 
        "credit_risk", 
        "liquidity_risk", 
        "operational_risk",
        "climate_risk",
        "cyber_risk",
        "ai_risk",
        "digitalization",
        "nonbank_intermediation",
        "global_architecture"
    ])
    
    # Network analysis settings
    NETWORK_CENTRALITY_MEASURES: List[str] = field(default_factory=lambda: [
        "degree_centrality",
        "betweenness_centrality", 
        "eigenvector_centrality",
        "katz_centrality"
    ])
    
    # Visualization settings
    DEFAULT_CHART_THEME: str = "plotly_white"
    COLOR_PALETTE: Dict[str, str] = field(default_factory=lambda: {
        "market_risk": "#FF5733",
        "credit_risk": "#33FF57", 
        "liquidity_risk": "#3357FF",
        "operational_risk": "#F033FF",
        "climate_risk": "#33FFF0",
        "cyber_risk": "#F0FF33",
        "ai_risk": "#FF33F0",
        "digitalization": "#33F0FF",
        "nonbank_intermediation": "#F0F033",
        "global_architecture": "#F03333"
    })
    
    # Dashboard settings
    RUN_DASHBOARD: bool = True
    DASHBOARD_HOST: str = "127.0.0.1"
    DASHBOARD_PORT: int = 8050
    DASHBOARD_DEBUG: bool = DEBUG
    
    # Model settings
    USE_ML_MODELS: bool = True
    MODEL_SAVE_DIR: Path = BASE_DIR / "models"
    
    # Stress test settings
    STRESS_TEST_SCENARIOS: List[str] = field(default_factory=lambda: [
        "baseline",
        "adverse",
        "severely_adverse", 
        "climate_transition",
        "cyber_attack",
        "liquidity_freeze"
    ])
    
    # Risk thresholds
    RISK_THRESHOLDS: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "market_risk": {"low": 0.2, "medium": 0.5, "high": 0.8},
        "credit_risk": {"low": 0.15, "medium": 0.4, "high": 0.7},
        "liquidity_risk": {"low": 0.1, "medium": 0.3, "high": 0.6},
        "operational_risk": {"low": 0.2, "medium": 0.4, "high": 0.7},
        "climate_risk": {"low": 0.2, "medium": 0.5, "high": 0.75},
        "cyber_risk": {"low": 0.25, "medium": 0.5, "high": 0.8},
    })


# Create settings instance
settings = Settings() 