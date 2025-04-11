"""
Risk registry for managing different risk types in the Financial Risk Analysis System
"""
import logging
from typing import Dict, List, Any, Optional, Type
import importlib
from pathlib import Path

from src.core.config import settings


class RiskMetric:
    """Base class for risk metrics"""
    
    def __init__(self, name: str, description: str, unit: str = None, threshold_low: float = None, 
                 threshold_medium: float = None, threshold_high: float = None):
        """Initialize a risk metric
        
        Args:
            name: Name of the metric
            description: Description of what the metric measures
            unit: Unit of measurement (e.g., %, USD, etc.)
            threshold_low: Threshold for low risk
            threshold_medium: Threshold for medium risk
            threshold_high: Threshold for high risk
        """
        self.name = name
        self.description = description
        self.unit = unit
        self.threshold_low = threshold_low
        self.threshold_medium = threshold_medium
        self.threshold_high = threshold_high
        
    def assess_risk_level(self, value: float) -> str:
        """Assess the risk level based on the value and thresholds
        
        Args:
            value: The value to assess
            
        Returns:
            Risk level as a string ('low', 'medium', 'high', 'critical')
        """
        if self.threshold_high is not None and value >= self.threshold_high:
            return "critical"
        elif self.threshold_medium is not None and value >= self.threshold_medium:
            return "high"
        elif self.threshold_low is not None and value >= self.threshold_low:
            return "medium"
        else:
            return "low"
    
    def __str__(self):
        return f"{self.name} ({self.unit}): {self.description}"


class RiskCategory:
    """Base class for risk categories"""
    
    def __init__(self, name: str, description: str):
        """Initialize a risk category
        
        Args:
            name: Name of the risk category
            description: Description of the risk category
        """
        self.name = name
        self.description = description
        self.metrics: Dict[str, RiskMetric] = {}
        
    def add_metric(self, metric: RiskMetric):
        """Add a metric to this risk category
        
        Args:
            metric: The metric to add
        """
        self.metrics[metric.name] = metric
        
    def get_metric(self, metric_name: str) -> Optional[RiskMetric]:
        """Get a metric by name
        
        Args:
            metric_name: Name of the metric to get
            
        Returns:
            The metric, or None if not found
        """
        return self.metrics.get(metric_name)
    
    def get_all_metrics(self) -> List[RiskMetric]:
        """Get all metrics for this category
        
        Returns:
            List of all metrics
        """
        return list(self.metrics.values())
    
    def __str__(self):
        return f"{self.name}: {self.description} ({len(self.metrics)} metrics)"


class RiskRegistry:
    """Registry for all risk categories and metrics"""
    
    def __init__(self):
        """Initialize the risk registry"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing RiskRegistry")
        
        # Initialize risk categories
        self.categories: Dict[str, RiskCategory] = {}
        
        # Initialize with standard risk categories
        self._init_market_risk()
        self._init_credit_risk()
        self._init_liquidity_risk()
        self._init_operational_risk()
        self._init_climate_risk()
        self._init_cyber_risk()
        self._init_ai_risk()
        self._init_digitalization_risk()
        self._init_nonbank_intermediation_risk()
        self._init_global_architecture_risk()
        
        self.logger.info(f"RiskRegistry initialized with {len(self.categories)} categories")
    
    def add_category(self, category: RiskCategory):
        """Add a risk category to the registry
        
        Args:
            category: The risk category to add
        """
        self.categories[category.name] = category
        
    def get_category(self, category_name: str) -> Optional[RiskCategory]:
        """Get a risk category by name
        
        Args:
            category_name: Name of the category to get
            
        Returns:
            The category, or None if not found
        """
        return self.categories.get(category_name)
    
    def get_all_categories(self) -> List[RiskCategory]:
        """Get all risk categories
        
        Returns:
            List of all risk categories
        """
        return list(self.categories.values())
    
    def get_metric(self, category_name: str, metric_name: str) -> Optional[RiskMetric]:
        """Get a metric by category and name
        
        Args:
            category_name: Name of the category
            metric_name: Name of the metric
            
        Returns:
            The metric, or None if not found
        """
        category = self.get_category(category_name)
        if category:
            return category.get_metric(metric_name)
        return None
    
    def _init_market_risk(self):
        """Initialize market risk category and metrics"""
        category = RiskCategory(
            name="market_risk",
            description="Risk arising from movements in market prices including equity, interest rates, FX, and commodity risks"
        )
        
        # Add standard market risk metrics
        category.add_metric(RiskMetric(
            name="value_at_risk",
            description="Maximum potential loss at a given confidence level",
            unit="USD",
            threshold_low=1000000,
            threshold_medium=5000000,
            threshold_high=10000000
        ))
        
        category.add_metric(RiskMetric(
            name="expected_shortfall",
            description="Expected loss beyond VaR threshold",
            unit="USD",
            threshold_low=1500000,
            threshold_medium=7000000,
            threshold_high=15000000
        ))
        
        category.add_metric(RiskMetric(
            name="volatility",
            description="Annualized standard deviation of returns",
            unit="%",
            threshold_low=15,
            threshold_medium=25,
            threshold_high=35
        ))
        
        category.add_metric(RiskMetric(
            name="stress_loss",
            description="Potential loss under stress scenario",
            unit="USD",
            threshold_low=5000000,
            threshold_medium=20000000,
            threshold_high=50000000
        ))
        
        category.add_metric(RiskMetric(
            name="tail_risk",
            description="Severity of extreme events beyond expected distributions",
            unit="risk score",
            threshold_low=50,
            threshold_medium=70,
            threshold_high=85
        ))
        
        self.add_category(category)
    
    def _init_credit_risk(self):
        """Initialize credit risk category and metrics"""
        category = RiskCategory(
            name="credit_risk",
            description="Risk of loss resulting from borrower or counterparty default"
        )
        
        # Add standard credit risk metrics
        category.add_metric(RiskMetric(
            name="probability_of_default",
            description="Likelihood of default over a specific time horizon",
            unit="%",
            threshold_low=1,
            threshold_medium=5,
            threshold_high=10
        ))
        
        category.add_metric(RiskMetric(
            name="loss_given_default",
            description="Portion of exposure likely to be lost in event of default",
            unit="%",
            threshold_low=30,
            threshold_medium=50,
            threshold_high=70
        ))
        
        category.add_metric(RiskMetric(
            name="exposure_at_default",
            description="Amount of exposure when default occurs",
            unit="USD",
            threshold_low=1000000,
            threshold_medium=5000000,
            threshold_high=10000000
        ))
        
        category.add_metric(RiskMetric(
            name="credit_var",
            description="Maximum potential credit loss at a given confidence level",
            unit="USD",
            threshold_low=2000000,
            threshold_medium=8000000,
            threshold_high=15000000
        ))
        
        category.add_metric(RiskMetric(
            name="concentration_risk",
            description="Risk arising from concentrated exposures to sectors/entities",
            unit="risk score",
            threshold_low=40,
            threshold_medium=60,
            threshold_high=80
        ))
        
        self.add_category(category)
    
    def _init_liquidity_risk(self):
        """Initialize liquidity risk category and metrics"""
        category = RiskCategory(
            name="liquidity_risk",
            description="Risk of insufficient liquid assets to meet obligations or fund asset purchases"
        )
        
        # Add standard liquidity risk metrics
        category.add_metric(RiskMetric(
            name="liquidity_coverage_ratio",
            description="Ratio of high-quality liquid assets to net cash outflows",
            unit="ratio",
            threshold_low=1.2,
            threshold_medium=1.1,
            threshold_high=1.0
        ))
        
        category.add_metric(RiskMetric(
            name="net_stable_funding_ratio",
            description="Ratio of available stable funding to required stable funding",
            unit="ratio",
            threshold_low=1.1,
            threshold_medium=1.05,
            threshold_high=1.0
        ))
        
        category.add_metric(RiskMetric(
            name="bid_ask_spread",
            description="Difference between bid and ask prices",
            unit="basis points",
            threshold_low=10,
            threshold_medium=25,
            threshold_high=50
        ))
        
        category.add_metric(RiskMetric(
            name="market_depth",
            description="Amount that can be traded at current prices",
            unit="USD",
            threshold_low=5000000,
            threshold_medium=2000000,
            threshold_high=500000
        ))
        
        category.add_metric(RiskMetric(
            name="funding_concentration",
            description="Concentration of funding sources",
            unit="risk score",
            threshold_low=30,
            threshold_medium=60,
            threshold_high=80
        ))
        
        self.add_category(category)
    
    def _init_operational_risk(self):
        """Initialize operational risk category and metrics"""
        category = RiskCategory(
            name="operational_risk",
            description="Risk of loss from inadequate or failed internal processes, people, and systems"
        )
        
        # Add standard operational risk metrics
        category.add_metric(RiskMetric(
            name="loss_frequency",
            description="Frequency of operational loss events",
            unit="events/month",
            threshold_low=5,
            threshold_medium=10,
            threshold_high=20
        ))
        
        category.add_metric(RiskMetric(
            name="loss_severity",
            description="Average severity of operational loss events",
            unit="USD",
            threshold_low=100000,
            threshold_medium=500000,
            threshold_high=1000000
        ))
        
        category.add_metric(RiskMetric(
            name="operational_var",
            description="Maximum potential operational loss at a given confidence level",
            unit="USD",
            threshold_low=1000000,
            threshold_medium=5000000,
            threshold_high=10000000
        ))
        
        category.add_metric(RiskMetric(
            name="control_effectiveness",
            description="Effectiveness of operational risk controls",
            unit="score",
            threshold_low=80,
            threshold_medium=60,
            threshold_high=40
        ))
        
        category.add_metric(RiskMetric(
            name="recovery_time",
            description="Time to recover from operational disruptions",
            unit="hours",
            threshold_low=4,
            threshold_medium=12,
            threshold_high=24
        ))
        
        self.add_category(category)
    
    def _init_climate_risk(self):
        """Initialize climate risk category and metrics"""
        category = RiskCategory(
            name="climate_risk",
            description="Risk arising from climate change impacts and transition to low-carbon economy"
        )
        
        # Add climate risk metrics
        category.add_metric(RiskMetric(
            name="physical_risk_exposure",
            description="Exposure to physical climate risks (floods, fires, etc.)",
            unit="score",
            threshold_low=30,
            threshold_medium=60,
            threshold_high=80
        ))
        
        category.add_metric(RiskMetric(
            name="transition_risk_exposure",
            description="Exposure to transition risks (policy, technology, market shifts)",
            unit="score",
            threshold_low=30,
            threshold_medium=60,
            threshold_high=80
        ))
        
        category.add_metric(RiskMetric(
            name="carbon_intensity",
            description="Carbon emissions relative to revenue or value",
            unit="tCO2e/$M",
            threshold_low=100,
            threshold_medium=250,
            threshold_high=500
        ))
        
        category.add_metric(RiskMetric(
            name="climate_var",
            description="Value at risk from climate scenarios",
            unit="USD",
            threshold_low=5000000,
            threshold_medium=15000000,
            threshold_high=30000000
        ))
        
        category.add_metric(RiskMetric(
            name="green_asset_ratio",
            description="Proportion of assets aligned with sustainable activities",
            unit="%",
            threshold_low=50,
            threshold_medium=25,
            threshold_high=10
        ))
        
        self.add_category(category)
    
    def _init_cyber_risk(self):
        """Initialize cyber risk category and metrics"""
        category = RiskCategory(
            name="cyber_risk",
            description="Risk arising from cyberattacks, data breaches, and system vulnerabilities"
        )
        
        # Add cyber risk metrics
        category.add_metric(RiskMetric(
            name="attack_frequency",
            description="Frequency of cyber attacks",
            unit="events/month",
            threshold_low=5,
            threshold_medium=20,
            threshold_high=50
        ))
        
        category.add_metric(RiskMetric(
            name="vulnerability_score",
            description="Measure of system vulnerabilities",
            unit="score",
            threshold_low=40,
            threshold_medium=70,
            threshold_high=90
        ))
        
        category.add_metric(RiskMetric(
            name="data_breach_impact",
            description="Potential financial impact of data breaches",
            unit="USD",
            threshold_low=1000000,
            threshold_medium=5000000,
            threshold_high=20000000
        ))
        
        category.add_metric(RiskMetric(
            name="recovery_capability",
            description="Capability to recover from cyber incidents",
            unit="score",
            threshold_low=80,
            threshold_medium=60,
            threshold_high=40
        ))
        
        category.add_metric(RiskMetric(
            name="third_party_risk",
            description="Cyber risk from third-party vendors and services",
            unit="score",
            threshold_low=30,
            threshold_medium=60,
            threshold_high=80
        ))
        
        self.add_category(category)
    
    def _init_ai_risk(self):
        """Initialize AI risk category and metrics"""
        category = RiskCategory(
            name="ai_risk",
            description="Risk arising from artificial intelligence systems and algorithms"
        )
        
        # Add AI risk metrics
        category.add_metric(RiskMetric(
            name="model_error",
            description="Frequency and impact of AI model errors",
            unit="score",
            threshold_low=30,
            threshold_medium=60,
            threshold_high=80
        ))
        
        category.add_metric(RiskMetric(
            name="algorithmic_bias",
            description="Presence of bias in AI algorithms",
            unit="score",
            threshold_low=20,
            threshold_medium=50,
            threshold_high=70
        ))
        
        category.add_metric(RiskMetric(
            name="ai_dependency",
            description="Dependency on AI systems for critical functions",
            unit="score",
            threshold_low=40,
            threshold_medium=70,
            threshold_high=90
        ))
        
        category.add_metric(RiskMetric(
            name="ai_explainability",
            description="Level of explainability in AI systems",
            unit="score",
            threshold_low=80,
            threshold_medium=50,
            threshold_high=20
        ))
        
        category.add_metric(RiskMetric(
            name="ai_governance",
            description="Effectiveness of AI governance framework",
            unit="score",
            threshold_low=80,
            threshold_medium=50,
            threshold_high=20
        ))
        
        self.add_category(category)
    
    def _init_digitalization_risk(self):
        """Initialize digitalization risk category and metrics"""
        category = RiskCategory(
            name="digitalization",
            description="Risk arising from digital transformation and technological change"
        )
        
        # Add digitalization risk metrics
        category.add_metric(RiskMetric(
            name="digital_infrastructure",
            description="Robustness of digital infrastructure",
            unit="score",
            threshold_low=80,
            threshold_medium=50,
            threshold_high=20
        ))
        
        category.add_metric(RiskMetric(
            name="digital_adoption",
            description="Level of digital adoption relative to peers",
            unit="percentile",
            threshold_low=70,
            threshold_medium=40,
            threshold_high=20
        ))
        
        category.add_metric(RiskMetric(
            name="tech_concentration",
            description="Concentration in technology providers",
            unit="score",
            threshold_low=30,
            threshold_medium=60,
            threshold_high=80
        ))
        
        category.add_metric(RiskMetric(
            name="digital_disruption",
            description="Exposure to digital business model disruption",
            unit="score",
            threshold_low=30,
            threshold_medium=60,
            threshold_high=80
        ))
        
        category.add_metric(RiskMetric(
            name="data_quality",
            description="Quality of data for digital operations",
            unit="score",
            threshold_low=80,
            threshold_medium=50,
            threshold_high=20
        ))
        
        self.add_category(category)
    
    def _init_nonbank_intermediation_risk(self):
        """Initialize nonbank intermediation risk category and metrics"""
        category = RiskCategory(
            name="nonbank_intermediation",
            description="Risk arising from nonbank financial intermediation activities"
        )
        
        # Add nonbank intermediation risk metrics
        category.add_metric(RiskMetric(
            name="nonbank_growth",
            description="Growth rate of nonbank financial sector",
            unit="%",
            threshold_low=5,
            threshold_medium=15,
            threshold_high=25
        ))
        
        category.add_metric(RiskMetric(
            name="interconnectedness",
            description="Interconnectedness between banks and nonbanks",
            unit="score",
            threshold_low=30,
            threshold_medium=60,
            threshold_high=80
        ))
        
        category.add_metric(RiskMetric(
            name="leverage",
            description="Leverage in nonbank entities",
            unit="ratio",
            threshold_low=5,
            threshold_medium=10,
            threshold_high=20
        ))
        
        category.add_metric(RiskMetric(
            name="maturity_transformation",
            description="Extent of maturity transformation in nonbank entities",
            unit="score",
            threshold_low=30,
            threshold_medium=60,
            threshold_high=80
        ))
        
        category.add_metric(RiskMetric(
            name="regulatory_arbitrage",
            description="Evidence of regulatory arbitrage activities",
            unit="score",
            threshold_low=20,
            threshold_medium=50,
            threshold_high=70
        ))
        
        self.add_category(category)
    
    def _init_global_architecture_risk(self):
        """Initialize global architecture risk category and metrics"""
        category = RiskCategory(
            name="global_architecture",
            description="Risk arising from changes in global financial architecture"
        )
        
        # Add global architecture risk metrics
        category.add_metric(RiskMetric(
            name="fragmentation",
            description="Financial system fragmentation into regional blocs",
            unit="score",
            threshold_low=20,
            threshold_medium=50,
            threshold_high=70
        ))
        
        category.add_metric(RiskMetric(
            name="reserve_currency",
            description="Changes in reserve currency composition",
            unit="score",
            threshold_low=20,
            threshold_medium=50,
            threshold_high=70
        ))
        
        category.add_metric(RiskMetric(
            name="capital_flow_volatility",
            description="Volatility in cross-border capital flows",
            unit="score",
            threshold_low=30,
            threshold_medium=60,
            threshold_high=80
        ))
        
        category.add_metric(RiskMetric(
            name="regulatory_divergence",
            description="Divergence in regulatory standards across regions",
            unit="score",
            threshold_low=20,
            threshold_medium=50,
            threshold_high=70
        ))
        
        category.add_metric(RiskMetric(
            name="geopolitical_tension",
            description="Impact of geopolitical tensions on financial system",
            unit="score",
            threshold_low=30,
            threshold_medium=60,
            threshold_high=80
        ))
        
        self.add_category(category) 