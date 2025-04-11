"""
Extended Risk Visualization Module

This module provides functionality to visualize extended risk types:
- Operational Risk
- Climate Risk
- Cyber Risk
- AI Risk
- Digitalization Risk

It generates various visualizations including time series plots, radar charts,
heatmaps, and comprehensive risk dashboards.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

class ExtendedRiskVisualizer:
    """
    Class for visualizing extended risk data for various risk categories.
    """
    
    def __init__(self, output_dir="reports/visualizations"):
        """
        Initialize the risk visualizer.
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default visualization style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Define color schemes for different risk types
        self.color_schemes = {
            'operational_risk': 'YlOrRd',
            'climate_risk': 'BuGn',
            'cyber_risk': 'PuRd',
            'ai_risk': 'BuPu',
            'digitalization_risk': 'OrRd'
        }
        
        # Mapping for more readable titles
        self.readable_names = {
            'operational_risk': 'Operational Risk',
            'climate_risk': 'Climate Risk',
            'cyber_risk': 'Cyber Risk',
            'ai_risk': 'AI Risk',
            'digitalization_risk': 'Digitalization Risk',
            'process_failure': 'Process Failure',
            'human_error': 'Human Error',
            'system_failure': 'System Failure',
            'legal_risk': 'Legal Risk',
            'regulatory_compliance': 'Regulatory Compliance',
            'fraud_risk': 'Fraud Risk',
            'transition_risk': 'Transition Risk',
            'physical_risk': 'Physical Risk',
            'regulatory_risk': 'Regulatory Risk',
            'market_risk': 'Market Risk',
            'technology_risk': 'Technology Risk',
            'reputation_risk': 'Reputation Risk',
            'data_breach': 'Data Breach',
            'system_outage': 'System Outage',
            'ddos': 'DDoS Attack',
            'ransomware': 'Ransomware',
            'phishing': 'Phishing',
            'insider_threat': 'Insider Threat',
            'model_risk': 'Model Risk',
            'data_quality': 'Data Quality',
            'bias': 'Bias',
            'explainability': 'Explainability',
            'stability': 'Stability',
            'legacy_systems': 'Legacy Systems',
            'digital_transformation': 'Digital Transformation',
            'tech_debt': 'Technical Debt',
            'innovation_gap': 'Innovation Gap',
            'digital_competence': 'Digital Competence',
            'data_management': 'Data Management',
            'overall': 'Overall'
        }
        
    def load_data(self, data_path):
        """
        Load risk data from a file.
        
        Args:
            data_path (str): Path to the data file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        
        # Convert date column to datetime if it exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def plot_time_series(self, df, risk_type, metrics=None, scenario='base', figsize=(12, 8)):
        """
        Create time series plots for risk metrics.
        
        Args:
            df (pd.DataFrame): Risk data
            risk_type (str): Type of risk
            metrics (list, optional): List of metrics to plot. If None, use all except 'overall'
            scenario (str): Scenario name for the title
            figsize (tuple): Figure size
            
        Returns:
            str: Path to the saved figure
        """
        if 'date' not in df.columns:
            raise ValueError("DataFrame must contain a 'date' column")
        
        # Determine which metrics to plot
        all_metrics = [col for col in df.columns if col != 'date']
        if metrics is None:
            # Include all metrics except 'overall'
            metrics = [m for m in all_metrics if m != 'overall']
        
        # Set up plot
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize)
        
        # Plot individual metrics
        ax = axes[0]
        for metric in metrics:
            if metric in df.columns:
                ax.plot(df['date'], df[metric], label=self.readable_names.get(metric, metric))
        
        ax.set_title(f"{self.readable_names.get(risk_type, risk_type)} - {scenario.replace('_', ' ').title()} Scenario", fontsize=14)
        ax.set_ylabel("Risk Score", fontsize=12)
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)
        
        # Plot overall risk separately
        ax = axes[1]
        if 'overall' in df.columns:
            ax.plot(df['date'], df['overall'], color='red', linewidth=2, label='Overall Risk')
            
            # Add a threshold line at 0.6 (high risk)
            ax.axhline(y=0.6, color='darkred', linestyle='--', alpha=0.7, label='High Risk Threshold')
            
            # Add a threshold line at 0.3 (medium risk)
            ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Medium Risk Threshold')
        
        ax.set_title(f"Overall {self.readable_names.get(risk_type, risk_type)}", fontsize=14)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Risk Score", fontsize=12)
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)
        
        # Set limits for both y-axes
        for ax in axes:
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(self.output_dir, f"{risk_type}_{scenario}_time_series.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Time series plot saved to {output_file}")
        return output_file
    
    def create_radar_chart(self, df, risk_type, scenario='base', figsize=(10, 10)):
        """
        Create a radar chart of the latest risk metrics.
        
        Args:
            df (pd.DataFrame): Risk data
            risk_type (str): Type of risk
            scenario (str): Scenario name for the title
            figsize (tuple): Figure size
            
        Returns:
            str: Path to the saved figure
        """
        # Get the most recent data point
        latest_data = df.iloc[-1].copy()
        
        # Remove date column if present
        if 'date' in latest_data:
            latest_data = latest_data.drop('date')
        
        # Exclude overall for the radar chart
        if 'overall' in latest_data:
            latest_data = latest_data.drop('overall')
        
        metrics = latest_data.index.tolist()
        values = latest_data.values
        
        # Number of variables
        n = len(metrics)
        
        # Create angles for each metric
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        
        # Close the plot
        values = np.append(values, values[0])
        angles = np.append(angles, angles[0])
        metrics = metrics + [metrics[0]]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        
        # Plot data
        ax.plot(angles, values, linewidth=2, linestyle='solid')
        
        # Fill area
        ax.fill(angles, values, alpha=0.25)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([self.readable_names.get(m, m) for m in metrics[:-1]])
        
        # Add risk zones (concentric circles)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2 (Low)', '0.4 (Medium)', '0.6 (High)', '0.8 (Critical)'])
        ax.set_ylim(0, 1)
        
        # Add title
        plt.title(f"{self.readable_names.get(risk_type, risk_type)} Profile - {scenario.replace('_', ' ').title()} Scenario", size=15)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(self.output_dir, f"{risk_type}_{scenario}_radar.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Radar chart saved to {output_file}")
        return output_file
    
    def create_heatmap(self, dfs, scenario='base', figsize=(15, 10)):
        """
        Create a heatmap showing the overall risk levels across different risk types.
        
        Args:
            dfs (dict): Dictionary mapping risk types to DataFrames
            scenario (str): Scenario name for the title
            figsize (tuple): Figure size
            
        Returns:
            str: Path to the saved figure
        """
        # Extract the most recent 'overall' value for each risk type
        data = {}
        for risk_type, df in dfs.items():
            if 'overall' in df.columns:
                data[risk_type] = df['overall'].iloc[-1]
        
        # Create a new DataFrame for the heatmap
        risk_types = list(data.keys())
        risk_values = [data[rt] for rt in risk_types]
        
        # Sort by risk value (highest first)
        sorted_indices = np.argsort(risk_values)[::-1]
        sorted_types = [risk_types[i] for i in sorted_indices]
        sorted_values = [risk_values[i] for i in sorted_indices]
        
        # Create a DataFrame with human-readable names
        df_heatmap = pd.DataFrame({
            'Risk Type': [self.readable_names.get(rt, rt) for rt in sorted_types],
            'Risk Score': sorted_values
        })
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create custom colormap: green -> yellow -> red
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['darkgreen', 'yellowgreen', 'yellow', 'orange', 'red'])
        
        # Plot heatmap
        sns.barplot(x='Risk Score', y='Risk Type', data=df_heatmap, ax=ax, palette=cmap(sorted_values))
        
        # Add risk values as text
        for i, v in enumerate(sorted_values):
            ax.text(v + 0.01, i, f"{v:.3f}", va='center')
        
        # Add colored background based on risk levels
        ax.axvspan(0, 0.3, alpha=0.1, color='green', label='Low Risk')
        ax.axvspan(0.3, 0.6, alpha=0.1, color='yellow', label='Medium Risk')
        ax.axvspan(0.6, 1.0, alpha=0.1, color='red', label='High Risk')
        
        # Add title and labels
        ax.set_title(f"Overall Risk Comparison - {scenario.replace('_', ' ').title()} Scenario", fontsize=15)
        ax.set_xlabel("Risk Score", fontsize=12)
        ax.set_ylabel("Risk Type", fontsize=12)
        
        # Set limits
        ax.set_xlim(0, 1)
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(self.output_dir, f"risk_comparison_{scenario}_heatmap.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Risk heatmap saved to {output_file}")
        return output_file
    
    def generate_risk_report(self, dfs, scenario='base'):
        """
        Generate a comprehensive risk report with multiple visualizations.
        
        Args:
            dfs (dict): Dictionary mapping risk types to DataFrames
            scenario (str): Scenario name for the title
            
        Returns:
            str: Path to the saved report
        """
        # Create a subfolder for this report
        report_dir = os.path.join(self.output_dir, f"report_{scenario}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate all visualizations
        files = []
        
        # Time series for each risk type
        for risk_type, df in dfs.items():
            file = self.plot_time_series(df, risk_type, scenario=scenario)
            files.append(file)
        
        # Radar charts for each risk type
        for risk_type, df in dfs.items():
            file = self.create_radar_chart(df, risk_type, scenario=scenario)
            files.append(file)
        
        # Heatmap comparing all risk types
        file = self.create_heatmap(dfs, scenario=scenario)
        files.append(file)
        
        # Create an HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Extended Risk Report - {scenario.replace('_', ' ').title()} Scenario</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333366; }}
                .visualization {{ margin: 20px 0; }}
                img {{ max-width: 100%; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>Extended Risk Analysis Report</h1>
            <h2>{scenario.replace('_', ' ').title()} Scenario</h2>
            <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Risk Overview</h2>
            <div class="visualization">
                <img src="../{os.path.basename(files[-1])}" alt="Risk Comparison Heatmap">
                <p>Overall risk comparison across all risk types.</p>
            </div>
            
            <h2>Detailed Risk Analysis</h2>
        """
        
        # Add sections for each risk type
        risk_types = list(dfs.keys())
        for risk_type in risk_types:
            time_series_file = [f for f in files if f"{risk_type}_{scenario}_time_series.png" in f][0]
            radar_file = [f for f in files if f"{risk_type}_{scenario}_radar.png" in f][0]
            
            html_content += f"""
            <h3>{self.readable_names.get(risk_type, risk_type)}</h3>
            <div class="visualization">
                <img src="../{os.path.basename(time_series_file)}" alt="{risk_type} Time Series">
                <p>Time series analysis of {self.readable_names.get(risk_type, risk_type)} metrics.</p>
            </div>
            <div class="visualization">
                <img src="../{os.path.basename(radar_file)}" alt="{risk_type} Radar Chart">
                <p>Radar chart showing the distribution of {self.readable_names.get(risk_type, risk_type)} metrics.</p>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Save the HTML report
        report_file = os.path.join(report_dir, "index.html")
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Risk report generated at {report_file}")
        return report_file

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    visualizer = ExtendedRiskVisualizer()
    
    # Load sample data (this assumes data was generated using the ExtendedRiskSimulator)
    data_dir = "data/simulated"
    scenario = "base"
    
    try:
        dfs = {}
        for risk_type in ['operational_risk', 'climate_risk', 'cyber_risk', 'ai_risk', 'digitalization_risk']:
            data_file = os.path.join(data_dir, f"{risk_type}_{scenario}.csv")
            dfs[risk_type] = visualizer.load_data(data_file)
        
        # Generate visualizations
        for risk_type, df in dfs.items():
            visualizer.plot_time_series(df, risk_type, scenario=scenario)
            visualizer.create_radar_chart(df, risk_type, scenario=scenario)
        
        # Create comparison heatmap
        visualizer.create_heatmap(dfs, scenario=scenario)
        
        # Generate comprehensive report
        report_file = visualizer.generate_risk_report(dfs, scenario=scenario)
        logger.info(f"Sample visualizations and report generated. Report: {report_file}")
        
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        logger.info("Please run the extended_risk_simulation.py script first to generate sample data.") 