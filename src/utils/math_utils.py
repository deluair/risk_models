"""
Mathematical utility functions for risk calculations
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from scipy import stats


def calculate_var(returns: np.ndarray, confidence_level: float = 0.95, window: int = None) -> float:
    """Calculate Value at Risk (VaR) using historical simulation method
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level (default: 0.95)
        window: Window to use for calculation (default: all data)
    
    Returns:
        VaR value
    """
    if window is not None and window < len(returns):
        returns = returns[-window:]
    
    return np.percentile(returns, 100 * (1 - confidence_level))


def calculate_es(returns: np.ndarray, confidence_level: float = 0.95, window: int = None) -> float:
    """Calculate Expected Shortfall (ES) / Conditional VaR
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level (default: 0.95)
        window: Window to use for calculation (default: all data)
    
    Returns:
        ES value
    """
    if window is not None and window < len(returns):
        returns = returns[-window:]
    
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()


def calculate_volatility(returns: np.ndarray, window: int = None, annualize: bool = True,
                        trading_days: int = 252) -> Union[float, np.ndarray]:
    """Calculate volatility of returns
    
    Args:
        returns: Array of returns
        window: Rolling window to use (default: None = use all data)
        annualize: Whether to annualize the volatility (default: True)
        trading_days: Number of trading days in a year (default: 252)
    
    Returns:
        Volatility value or array of volatilities
    """
    if window is None:
        vol = returns.std()
        if annualize:
            vol *= np.sqrt(trading_days)
        return vol
    else:
        vol = pd.Series(returns).rolling(window=window).std().values
        if annualize:
            vol *= np.sqrt(trading_days)
        return vol


def calculate_beta(returns: np.ndarray, market_returns: np.ndarray, window: int = None) -> Union[float, np.ndarray]:
    """Calculate beta of returns relative to market returns
    
    Args:
        returns: Array of returns
        market_returns: Array of market returns
        window: Rolling window to use (default: None = use all data)
    
    Returns:
        Beta value or array of betas
    """
    if window is None:
        covariance = np.cov(returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        return covariance / market_variance
    else:
        covariances = pd.Series(returns).rolling(window=window).cov(pd.Series(market_returns)).values
        market_variances = pd.Series(market_returns).rolling(window=window).var().values
        return covariances / market_variances


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, window: int = None,
                          annualize: bool = True, trading_days: int = 252) -> Union[float, np.ndarray]:
    """Calculate Sharpe ratio of returns
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (default: 0.0)
        window: Rolling window to use (default: None = use all data)
        annualize: Whether to annualize the ratio (default: True)
        trading_days: Number of trading days in a year (default: 252)
    
    Returns:
        Sharpe ratio value or array of Sharpe ratios
    """
    if window is None:
        mean_return = returns.mean()
        if annualize:
            mean_return *= trading_days
        
        volatility = calculate_volatility(returns, window=None, annualize=annualize, trading_days=trading_days)
        return (mean_return - risk_free_rate) / volatility
    else:
        roll_mean = pd.Series(returns).rolling(window=window).mean().values
        if annualize:
            roll_mean *= trading_days
        
        roll_vol = calculate_volatility(returns, window=window, annualize=annualize, trading_days=trading_days)
        return (roll_mean - risk_free_rate) / roll_vol


def calculate_drawdown(returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate drawdown statistics from returns
    
    Args:
        returns: Array of returns
    
    Returns:
        Tuple of (drawdown series, maximum drawdown, drawdown duration series)
    """
    # Calculate wealth index
    wealth_index = (1 + returns).cumprod()
    
    # Calculate previous peaks
    previous_peaks = np.maximum.accumulate(wealth_index)
    
    # Calculate drawdown
    drawdown = wealth_index / previous_peaks - 1
    
    # Calculate maximum drawdown
    max_drawdown = drawdown.min()
    
    # Calculate drawdown duration
    drawdown_duration = np.zeros_like(drawdown)
    duration = 0
    
    for i in range(len(drawdown)):
        if drawdown[i] < 0:
            duration += 1
        else:
            duration = 0
        drawdown_duration[i] = duration
    
    return drawdown, max_drawdown, drawdown_duration


def calculate_correlation_matrix(returns_df: pd.DataFrame, window: int = None) -> pd.DataFrame:
    """Calculate correlation matrix for returns
    
    Args:
        returns_df: DataFrame of returns
        window: Rolling window to use (default: None = use all data)
    
    Returns:
        Correlation matrix
    """
    if window is None:
        return returns_df.corr()
    else:
        return returns_df.rolling(window=window).corr()


def calculate_tail_risk_metrics(returns: np.ndarray) -> Dict[str, float]:
    """Calculate tail risk metrics for returns
    
    Args:
        returns: Array of returns
    
    Returns:
        Dictionary of tail risk metrics
    """
    # Calculate skewness and kurtosis
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    
    # Calculate Jarque-Bera test for normality
    jb_stat, jb_pvalue = stats.jarque_bera(returns)
    
    # Calculate Conditional Tail Expectation (CTE)
    var_95 = calculate_var(returns, 0.95)
    var_99 = calculate_var(returns, 0.99)
    cte_95 = calculate_es(returns, 0.95)
    cte_99 = calculate_es(returns, 0.99)
    
    return {
        'skewness': skewness,
        'excess_kurtosis': kurtosis,
        'jarque_bera_stat': jb_stat,
        'jarque_bera_pvalue': jb_pvalue,
        'var_95': var_95,
        'var_99': var_99,
        'cte_95': cte_95,
        'cte_99': cte_99
    }


def calculate_extreme_value_metrics(returns: np.ndarray, threshold: Optional[float] = None,
                                  method: str = 'block_maxima') -> Dict[str, Any]:
    """Calculate Extreme Value Theory (EVT) metrics for returns
    
    Args:
        returns: Array of returns
        threshold: Threshold for Peaks Over Threshold method (default: None = 95th percentile)
        method: Method to use ('block_maxima' or 'pot')
    
    Returns:
        Dictionary of EVT metrics
    """
    if method not in ['block_maxima', 'pot']:
        raise ValueError("Method must be 'block_maxima' or 'pot'")
    
    if method == 'block_maxima':
        # Divide data into blocks and find maximum/minimum in each block
        block_size = int(np.sqrt(len(returns)))
        n_blocks = len(returns) // block_size
        block_maxima = np.zeros(n_blocks)
        block_minima = np.zeros(n_blocks)
        
        for i in range(n_blocks):
            start = i * block_size
            end = (i + 1) * block_size
            block = returns[start:end]
            block_maxima[i] = block.max()
            block_minima[i] = block.min()
        
        # Fit Generalized Extreme Value (GEV) distribution to maxima
        shape_max, loc_max, scale_max = stats.genextreme.fit(block_maxima)
        
        # Fit GEV distribution to minima (negative of minima for losses)
        shape_min, loc_min, scale_min = stats.genextreme.fit(-block_minima)
        
        return {
            'method': 'block_maxima',
            'gev_shape_max': shape_max,
            'gev_loc_max': loc_max,
            'gev_scale_max': scale_max,
            'gev_shape_min': shape_min,
            'gev_loc_min': loc_min,
            'gev_scale_min': scale_min
        }
    else:  # Peaks Over Threshold
        if threshold is None:
            threshold = np.percentile(returns, 95)
        
        # Extract exceedances over threshold
        exceedances = returns[returns > threshold] - threshold
        
        if len(exceedances) < 10:
            return {
                'method': 'pot',
                'error': 'Insufficient exceedances for fitting'
            }
        
        # Fit Generalized Pareto Distribution (GPD) to exceedances
        shape, loc, scale = stats.genpareto.fit(exceedances)
        
        return {
            'method': 'pot',
            'threshold': threshold,
            'exceedances_count': len(exceedances),
            'gpd_shape': shape,
            'gpd_loc': loc,
            'gpd_scale': scale
        }


def calculate_absorption_ratio(returns_df: pd.DataFrame, n_factors: int = None) -> float:
    """Calculate absorption ratio as a measure of systemic risk
    
    Args:
        returns_df: DataFrame of returns
        n_factors: Number of factors to use (default: None = 1/5 of variables)
    
    Returns:
        Absorption ratio
    """
    # Calculate correlation matrix
    corr_matrix = returns_df.corr().values
    
    # Perform eigenvalue decomposition
    eigenvalues, _ = np.linalg.eigh(corr_matrix)
    eigenvalues = eigenvalues[::-1]  # Sort in descending order
    
    # Determine number of factors if not specified
    if n_factors is None:
        n_factors = max(1, len(eigenvalues) // 5)
    
    # Calculate absorption ratio
    total_variance = np.sum(eigenvalues)
    variance_explained = np.sum(eigenvalues[:n_factors])
    
    return variance_explained / total_variance 