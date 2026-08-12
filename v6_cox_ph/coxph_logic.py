"""
Cox Proportional Hazards mathematical logic.

This module contains the core mathematical computations for the Cox-PH model,
separated from the orchestration logic in central.py and partial.py.
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
from scipy.linalg import solve
from typing import Tuple, List, Dict, Any

from vantage6_strongaya_general.miscellaneous import safe_log


def compute_derivatives(
    summed_agg1: np.ndarray,
    summed_agg2: np.ndarray,
    summed_agg3: np.ndarray,
    aggregated_time_events: pd.DataFrame,
    z_sum: pd.Series
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the primary and secondary derivatives for the Cox-PH model.
    
    These derivatives are used in the Newton-Raphson optimization to find
    the maximum partial likelihood estimates of the regression coefficients.
    
    Parameters
    ----------
    summed_agg1 : np.ndarray
        Aggregated sum of exp(beta * X) across all nodes for each event time
    summed_agg2 : np.ndarray
        Aggregated sum of X * exp(beta * X) across all nodes for each event time
    summed_agg3 : np.ndarray
        Aggregated sum of outer products X * X^T * exp(beta * X) across all nodes
        for each event time
    aggregated_time_events : pd.DataFrame
        DataFrame containing unique event times and their frequencies
    z_sum : pd.Series
        Sum of explanatory variables for all event cases
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - primary_derivative: First derivative of the partial log-likelihood
        - secondary_derivative: Second derivative (Hessian) of the partial log-likelihood
    """
    safe_log("debug", "Computing derivatives for Cox-PH model")
    
    n_covs = len(z_sum)
    tot_p1 = np.zeros(n_covs)
    tot_p2 = np.zeros((n_covs, n_covs))
    
    # Iterate over each row in the DataFrame
    for index, row in aggregated_time_events.iterrows():
        time_col = aggregated_time_events.columns[0]  # Get the time column name
        
        # Get frequency for this time point
        freq = row.get('freq', 1) if 'freq' in row else 1
        
        # Skip if we don't have data for this time point
        if index >= len(summed_agg1):
            continue
            
        s1_value = summed_agg1[index]
        s2_value = summed_agg2[index] if isinstance(summed_agg2, np.ndarray) else np.array(summed_agg2[index])
        s3_value = summed_agg3[index] if isinstance(summed_agg3, np.ndarray) else np.array(summed_agg3[index])
        
        # Handle potential NaN or zero values
        if s1_value <= 0 or np.isnan(s1_value):
            safe_log("warning", f"Invalid s1_value at index {index}: {s1_value}")
            continue
            
        # Compute the primary derivative component
        s1 = freq * (s2_value / s1_value)
        
        # Compute the first part of the secondary derivative component
        first_part = s3_value / s1_value
        
        # Compute the second part of the secondary derivative component
        # The numerator is the outer product of agg2
        if isinstance(s2_value, (list, np.ndarray)):
            s2_array = np.array(s2_value)
            numerator = np.outer(s2_array, s2_array)
        else:
            numerator = np.zeros((n_covs, n_covs))
            
        denominator = s1_value * s1_value
        second_part = numerator / denominator if denominator > 0 else np.zeros((n_covs, n_covs))
        
        s2 = freq * (first_part - second_part)
        
        tot_p1 += s1
        tot_p2 += s2
    
    # Compute the primary and secondary derivatives
    primary_derivative = z_sum.values - tot_p1
    secondary_derivative = -tot_p2
    
    safe_log("debug", f"Primary derivative shape: {primary_derivative.shape}")
    safe_log("debug", f"Secondary derivative shape: {secondary_derivative.shape}")
    
    return primary_derivative, secondary_derivative


def compute_model_results(
    beta: np.ndarray,
    secondary_derivative: np.ndarray,
    z_sum: pd.Series,
    aggregated_time_events: pd.DataFrame,
    summed_agg1: np.ndarray
) -> Dict[str, Any]:
    """
    Compute final model results after convergence.
    
    Parameters
    ----------
    beta : np.ndarray
        Final estimated regression coefficients
    secondary_derivative : np.ndarray
        Final Hessian matrix (negative second derivative)
    z_sum : pd.Series
        Sum of explanatory variables for all event cases
    aggregated_time_events : pd.DataFrame
        DataFrame containing unique event times and their frequencies
    summed_agg1 : np.ndarray
        Final aggregated sum of exp(beta * X) for each event time
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - coefficients: DataFrame with model coefficients and statistics
        - fisher_info: Fisher information matrix
        - standard_errors: Standard errors for each coefficient
        - zvalues: Z-values for each coefficient
        - pvalues: P-values for each coefficient
        - overall_p_value: Overall model p-value (Wald test)
        - aic: Akaike Information Criterion
        - n_params: Number of parameters
        - warnings: List of warning messages
    """
    safe_log("debug", "Computing final model results")
    
    n_covs = len(beta)
    expl_vars = [f"var_{i}" for i in range(n_covs)]  # Will be replaced with actual names
    
    # Computing the standard errors
    try:
        fisher = np.linalg.inv(-secondary_derivative)
        SErrors = []
        for k in range(fisher.shape[0]):
            SErrors.append(np.sqrt(fisher[k, k]))
        SErrors = np.array(SErrors)
    except np.linalg.LinAlgError as e:
        safe_log("warning", f"Could not invert Hessian matrix: {e}")
        fisher = np.zeros((n_covs, n_covs))
        SErrors = np.array([np.nan] * n_covs)
    
    # Calculating P and Z values
    with np.errstate(divide='ignore', invalid='ignore'):
        zvalues = np.zeros(n_covs)
        pvalues = np.ones(n_covs)
        
        for i in range(n_covs):
            if SErrors[i] > 0 and not np.isnan(SErrors[i]):
                zvalues[i] = (np.exp(beta[i]) - 1) / SErrors[i]
                pvalues[i] = 2 * norm.cdf(-abs(zvalues[i]))
            else:
                zvalues[i] = np.nan
                pvalues[i] = np.nan
    
    # Calculate overall model significance using Wald test
    # Reference: Andersen & Gill (1982) "Cox's regression model for counting processes"
    degrees_of_freedom = len(beta)
    try:
        wald_statistic = np.dot(beta, np.dot(-secondary_derivative, beta))
        overall_p_value = chi2.sf(wald_statistic, degrees_of_freedom)
    except Exception as e:
        safe_log("warning", f"Could not compute Wald statistic: {e}")
        wald_statistic = np.nan
        overall_p_value = np.nan
    
    # Compute AIC for model comparison
    # Reference: Cox (1972) "Regression models and life tables" - defines partial likelihood
    try:
        # Cox partial log-likelihood: L(beta) = sum[beta'x_i - log(sum_j exp(beta'x_j))]
        # First term: linear predictor contribution for all events
        linear_part = np.dot(z_sum.values, beta)
        
        # Second term: log of risk set sums (denominator terms)
        risk_set_part = 0
        if hasattr(summed_agg1, '__len__') and len(summed_agg1) > 0:
            for i in range(len(aggregated_time_events)):
                if i < len(summed_agg1) and summed_agg1[i] > 0:
                    freq = aggregated_time_events.iloc[i].get('freq', 1)
                    # Check for numerical issues before computing log
                    if summed_agg1[i] <= 0:
                        safe_log("warning", f"Risk set sum is non-positive at time index {i}: {summed_agg1[i]}")
                        continue
                    risk_set_part += freq * np.log(summed_agg1[i])
        
        log_likelihood = linear_part - risk_set_part
        n_params = len(beta)  # degrees of freedom
        
        # Check for numerical issues in log-likelihood
        if np.isnan(log_likelihood) or np.isinf(log_likelihood):
            raise ValueError(f"Invalid log-likelihood: {log_likelihood}")
        
        # AIC = -2 * log-likelihood + 2 * k (Akaike, 1974)
        aic = -2 * log_likelihood + 2 * n_params
        
    except (ValueError, IndexError, FloatingPointError) as e:
        safe_log("warning", f"Could not compute AIC due to numerical/data issue: {e}")
        aic = np.nan
        n_params = len(beta)
    except Exception as e:
        safe_log("warning", f"Unexpected error computing AIC: {e}")
        aic = np.nan
        n_params = len(beta)
    
    # 95% CI = beta +/- 1.96 * SE
    # Create results DataFrame
    results_data = {
        "Coef": np.around(beta, 5),
        "Exp(coef)": np.around(np.exp(beta), 5),
        "SE": np.around(SErrors, 5),
        "lower_CI": np.around(np.exp(beta - 1.96 * SErrors), 5),
        "upper_CI": np.around(np.exp(beta + 1.96 * SErrors), 5),
        "Z": zvalues,
        "p-value": pvalues
    }
    
    # Collect warnings for perfect prediction
    warnings = []
    threshold = 10
    for i, (coef, se) in enumerate(zip(beta, SErrors)):
        if (
                abs(coef) > threshold or np.isinf(coef) or np.isnan(coef) or
                abs(se) > threshold or np.isinf(se) or np.isnan(se)
        ):
            var_name = expl_vars[i] if i < len(expl_vars) else f"var_{i}"
            msg = (
                f"Warning: Covariate '{var_name}' may perfectly predict the event "
                f"(coef={coef}, SE={se}). Results may be unreliable."
            )
            safe_log("warning", msg)
            warnings.append(msg)
    
    return {
        "coefficients": results_data,
        "fisher_info": fisher,
        "standard_errors": SErrors,
        "zvalues": zvalues,
        "pvalues": pvalues,
        "overall_p_value": float(overall_p_value) if not np.isnan(overall_p_value) else None,
        "aic": float(aic) if not np.isnan(aic) else None,
        "n_params": int(n_params),
        "warnings": warnings
    }


def format_results_dataframe(
    results_data: Dict[str, List[float]],
    expl_vars: List[str]
) -> pd.DataFrame:
    """
    Format the results data into a properly indexed DataFrame.
    
    Parameters
    ----------
    results_data : Dict[str, List[float]]
        Dictionary containing result columns
    expl_vars : List[str]
        List of explanatory variable names
        
    Returns
    -------
    pd.DataFrame
        Formatted results DataFrame with variable names as index
    """
    results = pd.DataFrame(results_data)
    results['Var'] = expl_vars
    results = results.set_index("Var")
    return results
