"""
Cox Proportional Hazards mathematical logic.

This module contains the core mathematical computations for the Cox-PH model,
separated from the orchestration logic in central.py and partial.py.
"""

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.stats import chi2, norm

from vantage6.algorithm.tools.util import info, warn

from typing import Any, Dict, List, Tuple


def compute_derivatives(
    summed_agg1: np.ndarray,
    summed_agg2: np.ndarray,
    summed_agg3: np.ndarray,
    aggregated_time_events: pd.DataFrame,
    z_sum: pd.Series,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the primary and secondary derivatives for the Cox-PH model.

    These derivatives are used in the Newton-Raphson optimisation to find
    the maximum partial likelihood estimates of the regression coefficients.

    Parameters
    ----------
    summed_agg1 : np.ndarray
        Aggregated sum of exp(beta * X) across all nodes for each event time.
    summed_agg2 : np.ndarray
        Aggregated sum of X * exp(beta * X) across all nodes for each event time.
    summed_agg3 : np.ndarray
        Aggregated sum of outer products X * X^T * exp(beta * X) across all
        nodes for each event time.
    aggregated_time_events : pd.DataFrame
        DataFrame containing unique event times and their frequencies.
    z_sum : pd.Series
        Sum of explanatory variables for all event cases.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - primary_derivative: First derivative of the partial log-likelihood.
        - secondary_derivative: Second derivative (Hessian) of the partial
          log-likelihood.
    """
    info("Computing derivatives for Cox-PH model")

    n_covs = len(z_sum)
    tot_p1 = np.zeros(n_covs)
    tot_p2 = np.zeros((n_covs, n_covs))

    for index, row in aggregated_time_events.iterrows():
        freq = row.get("freq", 1)

        if index >= len(summed_agg1):
            continue

        s1_value = summed_agg1[index]
        s2_value = (
            summed_agg2[index]
            if isinstance(summed_agg2, np.ndarray)
            else np.array(summed_agg2[index])
        )
        s3_value = (
            summed_agg3[index]
            if isinstance(summed_agg3, np.ndarray)
            else np.array(summed_agg3[index])
        )

        if s1_value <= 0 or np.isnan(s1_value):
            warn(f"Invalid s1_value at index {index}: {s1_value}")
            continue

        # Primary derivative component
        s1 = freq * (s2_value / s1_value)

        # Secondary derivative component
        first_part = s3_value / s1_value

        if isinstance(s2_value, (list, np.ndarray)):
            s2_array = np.array(s2_value)
            numerator = np.outer(s2_array, s2_array)
        else:
            numerator = np.zeros((n_covs, n_covs))

        denominator = s1_value * s1_value
        second_part = (
            numerator / denominator if denominator > 0 else np.zeros((n_covs, n_covs))
        )

        s2 = freq * (first_part - second_part)

        tot_p1 += s1
        tot_p2 += s2

    primary_derivative = z_sum.values - tot_p1
    secondary_derivative = -tot_p2

    return primary_derivative, secondary_derivative


def compute_model_results(
    beta: np.ndarray,
    secondary_derivative: np.ndarray,
    z_sum: pd.Series,
    aggregated_time_events: pd.DataFrame,
    summed_agg1: np.ndarray,
    expl_vars: List[str],
) -> Dict[str, Any]:
    """
    Compute final model results after convergence.

    Parameters
    ----------
    beta : np.ndarray
        Final estimated regression coefficients.
    secondary_derivative : np.ndarray
        Final Hessian matrix (negative second derivative).
    z_sum : pd.Series
        Sum of explanatory variables for all event cases.
    aggregated_time_events : pd.DataFrame
        DataFrame containing unique event times and their frequencies.
    summed_agg1 : np.ndarray
        Final aggregated sum of exp(beta * X) for each event time.
    expl_vars : List[str]
        Names of the explanatory variables.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing model coefficients, statistics, and warnings.
    """
    info("Computing final model results")

    n_covs = len(beta)

    # Standard errors from the Fisher information matrix
    try:
        fisher = np.linalg.inv(-secondary_derivative)
        serrors = np.array([np.sqrt(fisher[k, k]) for k in range(fisher.shape[0])])
    except np.linalg.LinAlgError as e:
        warn(f"Could not invert Hessian matrix: {e}")
        fisher = np.zeros((n_covs, n_covs))
        serrors = np.array([np.nan] * n_covs)

    # Z-values and p-values
    with np.errstate(divide="ignore", invalid="ignore"):
        zvalues = np.zeros(n_covs)
        pvalues = np.ones(n_covs)
        for i in range(n_covs):
            if serrors[i] > 0 and not np.isnan(serrors[i]):
                zvalues[i] = (np.exp(beta[i]) - 1) / serrors[i]
                pvalues[i] = 2 * norm.cdf(-abs(zvalues[i]))
            else:
                zvalues[i] = np.nan
                pvalues[i] = np.nan

    # Overall model significance (Wald test)
    degrees_of_freedom = len(beta)
    try:
        wald_statistic = np.dot(beta, np.dot(-secondary_derivative, beta))
        overall_p_value = float(chi2.sf(wald_statistic, degrees_of_freedom))
    except Exception as e:
        warn(f"Could not compute Wald statistic: {e}")
        overall_p_value = None

    # AIC for model comparison
    try:
        linear_part = np.dot(z_sum.values, beta)

        risk_set_part = 0
        if hasattr(summed_agg1, "__len__") and len(summed_agg1) > 0:
            for i in range(len(aggregated_time_events)):
                if i < len(summed_agg1) and summed_agg1[i] > 0:
                    freq = aggregated_time_events.iloc[i].get("freq", 1)
                    if summed_agg1[i] <= 0:
                        warn(
                            f"Risk set sum is non-positive at time index {i}: "
                            f"{summed_agg1[i]}"
                        )
                        continue
                    risk_set_part += freq * np.log(summed_agg1[i])

        log_likelihood = linear_part - risk_set_part
        n_params = len(beta)

        if np.isnan(log_likelihood) or np.isinf(log_likelihood):
            raise ValueError(f"Invalid log-likelihood: {log_likelihood}")

        aic = float(-2 * log_likelihood + 2 * n_params)
    except (ValueError, IndexError, FloatingPointError) as e:
        warn(f"Could not compute AIC due to numerical/data issue: {e}")
        aic = None
    except Exception as e:
        warn(f"Unexpected error computing AIC: {e}")
        aic = None

    # Results data
    results_data = {
        "Coef": np.around(beta, 5),
        "Exp(coef)": np.around(np.exp(beta), 5),
        "SE": np.around(serrors, 5),
        "lower_CI": np.around(np.exp(beta - 1.96 * serrors), 5),
        "upper_CI": np.around(np.exp(beta + 1.96 * serrors), 5),
        "Z": zvalues,
        "p-value": pvalues,
    }

    # Perfect prediction warnings
    warnings = []
    threshold = 10
    for i, (coef, se) in enumerate(zip(beta, serrors)):
        if (
            abs(coef) > threshold
            or np.isinf(coef)
            or np.isnan(coef)
            or abs(se) > threshold
            or np.isinf(se)
            or np.isnan(se)
        ):
            var_name = expl_vars[i] if i < len(expl_vars) else f"var_{i}"
            msg = (
                f"Warning: Covariate '{var_name}' may perfectly predict the event "
                f"(coef={coef}, SE={se}). Results may be unreliable."
            )
            warn(msg)
            warnings.append(msg)

    return {
        "results_data": results_data,
        "fisher_info": fisher,
        "standard_errors": serrors,
        "zvalues": zvalues,
        "pvalues": pvalues,
        "overall_p_value": overall_p_value,
        "aic": aic,
        "n_params": int(len(beta)),
        "warnings": warnings,
    }


def format_results_dataframe(
    results_data: Dict[str, Any], expl_vars: List[str]
) -> pd.DataFrame:
    """
    Format the results data into a properly indexed DataFrame.

    Parameters
    ----------
    results_data : Dict[str, Any]
        Dictionary containing result columns.
    expl_vars : List[str]
        List of explanatory variable names.

    Returns
    -------
    pd.DataFrame
        Formatted results DataFrame with variable names as index.
    """
    results = pd.DataFrame(results_data)
    results["Var"] = expl_vars
    results = results.set_index("Var")
    return results


def update_beta(
    beta: np.ndarray,
    primary_derivative: np.ndarray,
    secondary_derivative: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Perform one Newton-Raphson update step.

    Parameters
    ----------
    beta : np.ndarray
        Current beta coefficients.
    primary_derivative : np.ndarray
        First derivative of the partial log-likelihood.
    secondary_derivative : np.ndarray
        Second derivative (Hessian) of the partial log-likelihood.

    Returns
    -------
    Tuple[np.ndarray, float]
        Updated beta coefficients and the delta (max absolute change).
    """
    beta_old = np.array(beta)
    beta = beta_old - solve(secondary_derivative, primary_derivative)
    delta = float(max(abs(beta - beta_old)))
    return beta, delta
