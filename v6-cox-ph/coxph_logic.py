"""
Cox Proportional Hazards mathematical logic.

This module contains the core mathematical computations for the Cox-PH model,
separated from the orchestration logic in central.py and partial.py.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from vantage6_strongaya_general.miscellaneous import safe_log

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
    safe_log("info", "Computing derivatives for Cox-PH model")

    n_covs = len(z_sum)
    n_times = len(aggregated_time_events)
    tot_p1 = np.zeros(n_covs)
    tot_p2 = np.zeros((n_covs, n_covs))

    freqs = aggregated_time_events["freq"].to_numpy()

    if len(summed_agg1) != n_times:
        raise ValueError(
            f"Length mismatch: aggregated_time_events has {n_times} rows but "
            f"summed_agg1 has {len(summed_agg1)} entries"
        )

    for pos in range(n_times):
        freq = freqs[pos]

        s1_value = summed_agg1[pos]
        s2_value = np.asarray(summed_agg2[pos])
        s3_value = np.asarray(summed_agg3[pos])

        if s1_value <= 0 or np.isnan(s1_value):
            safe_log("warning", f"Invalid s1_value at position {pos}: {s1_value}")
            continue

        # Primary derivative component
        s1 = freq * (s2_value / s1_value)

        # Secondary derivative component
        first_part = s3_value / s1_value
        numerator = np.outer(s2_value, s2_value)
        denominator = s1_value * s1_value
        second_part = numerator / denominator

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
    converged: bool = True,
    n_iterations: int | None = None,
) -> Dict[str, Any]:
    """
    Compute final model results after convergence.

    Parameters
    ----------
    beta : np.ndarray
        Final estimated regression coefficients.
    secondary_derivative : np.ndarray
        Final Hessian matrix (negative second derivative), evaluated at the
        same ``beta`` that is reported.
    z_sum : pd.Series
        Sum of explanatory variables for all event cases.
    aggregated_time_events : pd.DataFrame
        DataFrame containing unique event times and their frequencies.
    summed_agg1 : np.ndarray
        Final aggregated sum of exp(beta * X) for each event time, evaluated
        at the same ``beta`` that is reported.
    expl_vars : List[str]
        Names of the explanatory variables.
    converged : bool
        Whether the Newton-Raphson optimiser converged. When ``False`` a
        warning is appended so users know SE/p-values may be unreliable.
    n_iterations : int | None
        Number of Newton-Raphson iterations performed.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing model coefficients, statistics, and warnings.
    """
    safe_log("info", "Computing final model results")

    n_covs = len(beta)

    # Standard errors from the covariance matrix (inverse of the observed
    # Fisher information, i.e. the inverse of the negative Hessian).
    try:
        covariance = np.linalg.inv(-secondary_derivative)
        serrors = np.array(
            [np.sqrt(covariance[k, k]) for k in range(covariance.shape[0])]
        )
    except np.linalg.LinAlgError as e:
        safe_log("warning", f"Could not invert Hessian matrix: {e}")
        covariance = np.zeros((n_covs, n_covs))
        serrors = np.array([np.nan] * n_covs)

    # Z-values and p-values (Wald test: Z = beta / SE)
    with np.errstate(divide="ignore", invalid="ignore"):
        zvalues = np.zeros(n_covs)
        pvalues = np.ones(n_covs)
        for i in range(n_covs):
            if serrors[i] > 0 and not np.isnan(serrors[i]):
                zvalues[i] = beta[i] / serrors[i]
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
        safe_log("warning", f"Could not compute Wald statistic: {e}")
        overall_p_value = None

    # AIC for model comparison
    try:
        linear_part = np.dot(z_sum.values, beta)

        risk_set_part = 0
        freqs = aggregated_time_events["freq"].to_numpy()
        if hasattr(summed_agg1, "__len__") and len(summed_agg1) > 0:
            for i in range(len(freqs)):
                if i < len(summed_agg1) and summed_agg1[i] > 0:
                    risk_set_part += freqs[i] * np.log(summed_agg1[i])

        log_likelihood = linear_part - risk_set_part
        n_params = len(beta)

        if np.isnan(log_likelihood) or np.isinf(log_likelihood):
            raise ValueError(f"Invalid log-likelihood: {log_likelihood}")

        aic = float(-2 * log_likelihood + 2 * n_params)
    except (ValueError, IndexError, FloatingPointError) as e:
        safe_log("warning", f"Could not compute AIC due to numerical/data issue: {e}")
        aic = None
    except Exception as e:
        safe_log("warning", f"Unexpected error computing AIC: {e}")
        aic = None

    # Results data — full precision (FR-A4); rounding is a client concern.
    results_data = {
        "Coef": beta,
        "Exp(coef)": np.exp(beta),
        "SE": serrors,
        "lower_CI": np.exp(beta - 1.96 * serrors),
        "upper_CI": np.exp(beta + 1.96 * serrors),
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
            safe_log("warning", msg)
            warnings.append(msg)

    if not converged:
        n_iter_str = str(n_iterations) if n_iterations is not None else "unknown"
        msg = (
            f"Newton-Raphson did not converge in {n_iter_str} iterations; "
            f"SE/p-values may be unreliable; statistics are reported at the "
            f"last evaluated beta"
        )
        warnings.append(msg)

    return {
        "results_data": results_data,
        "covariance": covariance,
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
