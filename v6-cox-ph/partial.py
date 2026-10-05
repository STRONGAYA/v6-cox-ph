"""
This file contains all partial algorithm functions, that are normally executed
on all nodes for which the algorithm is executed.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled). From there, they are sent to the partial task
or directly to the user (if they requested partial results).
"""

import numpy as np
import pandas as pd
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.decorators import algorithm_client, data
from vantage6.algorithm.tools.exceptions import PrivacyThresholdViolation
from vantage6.algorithm.tools.util import info

from .privacy_guards import (
    guarded_risk_set_masks,
    prepare_node_data,
    prepare_time_column,
    validate_iteration_input,
)


@data(1)
@algorithm_client
def compute_summed_z(
    client: AlgorithmClient,
    df: pd.DataFrame,
    time_col: str,
    outcome_col: str,
    expl_vars: list,
) -> dict:
    """
    Compute the sum of the specified explanatory variables for the outcome events.

    Parameters
    ----------
    client : AlgorithmClient
        The client instance used to interact with the vantage6 server.
    df : pd.DataFrame
        The DataFrame containing the data.
    time_col : str
        The name of the column containing the time data.
    outcome_col : str
        The name of the column containing the outcome data.
    expl_vars : list
        A list of explanatory variables to be used in the computation.

    Returns
    -------
    dict
        A dictionary containing the sum of the explanatory variables.
    """
    info("Computing summed Z statistics")

    df, settings, threshold_met = prepare_node_data(client, df, time_col, outcome_col, expl_vars, need_outcome=True)

    if not threshold_met:
        raise PrivacyThresholdViolation("Sample size threshold not met: refusing to share aggregates.")

    df = prepare_time_column(df, time_col, settings, outcome_col)

    events = df[df[outcome_col] == 1]
    z_sum = events[expl_vars].sum().to_dict()

    # Also return per-time event counts from the same NaN-dropped,
    # censored DataFrame so that central can build aggregated_time_events
    # that are consistent with z_sum and the risk sets (FR-B4).
    times = events.groupby(time_col, as_index=False).count()
    times = times.sort_values(by=time_col)[[time_col, outcome_col]]
    times["freq"] = times[outcome_col]
    times = times.drop(columns=outcome_col)
    return {"sum": z_sum, "times": times.to_dict()}


@data(1)
@algorithm_client
def perform_iteration(
    client: AlgorithmClient,
    df: pd.DataFrame,
    time_col: str,
    expl_vars: list,
    beta: np.ndarray,
    unique_time_events: list,
) -> dict:
    """
    Perform an iteration of the algorithm, computing the necessary aggregates.

    Parameters
    ----------
    client : AlgorithmClient
        The client instance used to interact with the vantage6 server.
    df : pd.DataFrame
        The DataFrame containing the data.
    time_col : str
        The name of the column containing the time data.
    expl_vars : list
        A list of explanatory variables to be used in the computation.
    beta : np.ndarray
        The current estimate of the beta coefficients.
    unique_time_events : list
        A list of unique time events.

    Returns
    -------
    dict
        A dictionary containing the aggregates computed during the iteration.
    """
    info("Computing aggregates for the derivation of the partial likelihood")

    df, settings, threshold_met = prepare_node_data(client, df, time_col, None, expl_vars, need_outcome=False)

    # perform_iteration does not have the outcome column available, so the
    # threshold is checked on rows only.
    if not threshold_met:
        raise PrivacyThresholdViolation("Sample size threshold not met: refusing to share aggregates.")

    beta, unique_time_events = validate_iteration_input(beta, unique_time_events, expl_vars, settings)

    df = prepare_time_column(df, time_col, settings)

    num_unique_time_events = len(unique_time_events)
    num_explanatory_vars = len(expl_vars)

    masks = guarded_risk_set_masks(df[time_col], unique_time_events, settings.min_risk_set_change)
    X_all = df[expl_vars].to_numpy(dtype=float)

    agg1: list = []
    agg2: list = []
    agg3: list = []

    for i in range(num_unique_time_events):
        mask = masks[i]
        n_in_set = int(mask.sum())
        if n_in_set == 0:
            agg1.append(0)
            agg2.append(pd.Series(np.zeros(num_explanatory_vars), index=expl_vars))
            agg3.append(np.zeros((num_explanatory_vars, num_explanatory_vars)))
        else:
            X = X_all[mask]
            ebz = np.exp(X @ beta)
            agg1.append(float(ebz.sum()))
            agg2.append(pd.Series((X * ebz[:, None]).sum(axis=0), index=expl_vars))
            agg3.append((X * ebz[:, None]).T @ X)

    agg2 = pd.DataFrame(agg2).to_dict()
    agg3 = [array.tolist() for array in agg3]

    return {"agg1": agg1, "agg2": agg2, "agg3": agg3}
