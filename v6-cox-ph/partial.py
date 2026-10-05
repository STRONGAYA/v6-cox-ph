"""
This file contains all partial algorithm functions, that are normally executed
on all nodes for which the algorithm is executed.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled). From there, they are sent to the partial task
or directly to the user (if they requested partial results).
"""

import pandas as pd
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.decorators import algorithm_client, data
from vantage6.algorithm.tools.exceptions import PrivacyThresholdViolation
from vantage6.algorithm.tools.util import info

from .miscellaneous import IterationResult, SummedZResult
from .privacy_guards import (
    guarded_risk_set_aggregates,
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
    # Sum of squares over the event cases: with z_sum this yields the pooled
    # covariate centre and spread over events without another round-trip.
    sum_squares = events[expl_vars].pow(2).sum().to_dict()

    # Also return per-time event counts from the same NaN-dropped,
    # censored DataFrame so that central can build aggregated_time_events
    # that are consistent with z_sum and the risk sets (FR-B4).
    times = events.groupby(time_col, as_index=False).count()
    times = times.sort_values(by=time_col)[[time_col, outcome_col]]
    times["freq"] = times[outcome_col]
    times = times.drop(columns=outcome_col)

    # The node's privacy settings (configuration, not data) let central
    # report which guards were active during the run.
    privacy_settings = {
        "sample_size_threshold": settings.sample_size_threshold,
        "min_risk_set_change": settings.min_risk_set_change,
        "time_bin_width": settings.time_bin_width,
    }
    # The result dict is built from the wire model, so its keys and nesting
    # are pinned in one place (SummedZResult).
    return SummedZResult(
        organization_id=client.organization_id,
        sum=z_sum,
        sum_squares=sum_squares,
        times=times.to_dict(),
        privacy_settings=privacy_settings,
    ).model_dump()


@data(1)
@algorithm_client
def perform_iteration(
    client: AlgorithmClient,
    df: pd.DataFrame,
    time_col: str,
    outcome_col: str,
    expl_vars: list,
    beta: list,
    centre: list,
    scale: list,
    unique_time_events: list,
) -> dict:
    """
    Perform an iteration of the algorithm, computing the necessary aggregates.

    The covariates are standardised on the node as ``(x - centre) / scale``
    before the risk-set aggregates are computed: the partial likelihood is
    invariant to this shared affine transform, while centring prevents ``exp``
    overflow from large offsets and scaling fixes the Hessian conditioning
    from large spreads. Central back-transforms the coefficients it reports.

    Parameters
    ----------
    client : AlgorithmClient
        The client instance used to interact with the vantage6 server.
    df : pd.DataFrame
        The DataFrame containing the data.
    time_col : str
        The name of the column containing the time data.
    outcome_col : str
        The name of the column containing the outcome data. Rows with a
        missing outcome are dropped so that tail censoring uses the same
        rows as ``compute_summed_z``.
    expl_vars : list
        A list of explanatory variables to be used in the computation.
    beta : list
        The current estimate of the beta coefficients (standardised space),
        as it arrives on the wire (a JSON list).
    centre : list
        The pooled covariate means over the event cases.
    scale : list
        The pooled covariate standard deviations over the event cases.
    unique_time_events : list
        A list of unique time events.

    Returns
    -------
    dict
        A dictionary containing the aggregates computed during the iteration.
    """
    info("Computing aggregates for the derivation of the partial likelihood")

    df, settings, threshold_met = prepare_node_data(client, df, time_col, outcome_col, expl_vars, need_outcome=True)

    # A node that passed compute_summed_z passes this check as well (the same
    # rows are analysed); the threshold here is defence in depth.
    if not threshold_met:
        raise PrivacyThresholdViolation("Sample size threshold not met: refusing to share aggregates.")

    beta_arr, centre_arr, scale_arr, unique_time_events = validate_iteration_input(
        beta, centre, scale, unique_time_events, expl_vars, settings
    )

    df = prepare_time_column(df, time_col, settings, outcome_col)

    X_all = (df[expl_vars].to_numpy(dtype=float) - centre_arr) / scale_arr

    agg1, agg2, agg3 = guarded_risk_set_aggregates(
        df[time_col], unique_time_events, settings.min_risk_set_change, X_all, beta_arr
    )

    # The result dict is built from the wire model, so its keys and nesting
    # are pinned in one place (IterationResult).
    return IterationResult(
        organization_id=client.organization_id,
        agg1=agg1.tolist(),
        agg2=agg2.tolist(),
        agg3=agg3.tolist(),
    ).model_dump()
