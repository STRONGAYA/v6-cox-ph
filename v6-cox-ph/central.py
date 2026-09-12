"""
This file contains all central algorithm functions. It is important to note
that the central method is executed on a node, just like any other method.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled).
"""

import math
from typing import Optional

import numpy as np
import pandas as pd
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.tools.exceptions import UserInputError
from vantage6_strongaya_general.miscellaneous import (
    collect_organisation_ids,
    safe_log,
)

from .coxph_logic import compute_derivatives, compute_model_results
from .coxph_logic import format_results_dataframe, update_beta
from .miscellaneous import validate_coxph_input


@algorithm_client
def central(
    client: AlgorithmClient,
    time_col: str,
    outcome_col: str,
    expl_vars: list,
    organization_ids: Optional[list] = None,
) -> dict:
    """
    Central function for the federated Cox Proportional Hazards algorithm.

    This function orchestrates the federated computation by:
    1. Validating input parameters
    2. Collecting organisation IDs
    3. Dispatching subtasks to compute unique event times
    4. Dispatching subtasks to compute summed Z statistics
    5. Iteratively optimising beta coefficients using Newton-Raphson
    6. Computing final model statistics and returning results

    Parameters
    ----------
    client : AlgorithmClient
        The vantage6 client instance.
    time_col : str
        Name of the column containing time data.
    outcome_col : str
        Name of the column containing outcome/event data (1=event, 0=censored).
    expl_vars : list
        List of explanatory variable names to include in the model.
    organization_ids : list, optional
        List of organisation IDs to include. If None, all organisations are used.

    Returns
    -------
    dict
        Dictionary containing model results, p-values, AIC, and warnings.
    """
    # Validate input parameters
    try:
        validated = validate_coxph_input(
            time_col=time_col,
            outcome_col=outcome_col,
            expl_vars=expl_vars,
            organization_ids=organization_ids,
        )
    except UserInputError:
        raise

    time_col = validated.time_col
    outcome_col = validated.outcome_col
    expl_vars = validated.expl_vars
    organization_ids = validated.organization_ids

    # STRONG AYA: collect organisation IDs
    ids = collect_organisation_ids(organization_ids, client)

    excluded_ids = []
    safe_log("info", f"Sending task to organisations {ids}")

    n_covs = len(expl_vars)
    epochs = 10

    # Subtask: get unique event times
    safe_log("info", "Defining input parameters for subtask — get unique event times")
    input_ = {
        "method": "get_unique_event_times",
        "kwargs": {
            "time_col": time_col,
            "outcome_col": outcome_col,
        },
    }

    n_loops = 0
    n_threshold_met = False
    while not n_threshold_met:
        _excluded_ids = []
        if n_loops > 2:
            safe_log(
                "error",
                "Sample size violations should be eliminated yet criteria "
                "are not met. Exiting",
            )
            raise ValueError(
                "Sample size violations should be eliminated yet criteria "
                "are not met. Exiting"
            )

        n_loops += 1
        safe_log("info", "Creating subtask for all selected organisations")
        task = client.task.create(
            input_=input_,
            organizations=ids,
            name="Unique event times",
            description="Getting unique event times and their counts",
        )

        safe_log("info", "Waiting for results")
        results = client.wait_for_results(task_id=task.get("id"))
        safe_log("info", "Results obtained!")

        unique_time_events = []
        for output in results:
            if "N-Threshold not met" in output:
                safe_log(
                    "warning",
                    f"Insufficient samples for organisation "
                    f"{output['N-Threshold not met']}. Excluding from analysis.",
                )
                ids.remove(output["N-Threshold not met"])
                excluded_ids.append(output["N-Threshold not met"])
                _excluded_ids.append(output["N-Threshold not met"])
                continue

            output = pd.DataFrame.from_dict(output["times"])
            unique_time_events.append(output)

        if len(_excluded_ids) == 0:
            n_threshold_met = True
        elif len(ids) == 0:
            safe_log(
                "warning", "No organisations meet the minimal sample size threshold."
            )
            return {"excluded_organizations": excluded_ids, "table": np.nan}

    aggregated_time_events = pd.concat(unique_time_events)
    aggregated_time_events = aggregated_time_events.groupby(
        time_col, as_index=False
    ).sum()

    unique_time_events = aggregated_time_events[time_col].tolist()

    # Subtask: compute summed Z
    safe_log("info", "Defining input parameters for subtask — compute summed Z")
    input_ = {
        "method": "compute_summed_z",
        "kwargs": {
            "outcome_col": outcome_col,
            "expl_vars": expl_vars,
        },
    }

    safe_log("info", "Creating subtask for all organisations")
    task = client.task.create(
        input_=input_,
        organizations=ids,
        name="Summed Z statistic",
        description="Computing the summed Z statistic",
    )

    safe_log("info", "Waiting for results")
    results = client.wait_for_results(task_id=task.get("id"))
    safe_log("info", "Results obtained!")

    z_sum = None
    for output in results:
        if z_sum is None:
            z_sum = pd.Series(output["sum"])
        else:
            z_sum += pd.Series(output["sum"])

    beta: np.ndarray = np.zeros(n_covs)

    for epoch in range(epochs):
        # Serialise beta for vantage6
        beta_serialised: list = beta.tolist()

        safe_log("info", "Defining input parameters for subtask — perform iteration")
        input_ = {
            "method": "perform_iteration",
            "kwargs": {
                "time_col": time_col,
                "expl_vars": expl_vars,
                "beta": beta_serialised,
                "unique_time_events": unique_time_events,
            },
        }

        # Deserialise beta
        beta = np.array(beta_serialised)

        safe_log("info", "Creating subtask for all organisations")
        task = client.task.create(
            input_=input_,
            organizations=ids,
            name="Start iteration",
            description="Iterating to find the optimal beta",
        )

        safe_log("info", "Waiting for results")
        results = client.wait_for_results(task_id=task.get("id"))
        safe_log("info", "Results obtained!")

        n_times = len(unique_time_events)
        summed_agg1: np.ndarray = np.zeros(n_times)
        summed_agg2: np.ndarray = np.zeros((n_times, n_covs))
        summed_agg3: np.ndarray = np.zeros((n_times, n_covs, n_covs))

        for output in results:
            summed_agg1 += np.array(output["agg1"])
            summed_agg2 += np.array(pd.DataFrame.from_dict(output["agg2"]))
            summed_agg3 += np.array([np.array(lst) for lst in output["agg3"]])

        primary_derivative, secondary_derivative = compute_derivatives(
            summed_agg1,
            summed_agg2,
            summed_agg3,
            aggregated_time_events,
            z_sum,
        )

        beta, delta = update_beta(beta, primary_derivative, secondary_derivative)

        if math.isnan(delta):
            safe_log("warning", "Delta has turned into a NaN")
            break

        if delta <= 0.000001:
            safe_log("info", "Betas have settled! Finished iterating!")
            break

    # Compute final model results
    model = compute_model_results(
        beta=beta,
        secondary_derivative=secondary_derivative,
        z_sum=z_sum,
        aggregated_time_events=aggregated_time_events,
        summed_agg1=summed_agg1,
        expl_vars=expl_vars,
    )

    results_df = format_results_dataframe(model["results_data"], expl_vars)

    return {
        "included_organizations": ids,
        "excluded_organizations": excluded_ids,
        "model": results_df.to_json(),
        "overall_p_value": model["overall_p_value"],
        "aic": model["aic"],
        "degrees_of_freedom": model["n_params"],
        "warnings": model["warnings"],
    }
