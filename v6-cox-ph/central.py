"""
This file contains all central algorithm functions. It is important to note
that the central method is executed on a node, just like any other method.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled).
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.linalg import solve
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.tools.exceptions import AlgorithmError, UserInputError
from vantage6.algorithm.tools.util import error, info, warn

from .coxph_logic import compute_derivatives, compute_model_results
from .coxph_logic import format_results_dataframe
from .miscellaneous import validate_coxph_input

# Maximum Newton-Raphson iterations. Module-level so tests can monkeypatch
# it to force non-convergence.
MAX_ITERATIONS = 10


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

    # Collect all organisations unless specified
    ids: list
    if organization_ids is None:
        organisations = client.organization.list()
        ids = [organisation.get("id") for organisation in organisations]
    else:
        ids = list(organization_ids)

    excluded_ids = []
    info(f"Sending task to organisations {ids}")

    n_covs = len(expl_vars)
    epochs = MAX_ITERATIONS

    # Subtask: get unique event times
    info("Defining input parameters for subtask — get unique event times")
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
            error("Sample size violations should be eliminated yet criteria " "are not met. Exiting")
            raise ValueError("Sample size violations should be eliminated yet criteria " "are not met. Exiting")

        n_loops += 1
        info("Creating subtask for all selected organisations")
        task = client.task.create(
            input_=input_,
            organizations=ids,
            name="Unique event times",
            description="Getting unique event times and their counts",
        )

        info("Waiting for results")
        results = client.wait_for_results(task_id=task.get("id"))
        info("Results obtained!")

        unique_time_events = []
        for output in results:
            if "N-Threshold not met" in output:
                warn(
                    f"Insufficient samples for organisation "
                    f"{output['N-Threshold not met']}. Excluding from analysis."
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
            warn("No organisations meet the minimal sample size threshold.")
            return {
                "included_organizations": [],
                "excluded_organizations": excluded_ids,
                "model": None,
                "overall_p_value": None,
                "aic": None,
                "degrees_of_freedom": n_covs,
                "warnings": ["No organisations meet the minimal sample size threshold."],
                "converged": False,
                "n_iterations": 0,
            }

    # Subtask: compute summed Z
    info("Defining input parameters for subtask — compute summed Z")
    input_ = {
        "method": "compute_summed_z",
        "kwargs": {
            "time_col": time_col,
            "outcome_col": outcome_col,
            "expl_vars": expl_vars,
        },
    }

    info("Creating subtask for all organisations")
    task = client.task.create(
        input_=input_,
        organizations=ids,
        name="Summed Z statistic",
        description="Computing the summed Z statistic",
    )

    info("Waiting for results")
    results = client.wait_for_results(task_id=task.get("id"))
    info("Results obtained!")

    z_sum = None
    time_event_dfs = []
    for i, output in enumerate(results):
        _validate_zsum_result(output, expl_vars, time_col, org_id=ids[i] if i < len(ids) else i)
        if z_sum is None:
            z_sum = pd.Series(output["sum"])
        else:
            z_sum += pd.Series(output["sum"])
        # Collect per-time event counts (NaN-consistent with z_sum)
        time_event_dfs.append(pd.DataFrame.from_dict(output["times"]))

    # Build aggregated_time_events from compute_summed_z results, not from
    # get_unique_event_times. The latter drops NaN only in time/outcome
    # columns, while compute_summed_z also drops NaN in expl_vars — so
    # event counts from compute_summed_z are consistent with z_sum and the
    # risk sets (FR-B4).
    aggregated_time_events = pd.concat(time_event_dfs)
    aggregated_time_events = aggregated_time_events.groupby(time_col, as_index=False).sum()

    unique_time_events = aggregated_time_events[time_col].tolist()

    beta: np.ndarray = np.zeros(n_covs)

    # Variables that must stay consistent with the reported beta: the
    # secondary derivative (Hessian) and summed_agg1 are evaluated at the
    # same beta that is ultimately reported. By testing convergence *before*
    # applying the Newton step we keep all three in sync without an extra
    # round-trip.
    secondary_derivative: np.ndarray = np.zeros((n_covs, n_covs))
    summed_agg1: np.ndarray = np.zeros(0)
    converged = False
    convergence_cause: str | None = None
    epoch = 0

    for epoch in range(epochs):
        # Serialise beta for vantage6
        beta_serialised: list = beta.tolist()

        info("Defining input parameters for subtask — perform iteration")
        input_ = {
            "method": "perform_iteration",
            "kwargs": {
                "time_col": time_col,
                "expl_vars": expl_vars,
                "beta": beta_serialised,
                "unique_time_events": unique_time_events,
            },
        }

        info("Creating subtask for all organisations")
        task = client.task.create(
            input_=input_,
            organizations=ids,
            name="Start iteration",
            description="Iterating to find the optimal beta",
        )

        info("Waiting for results")
        results = client.wait_for_results(task_id=task.get("id"))
        info("Results obtained!")

        n_times = len(unique_time_events)
        summed_agg1 = np.zeros(n_times)
        summed_agg2 = np.zeros((n_times, n_covs))
        summed_agg3 = np.zeros((n_times, n_covs, n_covs))

        for i, output in enumerate(results):
            org_id = ids[i] if i < len(ids) else i
            _validate_iteration_result(output, n_times, n_covs, expl_vars, org_id)
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

        try:
            step = solve(secondary_derivative, primary_derivative)
        except np.linalg.LinAlgError as e:
            convergence_cause = f"Hessian is singular: {e}"
            warn(convergence_cause)
            break

        delta = float(np.max(np.abs(step)))

        if not np.isfinite(delta):
            convergence_cause = "Newton step is not finite"
            warn(convergence_cause)
            break

        if delta <= 0.000001:
            info("Betas have settled! Finished iterating!")
            converged = True
            break

        if epoch == epochs - 1:
            # FR-A1: do NOT apply the last step so the reported beta,
            # Hessian and summed_agg1 stay consistent.
            break

        beta = beta - step

    n_iterations = epoch + 1
    central_warnings = []
    if not converged:
        msg = (
            f"Newton-Raphson did not converge in {n_iterations} iterations; "
            f"SE/p-values may be unreliable; statistics are reported at the "
            f"last evaluated beta"
        )
        if convergence_cause:
            msg += f"; cause: {convergence_cause}"
        warn(msg)
        central_warnings.append(msg)

    # Compute final model results — beta, secondary_derivative and
    # summed_agg1 are all evaluated at the reported beta.
    model = compute_model_results(
        beta=beta,
        secondary_derivative=secondary_derivative,
        z_sum=z_sum,
        aggregated_time_events=aggregated_time_events,
        summed_agg1=summed_agg1,
        expl_vars=expl_vars,
        converged=converged,
        n_iterations=n_iterations,
    )

    results_df = format_results_dataframe(model["results_data"], expl_vars)

    return {
        "included_organizations": ids,
        "excluded_organizations": excluded_ids,
        "model": results_df.to_json(double_precision=15),
        "overall_p_value": model["overall_p_value"],
        "aic": model["aic"],
        "degrees_of_freedom": model["n_params"],
        "warnings": model["warnings"] + central_warnings,
        "converged": converged,
        "n_iterations": n_iterations,
    }


def _validate_iteration_result(output: dict, n_times: int, n_covs: int, expl_vars: list, org_id: int) -> None:
    """Validate a ``perform_iteration`` sub-task result (FR-A3).

    Raises ``AlgorithmError`` naming the organisation on any structural or
    numerical problem.
    """
    if not isinstance(output, dict):
        raise AlgorithmError(
            f"Organisation {org_id}: perform_iteration returned " f"{type(output).__name__}, expected a dict"
        )
    for key in ("agg1", "agg2", "agg3"):
        if key not in output:
            raise AlgorithmError(f"Organisation {org_id}: perform_iteration result missing " f"key '{key}'")
    agg1 = np.asarray(output["agg1"], dtype=float)
    if agg1.ndim != 1 or len(agg1) != n_times:
        raise AlgorithmError(f"Organisation {org_id}: agg1 has length {len(agg1)}, " f"expected {n_times}")
    if not np.all(np.isfinite(agg1)):
        raise AlgorithmError(f"Organisation {org_id}: agg1 contains non-finite values")
    agg2_df = pd.DataFrame.from_dict(output["agg2"])
    if list(agg2_df.columns) != list(expl_vars):
        raise AlgorithmError(
            f"Organisation {org_id}: agg2 columns {list(agg2_df.columns)} " f"do not match expl_vars {list(expl_vars)}"
        )
    if agg2_df.shape[0] != n_times:
        raise AlgorithmError(f"Organisation {org_id}: agg2 has {agg2_df.shape[0]} rows, " f"expected {n_times}")
    agg3 = np.array([np.array(lst) for lst in output["agg3"]])
    if agg3.shape != (n_times, n_covs, n_covs):
        raise AlgorithmError(
            f"Organisation {org_id}: agg3 has shape {agg3.shape}, " f"expected ({n_times}, {n_covs}, {n_covs})"
        )


def _validate_zsum_result(output: dict, expl_vars: list, time_col: str, org_id: int | None = None) -> None:
    """Validate a ``compute_summed_z`` sub-task result (FR-A3).

    Raises ``AlgorithmError`` if the result is not a dict with a ``sum`` key
    whose entries match ``expl_vars``, or a ``times`` key with the per-time
    event counts (columns ``time_col`` and ``freq``).
    """
    if not isinstance(output, dict) or "sum" not in output:
        raise AlgorithmError(f"Organisation {org_id}: compute_summed_z result missing 'sum' key")
    sum_dict = output["sum"]
    if not isinstance(sum_dict, dict):
        raise AlgorithmError(f"Organisation {org_id}: compute_summed_z 'sum' is not a dict")
    missing = [v for v in expl_vars if v not in sum_dict]
    if missing:
        raise AlgorithmError(f"Organisation {org_id}: compute_summed_z 'sum' missing " f"variables {missing}")
    if "times" not in output:
        raise AlgorithmError(f"Organisation {org_id}: compute_summed_z result missing 'times' key")
    times_df = pd.DataFrame.from_dict(output["times"])
    if time_col not in times_df.columns or "freq" not in times_df.columns:
        raise AlgorithmError(
            f"Organisation {org_id}: compute_summed_z 'times' must have columns "
            f"'{time_col}' and 'freq', got {list(times_df.columns)}"
        )
    if len(times_df) > 0:
        freqs = times_df["freq"].to_numpy()
        if not np.all(np.isfinite(freqs)) or np.any(freqs < 0):
            raise AlgorithmError(
                f"Organisation {org_id}: compute_summed_z 'freq' contains " f"non-finite or negative values"
            )
