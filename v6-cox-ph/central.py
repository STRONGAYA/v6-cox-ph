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
from scipy.stats import chi2
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.tools.exceptions import AlgorithmError
from vantage6.algorithm.tools.util import info, warn

from .coxph_logic import back_transform_results, compute_derivatives, compute_model_results
from .coxph_logic import partial_log_likelihood, pooled_standardisation
from .coxph_logic import survival_curves as survival_curves_from_aggregates
from .coxph_logic import format_results_dataframe
from .miscellaneous import validate_coxph_input

# Maximum Newton-Raphson iterations. Module-level so tests can monkeypatch
# it to force non-convergence.
MAX_ITERATIONS = 20


@algorithm_client
def central(
    client: AlgorithmClient,
    time_col: str,
    outcome_col: str,
    expl_vars: list,
    organization_ids: Optional[list] = None,
    covariate_profiles: Optional[list] = None,
) -> dict:
    """
    Central function for the federated Cox Proportional Hazards algorithm.

    This function orchestrates the federated computation by:
    1. Validating input parameters
    2. Collecting organisation IDs
    3. Dispatching subtasks to compute summed Z statistics (and with them the
       event-time grid)
    4. Iteratively optimising beta coefficients using Newton-Raphson
    5. Computing final model statistics and returning results

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
        Organisations that do not meet the sample-size threshold fail the
        whole run with ``PrivacyThresholdViolation`` — select them explicitly
        here rather than relying on automatic exclusion.

    Returns
    -------
    dict
        Dictionary containing model results, p-values, AIC, and warnings.
    """
    # Validate input parameters
    validated = validate_coxph_input(
        time_col=time_col,
        outcome_col=outcome_col,
        expl_vars=expl_vars,
        organization_ids=organization_ids,
        covariate_profiles=covariate_profiles,
    )

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

    info(f"Sending task to organisations {ids}")

    n_covs = len(expl_vars)
    epochs = MAX_ITERATIONS

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
    _require_all_organisations_answered(results, ids)

    z_sum = None
    sum_squares = None
    time_event_dfs = []
    node_privacy_settings = []
    for output in results:
        org_id = _result_org_id(output, ids)
        _validate_zsum_result(output, expl_vars, time_col, org_id=org_id)
        if z_sum is None:
            z_sum = pd.Series(output["sum"])
            sum_squares = pd.Series(output["sum_squares"])
        else:
            z_sum += pd.Series(output["sum"])
            sum_squares += pd.Series(output["sum_squares"])
        # Collect per-time event counts (NaN-consistent with z_sum)
        time_event_dfs.append(pd.DataFrame.from_dict(output["times"]))
        node_privacy_settings.append(output["privacy_settings"])

    # Build aggregated_time_events from compute_summed_z results: it drops
    # NaN in time, outcome and expl_vars, so its event counts are consistent
    # with z_sum and the risk sets (FR-B4).
    aggregated_time_events = pd.concat(time_event_dfs)
    aggregated_time_events = aggregated_time_events.groupby(time_col, as_index=False).sum()

    unique_time_events = aggregated_time_events[time_col].tolist()

    # Standardisation: the pooled covariate centre and spread over the event
    # cases, rounded to two significant figures so the nodes learn less. The
    # partial likelihood is invariant to this shared affine transform; the
    # nodes' aggregates are computed on (x - centre) / scale and the reported
    # coefficients are transformed back.
    assert z_sum is not None and sum_squares is not None
    n_events = float(aggregated_time_events["freq"].sum())
    centre, scale = pooled_standardisation(z_sum.to_numpy(dtype=float), sum_squares.to_numpy(dtype=float), n_events)
    info("Dispatching standardised covariates (centre and scale to 2 significant figures).")

    # Aggregate summary of the guards that were active on the nodes (D3):
    # configuration, not data; no per-node breakdown. A warning is added
    # when any guard was active, because the estimates are then approximate.
    max_min_risk_set_change = max(int(s.get("min_risk_set_change") or 1) for s in node_privacy_settings)
    time_binning = any(s.get("time_bin_width") for s in node_privacy_settings)
    guards_active = max_min_risk_set_change > 1 or time_binning
    privacy_guards = {
        "active": guards_active,
        "max_min_risk_set_change": max_min_risk_set_change,
        "time_binning": time_binning,
    }
    guards_warning = None
    if guards_active:
        guards_warning = (
            "Privacy guards were active on one or more nodes "
            f"(maximum minimum risk-set change k={max_min_risk_set_change}, "
            f"time binning {'on' if time_binning else 'off'}); "
            "the estimates are approximate."
        )
        warn(guards_warning)

    # The gradient in the standardised space needs the event-case covariate
    # sums in that space as well.
    z_sum_star = pd.Series((z_sum.to_numpy(dtype=float) - centre * n_events) / scale, index=z_sum.index)

    beta: np.ndarray = np.zeros(n_covs)

    # The reported beta, Hessian and summed_agg1 are always evaluated at the
    # same beta: the last *accepted* evaluation. A decrease of the
    # log-likelihood is only detectable at the next round-trip, so the
    # previous accepted state is kept and step-halving backs off toward it.
    accepted: dict | None = None  # beta / secondary / summed_agg1 / log_likelihood
    log_likelihood_null: float | None = None
    step_halvings = 0
    last_step: np.ndarray | None = None
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
                "outcome_col": outcome_col,
                "expl_vars": expl_vars,
                "beta": beta_serialised,
                "centre": centre.tolist(),
                "scale": scale.tolist(),
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
        _require_all_organisations_answered(results, ids)

        n_times = len(unique_time_events)
        summed_agg1 = np.zeros(n_times)
        summed_agg2 = np.zeros((n_times, n_covs))
        summed_agg3 = np.zeros((n_times, n_covs, n_covs))

        for output in results:
            org_id = _result_org_id(output, ids)
            _validate_iteration_result(output, n_times, n_covs, expl_vars, org_id)
            summed_agg1 += np.array(output["agg1"])
            summed_agg2 += np.asarray(output["agg2"], dtype=float)
            summed_agg3 += np.array(output["agg3"])

        # The log-likelihood at the current beta, from aggregates that are
        # already collected (no extra round-trip).
        ll_here = partial_log_likelihood(beta, z_sum_star, aggregated_time_events, summed_agg1)
        if log_likelihood_null is None and not beta.any():
            log_likelihood_null = ll_here

        if accepted is not None and ll_here < accepted["log_likelihood"] - 1e-12:
            # The step away from the accepted beta decreased the likelihood:
            # halve it and retry from the accepted point. The accepted state
            # is reported if the budget runs out mid-halving.
            step_halvings += 1
            warn("Newton step decreased the log-likelihood; halving the step.")
            last_step = np.asarray(last_step, dtype=float) * 0.5
            beta = accepted["beta"] - last_step
            continue

        primary_derivative, secondary_derivative = compute_derivatives(
            summed_agg1,
            summed_agg2,
            summed_agg3,
            aggregated_time_events,
            z_sum_star,
        )

        # Accept this evaluation: the reported beta, Hessian and summed_agg1
        # stay evaluated at the same beta.
        accepted = {
            "beta": beta,
            "secondary": secondary_derivative,
            "summed_agg1": summed_agg1,
            "log_likelihood": ll_here,
        }

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

        last_step = step
        beta = beta - step

    n_iterations = epoch + 1
    if guards_warning is not None:
        central_warnings = [guards_warning]
    else:
        central_warnings = []
    if accepted is None:
        raise AlgorithmError("The optimiser never accepted an evaluation; no model can be reported.")
    beta = accepted["beta"]
    secondary_derivative = accepted["secondary"]
    summed_agg1 = accepted["summed_agg1"]
    log_likelihood = accepted["log_likelihood"]
    if log_likelihood_null is None:
        log_likelihood_null = log_likelihood
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
    # summed_agg1 are all evaluated at the reported beta (in the
    # standardised space). Back-transform the reported statistics to the
    # original covariates afterwards.
    model = compute_model_results(
        beta=beta,
        secondary_derivative=secondary_derivative,
        z_sum=z_sum_star,
        aggregated_time_events=aggregated_time_events,
        summed_agg1=summed_agg1,
        expl_vars=expl_vars,
        converged=converged,
        n_iterations=n_iterations,
    )
    model = back_transform_results(model, scale)

    results_df = format_results_dataframe(model["results_data"], expl_vars)

    # Baseline cumulative hazard and survival curves, computed centrally from
    # quantities that are already aggregated (see coxph_logic): no new
    # partial, no new data leaves a node.
    baseline_cumulative_hazard, survival_curves = survival_curves_from_aggregates(
        aggregated_time_events["freq"].to_numpy(dtype=float),
        summed_agg1,
        aggregated_time_events[time_col].to_numpy(dtype=float),
        beta,
        centre,
        scale,
        expl_vars,
        validated.covariate_profiles,
    )

    lr_statistic = 2 * (log_likelihood - log_likelihood_null)
    return {
        "model": results_df.to_dict(orient="index"),
        "overall_p_value": model["overall_p_value"],
        "aic": model["aic"],
        "degrees_of_freedom": model["n_params"],
        "warnings": model["warnings"] + central_warnings,
        "converged": converged,
        "n_iterations": n_iterations,
        "log_likelihood": log_likelihood,
        "log_likelihood_null": log_likelihood_null,
        "lr_statistic": lr_statistic,
        "lr_p_value": float(chi2.sf(lr_statistic, n_covs)),
        "n_events": int(aggregated_time_events["freq"].sum()),
        "covariance": np.asarray(model["covariance"]).tolist(),
        "algorithm_version": _algorithm_version(),
        "baseline_cumulative_hazard": baseline_cumulative_hazard,
        "survival_curves": survival_curves,
        "privacy_guards": privacy_guards,
    }


def _algorithm_version() -> str:
    """The installed algorithm package version, for result traceability."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("v6-cox-ph")
    except PackageNotFoundError:
        return "unknown"


def _result_org_id(output: dict, expected_ids: list) -> int:
    """Return the organisation a partial result says it came from.

    The order of ``wait_for_results`` is not the dispatch order (the
    organisation id list is deduplicated and the server orders freely), so
    results carry their own ``organization_id`` and errors are attributed
    by it. A result without a known id raises ``AlgorithmError``.

    Raises
    ------
    AlgorithmError
        If the result is not a dict or its ``organization_id`` is not one
        of the dispatched organisations.
    """
    if not isinstance(output, dict):
        raise AlgorithmError(f"Expected a dict result, got {type(output).__name__}")
    org_id = output.get("organization_id")
    if org_id is None or org_id not in expected_ids:
        raise AlgorithmError(
            f"Result carries organization_id {org_id!r}, which is not one of " f"the dispatched organisations."
        )
    return int(org_id)


def _require_all_organisations_answered(results, ids: list) -> None:
    """Fail closed unless every dispatched organisation returned a result.

    ``wait_for_results`` returns the decoded results of the runs that have
    one; a node whose run failed (for example
    ``PrivacyThresholdViolation``) is simply absent from the list.
    Aggregating the survivors would silently compute a model over a subset
    of the collaboration — the run must fail instead (D4). The message
    names the counts, never the data.

    Raises
    ------
    AlgorithmError
        When fewer results came back than organisations were dispatched.
    """
    valid = [r for r in results if isinstance(r, dict)]
    if len(valid) != len(ids):
        raise AlgorithmError(
            f"Only {len(valid)} of {len(ids)} organisations returned a result; "
            f"refusing to aggregate a subset. The run fails closed."
        )


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
    agg2 = np.asarray(output["agg2"], dtype=float)
    if agg2.shape != (n_times, n_covs):
        raise AlgorithmError(f"Organisation {org_id}: agg2 has shape {agg2.shape}, " f"expected ({n_times}, {n_covs})")
    if not np.all(np.isfinite(agg2)):
        raise AlgorithmError(f"Organisation {org_id}: agg2 contains non-finite values")
    agg3 = np.array([np.array(lst) for lst in output["agg3"]])
    if agg3.shape != (n_times, n_covs, n_covs):
        raise AlgorithmError(
            f"Organisation {org_id}: agg3 has shape {agg3.shape}, " f"expected ({n_times}, {n_covs}, {n_covs})"
        )


def _validate_zsum_result(output: dict, expl_vars: list, time_col: str, org_id: int | None = None) -> None:
    """Validate a ``compute_summed_z`` sub-task result (FR-A3).

    Raises ``AlgorithmError`` if the result is not a dict with a ``sum`` key
    whose entries match ``expl_vars``, a ``sum_squares`` key with the same
    entries, a ``times`` key with the per-time event counts (columns
    ``time_col`` and ``freq``) and a ``privacy_settings`` key.
    """
    if not isinstance(output, dict) or "sum" not in output:
        raise AlgorithmError(f"Organisation {org_id}: compute_summed_z result missing 'sum' key")
    sum_dict = output["sum"]
    if not isinstance(sum_dict, dict):
        raise AlgorithmError(f"Organisation {org_id}: compute_summed_z 'sum' is not a dict")
    missing = [v for v in expl_vars if v not in sum_dict]
    if missing:
        raise AlgorithmError(f"Organisation {org_id}: compute_summed_z 'sum' missing " f"variables {missing}")
    if "sum_squares" not in output:
        raise AlgorithmError(f"Organisation {org_id}: compute_summed_z result missing 'sum_squares' key")
    sum_squares = output["sum_squares"]
    if not isinstance(sum_squares, dict):
        raise AlgorithmError(f"Organisation {org_id}: compute_summed_z 'sum_squares' is not a dict")
    missing_sq = [v for v in expl_vars if v not in sum_squares]
    if missing_sq:
        raise AlgorithmError(
            f"Organisation {org_id}: compute_summed_z 'sum_squares' missing " f"variables {missing_sq}"
        )
    if "privacy_settings" not in output:
        raise AlgorithmError(f"Organisation {org_id}: compute_summed_z result missing 'privacy_settings' key")
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
