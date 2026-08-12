"""
Central algorithm functions for v6-cox-ph.

This file contains the central orchestration logic for the federated Cox-PH algorithm.
It coordinates subtasks dispatched to nodes, collects partial results, and aggregates them.

Note: This file should only contain orchestration logic. Mathematical computations
are delegated to coxph_logic.py module.
"""

import math
import numpy as np
import pandas as pd

from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.tools.exceptions import UserInputError
from vantage6.algorithm.client import AlgorithmClient

from vantage6_strongaya_general.miscellaneous import (
    safe_log,
    collect_organisation_ids,
    check_partial_result_presence
)

from .miscellaneous import validate_coxph_input, PrivacyThresholdConfig
from .coxph_logic import compute_derivatives, compute_model_results, format_results_dataframe


@algorithm_client
def central(
    client: AlgorithmClient,
    time_col: str,
    outcome_col: str,
    expl_vars: list,
    organization_ids: list = None
) -> dict:
    """
    Central function for the federated Cox Proportional Hazards algorithm.
    
    This function orchestrates the federated computation by:
    1. Validating input parameters
    2. Collecting organization IDs
    3. Dispatching subtasks to compute unique event times
    4. Dispatching subtasks to compute summed Z statistics
    5. Iteratively optimizing beta coefficients using Newton-Raphson
    6. Computing final model statistics and returning results
    
    Parameters
    ----------
    client : AlgorithmClient
        The vantage6 client instance
    time_col : str
        Name of the column containing time data
    outcome_col : str
        Name of the column containing outcome/event data (1=event, 0=censored)
    expl_vars : list
        List of explanatory variable names to include in the model
    organization_ids : list, optional
        List of organization IDs to include. If None, all organizations are used.
        
    Returns
    -------
    dict
        Dictionary containing:
        - included_organizations: List of organization IDs that contributed
        - excluded_organizations: List of organization IDs excluded due to privacy
        - model: JSON string of the results DataFrame
        - overall_p_value: Overall model p-value from Wald test
        - aic: Akaike Information Criterion
        - degrees_of_freedom: Number of degrees of freedom
        - warnings: List of warning messages
    """
    safe_log("info", "Starting Cox-PH central algorithm")
    
    # Validate input parameters
    safe_log("debug", "Validating input parameters")
    try:
        validated_input = validate_coxph_input(
            time_col=time_col,
            outcome_col=outcome_col,
            expl_vars=expl_vars,
            organization_ids=organization_ids
        )
        time_col = validated_input.time_col
        outcome_col = validated_input.outcome_col
        expl_vars = validated_input.expl_vars
        organization_ids = validated_input.organization_ids
    except UserInputError as e:
        safe_log("error", f"Input validation failed: {e}")
        raise
    
    # Collect all organizations that participate in this collaboration
    safe_log("debug", "Collecting organization IDs")
    ids = collect_organisation_ids(organization_ids, client)
    safe_log("info", f"Sending tasks to organizations {ids}")
    
    n_covs = len(expl_vars)
    epochs = 10
    
    # ========================================================================
    # Step 1: Get unique event times from all organizations
    # ========================================================================
    safe_log("info", "Step 1: Computing unique event times")
    
    input_ = {
        "method": "get_unique_event_times",
        "kwargs": {
            "time_col": time_col,
            "outcome_col": outcome_col
        },
    }
    
    n_loops = 0
    n_threshold_met = False
    excluded_ids = []
    
    while not n_threshold_met:
        _excluded_ids = []
        
        if n_loops > 2:
            safe_log("error", "Sample size violations should be eliminated yet criteria are not met. Exiting")
            raise ValueError("Sample size violations should be eliminated yet criteria are not met. Exiting")
        
        n_loops += 1
        
        # Create subtask for all selected organizations
        safe_log("debug", "Creating subtask for unique event times")
        task = client.task.create(
            input_=input_,
            organizations=ids,
            name="Unique event times",
            description="Getting unique event times and their counts"
        )
        
        # Wait for results
        safe_log("debug", "Waiting for unique event times results")
        results = client.wait_for_results(task_id=task.get("id"))
        safe_log("info", "Unique event times results obtained")
        
        # Check for privacy threshold violations
        results_without_errors = check_partial_result_presence(results, ids)
        
        unique_time_events = []
        for output in results:
            # Exclude organizations that do not meet the N-threshold
            if "N-Threshold not met" in output:
                org_id = output["N-Threshold not met"]
                safe_log("warning", f"Insufficient samples for organization {org_id}. Excluding from analysis.")
                if org_id in ids:
                    ids.remove(org_id)
                excluded_ids.append(org_id)
                _excluded_ids.append(org_id)
                continue
            
            if "times" in output:
                output_df = pd.DataFrame.from_dict(output["times"])
                unique_time_events.append(output_df)
        
        if len(_excluded_ids) == 0:
            n_threshold_met = True
        elif len(ids) == 0:
            safe_log("warning", "No organizations meet the minimal sample size threshold, returning NaN.")
            return {
                "excluded_organizations": excluded_ids,
                "included_organizations": [],
                "table": None,
                "model": None
            }
    
    # Aggregate unique time events across all organizations
    safe_log("debug", "Aggregating unique time events")
    aggregated_time_events = pd.concat(unique_time_events)
    aggregated_time_events = aggregated_time_events.groupby(time_col, as_index=False).sum()
    
    # Get the list of unique_time_events
    unique_time_events_list = aggregated_time_events[time_col].tolist()
    safe_log("info", f"Found {len(unique_time_events_list)} unique event times")
    
    # ========================================================================
    # Step 2: Compute summed Z statistics
    # ========================================================================
    safe_log("info", "Step 2: Computing summed Z statistics")
    
    input_ = {
        "method": "compute_summed_z",
        "kwargs": {
            "outcome_col": outcome_col,
            "expl_vars": expl_vars,
        }
    }
    
    safe_log("debug", "Creating subtask for summed Z computation")
    task = client.task.create(
        input_=input_,
        organizations=ids,
        name="Summed Z statistic",
        description="Computing the summed Z statistic"
    )
    
    safe_log("debug", "Waiting for summed Z results")
    results = client.wait_for_results(task_id=task.get("id"))
    safe_log("info", "Summed Z results obtained")
    
    # Aggregate Z sums
    z_sum = pd.Series(np.zeros(n_covs), index=expl_vars)
    for output in results:
        if "sum" in output:
            output_sum = pd.Series(output["sum"])
            z_sum = z_sum.add(output_sum, fill_value=0)
    
    safe_log("debug", f"Z sum computed: {z_sum.to_dict()}")
    
    # ========================================================================
    # Step 3: Iterative optimization (Newton-Raphson)
    # ========================================================================
    safe_log("info", "Step 3: Starting iterative optimization")
    
    beta = np.zeros(n_covs)
    
    for epoch in range(epochs):
        safe_log("debug", f"Starting iteration {epoch + 1}/{epochs}")
        
        # JSON-serialize beta for Vantage6
        beta_list = beta.tolist()
        
        # Define input parameters for iteration subtask
        input_ = {
            "method": "perform_iteration",
            "kwargs": {
                'time_col': time_col,
                "expl_vars": expl_vars,
                'beta': beta_list,
                'unique_time_events': unique_time_events_list
            }
        }
        
        # Create subtask for all organizations
        safe_log("debug", "Creating subtask for iteration")
        task = client.task.create(
            input_=input_,
            organizations=ids,
            name="Start iteration",
            description="Iterating to find the optimal beta"
        )
        
        # Wait for results
        safe_log("debug", "Waiting for iteration results")
        results = client.wait_for_results(task_id=task.get("id"))
        safe_log("info", f"Iteration {epoch + 1} results obtained")
        
        # Aggregate results
        summed_agg1 = np.zeros(len(unique_time_events_list))
        summed_agg2 = np.zeros((len(unique_time_events_list), n_covs))
        summed_agg3 = np.zeros((len(unique_time_events_list), n_covs, n_covs))
        
        for output in results:
            if 'agg1' in output:
                agg1_array = np.array(output['agg1'])
                summed_agg1 += agg1_array
                
            if 'agg2' in output:
                agg2_df = pd.DataFrame.from_dict(output['agg2'])
                agg2_array = agg2_df.values
                summed_agg2 += agg2_array
                
            if 'agg3' in output:
                agg3_array = np.array([np.array(lst) for lst in output['agg3']])
                summed_agg3 += agg3_array
        
        # Compute derivatives
        safe_log("debug", "Computing derivatives")
        primary_derivative, secondary_derivative = compute_derivatives(
            summed_agg1, summed_agg2, summed_agg3,
            aggregated_time_events, z_sum
        )
        
        # Update beta
        beta_old = np.array(beta)
        try:
            beta = beta_old - solve(secondary_derivative, primary_derivative)
        except np.linalg.LinAlgError as e:
            safe_log("warning", f"Linear algebra error in solve: {e}")
            break
            
        delta = float(max(abs(beta - beta_old)))
        safe_log("debug", f"Delta: {delta}")
        
        if math.isnan(delta):
            safe_log("warning", "Delta has turned into a NaN")
            break
        
        if delta <= 0.000001:
            safe_log("info", "Betas have settled! Finished iterating!")
            break
    
    # ========================================================================
    # Step 4: Compute final model results
    # ========================================================================
    safe_log("info", "Step 4: Computing final model results")
    
    model_results = compute_model_results(
        beta, secondary_derivative, z_sum, aggregated_time_events, summed_agg1
    )
    
    # Format results DataFrame
    results_df = format_results_dataframe(model_results["coefficients"], expl_vars)
    
    # Prepare return dictionary
    return_dict = {
        "included_organizations": ids,
        "excluded_organizations": excluded_ids,
        "model": results_df.to_json(),
        "overall_p_value": model_results["overall_p_value"],
        "aic": model_results["aic"],
        "degrees_of_freedom": model_results["n_params"],
        "warnings": model_results["warnings"]
    }
    
    safe_log("info", "Cox-PH central algorithm completed successfully")
    return return_dict
