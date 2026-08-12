"""
Partial algorithm functions for v6-cox-ph.

This file contains the partial functions that are executed on each node with access
to local data. It handles data pipeline steps and delegates computation to
mathematical logic modules.

Note: This is the ONLY place where data querying (via v6-tools-rdf) occurs.
"""

import numpy as np
import pandas as pd

from vantage6.algorithm.tools.decorators import algorithm_client, data
from vantage6.algorithm.tools.exceptions import UserInputError
from vantage6.algorithm.client import AlgorithmClient

from vantage6_strongaya_general.miscellaneous import (
    safe_log,
    mask_unnecessary_variables,
    apply_sample_size_threshold,
    set_datatypes,
    apply_data_stratification,
    collect_organisation_ids
)
from vantage6_strongaya_rdf.collect_sparql_data import collect_sparql_data

from .miscellaneous import PrivacyThresholdConfig


@data(1)
@algorithm_client
def get_unique_event_times(
    client: AlgorithmClient,
    df: pd.DataFrame,
    time_col: str,
    outcome_col: str
) -> dict:
    """
    Compute unique event times from the local data.
    
    This function:
    1. Checks for RDF endpoint and queries if present
    2. Masks unnecessary variables
    3. Sets datatypes
    4. Applies sample size threshold
    5. Computes unique event times for cases where outcome = 1
    
    Parameters
    ----------
    client : AlgorithmClient
        The vantage6 client instance
    df : pd.DataFrame
        The local data DataFrame
    time_col : str
        Name of the column containing time data
    outcome_col : str
        Name of the column containing outcome/event data
        
    Returns
    -------
    dict
        Dictionary containing:
        - times: Dictionary of unique event times and their frequencies
        OR
        - N-Threshold not met: Organization ID if sample size is insufficient
    """
    safe_log("info", "Computing unique event times")
    
    # Define variables to analyse
    variables_to_analyse = [time_col, outcome_col]
    
    # Step 1: Check for RDF endpoint and query if present
    safe_log("debug", "Checking for RDF endpoint")
    if "endpoint" in df.columns:
        safe_log("debug", "RDF endpoint detected, querying data")
        df = collect_sparql_data(variables_to_analyse, endpoint=df["endpoint"].iloc[0])
    
    # Step 2: Mask unnecessary variables
    safe_log("debug", "Masking unnecessary variables")
    df = mask_unnecessary_variables(df, variables_to_analyse)
    
    # Step 3: Set datatypes
    safe_log("debug", "Setting datatypes")
    df = set_datatypes(df)
    
    # Step 4: Validate that requested variables exist
    safe_log("debug", "Validating requested variables exist")
    for var in variables_to_analyse:
        if var not in df.columns:
            raise UserInputError(f"Variable '{var}' not found in data")
    
    # Step 5: Apply sample size threshold
    safe_log("debug", "Applying sample size threshold")
    try:
        df = apply_sample_size_threshold(client, df, variables_to_analyse)
    except Exception as e:
        safe_log("warning", f"Sample size threshold not met: {e}")
        return {"N-Threshold not met": client.organization_id}
    
    # Check if we have enough events
    event_count = df[df[outcome_col] == 1].shape[0]
    if event_count <= PrivacyThresholdConfig.MIN_EVENT_COUNT:
        safe_log("warning", f"Insufficient number of events ({event_count} <= {PrivacyThresholdConfig.MIN_EVENT_COUNT})")
        return {"N-Threshold not met": client.organization_id}
    
    # Step 6: Compute unique event times
    safe_log("debug", "Computing unique event times for outcome=1")
    times = df[df[outcome_col] == 1].groupby(time_col, as_index=False).count()
    times = times.sort_values(by=time_col)[[time_col, outcome_col]]
    times['freq'] = times[outcome_col]
    times = times.drop(columns=outcome_col)
    
    safe_log("info", f"Found {len(times)} unique event times")
    return {'times': times.to_dict()}


@data(1)
@algorithm_client
def compute_summed_z(
    client: AlgorithmClient,
    df: pd.DataFrame,
    outcome_col: str,
    expl_vars: list
) -> dict:
    """
    Compute the sum of explanatory variables for event cases.
    
    This function:
    1. Checks for RDF endpoint and queries if present
    2. Masks unnecessary variables
    3. Sets datatypes
    4. Applies sample size threshold
    5. Computes sum of explanatory variables for cases where outcome = 1
    
    Parameters
    ----------
    client : AlgorithmClient
        The vantage6 client instance
    df : pd.DataFrame
        The local data DataFrame
    outcome_col : str
        Name of the column containing outcome/event data
    expl_vars : list
        List of explanatory variable names
        
    Returns
    -------
    dict
        Dictionary containing:
        - sum: Dictionary of summed explanatory variables for event cases
    """
    safe_log("info", "Computing summed z statistics")
    
    # Define variables to analyse
    variables_to_analyse = [outcome_col] + expl_vars
    
    # Step 1: Check for RDF endpoint and query if present
    safe_log("debug", "Checking for RDF endpoint")
    if "endpoint" in df.columns:
        safe_log("debug", "RDF endpoint detected, querying data")
        df = collect_sparql_data(variables_to_analyse, endpoint=df["endpoint"].iloc[0])
    
    # Step 2: Mask unnecessary variables
    safe_log("debug", "Masking unnecessary variables")
    df = mask_unnecessary_variables(df, variables_to_analyse)
    
    # Step 3: Set datatypes
    safe_log("debug", "Setting datatypes")
    df = set_datatypes(df)
    
    # Step 4: Validate that requested variables exist
    safe_log("debug", "Validating requested variables exist")
    for var in variables_to_analyse:
        if var not in df.columns:
            raise UserInputError(f"Variable '{var}' not found in data")
    
    # Step 5: Apply sample size threshold
    safe_log("debug", "Applying sample size threshold")
    try:
        df = apply_sample_size_threshold(client, df, variables_to_analyse)
    except Exception as e:
        safe_log("warning", f"Sample size threshold not met: {e}")
        return {"N-Threshold not met": client.organization_id}
    
    # Step 6: Compute summed Z statistics
    safe_log("debug", "Computing sum of explanatory variables for event cases")
    z_sum = (df[df[outcome_col] == 1][expl_vars].sum().to_dict())
    
    safe_log("info", f"Computed Z sum: {z_sum}")
    return {'sum': z_sum}


@data(1)
@algorithm_client
def perform_iteration(
    client: AlgorithmClient,
    df: pd.DataFrame,
    time_col: str,
    expl_vars: list,
    beta: list,
    unique_time_events: list
) -> dict:
    """
    Perform one iteration of the Newton-Raphson optimization for Cox-PH.
    
    This function:
    1. Checks for RDF endpoint and queries if present
    2. Masks unnecessary variables
    3. Sets datatypes
    4. Applies sample size threshold
    5. Computes aggregates needed for derivative computation
    
    Parameters
    ----------
    client : AlgorithmClient
        The vantage6 client instance
    df : pd.DataFrame
        The local data DataFrame
    time_col : str
        Name of the column containing time data
    expl_vars : list
        List of explanatory variable names
    beta : list
        Current estimate of beta coefficients
    unique_time_events : list
        List of unique event times
        
    Returns
    -------
    dict
        Dictionary containing:
        - agg1: List of aggregated exp(beta * X) sums
        - agg2: List of aggregated X * exp(beta * X) sums
        - agg3: List of aggregated outer products X * X^T * exp(beta * X)
    """
    safe_log("info", "Computing aggregates for Cox-PH iteration")
    
    # Define variables to analyse
    variables_to_analyse = [time_col] + expl_vars
    
    # Step 1: Check for RDF endpoint and query if present
    safe_log("debug", "Checking for RDF endpoint")
    if "endpoint" in df.columns:
        safe_log("debug", "RDF endpoint detected, querying data")
        df = collect_sparql_data(variables_to_analyse, endpoint=df["endpoint"].iloc[0])
    
    # Step 2: Mask unnecessary variables
    safe_log("debug", "Masking unnecessary variables")
    df = mask_unnecessary_variables(df, variables_to_analyse)
    
    # Step 3: Set datatypes
    safe_log("debug", "Setting datatypes")
    df = set_datatypes(df)
    
    # Step 4: Validate that requested variables exist
    safe_log("debug", "Validating requested variables exist")
    for var in variables_to_analyse:
        if var not in df.columns:
            raise UserInputError(f"Variable '{var}' not found in data")
    
    # Step 5: Apply sample size threshold
    safe_log("debug", "Applying sample size threshold")
    try:
        df = apply_sample_size_threshold(client, df, variables_to_analyse)
    except Exception as e:
        safe_log("warning", f"Sample size threshold not met: {e}")
        return {"N-Threshold not met": client.organization_id}
    
    # Step 6: Compute aggregates
    safe_log("debug", "Computing aggregates for derivative computation")
    
    # Deserialize beta values
    beta = np.array(beta)
    num_unique_time_events = len(unique_time_events)
    num_explanatory_vars = len(expl_vars)
    
    agg1 = []
    agg2 = []
    agg3 = []
    
    for i in range(num_unique_time_events):
        # Get risk set at time t_i (all subjects with time >= t_i)
        R_i = df[df[time_col] >= unique_time_events[i]][expl_vars]
        
        # Check if R_i is empty
        if not R_i.empty:
            # Compute exp(beta * X) for each subject in risk set
            ebz = np.exp(np.dot(np.array(R_i), beta))
            agg1.append(sum(ebz))
            
            # Compute z_ebz = X * exp(beta * X)
            func = lambda x: np.asarray(x) * np.asarray(ebz)
            z_ebz = R_i.apply(func)
            agg2.append(z_ebz.sum())
            
            # Compute outer products
            summed = np.zeros((num_explanatory_vars, num_explanatory_vars))
            for j in range(len(R_i)):
                summed = summed + np.outer(np.array(z_ebz)[j], np.array(R_i)[j].T)
            agg3.append(summed)
        else:
            agg1.append(0)
            agg2.append(pd.Series(np.zeros(num_explanatory_vars), index=expl_vars))
            agg3.append(np.zeros((num_explanatory_vars, num_explanatory_vars)))
    
    # JSON-serialize the results
    safe_log("debug", "Serializing results for JSON")
    agg2_dict = pd.DataFrame(agg2).to_dict()
    agg3_list = [array.tolist() for array in agg3]
    
    safe_log("info", "Iteration computation completed")
    return {
        'agg1': agg1,
        'agg2': agg2_dict,
        'agg3': agg3_list
    }
