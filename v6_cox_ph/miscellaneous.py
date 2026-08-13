"""
Miscellaneous utilities for the v6_cox_ph algorithm.

This module contains:
- Pydantic models for input validation
- Helper functions for data processing
- Type definitions and constants
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, field_validator, Field
from vantage6.algorithm.tools.exceptions import UserInputError


class CoxPHInput(BaseModel):
    """
    Pydantic model for validating Cox-PH algorithm input parameters.

    This validates all user-supplied parameters that travel over the wire
    between central and partial functions.
    """

    time_col: str = Field(..., description="Name of the column containing time data")
    outcome_col: str = Field(
        ..., description="Name of the column containing outcome/event data"
    )
    expl_vars: List[str] = Field(..., description="List of explanatory variable names")
    organization_ids: Optional[List[int]] = Field(
        default=None,
        description="List of organization IDs to include. If None, all organizations are used.",
    )

    @field_validator("time_col", "outcome_col")
    @classmethod
    def validate_column_names(cls, v: str) -> str:
        """Validate that column names are non-empty strings."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Column names must be non-empty strings")
        return v.strip()

    @field_validator("expl_vars")
    @classmethod
    def validate_expl_vars(cls, v: List[str]) -> List[str]:
        """Validate that explanatory variables list is non-empty."""
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("At least one explanatory variable must be specified")
        # Validate each variable name
        for var in v:
            if not isinstance(var, str) or not var.strip():
                raise ValueError(f"Invalid variable name: {var}")
        return [var.strip() for var in v]

    @field_validator("organization_ids")
    @classmethod
    def validate_organization_ids(cls, v: Optional[List[int]]) -> Optional[List[int]]:
        """Validate organization IDs if provided."""
        if v is None:
            return None
        if not isinstance(v, list):
            raise ValueError("organization_ids must be a list of integers")
        for org_id in v:
            if not isinstance(org_id, int) or org_id < 0:
                raise ValueError(
                    f"Invalid organization ID: {org_id}. Must be non-negative integer."
                )
        return list(set(v))  # Remove duplicates


class PartialResult(BaseModel):
    """
    Pydantic model for validating partial results from node computations.
    """

    agg1: Optional[List[float]] = Field(
        default=None, description="First aggregate values"
    )
    agg2: Optional[Dict[str, Any]] = Field(
        default=None, description="Second aggregate values"
    )
    agg3: Optional[List[List[float]]] = Field(
        default=None, description="Third aggregate values"
    )
    sum: Optional[Dict[str, float]] = Field(
        default=None, description="Sum of explanatory variables"
    )
    times: Optional[Dict[str, Any]] = Field(
        default=None, description="Unique event times data"
    )

    @field_validator("agg1", "agg3")
    @classmethod
    def validate_numeric_lists(cls, v: Optional[List]) -> Optional[List]:
        """Validate that numeric lists contain only numbers."""
        if v is None:
            return None
        if not isinstance(v, list):
            raise ValueError("Must be a list")
        return v


class PrivacyThresholdConfig:
    """Configuration for privacy threshold checks."""

    MIN_SAMPLE_SIZE = 10
    MIN_EVENT_COUNT = 5
    MAX_ITERATIONS = 10
    CONVERGENCE_THRESHOLD = 0.000001


class StratificationDetails:
    """
    Type definition for stratification details.

    This follows the STRONG AYA data standard for stratification.
    """

    pass


class VariableDetails:
    """
    Type definition for variable details.

    This follows the STRONG AYA data standard for variable specifications.
    """

    pass


def validate_coxph_input(
    time_col: str,
    outcome_col: str,
    expl_vars: List[str],
    organization_ids: Optional[List[int]] = None,
) -> CoxPHInput:
    """
    Validate Cox-PH algorithm input parameters.

    Parameters
    ----------
    time_col : str
        Name of the time column
    outcome_col : str
        Name of the outcome column
    expl_vars : List[str]
        List of explanatory variable names
    organization_ids : Optional[List[int]]
        List of organization IDs to include

    Returns
    -------
    CoxPHInput
        Validated input model

    Raises
    ------
    UserInputError
        If input validation fails
    """
    try:
        return CoxPHInput(
            time_col=time_col,
            outcome_col=outcome_col,
            expl_vars=expl_vars,
            organization_ids=organization_ids,
        )
    except Exception as e:
        raise UserInputError(f"Invalid Cox-PH input: {e}")


def validate_partial_result(result: Dict[str, Any]) -> PartialResult:
    """
    Validate a partial result from a node computation.

    Parameters
    ----------
    result : Dict[str, Any]
        The partial result to validate

    Returns
    -------
    PartialResult
        Validated partial result model

    Raises
    ------
    UserInputError
        If result validation fails
    """
    try:
        return PartialResult(**result)
    except Exception as e:
        raise UserInputError(f"Invalid partial result: {e}")


def check_event_count(
    df: pd.DataFrame, outcome_col: str, min_events: int = None
) -> bool:
    """
    Check if the data has sufficient number of events.

    Parameters
    ----------
    df : pd.DataFrame
        The data to check
    outcome_col : str
        Name of the outcome column
    min_events : int, optional
        Minimum number of events required. If None, uses PrivacyThresholdConfig.MIN_EVENT_COUNT

    Returns
    -------
    bool
        True if sufficient events, False otherwise
    """
    if min_events is None:
        min_events = PrivacyThresholdConfig.MIN_EVENT_COUNT

    event_count = df[df[outcome_col] == 1].shape[0]
    return event_count > min_events


def check_data_quality(
    df: pd.DataFrame, time_col: str, outcome_col: str
) -> Dict[str, Any]:
    """
    Perform comprehensive data quality checks.

    Parameters
    ----------
    df : pd.DataFrame
        The data to check
    time_col : str
        Name of the time column
    outcome_col : str
        Name of the outcome column

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - has_time: Whether time column exists
        - has_outcome: Whether outcome column exists
        - event_count: Number of events
        - censored_count: Number of censored observations
        - total_count: Total number of observations
        - time_range: Range of time values
        - has_negative_time: Whether there are negative time values
        - outcome_values: Unique values in outcome column
    """
    result = {
        "has_time": time_col in df.columns,
        "has_outcome": outcome_col in df.columns,
        "event_count": 0,
        "censored_count": 0,
        "total_count": len(df),
        "time_range": None,
        "has_negative_time": False,
        "outcome_values": [],
    }

    if result["has_time"]:
        time_series = df[time_col]
        result["time_range"] = (time_series.min(), time_series.max())
        result["has_negative_time"] = (time_series < 0).any()

    if result["has_outcome"]:
        outcome_series = df[outcome_col]
        result["outcome_values"] = outcome_series.unique().tolist()
        result["event_count"] = (outcome_series == 1).sum()
        result["censored_count"] = (outcome_series == 0).sum()

    return result
