"""
Miscellaneous utilities for the Cox-PH algorithm.

This module contains:
- Pydantic models for input validation
- Helper functions for data quality checks
"""

from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator
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
        description=(
            "List of organisation IDs to include. If None, all "
            "organisations are used."
        ),
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
        for var in v:
            if not isinstance(var, str) or not var.strip():
                raise ValueError(f"Invalid variable name: {var}")
        return [var.strip() for var in v]

    @field_validator("organization_ids")
    @classmethod
    def validate_organization_ids(cls, v: Optional[List[int]]) -> Optional[List[int]]:
        """Validate organisation IDs if provided."""
        if v is None:
            return None
        if not isinstance(v, list):
            raise ValueError("organization_ids must be a list of integers")
        for org_id in v:
            if not isinstance(org_id, int) or org_id < 0:
                raise ValueError(
                    f"Invalid organisation ID: {org_id}. "
                    "Must be a non-negative integer."
                )
        return list(set(v))


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
        Name of the time column.
    outcome_col : str
        Name of the outcome column.
    expl_vars : List[str]
        List of explanatory variable names.
    organization_ids : Optional[List[int]]
        List of organisation IDs to include.

    Returns
    -------
    CoxPHInput
        Validated input model.

    Raises
    ------
    UserInputError
        If input validation fails.
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


def check_event_count(df: pd.DataFrame, outcome_col: str, min_events: int = 5) -> bool:
    """
    Check if the data has a sufficient number of events.

    Parameters
    ----------
    df : pd.DataFrame
        The data to check.
    outcome_col : str
        Name of the outcome column.
    min_events : int
        Minimum number of events required (default 5).

    Returns
    -------
    bool
        True if sufficient events, False otherwise.
    """
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
        The data to check.
    time_col : str
        Name of the time column.
    outcome_col : str
        Name of the outcome column.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing data quality information.
    """
    result: Dict[str, Any] = {
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
        result["time_range"] = (float(time_series.min()), float(time_series.max()))
        result["has_negative_time"] = bool((time_series < 0).any())

    if result["has_outcome"]:
        outcome_series = df[outcome_col]
        result["outcome_values"] = outcome_series.unique().tolist()
        result["event_count"] = int((outcome_series == 1).sum())
        result["censored_count"] = int((outcome_series == 0).sum())

    return result
