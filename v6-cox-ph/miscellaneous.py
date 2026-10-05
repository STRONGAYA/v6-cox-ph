"""
Miscellaneous utilities for the Cox-PH algorithm.

This module contains the Pydantic models for the wire contract: user input
(``CoxPHInput``), the payload central sends to ``perform_iteration``
(``IterationInput``) and, in a later step, the partial results.
"""

from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field, ValidationInfo, ValidationError, field_validator, model_validator
from vantage6.algorithm.tools.exceptions import UserInputError


def format_validation_error(exc: ValidationError) -> str:
    """Format a pydantic error without echoing the offending input.

    Pydantic's default message embeds the input value; for wire payloads
    that value can be an aggregate array, and the message ends up in the
    container log the researcher reads. One line per error, no input, no URL.
    """
    parts = []
    for error in exc.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in error["loc"])
        parts.append(f"{location}: {error['msg']}" if location else error["msg"])
    return "; ".join(parts)


class CoxPHInput(BaseModel):
    """
    Pydantic model for validating Cox-PH algorithm input parameters.

    This validates all user-supplied parameters that travel over the wire
    between central and partial functions.
    """

    time_col: str = Field(..., description="Name of the column containing time data")
    outcome_col: str = Field(..., description="Name of the column containing outcome/event data")
    expl_vars: List[str] = Field(..., description="List of explanatory variable names")
    organization_ids: Optional[List[int]] = Field(
        default=None,
        description=("List of organisation IDs to include. If None, all " "organisations are used."),
    )
    covariate_profiles: Optional[List[Dict[str, float]]] = Field(
        default=None,
        description=(
            "Optional covariate profiles for survival curves. Each entry maps "
            "explanatory variable names to values; variables left out are taken "
            "at the pooled event-case mean (the baseline profile)."
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
                raise ValueError(f"Invalid organisation ID: {org_id}. " "Must be a non-negative integer.")
        return list(set(v))

    @model_validator(mode="after")
    def validate_covariate_profiles(self) -> "CoxPHInput":
        """Covariate profiles may only name known explanatory variables."""
        if self.covariate_profiles is None:
            return self
        for i, profile in enumerate(self.covariate_profiles):
            if not isinstance(profile, dict):
                raise ValueError(f"Covariate profile {i} must be a mapping of variable names to values")
            unknown = [var for var in profile if var not in self.expl_vars]
            if unknown:
                raise ValueError(
                    f"Covariate profile {i} names unknown variables {unknown}; "
                    f"known explanatory variables are {self.expl_vars}"
                )
        return self


def validate_coxph_input(
    time_col: str,
    outcome_col: str,
    expl_vars: List[str],
    organization_ids: Optional[List[int]] = None,
    covariate_profiles: Optional[List[Dict[str, float]]] = None,
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
    covariate_profiles : Optional[List[Dict[str, float]]]
        Optional covariate profiles for survival curves.

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
            covariate_profiles=covariate_profiles,
        )
    except Exception as e:
        raise UserInputError(f"Invalid Cox-PH input: {e}")


class IterationInput(BaseModel):
    """
    Wire payload ``central`` sends to ``perform_iteration`` each round-trip.

    Validated on the node before any aggregate is computed. The expected
    lengths (one entry per explanatory variable) and the node's privacy
    settings are not part of the payload: they arrive via the validation
    ``context`` (``expl_vars``, ``settings``).

    Structure is validated by pydantic; the numeric checks (finiteness,
    positivity, monotonicity, the bin grid) run through numpy on the plain
    lists, which is faster than element-wise validation and adds no safety.
    """

    beta: List[float]
    centre: List[float]
    scale: List[float]
    unique_time_events: List[float]

    @field_validator("beta", "centre", "scale")
    @classmethod
    def _finite_vector(cls, v: List[float], info: ValidationInfo) -> List[float]:
        if not np.all(np.isfinite(np.asarray(v, dtype=float))):
            raise ValueError(f"{info.field_name} contains non-finite values (NaN or inf).")
        return v

    @field_validator("scale")
    @classmethod
    def _positive_scale(cls, v: List[float]) -> List[float]:
        if np.any(np.asarray(v, dtype=float) <= 0):
            raise ValueError("scale must be strictly positive for every explanatory variable.")
        return v

    @field_validator("unique_time_events")
    @classmethod
    def _valid_grid(cls, v: List[float]) -> List[float]:
        if len(v) == 0:
            raise ValueError("unique_time_events must not be empty.")
        grid = np.asarray(v, dtype=float)
        if not np.all(np.isfinite(grid)):
            raise ValueError("unique_time_events contains non-finite values.")
        if not np.all(np.diff(grid) > 0):
            # Strictly increasing covers both the sortedness and the
            # no-duplicates rules.
            raise ValueError("unique_time_events must be strictly increasing (sorted, without duplicates).")
        return v

    @model_validator(mode="after")
    def _lengths_and_bin_grid(self, info: ValidationInfo) -> "IterationInput":
        context = info.context or {}
        expl_vars = context.get("expl_vars", [])
        for name in ("beta", "centre", "scale"):
            vector = getattr(self, name)
            if len(vector) != len(expl_vars):
                raise ValueError(
                    f"{name} must have length {len(expl_vars)} (one per explanatory "
                    f"variable), got shape {np.asarray(vector).shape}"
                )
        settings = context.get("settings")
        width = getattr(settings, "time_bin_width", None)
        if width:
            grid = np.asarray(self.unique_time_events, dtype=float)
            binned = np.floor(grid / width) * width
            off_grid = ~np.isclose(binned, grid)
            if np.any(off_grid):
                raise ValueError(
                    f"unique_time_events contains time {grid[off_grid][0]} that is "
                    f"not on the bin grid (width={width})."
                )
        return self
