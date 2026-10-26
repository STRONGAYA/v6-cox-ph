"""
Miscellaneous utilities for the Cox-PH algorithm.

This module contains the Pydantic models for the wire contract: user input
(``CoxPHInput``), the payload central sends to ``perform_iteration``
(``IterationInput``) and the partial results (``SummedZResult``,
``IterationResult``), which the partials build and central validates.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    ValidationError,
    field_validator,
    model_validator,
)
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


# Stratification variable names must be identifier-shaped, or prefixed names
# such as ``ncit:C28421`` (the STRONG AYA semantic map names RDF variables by
# their class): ``apply_data_stratification`` interpolates them into a
# dataframe query between backticks, so anything beyond this pattern (a
# backtick above all) is rejected before the query is built.
STRATIFICATION_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(:[A-Za-z0-9_]+)?$")


def validate_stratification_spec(spec: object) -> Dict[str, Any]:
    """Validate a ``variables_to_stratify`` mapping (STRONG AYA data standard).

    A mapping of stratification variable names to either a non-empty list of
    values to keep or a ``{"start": ..., "end": ...}`` range (at least one
    bound, numeric when present, start <= end). The messages echo the
    user-supplied variable names only, never data values.

    Raises
    ------
    ValueError
        On any structural problem.
    """
    if not isinstance(spec, dict) or len(spec) == 0:
        raise ValueError("variables_to_stratify must be a non-empty mapping of variable names to values or ranges")
    for name, selection in spec.items():
        if not isinstance(name, str) or not STRATIFICATION_NAME_PATTERN.match(name):
            raise ValueError(
                f"stratification variable names must be identifier-shaped or prefixed names "
                f"(such as ncit:C28421), got '{name}'"
            )
        if isinstance(selection, dict):
            unknown_keys = sorted(set(selection) - {"start", "end"})
            if unknown_keys:
                raise ValueError(f"range for '{name}' may only have 'start'/'end' keys, got {unknown_keys}")
            if "start" not in selection and "end" not in selection:
                raise ValueError(f"range for '{name}' must define at least one of 'start'/'end'")
            for bound in ("start", "end"):
                value = selection.get(bound)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                    raise ValueError(f"range bound '{bound}' for '{name}' must be numeric")
            start, end = selection.get("start"), selection.get("end")
            if start is not None and end is not None and start > end:
                raise ValueError(f"range for '{name}' has start above end")
        elif isinstance(selection, list):
            if len(selection) == 0:
                raise ValueError(f"value list for '{name}' must not be empty")
            for value in selection:
                if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                    raise ValueError(f"values for '{name}' must be strings or numbers")
        else:
            raise ValueError(f"stratification for '{name}' must be a list of values or a range mapping")
    return spec


# Namespaces of the STRONG AYA semantic map (the ``prefixes`` of
# AYA_cancer_schema.jsonld), plus the SNOMED namespace the RDF library's
# queries declare. RDF data carries mapped terms as full IRIs; compacting
# them lets the outcome codes in ``outcome_col`` and the values in
# ``variables_to_stratify`` name a term either way (``ncit:C37987`` or its
# full IRI).
ONTOLOGY_NAMESPACES = {
    "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#": "ncit",
    "http://semanticscience.org/resource/": "sio",
    "http://www.cancerdata.org/roo/": "roo",
    "https://strongaya.eu/": "strongaya",
    "http://purl.bioontology.org/ontology/SNOMEDCT/": "sct",
    "http://snomed.info/id/": "sct",
    "http://id.nlm.nih.gov/mesh/": "mesh",
    "http://purl.obolibrary.org/obo/": "gsso",
}

# The STRONG AYA semantic map codes a missing or unspecified value of any
# variable, numeric ones included, with this term.
MISSING_TERM = "ncit:C54031"

OUTCOME_CODE_KEYS = ("event", "censored", "missing")

_NUMBER_PATTERN = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def term_key(value: object) -> Optional[Any]:
    """The comparable form of a coded value, from the data or the user input.

    A term IRI in a known namespace is compacted to its prefixed name
    (``http://...Thesaurus.owl#C37987`` and ``<...>`` both become
    ``ncit:C37987``); a number, or a string that reads as one, becomes a
    float, so ``1``, ``1.0`` and ``"1"`` are the same code; any other string
    is compared as written, without surrounding whitespace. Missing values
    (``None``, NaN, an empty string) have no key and return ``None``.
    """
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_, int, float, np.integer, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if not isinstance(value, str):
        return str(value)
    term = value.strip()
    if term.startswith("<") and term.endswith(">"):
        term = term[1:-1].strip()
    if not term:
        return None
    for namespace, prefix in ONTOLOGY_NAMESPACES.items():
        if term.startswith(namespace) and len(term) > len(namespace):
            return f"{prefix}:{term[len(namespace):]}"
    if _NUMBER_PATTERN.match(term):
        return float(term)
    return term


def validate_outcome_codes(codes: object, name: str) -> Dict[str, List[Any]]:
    """Validate the outcome codes of an ``outcome_col`` (STRONG AYA data standard).

    A mapping with the required non-empty lists ``event`` and ``censored``
    and an optional list ``missing``, each holding outcome codes (strings
    or numbers). No code may be listed under two keys, also not when one
    entry is a prefixed name and the other its full IRI. The messages echo
    the user-supplied names and codes only, never data values.

    Raises
    ------
    ValueError
        On any structural problem.
    """
    if not isinstance(codes, dict):
        raise ValueError(f"outcome_col '{name}' must map to an object with 'event' and 'censored' lists of codes")
    unknown_keys = sorted(str(key) for key in set(codes) - set(OUTCOME_CODE_KEYS))
    if unknown_keys:
        raise ValueError(
            f"outcome_col '{name}' may only list 'event', 'censored' and 'missing' codes, got {unknown_keys}"
        )
    listed_under: Dict[Any, str] = {}
    for key in OUTCOME_CODE_KEYS:
        if key not in codes:
            if key == "missing":
                continue
            raise ValueError(f"outcome_col '{name}' must list the '{key}' codes")
        listed = codes[key]
        if not isinstance(listed, list):
            raise ValueError(f"outcome_col '{name}' '{key}' must be a list of outcome codes")
        if key != "missing" and len(listed) == 0:
            raise ValueError(f"outcome_col '{name}' '{key}' must not be empty")
        for code in listed:
            if isinstance(code, bool) or not isinstance(code, (str, int, float)):
                raise ValueError(f"outcome_col '{name}' '{key}' codes must be strings or numbers")
            term = term_key(code)
            if term is None:
                raise ValueError(f"outcome_col '{name}' '{key}' codes must not be empty or non-finite")
            other = listed_under.get(term)
            if other is not None and other != key:
                raise ValueError(f"outcome code '{code}' is listed under both '{other}' and '{key}'")
            listed_under[term] = key
    return codes


def parse_outcome_col(outcome_col: object) -> Tuple[str, Optional[Dict[str, List[Any]]]]:
    """Split ``outcome_col`` into the column name and its outcome codes.

    ``outcome_col`` is the name of an outcome column coded 0/1, or (STRONG
    AYA) a mapping of one column name to the codes that mean an event, a
    censored observation and, optionally, a missing value::

        {"ncit:C25717": {"event": ["ncit:C99799"], "censored": ["ncit:C37987"], "missing": ["ncit:C54031"]}}

    Returns
    -------
    Tuple[str, Optional[Dict[str, List[Any]]]]
        The column name (without surrounding whitespace) and the validated
        codes, or ``None`` for a plain column name.

    Raises
    ------
    ValueError
        On any structural problem.
    """
    if isinstance(outcome_col, str):
        if not outcome_col.strip():
            raise ValueError("Column names must be non-empty strings")
        return outcome_col.strip(), None
    if not isinstance(outcome_col, dict) or len(outcome_col) != 1:
        raise ValueError("outcome_col must be a column name, or a mapping of one column name to its outcome codes")
    ((name, codes),) = outcome_col.items()
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Column names must be non-empty strings")
    return name.strip(), validate_outcome_codes(codes, name.strip())


class CoxPHInput(BaseModel):
    """
    Pydantic model for validating Cox-PH algorithm input parameters.

    This validates all user-supplied parameters that travel over the wire
    between central and partial functions.
    """

    time_col: str = Field(..., description="Name of the column containing time data")
    outcome_col: Union[str, Dict[str, Any]] = Field(
        ...,
        description=(
            "Name of the outcome column coded 0/1, or (STRONG AYA) a mapping of the "
            "column name to its 'event', 'censored' and optionally 'missing' codes"
        ),
    )
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
    variables_to_stratify: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional STRONG AYA stratification: a mapping of variable names to "
            "lists of values or {'start', 'end'} ranges. The run analyses only "
            "the rows inside the requested subpopulation; the sample-size "
            "thresholds apply to the stratified rows."
        ),
    )

    @field_validator("time_col")
    @classmethod
    def validate_column_names(cls, v: str) -> str:
        """Validate that column names are non-empty strings."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Column names must be non-empty strings")
        return v.strip()

    @field_validator("outcome_col")
    @classmethod
    def validate_outcome_col(cls, v: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """A column name, or a mapping of one column name to its outcome codes."""
        name, codes = parse_outcome_col(v)
        return name if codes is None else {name: codes}

    @property
    def outcome_name(self) -> str:
        """The name of the outcome column, without its outcome codes."""
        return parse_outcome_col(self.outcome_col)[0]

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

    @field_validator("variables_to_stratify")
    @classmethod
    def validate_variables_to_stratify(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate the stratification mapping (values or ranges)."""
        if v is None:
            return None
        return validate_stratification_spec(v)

    @model_validator(mode="after")
    def validate_column_semantics(self) -> "CoxPHInput":
        """The time, outcome and explanatory columns must be distinct.

        The messages echo only the user-supplied column names.
        """
        if self.time_col == self.outcome_name:
            raise ValueError(f"time_col and outcome_col must be different columns, both are '{self.time_col}'")
        duplicates = sorted({var for var in self.expl_vars if self.expl_vars.count(var) > 1})
        if duplicates:
            raise ValueError(f"expl_vars contains duplicate variables: {duplicates}")
        clash = [var for var in self.expl_vars if var in (self.time_col, self.outcome_name)]
        if clash:
            raise ValueError(
                f"expl_vars must not name the time or outcome column; found {clash} "
                f"(time_col='{self.time_col}', outcome_col='{self.outcome_name}')"
            )
        return self

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
    outcome_col: Union[str, Dict[str, Any]],
    expl_vars: List[str],
    organization_ids: Optional[List[int]] = None,
    covariate_profiles: Optional[List[Dict[str, float]]] = None,
    variables_to_stratify: Optional[Dict[str, Any]] = None,
) -> CoxPHInput:
    """
    Validate Cox-PH algorithm input parameters.

    Parameters
    ----------
    time_col : str
        Name of the time column.
    outcome_col : Union[str, Dict[str, Any]]
        Name of the outcome column, or a mapping of it to its outcome codes.
    expl_vars : List[str]
        List of explanatory variable names.
    organization_ids : Optional[List[int]]
        List of organisation IDs to include.
    covariate_profiles : Optional[List[Dict[str, float]]]
        Optional covariate profiles for survival curves.
    variables_to_stratify : Optional[Dict[str, Any]]
        Optional STRONG AYA stratification mapping (values or ranges).

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
            variables_to_stratify=variables_to_stratify,
        )
    except ValidationError as e:
        # The formatted message never echoes the offending input: pydantic's
        # default message embeds it, and the message ends up in the container
        # log the researcher reads.
        raise UserInputError(f"Invalid Cox-PH input: {format_validation_error(e)}") from None


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


class SummedZResult(BaseModel):
    """
    Result of the ``compute_summed_z`` sub-task.

    Built on the node (no context) and returned as ``model_dump()``;
    validated by central with the runtime context (``expected_ids``,
    ``expl_vars``, ``time_col``). The context checks only run when the
    context provides them. ``extra="forbid"``: central and the partials
    ship in one image, so an unexpected key is contract drift, not forward
    compatibility.
    """

    model_config = ConfigDict(extra="forbid")

    organization_id: int = Field(strict=True)
    sum: Dict[str, float]
    sum_squares: Dict[str, float]
    times: Dict[str, Any]
    privacy_settings: Dict[str, Any]

    @model_validator(mode="after")
    def _matches_request(self, info: ValidationInfo) -> "SummedZResult":
        context = info.context or {}
        expected_ids = context.get("expected_ids")
        if expected_ids is not None and self.organization_id not in expected_ids:
            raise ValueError(f"organization_id {self.organization_id!r} is not one of the dispatched organisations.")
        expl_vars = context.get("expl_vars")
        if expl_vars is not None:
            for name in ("sum", "sum_squares"):
                missing = [v for v in expl_vars if v not in getattr(self, name)]
                if missing:
                    raise ValueError(f"'{name}' missing variables {missing}")
        time_col = context.get("time_col")
        if time_col is not None:
            times_df = pd.DataFrame.from_dict(self.times)
            if time_col not in times_df.columns or "freq" not in times_df.columns:
                raise ValueError(f"'times' must have columns '{time_col}' and 'freq', got {list(times_df.columns)}")
            if len(times_df) > 0:
                freqs = times_df["freq"].to_numpy()
                if not np.all(np.isfinite(freqs)) or np.any(freqs < 0):
                    raise ValueError("'freq' contains non-finite or negative values")
        return self


class IterationResult(BaseModel):
    """
    Result of the ``perform_iteration`` sub-task.

    Built on the node (no context) and returned as ``model_dump()``;
    validated by central with the runtime context (``expected_ids``,
    ``n_times``, ``n_covs``). The context checks only run when the context
    provides them. ``extra="forbid"`` as on ``SummedZResult``.
    """

    model_config = ConfigDict(extra="forbid")

    organization_id: int = Field(strict=True)
    agg1: List
    agg2: List
    agg3: List

    @model_validator(mode="after")
    def _shapes_and_finiteness(self, info: ValidationInfo) -> "IterationResult":
        context = info.context or {}
        expected_ids = context.get("expected_ids")
        if expected_ids is not None and self.organization_id not in expected_ids:
            raise ValueError(f"organization_id {self.organization_id!r} is not one of the dispatched organisations.")
        n_times = context.get("n_times")
        n_covs = context.get("n_covs")
        if n_times is None or n_covs is None:
            return self
        agg1 = np.asarray(self.agg1, dtype=float)
        if agg1.ndim != 1 or len(agg1) != n_times:
            raise ValueError(f"agg1 has length {len(agg1)}, expected {n_times}")
        if not np.all(np.isfinite(agg1)):
            raise ValueError("agg1 contains non-finite values")
        agg2 = np.asarray(self.agg2, dtype=float)
        if agg2.shape != (n_times, n_covs):
            raise ValueError(f"agg2 has shape {agg2.shape}, expected ({n_times}, {n_covs})")
        if not np.all(np.isfinite(agg2)):
            raise ValueError("agg2 contains non-finite values")
        agg3 = np.asarray(self.agg3, dtype=float)
        if agg3.shape != (n_times, n_covs, n_covs):
            raise ValueError(f"agg3 has shape {agg3.shape}, expected ({n_times}, {n_covs}, {n_covs})")
        # Stricter than 2.0.0, which did not check agg3 for finiteness.
        if not np.all(np.isfinite(agg3)):
            raise ValueError("agg3 contains non-finite values")
        return self
