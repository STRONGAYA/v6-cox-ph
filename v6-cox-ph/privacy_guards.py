"""
Privacy guards for the federated Cox-PH partial functions.

These guards are pure functions (DataFrame/array in, DataFrame/array out)
so they can be unit-tested without vantage6. All partial functions run the
shared ``prepare_node_data`` step, in this order::

    ensure_spawned_by_central -> load_privacy_settings
    -> (validate_expl_vars, where expl_vars are used) -> drop_incomplete_rows
    -> validate_survival_columns -> select_rows (hook) -> check_sample_size
    -> (validate_iteration_input, perform_iteration only)
    -> prepare_time_column -> (function-specific work; perform_iteration
    builds its risk sets with guarded_risk_set_aggregates)

The settings are read from node environment variables (``algorithm_env``)
via ``get_env_var``:

- ``SAMPLE_SIZE_THRESHOLD`` (int, default 10) — a node is excluded / raises
  when its rows or events are not strictly greater than this threshold.
- ``COXPH_TIME_BIN_WIDTH`` (float, optional) — when set, event times are
  coarsened to a regular grid ``floor(t / w) * w``.
- ``COXPH_MIN_RISK_SET_CHANGE`` (int, default 5) — the minimum number of
  individuals by which consecutive shared risk-set aggregates must differ.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import jwt
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from vantage6.algorithm.tools.exceptions import (
    AlgorithmError,
    PrivacyViolation,
    UserInputError,
)

from .miscellaneous import format_validation_error
from vantage6.algorithm.tools.util import info, warn

# Environment variable names
ENV_SAMPLE_SIZE_THRESHOLD = "SAMPLE_SIZE_THRESHOLD"
ENV_TIME_BIN_WIDTH = "COXPH_TIME_BIN_WIDTH"
ENV_MIN_RISK_SET_CHANGE = "COXPH_MIN_RISK_SET_CHANGE"

# Defaults
DEFAULT_SAMPLE_SIZE_THRESHOLD = 10
DEFAULT_MIN_RISK_SET_CHANGE = 5


class PrivacySettings(BaseModel):
    """Privacy-related settings read from the node environment.

    A frozen Pydantic model: the bounds (positive threshold, ``k >= 1``,
    positive bin width) and the threshold-covers-``k`` rule are declared on
    the fields instead of hand-rolled. The values still come in through
    vantage6's ``get_env_var`` (which decodes base32-encoded
    ``algorithm_env``); ``pydantic-settings`` would read ``os.environ``
    directly and bypass that decoding, so it is not used.
    """

    model_config = ConfigDict(frozen=True)

    sample_size_threshold: int = Field(gt=0)
    time_bin_width: Optional[float] = Field(default=None, gt=0)
    min_risk_set_change: int = Field(ge=1)

    @model_validator(mode="after")
    def _threshold_covers_k(self) -> "PrivacySettings":
        if self.sample_size_threshold + 1 < self.min_risk_set_change:
            raise ValueError(
                f"SAMPLE_SIZE_THRESHOLD ({self.sample_size_threshold}) + 1 must be "
                f">= COXPH_MIN_RISK_SET_CHANGE ({self.min_risk_set_change}); otherwise "
                f"a node can pass the sample-size threshold but fail to guarantee "
                f"risk sets of size >= k."
            )
        return self


def load_privacy_settings() -> PrivacySettings:
    """Load privacy settings from node environment variables.

    Uses ``get_env_var`` so that base32-encoded ``algorithm_env`` values are
    decoded the same way as in the rest of vantage6.

    Returns
    -------
    PrivacySettings
        Validated settings. ``time_bin_width`` is ``None`` when binning is
        disabled.

    Raises
    ------
    UserInputError
        If a setting value is not a positive number.
    """
    from vantage6.algorithm.tools.util import get_env_var

    sample_size_threshold = get_env_var(
        ENV_SAMPLE_SIZE_THRESHOLD,
        default=str(DEFAULT_SAMPLE_SIZE_THRESHOLD),
        as_type="int",
    )
    min_risk_set_change = get_env_var(
        ENV_MIN_RISK_SET_CHANGE,
        default=str(DEFAULT_MIN_RISK_SET_CHANGE),
        as_type="int",
    )
    time_bin_width_raw = get_env_var(ENV_TIME_BIN_WIDTH, default=None)
    time_bin_width: Optional[float] = None
    if time_bin_width_raw is not None and str(time_bin_width_raw).strip() != "":
        try:
            time_bin_width = float(time_bin_width_raw)
        except (TypeError, ValueError) as e:
            raise UserInputError(
                f"{ENV_TIME_BIN_WIDTH} must be a positive float, got " f"{time_bin_width_raw!r}"
            ) from e

    try:
        return PrivacySettings(
            sample_size_threshold=sample_size_threshold,
            time_bin_width=time_bin_width,
            min_risk_set_change=min_risk_set_change,
        )
    except ValidationError as e:
        # The offending values are node configuration, not data; echoing them
        # is what the current messages do and is safe.
        raise UserInputError(f"Invalid privacy settings: {format_validation_error(e)}") from e


def ensure_spawned_by_central(client) -> None:
    """Refuse to run a partial unless it was spawned by the ``central`` task.

    The container JWT ``sub`` claim carries the running ``task_id``. We look
    up that task and require a non-empty ``parent`` field: the vantage6
    server only sets ``parent_id`` for container-created sub-tasks, so a user
    task (called directly) has ``parent = None`` and is refused.

    The guard is skipped only for ``MockAlgorithmClient`` (in-process tests).
    Any other client without a decodable token raises ``AlgorithmError``
    (fail closed). Any failure to obtain the task also fails closed.

    Raises
    ------
    PrivacyViolation
        If the running task has no parent (i.e. it was created directly by a
        user rather than by ``central``).
    AlgorithmError
        If the token is missing or cannot be decoded, or if the task lookup
        itself fails (fail closed).
    """
    from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

    if isinstance(client, MockAlgorithmClient):
        info("Skipping parent-task guard: mock client detected.")
        return

    token = getattr(client, "_access_token", None)
    if token is None:
        raise AlgorithmError(
            "No access token found on client; cannot verify parent task. " "Refusing to proceed (fail closed)."
        )

    try:
        payload = jwt.decode(
            token,
            options={
                "verify_signature": False,
                "verify_exp": False,
                "verify_sub": False,
            },
        )
        task_id = payload["sub"]["task_id"]
    except Exception as e:
        raise AlgorithmError(f"Could not decode task identity from container token: {e}") from e

    try:
        task = client.task.get(task_id)
    except Exception as e:
        raise AlgorithmError(f"Could not look up task {task_id} for parent-task guard: {e}") from e

    parent = task.get("parent") if isinstance(task, dict) else None
    if not parent:
        raise PrivacyViolation(
            "Partial functions may only be invoked as sub-tasks of the "
            "'central' function. Direct invocation is not permitted."
        )
    info("Parent-task guard passed: task has a parent.")


def select_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Marked hook for project row selection.

    A project layer may replace this function to select the rows its
    protocol analyses (an eligibility filter, an inlier filter, …). The
    hook is a no-op here and runs **before** ``check_sample_size``, so any
    rows it removes count against the threshold. Implementations must
    never add rows and must remove rows only — otherwise the aggregates
    across partials describe different row sets.
    """
    return df


def prepare_node_data(
    client,
    df: pd.DataFrame,
    time_col: str,
    outcome_col: Optional[str],
    expl_vars: Optional[list],
    *,
    need_outcome: bool,
) -> tuple[pd.DataFrame, PrivacySettings, bool]:
    """Run the shared node-side guard sequence once.

    Order (do not reorder or skip):

        ``ensure_spawned_by_central`` -> ``load_privacy_settings``
        -> ``validate_expl_vars`` (when ``expl_vars`` are given)
        -> ``drop_incomplete_rows`` -> ``validate_survival_columns``
        -> ``select_rows`` (hook) -> ``check_sample_size``

    Every partial calls this so that all aggregates central combines come
    from the same row set on each node. The threshold failure behaviour
    (return a marker vs. raise) stays with the caller.

    Parameters
    ----------
    client : AlgorithmClient
        The algorithm client (used only by the parent-task guard).
    df : pd.DataFrame
        The node's data.
    time_col : str
        Name of the time column.
    outcome_col : str | None
        Name of the outcome column, or ``None`` when the partial does not
        receive it (``perform_iteration`` until the outcome column joins
        the wire contract).
    expl_vars : list | None
        Explanatory variable names, or an empty list/None when the partial
        does not use covariates.
    need_outcome : bool
        Whether the sample-size threshold also counts events (rows and
        events must both exceed the threshold).

    Returns
    -------
    tuple[pd.DataFrame, PrivacySettings, bool]
        The prepared frame, the loaded privacy settings, and whether the
        sample-size threshold was met.
    """
    ensure_spawned_by_central(client)
    settings = load_privacy_settings()

    if expl_vars:
        validate_expl_vars(df, expl_vars, time_col, outcome_col)

    cols = [time_col]
    if outcome_col is not None:
        cols.append(outcome_col)
    if expl_vars:
        cols.extend(expl_vars)
    df = drop_incomplete_rows(df, cols)

    validate_survival_columns(df, time_col, outcome_col)

    df = select_rows(df)

    outcome = outcome_col if need_outcome else None
    threshold_met = check_sample_size(df, outcome, settings)

    return df, settings, threshold_met


def guarded_risk_set_aggregates(
    times: pd.Series,
    grid: list[float],
    k: int,
    X: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised risk-set aggregates with the minimum-change ("jump") guard.

    Rows are bucketed by grid point with ``searchsorted`` and summed per
    bucket with ``bincount`` (S0, S1 per covariate, S2 per covariate pair);
    reverse cumulative sums turn the bucket sums into risk-set sums. The
    jump guard becomes an *effective grid index* computed from the bucket
    counts alone: walking the grid while keeping the index ``h`` of the held
    risk set, ``removed = |R(h)| - |R(j+1)|``; if ``0 < removed < k`` the
    previous set is held (``h`` stays), otherwise ``h = j+1``. This matches
    the previous mask implementation exactly (see the reference copy in
    ``tests/unit/reference_risk_sets.py``).

    Complexity: O(N·p²) compute and O(N·p + T·p²) memory, instead of T×N
    boolean masks.

    Parameters
    ----------
    times : pd.Series
        The node's (binned, tail-censored) times.
    grid : list[float]
        The sorted event-time grid.
    k : int
        The minimum risk-set change; ``k <= 1`` disables the guard.
    X : np.ndarray
        The (standardised) covariate matrix, shape (N, p).
    beta : np.ndarray
        The coefficient vector, shape (p,).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``agg1`` (T,), ``agg2`` (T, p) and ``agg3`` (T, p, p): the shared
        risk-set sums S0(t) = sum exp(beta.x), S1(t) = sum x exp(beta.x) and
        S2(t) = sum x x^T exp(beta.x) per grid point.
    """
    grid_arr = np.asarray(grid, dtype=float)
    n_times = len(grid_arr)
    n_covs = X.shape[1]

    times_arr = pd.to_numeric(times, errors="coerce").to_numpy(dtype=float)
    weights = np.exp(X @ beta)

    # Bucket each row at the largest grid point that does not exceed its
    # time; a row in bucket b is in the risk sets of grid points 0..b.
    # Rows before the first grid point land in bucket -1 (in no risk set).
    finite = np.isfinite(times_arr)
    bucket = np.searchsorted(grid_arr, times_arr, side="right") - 1
    in_grid = (bucket >= 0) & finite
    bucket_clipped = np.where(in_grid, bucket, 0)

    # Per-bucket sums; reverse cumulative sums give the risk-set sums.
    counts = np.bincount(bucket_clipped[in_grid], minlength=n_times).astype(float)

    s0_bucket = np.bincount(bucket_clipped[in_grid], weights=weights[in_grid], minlength=n_times)
    s0 = np.cumsum(s0_bucket[::-1])[::-1]
    count_r = np.cumsum(counts[::-1])[::-1]  # |R(t_j)| per grid point

    s1_bucket = np.empty((n_times, n_covs))
    for c in range(n_covs):
        s1_bucket[:, c] = np.bincount(bucket_clipped[in_grid], weights=(weights * X[:, c])[in_grid], minlength=n_times)
    s1 = np.cumsum(s1_bucket[::-1], axis=0)[::-1]

    s2_bucket = np.empty((n_times, n_covs, n_covs))
    for c1 in range(n_covs):
        for c2 in range(c1, n_covs):
            s2_bucket[:, c1, c2] = np.bincount(
                bucket_clipped[in_grid], weights=(weights * X[:, c1] * X[:, c2])[in_grid], minlength=n_times
            )
            if c2 != c1:
                s2_bucket[:, c2, c1] = s2_bucket[:, c1, c2]
    s2 = np.cumsum(s2_bucket[::-1], axis=0)[::-1]

    # Effective grid index from the counts alone (the jump guard).
    if k is None or k <= 1:
        effective = np.arange(n_times)
    else:
        effective = np.empty(n_times, dtype=int)
        effective[0] = 0
        held = 0
        for j in range(1, n_times):
            removed = count_r[held] - count_r[j]
            if not (0 < removed < k):
                held = j
            effective[j] = held

    agg1 = s0[effective]
    agg2 = s1[effective]
    agg3 = s2[effective]
    return agg1, agg2, agg3


def validate_survival_columns(df: pd.DataFrame, time_col: str, outcome_col: Optional[str]) -> None:
    """Validate the survival columns on the prepared frame.

    The time column must be numeric, finite and non-negative; the outcome
    column must be binary (0/1 or boolean). A missing column raises
    ``UserInputError`` — it is a user-input problem that no fallback may
    hide. Error messages name columns and dtypes, never data values.

    Raises
    ------
    UserInputError
        On any validation failure.
    """
    if time_col not in df.columns:
        raise UserInputError(f"Time column '{time_col}' not found in data columns.")

    time_values = df[time_col]
    if not pd.api.types.is_numeric_dtype(time_values):
        raise UserInputError(f"Time column '{time_col}' must be numeric, got dtype {time_values.dtype}.")
    if not np.isfinite(time_values.to_numpy(dtype=float)).all():
        raise UserInputError(f"Time column '{time_col}' contains non-finite values.")
    if (time_values < 0).any():
        raise UserInputError(f"Time column '{time_col}' contains negative values.")

    if outcome_col is not None:
        if outcome_col not in df.columns:
            raise UserInputError(f"Outcome column '{outcome_col}' not found in data columns.")
        outcome_values = df[outcome_col]
        if not (pd.api.types.is_bool_dtype(outcome_values) or pd.api.types.is_numeric_dtype(outcome_values)):
            raise UserInputError(
                f"Outcome column '{outcome_col}' must be numeric or boolean, " f"got dtype {outcome_values.dtype}."
            )
        if not pd.api.types.is_bool_dtype(outcome_values) and not outcome_values.isin([0, 1]).all():
            raise UserInputError(
                f"Outcome column '{outcome_col}' must only contain 0 and 1 "
                f"(booleans are allowed); recode the data before calling."
            )


def check_sample_size(df: pd.DataFrame, outcome_col: Optional[str], settings: PrivacySettings) -> bool:
    """Check whether the node meets the sample-size threshold.

    Both rows and events must be strictly greater than the threshold. The
    warning on failure names the threshold but never the actual count: the
    container log is returned to the researcher with the run, and a
    below-threshold count is exactly what the threshold protects.

    Parameters
    ----------
    df : pd.DataFrame
        The node's data.
    outcome_col : str | None
        The outcome column name. When ``None`` (e.g. ``perform_iteration``
        where the outcome column is not available), only rows are checked.
    settings : PrivacySettings
        Loaded privacy settings.

    Returns
    -------
    bool
        ``True`` when rows > threshold and (when checked) events > threshold.
    """
    threshold = settings.sample_size_threshold
    n_rows = len(df)
    if n_rows <= threshold:
        warn(f"Sample size threshold not met: row count does not exceed {threshold}.")
        return False

    if outcome_col is not None:
        if outcome_col not in df.columns:
            raise UserInputError(f"Outcome column '{outcome_col}' not found in data columns.")
        n_events = int((df[outcome_col] == 1).sum())
        if n_events <= threshold:
            warn(f"Sample size threshold not met: event count does not exceed " f"{threshold}.")
            return False

    return True


def validate_iteration_input(
    beta,
    centre,
    scale,
    unique_time_events,
    expl_vars,
    settings: PrivacySettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Validate the wire input to ``perform_iteration``.

    A thin wrapper around the ``IterationInput`` Pydantic model: the model
    holds the wire shape (finite vectors, strictly positive scale, a
    strictly increasing grid, per-variable lengths and the bin-grid rule);
    this wrapper keeps the historical signature and return types
    (numpy arrays and a list of floats) and wraps ``ValidationError``
    into ``UserInputError``.

    Parameters
    ----------
    beta : array-like
        Current coefficient vector (standardised space).
    centre : array-like
        Pooled covariate means over the event cases.
    scale : array-like
        Pooled covariate standard deviations over the event cases.
    unique_time_events : list[float]
        The sorted event-time grid.
    expl_vars : list[str]
        Explanatory variable names (determine the expected lengths).
    settings : PrivacySettings
        Loaded privacy settings (used for the bin-grid check).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]
        The validated ``beta``, ``centre`` and ``scale`` as numpy arrays and
        the validated ``unique_time_events`` as a list of floats.

    Raises
    ------
    UserInputError
        On any validation failure reported by ``IterationInput``.
    """
    from .miscellaneous import IterationInput

    try:
        IterationInput.model_validate(
            {"beta": beta, "centre": centre, "scale": scale, "unique_time_events": unique_time_events},
            context={"expl_vars": expl_vars, "settings": settings},
        )
    except ValidationError as e:
        raise UserInputError(format_validation_error(e)) from e

    beta_arr = np.asarray(beta, dtype=float)
    centre_arr = np.asarray(centre, dtype=float)
    scale_arr = np.asarray(scale, dtype=float)
    return beta_arr, centre_arr, scale_arr, [float(t) for t in unique_time_events]


def bin_times(times: pd.Series, width: float | None) -> pd.Series:
    """Coarsen event times to a regular grid ``floor(t / w) * w``.

    When ``width`` is ``None`` or non-positive the times are returned
    unchanged.
    """
    if width is None or width <= 0:
        return times
    return pd.Series(
        np.floor(times.to_numpy(dtype=float) / width) * width,
        index=times.index,
    )


def tail_cutoff(times: pd.Series, k: int) -> float | None:
    """Return the k-th largest time, used as the tail-censoring cut-off.

    Rows with ``time > t_cut`` are administratively censored (their time is
    clamped to ``t_cut``), which guarantees that the largest shared risk set
    contains at least ``k`` individuals.

    Returns ``None`` when ``k <= 1`` (guard disabled).

    Raises ``PrivacyViolation`` when ``k > 1`` and fewer than ``k`` valid
    times remain — a node should have been excluded by the sample-size
    threshold before this is reached, so this is a defence-in-depth
    assertion.
    """
    if k is None or k <= 1:
        return None
    vals = pd.to_numeric(times, errors="coerce").dropna().to_numpy()
    if len(vals) < k:
        raise PrivacyViolation(
            f"Tail censoring requires at least {k} valid times but only " f"{len(vals)} are available."
        )
    return float(np.sort(vals)[-k])


def prepare_time_column(
    df: pd.DataFrame,
    time_col: str,
    settings: PrivacySettings,
    outcome_col: str | None = None,
) -> pd.DataFrame:
    """Bin and tail-censor the time column of a node's DataFrame.

    The transformation is applied in this order: bin the time column to the
    configured grid, compute the tail cut-off from the (binned) times, clamp
    rows past the cut-off to ``t_cut`` and, when an outcome column is given,
    set their event to 0 (administrative censoring).

    The input is not modified; a copy is returned.
    """
    out = df.copy()

    width = settings.time_bin_width
    if width is not None and width > 0:
        out[time_col] = bin_times(out[time_col], width)

    t_cut = tail_cutoff(out[time_col], settings.min_risk_set_change)
    if t_cut is not None:
        clamp_mask = out[time_col] > t_cut
        if clamp_mask.any():
            out.loc[clamp_mask, time_col] = t_cut
            if outcome_col is not None and outcome_col in out.columns:
                out.loc[clamp_mask, outcome_col] = 0

    return out


def validate_expl_vars(df: pd.DataFrame, expl_vars: list, time_col: str, outcome_col: str | None) -> None:
    """Validate explanatory variables on the node (FR-B3).

    Checks that every name in ``expl_vars`` is a column of ``df``, does not
    overlap with ``time_col`` or ``outcome_col``, and is numeric. Column
    names are user input and may be echoed in the error message; data
    values are never echoed.

    Raises
    ------
    UserInputError
        On any validation failure.
    """
    for var in expl_vars:
        if var not in df.columns:
            raise UserInputError(f"Explanatory variable '{var}' not found in data columns.")
        if var == time_col:
            raise UserInputError(f"Explanatory variable '{var}' must not equal time_col " f"'{time_col}'.")
        if outcome_col is not None and var == outcome_col:
            raise UserInputError(f"Explanatory variable '{var}' must not equal outcome_col " f"'{outcome_col}'.")
        if not pd.api.types.is_numeric_dtype(df[var]):
            raise UserInputError(f"Explanatory variable '{var}' must be numeric, got " f"dtype {df[var].dtype}.")


def drop_incomplete_rows(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Drop rows with NaN in any of the specified columns (FR-B4).

    Applied identically in all three partials before ``check_sample_size``
    so that thresholds apply to the analysed rows. The input is not
    modified; a copy is returned.
    """
    existing_cols = [c for c in cols if c in df.columns]
    if not existing_cols:
        return df.copy()
    return df.dropna(subset=existing_cols).copy()
