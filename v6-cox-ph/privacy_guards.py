"""
Privacy guards for the federated Cox-PH partial functions.

These guards are pure functions (DataFrame/array in, DataFrame/array out)
so they can be unit-tested without vantage6. They are called from all three
partial functions in the same order::

    ensure_spawned_by_central -> load_privacy_settings
    -> (validate_expl_vars, where expl_vars are used) -> drop_incomplete_rows
    -> check_sample_size -> (validate_iteration_input, perform_iteration only)
    -> prepare_time_column -> (function-specific work)

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

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import jwt
from vantage6.algorithm.tools.exceptions import (
    AlgorithmError,
    PrivacyViolation,
    UserInputError,
)
from vantage6_strongaya_general.miscellaneous import safe_log

# Environment variable names
ENV_SAMPLE_SIZE_THRESHOLD = "SAMPLE_SIZE_THRESHOLD"
ENV_TIME_BIN_WIDTH = "COXPH_TIME_BIN_WIDTH"
ENV_MIN_RISK_SET_CHANGE = "COXPH_MIN_RISK_SET_CHANGE"

# Defaults
DEFAULT_SAMPLE_SIZE_THRESHOLD = 10
DEFAULT_MIN_RISK_SET_CHANGE = 5


@dataclass(frozen=True)
class PrivacySettings:
    """Privacy-related settings read from the node environment."""

    sample_size_threshold: int
    time_bin_width: Optional[float]
    min_risk_set_change: int


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
    if sample_size_threshold is None or sample_size_threshold <= 0:
        raise UserInputError(
            f"{ENV_SAMPLE_SIZE_THRESHOLD} must be a positive integer, got " f"{sample_size_threshold!r}"
        )

    min_risk_set_change = get_env_var(
        ENV_MIN_RISK_SET_CHANGE,
        default=str(DEFAULT_MIN_RISK_SET_CHANGE),
        as_type="int",
    )
    if min_risk_set_change is None or min_risk_set_change < 1:
        raise UserInputError(
            f"{ENV_MIN_RISK_SET_CHANGE} must be a positive integer (>= 1), got " f"{min_risk_set_change!r}"
        )

    if sample_size_threshold + 1 < min_risk_set_change:
        raise UserInputError(
            f"{ENV_SAMPLE_SIZE_THRESHOLD} ({sample_size_threshold}) + 1 must be "
            f">= {ENV_MIN_RISK_SET_CHANGE} ({min_risk_set_change}); otherwise "
            f"a node can pass the sample-size threshold but fail to guarantee "
            f"risk sets of size >= k."
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
        if time_bin_width <= 0:
            raise UserInputError(f"{ENV_TIME_BIN_WIDTH} must be a positive float, got " f"{time_bin_width}")

    return PrivacySettings(
        sample_size_threshold=sample_size_threshold,
        time_bin_width=time_bin_width,
        min_risk_set_change=min_risk_set_change,
    )


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
        safe_log("info", "Skipping parent-task guard: mock client detected.")
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
    safe_log("info", "Parent-task guard passed: task has a parent.")


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
        safe_log("warning", f"Sample size threshold not met: row count does not exceed {threshold}.")
        return False

    if outcome_col is not None:
        if outcome_col not in df.columns:
            raise UserInputError(f"Outcome column '{outcome_col}' not found in data columns.")
        n_events = int((df[outcome_col] == 1).sum())
        if n_events <= threshold:
            safe_log("warning", f"Sample size threshold not met: event count does not exceed {threshold}.")
            return False

    return True


def validate_iteration_input(
    beta,
    unique_time_events,
    expl_vars,
    settings: PrivacySettings,
) -> tuple[np.ndarray, list[float]]:
    """Validate the wire input to ``perform_iteration``.

    Parameters
    ----------
    beta : array-like
        Current coefficient vector.
    unique_time_events : list[float]
        The sorted event-time grid.
    expl_vars : list[str]
        Explanatory variable names (used to determine expected length).
    settings : PrivacySettings
        Loaded privacy settings (used for the bin-grid check).

    Returns
    -------
    tuple[np.ndarray, list[float]]
        The validated ``beta`` as a numpy array and the validated
        ``unique_time_events`` as a list of floats.

    Raises
    ------
    UserInputError
        If ``beta`` is not finite or has the wrong length, or if
        ``unique_time_events`` is unsorted, contains duplicates/NaNs, or is
        off the bin grid when binning is active.
    """
    try:
        beta_arr = np.asarray(beta, dtype=float)
    except (TypeError, ValueError) as e:
        raise UserInputError(f"beta could not be converted to float: {e}") from e

    if beta_arr.ndim != 1 or len(beta_arr) != len(expl_vars):
        raise UserInputError(
            f"beta must have length {len(expl_vars)} (one per explanatory " f"variable), got shape {beta_arr.shape}"
        )
    if not np.all(np.isfinite(beta_arr)):
        raise UserInputError("beta contains non-finite values (NaN or inf).")

    if unique_time_events is None:
        raise UserInputError("unique_time_events must not be None.")

    try:
        grid = [float(t) for t in unique_time_events]
    except (TypeError, ValueError) as e:
        raise UserInputError(f"unique_time_events must be a list of numbers: {e}") from e

    if len(grid) == 0:
        raise UserInputError("unique_time_events must not be empty.")

    if not np.all(np.isfinite(grid)):
        raise UserInputError("unique_time_events contains non-finite values.")

    if grid != sorted(grid):
        raise UserInputError("unique_time_events must be sorted in ascending order.")

    if len(set(grid)) != len(grid):
        raise UserInputError("unique_time_events must not contain duplicates.")

    # When binning is active, every grid point must lie on the bin grid.
    width = settings.time_bin_width
    if width is not None and width > 0:
        for t in grid:
            binned = np.floor(t / width) * width
            if not np.isclose(binned, t):
                raise UserInputError(
                    f"unique_time_events contains time {t} that is not on the " f"bin grid (width={width})."
                )

    return beta_arr, grid


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


def guarded_risk_set_masks(times: pd.Series, grid: list[float], k: int) -> list[np.ndarray]:
    """Compute risk-set masks with the minimum-change ("jump") guard.

    Walking the grid from smallest to largest time, if moving from ``t_i`` to
    ``t_{i+1}`` would remove fewer than ``k`` (but more than zero)
    individuals from the risk set, the previous (larger) risk set is held.
    Consequently consecutive shared aggregates differ by 0 or by at least
    ``k`` individuals. When ``k <= 1`` the guard is a no-op and the masks are
    the plain ``times >= t`` masks.

    Returns
    -------
    list[np.ndarray]
        One boolean mask per grid point (aligned with ``grid``).
    """
    times_arr = pd.to_numeric(times, errors="coerce").to_numpy()

    if k is None or k <= 1:
        return [times_arr >= t for t in grid]

    masks: list[np.ndarray] = []
    current = times_arr >= grid[0]
    masks.append(current)
    for t in grid[1:]:
        cand = times_arr >= t
        removed = int(current.sum() - cand.sum())
        if 0 < removed < k:
            # Hold the previous risk set.
            masks.append(current)
        else:
            current = cand
            masks.append(current)
    return masks


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
