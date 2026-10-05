"""
Unit tests for the privacy guards module.

Covers settings parsing, the sample-size threshold, the parent-task guard,
and iteration-input validation. The time-binning / tail-censoring / jump-guard
functions are tested here as well (added in a later step).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import jwt

from vantage6.algorithm.tools.exceptions import (
    AlgorithmError,
    PrivacyViolation,
    UserInputError,
)

# Add the algorithm module to the path
algorithm_path = Path(__file__).parent.parent.parent / "v6-cox-ph"
sys.path.insert(0, str(algorithm_path))

from privacy_guards import (  # noqa: E402
    DEFAULT_MIN_RISK_SET_CHANGE,
    DEFAULT_SAMPLE_SIZE_THRESHOLD,
    PrivacySettings,
    bin_times,
    check_sample_size,
    drop_incomplete_rows,
    ensure_spawned_by_central,
    guarded_risk_set_masks,
    load_privacy_settings,
    prepare_node_data,
    prepare_time_column,
    tail_cutoff,
    validate_expl_vars,
    validate_iteration_input,
    validate_survival_columns,
)
import privacy_guards  # noqa: E402


@pytest.mark.unit
class TestLoadPrivacySettings:
    """Tests for load_privacy_settings."""

    def test_defaults(self, monkeypatch):
        """Default settings when no env vars are set."""
        monkeypatch.delenv("SAMPLE_SIZE_THRESHOLD", raising=False)
        monkeypatch.delenv("COXPH_TIME_BIN_WIDTH", raising=False)
        monkeypatch.delenv("COXPH_MIN_RISK_SET_CHANGE", raising=False)
        s = load_privacy_settings()
        assert s.sample_size_threshold == DEFAULT_SAMPLE_SIZE_THRESHOLD
        assert s.min_risk_set_change == DEFAULT_MIN_RISK_SET_CHANGE
        assert s.time_bin_width is None

    def test_custom_values(self, monkeypatch):
        """Custom values from env vars."""
        monkeypatch.setenv("SAMPLE_SIZE_THRESHOLD", "15")
        monkeypatch.setenv("COXPH_TIME_BIN_WIDTH", "10")
        monkeypatch.setenv("COXPH_MIN_RISK_SET_CHANGE", "8")
        s = load_privacy_settings()
        assert s.sample_size_threshold == 15
        assert s.min_risk_set_change == 8
        assert s.time_bin_width == 10.0

    def test_invalid_threshold(self, monkeypatch):
        """Non-positive threshold raises UserInputError."""
        monkeypatch.setenv("SAMPLE_SIZE_THRESHOLD", "0")
        with pytest.raises(UserInputError):
            load_privacy_settings()

    def test_invalid_bin_width(self, monkeypatch):
        """Non-positive bin width raises UserInputError."""
        monkeypatch.setenv("COXPH_TIME_BIN_WIDTH", "0")
        with pytest.raises(UserInputError):
            load_privacy_settings()

    def test_invalid_k(self, monkeypatch):
        """k < 1 raises UserInputError."""
        monkeypatch.setenv("COXPH_MIN_RISK_SET_CHANGE", "0")
        with pytest.raises(UserInputError):
            load_privacy_settings()

    def test_invalid_k_negative(self, monkeypatch):
        monkeypatch.setenv("COXPH_MIN_RISK_SET_CHANGE", "-1")
        with pytest.raises(UserInputError):
            load_privacy_settings()

    def test_threshold_plus_one_less_than_k(self, monkeypatch):
        """T + 1 < k raises UserInputError (FR-B2)."""
        monkeypatch.setenv("SAMPLE_SIZE_THRESHOLD", "3")
        monkeypatch.setenv("COXPH_MIN_RISK_SET_CHANGE", "5")
        with pytest.raises(UserInputError, match="must be"):
            load_privacy_settings()


@pytest.mark.unit
class TestCheckSampleSize:
    """Tests for check_sample_size."""

    @pytest.fixture
    def settings(self):
        return PrivacySettings(
            sample_size_threshold=10,
            time_bin_width=None,
            min_risk_set_change=5,
        )

    def test_sufficient_rows_and_events(self, settings):
        df = pd.DataFrame({"event": [1] * 20 + [0] * 5})
        assert check_sample_size(df, "event", settings) is True

    def test_insufficient_rows(self, settings):
        df = pd.DataFrame({"event": [1] * 5})
        assert check_sample_size(df, "event", settings) is False

    def test_insufficient_events(self, settings):
        df = pd.DataFrame({"event": [1] * 5 + [0] * 50})
        assert check_sample_size(df, "event", settings) is False

    def test_rows_only_when_no_outcome(self, settings):
        """When outcome_col is None, only rows are checked."""
        df = pd.DataFrame({"x": range(20)})
        assert check_sample_size(df, None, settings) is True
        df_small = pd.DataFrame({"x": range(5)})
        assert check_sample_size(df_small, None, settings) is False

    def test_boundary_strictly_greater(self, settings):
        """Threshold is strict: exactly threshold events fails."""
        df = pd.DataFrame({"event": [1] * 10 + [0] * 50})
        # 10 events, threshold 10 -> 10 > 10 is False
        assert check_sample_size(df, "event", settings) is False

    def test_missing_outcome_col_raises(self, settings):
        """FR-B5: missing outcome_col raises UserInputError."""
        df = pd.DataFrame({"x": range(20)})
        with pytest.raises(UserInputError, match="Outcome column"):
            check_sample_size(df, "event", settings)


@pytest.mark.unit
class TestEnsureSpawnedByCentral:
    """Tests for the parent-task guard."""

    def _make_client(self, token=None, task_dict=None, raises=None):
        """Build a fake client with controllable token/task.get."""

        class FakeTask:
            def get(self, task_id):
                if raises is not None:
                    raise raises
                return task_dict

        class FakeClient:
            def __init__(self):
                self.task = FakeTask()
                if token is not None:
                    self._access_token = token

        return FakeClient()

    def _make_token(self, task_id):
        payload = {"sub": {"task_id": task_id}}
        return jwt.encode(payload, key="unused", algorithm="HS256")

    def test_mock_client_skipped(self):
        """A MockAlgorithmClient is skipped (in-process test)."""
        from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

        client = MockAlgorithmClient(
            datasets=[[{"database": pd.DataFrame({"x": [1]}), "db_type": "csv"}]],
            module="v6-cox-ph",
            organization_ids=[1],
        )
        # Should not raise
        ensure_spawned_by_central(client)

    def test_non_mock_client_without_token_fails_closed(self):
        """FR-B1: a non-mock client without _access_token raises AlgorithmError."""
        client = self._make_client(token=None)
        with pytest.raises(AlgorithmError, match="No access token"):
            ensure_spawned_by_central(client)

    def test_direct_invocation_rejected(self):
        """A user task (parent is None) raises PrivacyViolation."""
        token = self._make_token(task_id=7)
        client = self._make_client(token=token, task_dict={"parent": None})
        with pytest.raises(PrivacyViolation):
            ensure_spawned_by_central(client)

    def test_subtask_passes(self):
        """A task with a parent passes the guard."""
        token = self._make_token(task_id=7)
        client = self._make_client(token=token, task_dict={"parent": {"id": 3}})
        ensure_spawned_by_central(client)

    def test_task_lookup_failure_fails_closed(self):
        """If task.get raises, the guard fails closed with AlgorithmError."""
        token = self._make_token(task_id=7)
        client = self._make_client(token=token, raises=RuntimeError("connection error"))
        with pytest.raises(AlgorithmError):
            ensure_spawned_by_central(client)

    def test_bad_token_fails_closed(self):
        """An undecodable token fails closed with AlgorithmError."""
        client = self._make_client(token="not-a-jwt", task_dict={"parent": None})
        with pytest.raises(AlgorithmError):
            ensure_spawned_by_central(client)


@pytest.mark.unit
class TestValidateIterationInput:
    """Tests for validate_iteration_input."""

    @pytest.fixture
    def settings(self):
        return PrivacySettings(
            sample_size_threshold=10,
            time_bin_width=None,
            min_risk_set_change=5,
        )

    @pytest.fixture
    def settings_binned(self):
        return PrivacySettings(
            sample_size_threshold=10,
            time_bin_width=10.0,
            min_risk_set_change=5,
        )

    def test_valid_input(self, settings):
        beta = [0.1, -0.2]
        grid = [1.0, 2.0, 3.0]
        b, g = validate_iteration_input(beta, grid, ["a", "b"], settings)
        np.testing.assert_array_equal(b, [0.1, -0.2])
        assert g == [1.0, 2.0, 3.0]

    def test_beta_wrong_length(self, settings):
        with pytest.raises(UserInputError):
            validate_iteration_input([0.1], [1.0], ["a", "b"], settings)

    def test_beta_non_finite(self, settings):
        with pytest.raises(UserInputError):
            validate_iteration_input([0.1, np.nan], [1.0], ["a", "b"], settings)

    def test_unsorted_grid(self, settings):
        with pytest.raises(UserInputError):
            validate_iteration_input([0.1], [3.0, 1.0], ["a"], settings)

    def test_duplicate_grid(self, settings):
        with pytest.raises(UserInputError):
            validate_iteration_input([0.1], [1.0, 1.0], ["a"], settings)

    def test_nan_in_grid(self, settings):
        with pytest.raises(UserInputError):
            validate_iteration_input([0.1], [1.0, float("nan")], ["a"], settings)

    def test_empty_grid(self, settings):
        with pytest.raises(UserInputError):
            validate_iteration_input([0.1], [], ["a"], settings)

    def test_off_grid_with_binning(self, settings_binned):
        with pytest.raises(UserInputError):
            validate_iteration_input([0.1], [5.0], ["a"], settings_binned)

    def test_on_grid_with_binning(self, settings_binned):
        b, g = validate_iteration_input([0.1], [10.0, 20.0], ["a"], settings_binned)
        assert g == [10.0, 20.0]


@pytest.mark.unit
class TestPrepareNodeData:
    """The shared node-preparation step runs every guard, in order.

    This is the executable form of the AGENTS.md rule that any step that
    removes rows runs before check_sample_size and identically in every
    partial: if the order or membership of this sequence changes, these
    tests fail.
    """

    @pytest.fixture
    def node_df(self):
        return pd.DataFrame(
            {
                "time": [float(i) for i in range(1, 31)],
                "event": [1] * 15 + [0] * 15,
                "age": [50.0] * 30,
                "treatment": [0, 1] * 15,
            }
        )

    @pytest.fixture
    def client(self):
        from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

        return MockAlgorithmClient(
            datasets=[[{"database": pd.DataFrame({"x": [1]}), "db_type": "csv"}]],
            module="v6-cox-ph",
            organization_ids=[1],
        )

    @pytest.fixture
    def default_env(self, monkeypatch):
        monkeypatch.delenv("SAMPLE_SIZE_THRESHOLD", raising=False)
        monkeypatch.delenv("COXPH_TIME_BIN_WIDTH", raising=False)
        monkeypatch.delenv("COXPH_MIN_RISK_SET_CHANGE", raising=False)

    def _record_guard_calls(self, monkeypatch):
        """Wrap the guards so their call order is recorded."""
        calls = []
        for name in [
            "ensure_spawned_by_central",
            "load_privacy_settings",
            "validate_expl_vars",
            "drop_incomplete_rows",
            "validate_survival_columns",
            "select_rows",
            "check_sample_size",
        ]:
            original = getattr(privacy_guards, name)

            def make_wrapper(guard_name, guard_original):
                def wrapper(*args, **kwargs):
                    calls.append(guard_name)
                    return guard_original(*args, **kwargs)

                return wrapper

            monkeypatch.setattr(privacy_guards, name, make_wrapper(name, original))
        return calls

    def test_guard_order_with_expl_vars(self, client, node_df, default_env, monkeypatch):
        """All guards run, in the documented order."""
        calls = self._record_guard_calls(monkeypatch)
        df, settings, threshold_met = prepare_node_data(
            client, node_df, "time", "event", ["age", "treatment"], need_outcome=True
        )
        assert calls == [
            "ensure_spawned_by_central",
            "load_privacy_settings",
            "validate_expl_vars",
            "drop_incomplete_rows",
            "validate_survival_columns",
            "select_rows",
            "check_sample_size",
        ]
        assert threshold_met is True
        assert len(df) == 30

    def test_guard_order_without_expl_vars(self, client, node_df, default_env, monkeypatch):
        """Without expl_vars, validate_expl_vars is skipped; order is stable."""
        calls = self._record_guard_calls(monkeypatch)
        _, _, threshold_met = prepare_node_data(client, node_df, "time", "event", [], need_outcome=True)
        assert calls == [
            "ensure_spawned_by_central",
            "load_privacy_settings",
            "drop_incomplete_rows",
            "validate_survival_columns",
            "select_rows",
            "check_sample_size",
        ]
        assert threshold_met is True

    def test_hook_runs_before_threshold(self, client, node_df, default_env, monkeypatch):
        """Rows removed by the select_rows hook count against the threshold."""
        monkeypatch.setattr(
            privacy_guards,
            "select_rows",
            lambda df: df.iloc[:5],
        )
        _, _, threshold_met = prepare_node_data(
            client, node_df, "time", "event", ["age", "treatment"], need_outcome=True
        )
        assert threshold_met is False

    def test_need_outcome_false_checks_rows_only(self, client, default_env, monkeypatch):
        """need_outcome=False checks rows only (perform_iteration today)."""
        monkeypatch.setenv("SAMPLE_SIZE_THRESHOLD", "10")
        df = pd.DataFrame({"time": [1.0] * 30, "age": [50.0] * 30})
        _, _, threshold_met = prepare_node_data(client, df, "time", None, ["age"], need_outcome=False)
        assert threshold_met is True
        df_small = pd.DataFrame({"time": [1.0] * 5, "age": [50.0] * 5})
        _, _, threshold_met = prepare_node_data(client, df_small, "time", None, ["age"], need_outcome=False)
        assert threshold_met is False

    def test_nan_rows_dropped_before_threshold(self, client, default_env, monkeypatch):
        """Rows with NaN in any analysed column are dropped before the threshold."""
        monkeypatch.setenv("SAMPLE_SIZE_THRESHOLD", "12")
        df = pd.DataFrame(
            {
                "time": [1.0] * 30,
                "event": [1] * 15 + [0] * 15,
                "age": [50.0] * 30,
                "treatment": [0, 1] * 15,
            }
        )
        # 12 rows get NaN in an analysed column: 15 events and 30 rows would
        # both pass threshold 12, but after dropping only 18 rows / 13 events
        # remain — rows still pass, events (13 > 12) pass too; push one more
        # NaN onto an event row to make events 12, which must fail (>12).
        df.loc[0:11, "age"] = np.nan
        df.loc[12, "age"] = np.nan  # row 12 is an event row
        _, _, threshold_met = prepare_node_data(client, df, "time", "event", ["age", "treatment"], need_outcome=True)
        assert threshold_met is False


@pytest.mark.unit
class TestValidateSurvivalColumns:
    """Tests for validate_survival_columns (input rules, no value echoes)."""

    def _df(self, time=None, event=None):
        time = [1.0, 2.0, 3.0, 4.0] if time is None else time
        event = [1, 0, 1, 0] if event is None else event
        return pd.DataFrame({"time": time, "event": event})

    def test_valid_input_passes(self):
        validate_survival_columns(self._df(), "time", "event")

    def test_boolean_outcome_passes(self):
        validate_survival_columns(self._df(event=[True, False, True, False]), "time", "event")

    def test_missing_time_column_raises(self):
        with pytest.raises(UserInputError, match="Time column 'time' not found"):
            validate_survival_columns(pd.DataFrame({"event": [1]}), "time", "event")

    def test_missing_outcome_column_raises(self):
        with pytest.raises(UserInputError, match="Outcome column 'event' not found"):
            validate_survival_columns(pd.DataFrame({"time": [1.0]}), "time", "event")

    def test_no_outcome_col_skips_outcome_checks(self):
        df = pd.DataFrame({"time": [1.0, 2.0]})
        validate_survival_columns(df, "time", None)

    def test_string_time_raises(self):
        with pytest.raises(UserInputError, match="Time column 'time' must be numeric"):
            validate_survival_columns(self._df(time=["a", "b", "c", "d"]), "time", "event")

    def test_negative_time_raises(self):
        with pytest.raises(UserInputError, match="negative"):
            validate_survival_columns(self._df(time=[1.0, -2.0, 3.0, 4.0]), "time", "event")

    def test_infinite_time_raises(self):
        with pytest.raises(UserInputError, match="non-finite"):
            validate_survival_columns(self._df(time=[1.0, np.inf, 3.0, 4.0]), "time", "event")

    def test_outcome_coded_1_2_raises(self):
        with pytest.raises(UserInputError, match="only contain 0 and 1"):
            validate_survival_columns(self._df(event=[1, 2, 1, 0]), "time", "event")

    def test_string_outcome_raises(self):
        with pytest.raises(UserInputError, match="numeric or boolean"):
            validate_survival_columns(self._df(event=["yes", "no", "yes", "no"]), "time", "event")

    def test_messages_never_echo_values(self):
        """Error messages name columns and dtypes, never data values."""
        cases = [
            (self._df(time=[1.0, -123.456, 3.0, 4.0]), "time", "event", "-123.456"),
            (self._df(event=[1, 7, 1, 0]), "time", "event", "'7'"),
        ]
        for df, time_col, outcome_col, forbidden in cases:
            with pytest.raises(UserInputError) as exc_info:
                validate_survival_columns(df, time_col, outcome_col)
            assert forbidden not in str(exc_info.value)


@pytest.mark.unit
class TestBinTimes:
    """Tests for bin_times."""

    def test_no_width_returns_unchanged(self):
        times = pd.Series([1.5, 7.3, 12.8])
        result = bin_times(times, None)
        pd.testing.assert_series_equal(result, times)

    def test_zero_width_returns_unchanged(self):
        times = pd.Series([1.5, 7.3, 12.8])
        result = bin_times(times, 0)
        pd.testing.assert_series_equal(result, times)

    def test_bins_to_grid(self):
        times = pd.Series([1.5, 7.3, 12.8, 20.0])
        result = bin_times(times, 10.0)
        np.testing.assert_array_equal(result, [0.0, 0.0, 10.0, 20.0])

    def test_bins_negative(self):
        times = pd.Series([-3.2, 5.0])
        result = bin_times(times, 10.0)
        np.testing.assert_array_equal(result, [-10.0, 0.0])


@pytest.mark.unit
class TestTailCutoff:
    """Tests for tail_cutoff."""

    def test_disabled_when_k_le_one(self):
        times = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert tail_cutoff(times, k=1) is None
        assert tail_cutoff(times, k=0) is None

    def test_kth_largest(self):
        times = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        # k=3 -> 3rd largest = 3.0
        assert tail_cutoff(times, k=3) == 3.0

    def test_kth_largest_with_duplicates(self):
        times = pd.Series([1.0, 2.0, 3.0, 3.0, 5.0])
        # sorted: [1,2,3,3,5], k=3 -> [-3] = 3.0
        assert tail_cutoff(times, k=3) == 3.0

    def test_fewer_than_k_raises(self):
        """FR-B2: fewer than k valid times raises PrivacyViolation."""
        times = pd.Series([1.0, 2.0])
        with pytest.raises(PrivacyViolation):
            tail_cutoff(times, k=5)


@pytest.mark.unit
class TestPrepareTimeColumn:
    """Tests for prepare_time_column."""

    @pytest.fixture
    def settings_default(self):
        return PrivacySettings(sample_size_threshold=10, time_bin_width=None, min_risk_set_change=5)

    @pytest.fixture
    def settings_binned(self):
        return PrivacySettings(
            sample_size_threshold=10,
            time_bin_width=10.0,
            min_risk_set_change=1,
        )

    @pytest.fixture
    def settings_k1(self):
        return PrivacySettings(sample_size_threshold=10, time_bin_width=None, min_risk_set_change=1)

    def test_no_binning_no_censor(self, settings_k1):
        df = pd.DataFrame({"time": [1.0, 5.0, 10.0], "event": [1, 0, 1]})
        out = prepare_time_column(df, "time", settings_k1, "event")
        pd.testing.assert_frame_equal(out, df)

    def test_does_not_modify_input(self, settings_binned):
        df = pd.DataFrame({"time": [1.5, 12.3], "event": [1, 1]})
        original = df.copy()
        prepare_time_column(df, "time", settings_binned, "event")
        pd.testing.assert_frame_equal(df, original)

    def test_bins_times(self, settings_binned):
        df = pd.DataFrame({"time": [1.5, 12.3], "event": [1, 0]})
        out = prepare_time_column(df, "time", settings_binned, "event")
        np.testing.assert_array_equal(out["time"].to_numpy(), [0.0, 10.0])

    def test_tail_censors_and_zeros_events(self, settings_default):
        # k=5, 10 rows; t_cut = 5th largest = 6.0; times 7-10 are clamped to 6.0
        df = pd.DataFrame(
            {
                "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "event": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            }
        )
        out = prepare_time_column(df, "time", settings_default, "event")
        # Rows 6-9 (index) are clamped to 6.0 and their events zeroed
        for i in range(6, 10):
            assert out.loc[i, "time"] == 6.0
            assert out.loc[i, "event"] == 0
        # Row 5 (time 6.0) is unchanged
        assert out.loc[5, "time"] == 6.0
        assert out.loc[5, "event"] == 0

    def test_no_outcome_col_does_not_zero_events(self, settings_default):
        # k=5, 10 rows; t_cut = 6.0; times 7-10 clamped to 6.0
        df = pd.DataFrame(
            {
                "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "other": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            }
        )
        out = prepare_time_column(df, "time", settings_default, outcome_col=None)
        # time clamped but no event column to zero
        for i in range(6, 10):
            assert out.loc[i, "time"] == 6.0
        assert "event" not in out.columns


@pytest.mark.unit
class TestGuardedRiskSetMasks:
    """Tests for guarded_risk_set_masks."""

    @pytest.fixture
    def times(self):
        # 10 individuals with distinct times
        return pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    def test_k1_is_plain_masks(self, times):
        grid = [2.0, 5.0, 8.0]
        masks = guarded_risk_set_masks(times, grid, k=1)
        assert len(masks) == 3
        assert masks[0].sum() == 9  # >= 2
        assert masks[1].sum() == 6  # >= 5
        assert masks[2].sum() == 3  # >= 8

    def test_holds_when_removed_below_k(self, times):
        """Moving from t to t+1 removing < k individuals holds the mask."""
        grid = [1.0, 2.0, 8.0]
        masks = guarded_risk_set_masks(times, grid, k=5)
        # At t=1: 10 at risk. At t=2: 9 at risk (removed 1 < 5) -> hold
        assert masks[0].sum() == 10
        assert masks[1].sum() == 10  # held
        # At t=8: 3 at risk (removed 6 from 10 >= 5) -> use cand
        assert masks[2].sum() == 3

    def test_consecutive_changes_are_zero_or_ge_k(self, times):
        """Every change in mask size is 0 or >= k."""
        grid = [float(i) for i in range(1, 11)]
        masks = guarded_risk_set_masks(times, grid, k=5)
        sizes = [int(m.sum()) for m in masks]
        for i in range(1, len(sizes)):
            change = sizes[i - 1] - sizes[i]
            assert change == 0 or change >= 5, f"change {change} at index {i} (sizes={sizes})"

    def test_tail_goes_to_empty_only_from_ge_k(self):
        """The final transition to an empty risk set must remove >= k."""
        times = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        grid = [1.0, 11.0]
        masks = guarded_risk_set_masks(times, grid, k=5)
        # t=1: 10 at risk. t=11: 0 at risk (removed 10 >= 5)
        assert masks[0].sum() == 10
        assert masks[1].sum() == 0


@pytest.mark.unit
class TestValidateExplVars:
    """Tests for validate_expl_vars (FR-B3)."""

    def test_valid_vars(self):
        df = pd.DataFrame({"time": [1.0], "event": [1], "age": [50], "tx": [1]})
        validate_expl_vars(df, ["age", "tx"], "time", "event")

    def test_missing_column(self):
        df = pd.DataFrame({"time": [1.0], "event": [1], "age": [50]})
        with pytest.raises(UserInputError, match="not found"):
            validate_expl_vars(df, ["age", "missing"], "time", "event")

    def test_overlaps_time_col(self):
        df = pd.DataFrame({"time": [1.0], "event": [1], "age": [50]})
        with pytest.raises(UserInputError, match="must not equal time_col"):
            validate_expl_vars(df, ["time"], "time", "event")

    def test_overlaps_outcome_col(self):
        df = pd.DataFrame({"time": [1.0], "event": [1], "age": [50]})
        with pytest.raises(UserInputError, match="must not equal outcome_col"):
            validate_expl_vars(df, ["event"], "time", "event")

    def test_non_numeric_column(self):
        df = pd.DataFrame({"time": [1.0], "event": [1], "name": ["alice"]})
        with pytest.raises(UserInputError, match="must be numeric"):
            validate_expl_vars(df, ["name"], "time", "event")


@pytest.mark.unit
class TestDropIncompleteRows:
    """Tests for drop_incomplete_rows (FR-B4)."""

    def test_drops_nan_rows(self):
        df = pd.DataFrame({"time": [1.0, np.nan, 3.0], "event": [1, 1, 0], "age": [50, 60, np.nan]})
        out = drop_incomplete_rows(df, ["time", "event", "age"])
        assert len(out) == 1
        assert out.index.tolist() == [0]

    def test_does_not_modify_input(self):
        df = pd.DataFrame({"time": [1.0, np.nan], "event": [1, 1]})
        original = df.copy()
        drop_incomplete_rows(df, ["time", "event"])
        pd.testing.assert_frame_equal(df, original)

    def test_nonexistent_col_ignored(self):
        df = pd.DataFrame({"time": [1.0, 2.0], "event": [1, 0]})
        out = drop_incomplete_rows(df, ["time", "missing"])
        assert len(out) == 2
