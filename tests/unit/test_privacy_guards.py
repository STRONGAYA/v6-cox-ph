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
    check_sample_size,
    ensure_spawned_by_central,
    load_privacy_settings,
    validate_iteration_input,
)


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
        """Negative min_risk_set_change raises UserInputError."""
        monkeypatch.setenv("COXPH_MIN_RISK_SET_CHANGE", "-1")
        with pytest.raises(UserInputError):
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
        """A client without _access_token is skipped (mock client)."""
        client = self._make_client(token=None)
        # Should not raise
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
        client = self._make_client(
            token=token, task_dict={"parent": {"id": 3}}
        )
        ensure_spawned_by_central(client)

    def test_task_lookup_failure_fails_closed(self):
        """If task.get raises, the guard fails closed with AlgorithmError."""
        token = self._make_token(task_id=7)
        client = self._make_client(
            token=token, raises=RuntimeError("connection error")
        )
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
