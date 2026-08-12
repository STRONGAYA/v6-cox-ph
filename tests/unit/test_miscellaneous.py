"""
Unit tests for miscellaneous utilities in v6-cox-ph.
"""

import pytest
import pandas as pd
import numpy as np

from v6_cox_ph.miscellaneous import (
    CoxPHInput,
    PartialResult,
    PrivacyThresholdConfig,
    validate_coxph_input,
    validate_partial_result,
    check_event_count,
    check_data_quality
)
from vantage6.algorithm.tools.exceptions import UserInputError


class TestCoxPHInput:
    def test_valid_input(self):
        input_data = {
            "time_col": "time",
            "outcome_col": "event",
            "expl_vars": ["age", "treatment"],
            "organization_ids": [1, 2, 3]
        }
        model = CoxPHInput(**input_data)
        assert model.time_col == "time"
        assert model.expl_vars == ["age", "treatment"]

    def test_empty_expl_vars_raises_error(self):
        input_data = {
            "time_col": "time",
            "outcome_col": "event",
            "expl_vars": [],
            "organization_ids": [1, 2]
        }
        with pytest.raises(ValueError):
            CoxPHInput(**input_data)


class TestCheckEventCount:
    def test_sufficient_events(self):
        df = pd.DataFrame({"time": [1, 2, 3, 4, 5], "event": [1, 1, 1, 1, 1]})
        result = check_event_count(df, "event", min_events=3)
        assert result is True

    def test_insufficient_events(self):
        df = pd.DataFrame({"time": [1, 2, 3], "event": [1, 0, 0]})
        result = check_event_count(df, "event", min_events=3)
        assert result is False


class TestCheckDataQuality:
    def test_complete_data(self):
        df = pd.DataFrame({"time": [1.0, 2.0, 3.0], "event": [1, 0, 1]})
        result = check_data_quality(df, "time", "event")
        assert result["has_time"] is True
        assert result["event_count"] == 2
        assert result["censored_count"] == 1
