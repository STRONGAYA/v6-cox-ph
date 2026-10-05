"""
Unit tests for miscellaneous Cox-PH functions.

Tests input validation, data quality checks, and event count checks.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from vantage6.algorithm.tools.exceptions import UserInputError

# Add the algorithm module to the path
algorithm_path = Path(__file__).parent.parent.parent / "v6-cox-ph"
sys.path.insert(0, str(algorithm_path))

from miscellaneous import (  # noqa: E402
    check_data_quality,
    validate_coxph_input,
)


@pytest.mark.unit
class TestValidateCoxphInput:
    """Tests for the validate_coxph_input function."""

    def test_validate_input_valid(self):
        """Test input validation with valid parameters."""
        validated = validate_coxph_input(
            time_col="time",
            outcome_col="event",
            expl_vars=["age", "treatment"],
            organization_ids=[1, 2],
        )
        assert validated.time_col == "time"
        assert validated.outcome_col == "event"
        assert validated.expl_vars == ["age", "treatment"]
        assert validated.organization_ids == [1, 2]

    def test_validate_input_no_organisation_ids(self):
        """Test input validation with no organisation IDs."""
        validated = validate_coxph_input(
            time_col="time",
            outcome_col="event",
            expl_vars=["age"],
        )
        assert validated.organization_ids is None

    def test_validate_input_empty_expl_vars(self):
        """Test that empty explanatory variables raises UserInputError."""
        with pytest.raises(UserInputError):
            validate_coxph_input(
                time_col="time",
                outcome_col="event",
                expl_vars=[],
                organization_ids=[1, 2],
            )

    def test_validate_input_empty_time_col(self):
        """Test that empty time column name raises UserInputError."""
        with pytest.raises(UserInputError):
            validate_coxph_input(
                time_col="",
                outcome_col="event",
                expl_vars=["age"],
            )

    def test_validate_input_empty_outcome_col(self):
        """Test that empty outcome column name raises UserInputError."""
        with pytest.raises(UserInputError):
            validate_coxph_input(
                time_col="time",
                outcome_col="",
                expl_vars=["age"],
            )

    def test_validate_input_whitespace_column_name(self):
        """Test that whitespace-only column names are rejected."""
        with pytest.raises(UserInputError):
            validate_coxph_input(
                time_col="  ",
                outcome_col="event",
                expl_vars=["age"],
            )

    def test_validate_input_invalid_expl_var_name(self):
        """Test that invalid variable names raise UserInputError."""
        with pytest.raises(UserInputError):
            validate_coxph_input(
                time_col="time",
                outcome_col="event",
                expl_vars=["age", ""],
            )

    def test_validate_input_negative_org_id(self):
        """Test that negative organisation IDs raise UserInputError."""
        with pytest.raises(UserInputError):
            validate_coxph_input(
                time_col="time",
                outcome_col="event",
                expl_vars=["age"],
                organization_ids=[-1],
            )

    def test_validate_input_deduplicates_org_ids(self):
        """Test that duplicate organisation IDs are deduplicated."""
        validated = validate_coxph_input(
            time_col="time",
            outcome_col="event",
            expl_vars=["age"],
            organization_ids=[1, 1, 2],
        )
        assert sorted(validated.organization_ids) == [1, 2]

    def test_validate_input_strips_whitespace(self):
        """Test that whitespace in column names is stripped."""
        validated = validate_coxph_input(
            time_col="  time  ",
            outcome_col=" event ",
            expl_vars=["  age  "],
        )
        assert validated.time_col == "time"
        assert validated.outcome_col == "event"
        assert validated.expl_vars == ["age"]


@pytest.mark.unit
class TestCheckDataQuality:
    """Tests for the check_data_quality function."""

    def test_check_data_quality_valid_data(self):
        """Test data quality checks with valid data."""
        df = pd.DataFrame(
            {
                "time": [5.0, 10.0, 15.0, 20.0],
                "event": [1, 0, 1, 1],
                "age": [50, 60, 70, 55],
            }
        )
        result = check_data_quality(df, "time", "event")
        assert result["has_time"] is True
        assert result["has_outcome"] is True
        assert result["event_count"] == 3
        assert result["censored_count"] == 1
        assert result["total_count"] == 4
        assert result["has_negative_time"] is False
        assert result["time_range"] == (5.0, 20.0)

    def test_check_data_quality_missing_time(self):
        """Test data quality with missing time column."""
        df = pd.DataFrame({"event": [1, 0, 1]})
        result = check_data_quality(df, "time", "event")
        assert result["has_time"] is False
        assert result["has_outcome"] is True

    def test_check_data_quality_missing_outcome(self):
        """Test data quality with missing outcome column."""
        df = pd.DataFrame({"time": [5.0, 10.0, 15.0]})
        result = check_data_quality(df, "time", "event")
        assert result["has_time"] is True
        assert result["has_outcome"] is False

    def test_check_data_quality_negative_time(self):
        """Test that negative time values are detected."""
        df = pd.DataFrame(
            {
                "time": [-5.0, 10.0, 15.0],
                "event": [1, 0, 1],
            }
        )
        result = check_data_quality(df, "time", "event")
        assert result["has_negative_time"] is True

    def test_check_data_quality_empty_df(self):
        """Test data quality with an empty DataFrame."""
        df = pd.DataFrame()
        result = check_data_quality(df, "time", "event")
        assert result["has_time"] is False
        assert result["has_outcome"] is False
        assert result["total_count"] == 0
