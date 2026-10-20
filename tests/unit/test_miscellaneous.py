"""
Unit tests for miscellaneous Cox-PH functions.

Tests input validation via the Pydantic model.
"""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from vantage6.algorithm.tools.exceptions import UserInputError

# Add the algorithm module to the path
algorithm_path = Path(__file__).parent.parent.parent / "v6-cox-ph"
sys.path.insert(0, str(algorithm_path))

from miscellaneous import (  # noqa: E402
    CoxPHInput,
    validate_coxph_input,
    validate_stratification_spec,
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

    def test_validate_input_time_col_equals_outcome_col(self):
        """Test that time_col == outcome_col raises UserInputError."""
        with pytest.raises(UserInputError, match="different columns"):
            validate_coxph_input(
                time_col="time",
                outcome_col="time",
                expl_vars=["age"],
            )

    def test_validate_input_duplicate_expl_vars(self):
        """Test that duplicate explanatory variables raise UserInputError."""
        with pytest.raises(UserInputError, match="duplicate variables"):
            validate_coxph_input(
                time_col="time",
                outcome_col="event",
                expl_vars=["age", "treatment", "age"],
            )

    def test_validate_input_expl_var_names_time_col(self):
        """Test that an explanatory variable naming the time column is rejected."""
        with pytest.raises(UserInputError, match="time or outcome column"):
            validate_coxph_input(
                time_col="time",
                outcome_col="event",
                expl_vars=["time", "age"],
            )

    def test_validate_input_expl_var_names_outcome_col(self):
        """Test that an explanatory variable naming the outcome column is rejected."""
        with pytest.raises(UserInputError, match="time or outcome column"):
            validate_coxph_input(
                time_col="time",
                outcome_col="event",
                expl_vars=["age", "event"],
            )

    def test_model_rejects_semantic_violations_directly(self):
        """The model itself raises ValidationError for semantic violations."""
        with pytest.raises(ValidationError):
            CoxPHInput(time_col="time", outcome_col="time", expl_vars=["age"])
        with pytest.raises(ValidationError):
            CoxPHInput(time_col="time", outcome_col="event", expl_vars=["age", "age"])
        with pytest.raises(ValidationError):
            CoxPHInput(time_col="time", outcome_col="event", expl_vars=["event"])

    def test_wrapped_error_is_formatted_without_input(self):
        """The UserInputError wraps the one-line formatted validation message."""
        with pytest.raises(UserInputError) as excinfo:
            validate_coxph_input(
                time_col="time",
                outcome_col="event",
                expl_vars=["age"],
                organization_ids=[-1],
            )
        message = str(excinfo.value)
        assert message.startswith("Invalid Cox-PH input:")
        assert "https://errors.pydantic.dev" not in message


@pytest.mark.unit
class TestStratificationSpec:
    """Tests for the variables_to_stratify validation (STRONG AYA standard)."""

    def test_valid_value_list(self):
        spec = {"sex": ["F", "M"], "treatment": [1]}
        assert validate_stratification_spec(spec) == spec

    def test_valid_range_both_bounds(self):
        spec = {"age": {"start": 40, "end": 60}}
        assert validate_stratification_spec(spec) == spec

    def test_valid_range_single_bound(self):
        assert validate_stratification_spec({"age": {"start": 40}}) == {"age": {"start": 40}}
        assert validate_stratification_spec({"age": {"end": 60}}) == {"age": {"end": 60}}
        assert validate_stratification_spec({"age": {"start": None, "end": 60}}) == {"age": {"start": None, "end": 60}}

    @pytest.mark.parametrize("spec", [{}, "not-a-dict", [], {"sex": "F"}, {"sex": 1}])
    def test_non_mapping_rejected(self, spec):
        with pytest.raises(ValueError):
            validate_stratification_spec(spec)

    @pytest.mark.parametrize("name", ["age at diagnosis", "age`", "age>=40", "1age", "", "age;drop"])
    def test_non_identifier_name_rejected(self, name):
        with pytest.raises(ValueError, match="identifier-shaped"):
            validate_stratification_spec({name: [1]})

    @pytest.mark.parametrize("selection", [[], ["F", None], ["F", True], ["F", ["nested"]], ["F", {"start": 1}]])
    def test_invalid_value_list_rejected(self, selection):
        with pytest.raises(ValueError):
            validate_stratification_spec({"sex": selection})

    @pytest.mark.parametrize(
        "selection",
        [
            {"start": 60, "end": 40},
            {"start": "40"},
            {"start": True},
            {"lower": 40},
            {"start": 40, "end": 60, "step": 5},
            {},
        ],
    )
    def test_invalid_range_rejected(self, selection):
        with pytest.raises(ValueError):
            validate_stratification_spec({"age": selection})

    def test_input_model_accepts_valid_stratification(self):
        validated = validate_coxph_input(
            time_col="time",
            outcome_col="event",
            expl_vars=["age", "treatment"],
            variables_to_stratify={"sex": ["F"], "age": {"start": 40, "end": 60}},
        )
        assert validated.variables_to_stratify == {"sex": ["F"], "age": {"start": 40, "end": 60}}

    def test_input_model_defaults_to_none(self):
        validated = validate_coxph_input(time_col="time", outcome_col="event", expl_vars=["age"])
        assert validated.variables_to_stratify is None

    def test_input_model_rejects_invalid_stratification(self):
        with pytest.raises(UserInputError):
            validate_coxph_input(
                time_col="time",
                outcome_col="event",
                expl_vars=["age"],
                variables_to_stratify={"age": {"start": 60, "end": 40}},
            )

    def test_wrapped_error_does_not_echo_values(self):
        """The formatted validation error never echoes the offending values."""
        with pytest.raises(UserInputError) as excinfo:
            validate_coxph_input(
                time_col="time",
                outcome_col="event",
                expl_vars=["age"],
                variables_to_stratify={"age": {"start": 61, "end": 40}},
            )
        message = str(excinfo.value)
        assert "61" not in message
        assert "40" not in message
