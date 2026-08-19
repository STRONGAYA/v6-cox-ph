"""
Comprehensive Vantage6 integration testing for v6-cox-ph algorithm.
"""

import pytest

from v6_cox_ph.miscellaneous import validate_coxph_input
from vantage6.algorithm.tools.exceptions import UserInputError


@pytest.fixture
def test_methods():
    """
    Fixture providing different algorithm methods to test with their specific kwargs templates.
    """
    return {
        "central": {
            "basic": {
                "time_col": None,
                "outcome_col": None,
                "expl_vars": None,
            },
            "organisation_selection": {
                "time_col": None,
                "outcome_col": None,
                "expl_vars": None,
                "organization_ids": None,
            },
            "parameter_galore": {
                "time_col": None,
                "outcome_col": None,
                "expl_vars": None,
                "organization_ids": None,
            },
        },
    }


@pytest.fixture
def coxph_variables():
    """Standard Cox-PH variable configuration for testing."""
    return {
        "time_col": "time",
        "outcome_col": "event",
        "expl_vars": ["age", "treatment"],
    }


class TestCoxPHAlgorithmIntegration:
    """Integration tests for the Cox-PH algorithm."""

    @pytest.mark.unit
    def test_validate_input_valid(self, coxph_variables):
        """Test input validation with valid parameters."""
        # This should not raise any exception
        validated = validate_coxph_input(
            time_col=coxph_variables["time_col"],
            outcome_col=coxph_variables["outcome_col"],
            expl_vars=coxph_variables["expl_vars"],
            organization_ids=None
        )
        
        assert validated.time_col == coxph_variables["time_col"]
        assert validated.outcome_col == coxph_variables["outcome_col"]
        assert validated.expl_vars == coxph_variables["expl_vars"]

    @pytest.mark.unit
    def test_validate_input_empty_expl_vars(self):
        """Test input validation with empty explanatory variables."""
        with pytest.raises(UserInputError):
            validate_coxph_input(
                time_col="time",
                outcome_col="event",
                expl_vars=[],
                organization_ids=[1, 2]
            )

    @pytest.mark.unit
    def test_validate_input_empty_column_name(self):
        """Test input validation with empty column name."""
        with pytest.raises(UserInputError):
            validate_coxph_input(
                time_col="",
                outcome_col="event",
                expl_vars=["age"],
                organization_ids=[1, 2]
            )
