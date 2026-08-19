"""
Unit tests for miscellaneous utilities in v6-cox-ph.
"""

import pytest
import pandas as pd

from v6_cox_ph.miscellaneous import (
    CoxPHInput,
    validate_coxph_input
)
from vantage6.algorithm.tools.exceptions import UserInputError


class TestCoxPHInput:
    """Tests for CoxPHInput Pydantic model."""

    def test_valid_input(self):
        """Test validation with valid input."""
        input_data = {
            "time_col": "time",
            "outcome_col": "event",
            "expl_vars": ["age", "treatment"],
            "organization_ids": [1, 2, 3]
        }
        
        model = CoxPHInput(**input_data)
        
        assert model.time_col == "time"
        assert model.outcome_col == "event"
        assert model.expl_vars == ["age", "treatment"]
        assert model.organization_ids == [1, 2, 3]

    def test_valid_input_no_org_ids(self):
        """Test validation with no organization IDs."""
        input_data = {
            "time_col": "time",
            "outcome_col": "event",
            "expl_vars": ["age", "treatment"],
            "organization_ids": None
        }
        
        model = CoxPHInput(**input_data)
        
        assert model.organization_ids is None

    def test_empty_expl_vars_raises_error(self):
        """Test that empty explanatory variables raises error."""
        input_data = {
            "time_col": "time",
            "outcome_col": "event",
            "expl_vars": [],
            "organization_ids": [1, 2]
        }
        
        with pytest.raises(ValueError):
            CoxPHInput(**input_data)

    def test_empty_time_col_raises_error(self):
        """Test that empty time column name raises error."""
        input_data = {
            "time_col": "",
            "outcome_col": "event",
            "expl_vars": ["age"],
            "organization_ids": [1, 2]
        }
        
        with pytest.raises(ValueError):
            CoxPHInput(**input_data)

    def test_duplicate_org_ids_removed(self):
        """Test that duplicate organization IDs are removed."""
        input_data = {
            "time_col": "time",
            "outcome_col": "event",
            "expl_vars": ["age"],
            "organization_ids": [1, 2, 2, 3, 1]  # Duplicates
        }
        
        model = CoxPHInput(**input_data)
        
        # Should have unique organization IDs
        assert model.organization_ids == [1, 2, 3]
