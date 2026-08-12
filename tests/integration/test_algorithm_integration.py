"""
Comprehensive Vantage6 integration testing for v6-cox-ph algorithm.
"""

import pytest
import json
import pandas as pd
import numpy as np

from json import JSONDecodeError
from typing import Any, Dict, Tuple
from vantage6.algorithm.tools.exceptions import (
    DataError,
    UserInputError,
    CollectResultsError,
    PrivacyThresholdViolation,
    InputError,
    AlgorithmError,
    CollectOrganizationError,
)


@pytest.fixture
def test_methods():
    """
    Fixture providing different algorithm methods to test with their specific kwargs templates.
    """
    return {
        "central": {
            "basic": {
                "time_col": None,  # Will be filled from config
                "outcome_col": None,  # Will be filled from config
                "expl_vars": None,  # Will be filled from config
            },
            "organisation_selection": {
                "time_col": None,
                "outcome_col": None,
                "expl_vars": None,
                "organization_ids": None,  # Will be filled from config
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
def test_configurations():
    """
    Fixture providing comprehensive test configurations for algorithm validation.
    """
    return {
        "standard_dataset": {
            "database_label": "coxph_test_data_1",
            "time_col": "time",
            "outcome_col": "event",
            "expl_vars": ["age", "treatment"],
            "organisation_subset": [1, 2],
            "expected_failure": False,
            "failure_reason": None,
            "expected_error_type": None,
        },
        "standard_dataset_all_orgs": {
            "database_label": "coxph_test_data_1",
            "time_col": "time",
            "outcome_col": "event",
            "expl_vars": ["age", "treatment"],
            "organisation_subset": [1, 2, 3],
            "expected_failure": False,
            "failure_reason": None,
            "expected_error_type": None,
        },
        "standard_dataset_single_covariate": {
            "database_label": "coxph_test_data_1",
            "time_col": "time",
            "outcome_col": "event",
            "expl_vars": ["age"],
            "organisation_subset": [1, 2],
            "expected_failure": False,
            "failure_reason": None,
            "expected_error_type": None,
        },
        "standard_dataset_bad_actor": {
            "database_label": "coxph_test_data_3",  # Small dataset
            "time_col": "time",
            "outcome_col": "event",
            "expl_vars": ["age", "treatment"],
            "organisation_subset": [3],
            "expected_failure": True,
            "failure_reason": "Small dataset may trigger privacy threshold",
            "expected_error_type": [PrivacyThresholdViolation, CollectResultsError],
        },
        "standard_dataset_incorrect_input": {
            "database_label": "coxph_test_data_1",
            "time_col": "nonexistent_column",
            "outcome_col": "event",
            "expl_vars": ["age", "treatment"],
            "organisation_subset": [1, 2],
            "expected_failure": True,
            "failure_reason": "Non-existent time column",
            "expected_error_type": [UserInputError, CollectResultsError],
        },
        "standard_dataset_empty_expl_vars": {
            "database_label": "coxph_test_data_1",
            "time_col": "time",
            "outcome_col": "event",
            "expl_vars": [],
            "organisation_subset": [1, 2],
            "expected_failure": True,
            "failure_reason": "Empty explanatory variables list",
            "expected_error_type": [UserInputError],
        },
    }


@pytest.fixture
def test_configurations_manual():
    """
    Manual test configurations for Cox-PH specific scenarios.
    """
    return {
        "perfect_prediction": {
            "time_col": "time",
            "outcome_col": "event",
            "expl_vars": ["age"],
            "organization_ids": [1, 2],
            "expected_failure": False,
            "check_warnings": True,  # Should check for perfect prediction warnings
        },
        "multiple_covariates": {
            "time_col": "time",
            "outcome_col": "event",
            "expl_vars": ["age", "treatment", "sex"],
            "organization_ids": [1, 2],
            "expected_failure": False,
        },
    }


class TestCoxPHAlgorithmIntegration:
    """Integration tests for the Cox-PH algorithm."""

    @pytest.mark.integration
    @pytest.mark.vantage6
    @pytest.mark.docker
    def test_central_basic_execution(self, authentication, variables_config):
        """Test basic execution of the central algorithm."""
        client = authentication
        
        # Run the central algorithm
        task = client.task.create(
            input_={
                "method": "central",
                "kwargs": {
                    "time_col": variables_config["time_col"],
                    "outcome_col": variables_config["outcome_col"],
                    "expl_vars": variables_config["expl_vars"],
                    "organization_ids": variables_config["organization_ids"],
                }
            },
            organizations=variables_config["organization_ids"],
        )
        
        results = client.wait_for_results(task_id=task.get("id"))
        
        # Validate results
        assert len(results) > 0, "No results returned"
        result = results[0]
        
        assert "model" in result, "Result should contain 'model' key"
        assert "overall_p_value" in result, "Result should contain 'overall_p_value' key"
        assert "aic" in result, "Result should contain 'aic' key"
        assert "included_organizations" in result, "Result should contain 'included_organizations' key"
        
        # Check that model is valid JSON
        if result["model"]:
            model_df = pd.read_json(result["model"], typ='frame')
            assert len(model_df) > 0, "Model DataFrame should not be empty"
            assert "Coef" in model_df.columns, "Model should have Coef column"
            assert "Exp(coef)" in model_df.columns, "Model should have Exp(coef) column"
            assert "SE" in model_df.columns, "Model should have SE column"

    @pytest.mark.integration
    @pytest.mark.vantage6
    @pytest.mark.docker
    def test_central_with_subset_organizations(self, authentication, variables_config):
        """Test execution with a subset of organizations."""
        client = authentication
        
        # Use only first two organizations
        org_subset = variables_config["organization_ids"][:2]
        
        task = client.task.create(
            input_={
                "method": "central",
                "kwargs": {
                    "time_col": variables_config["time_col"],
                    "outcome_col": variables_config["outcome_col"],
                    "expl_vars": variables_config["expl_vars"],
                    "organization_ids": org_subset,
                }
            },
            organizations=org_subset,
        )
        
        results = client.wait_for_results(task_id=task.get("id"))
        
        # Validate results
        assert len(results) > 0, "No results returned"
        result = results[0]
        
        assert "included_organizations" in result
        assert "excluded_organizations" in result
        
        # Check that the correct organizations are included
        included = result["included_organizations"]
        excluded = result["excluded_organizations"]
        
        # All requested organizations should be either included or excluded
        all_orgs = included + excluded
        for org in org_subset:
            assert org in all_orgs, f"Organization {org} should be in included or excluded"

    @pytest.mark.integration
    @pytest.mark.vantage6
    @pytest.mark.docker
    def test_central_with_single_covariate(self, authentication, variables_config):
        """Test execution with a single explanatory variable."""
        client = authentication
        
        single_covariate = [variables_config["expl_vars"][0]]
        
        task = client.task.create(
            input_={
                "method": "central",
                "kwargs": {
                    "time_col": variables_config["time_col"],
                    "outcome_col": variables_config["outcome_col"],
                    "expl_vars": single_covariate,
                    "organization_ids": variables_config["organization_ids"],
                }
            },
            organizations=variables_config["organization_ids"],
        )
        
        results = client.wait_for_results(task_id=task.get("id"))
        
        # Validate results
        assert len(results) > 0, "No results returned"
        result = results[0]
        
        # Check that model has only one row (one covariate)
        if result["model"]:
            model_df = pd.read_json(result["model"], typ='frame')
            assert len(model_df) == 1, "Model should have exactly one row for single covariate"

    @pytest.mark.integration
    @pytest.mark.vantage6
    @pytest.mark.docker
    def test_invalid_input_non_existent_column(self, authentication, variables_config):
        """Test that non-existent columns trigger appropriate errors."""
        client = authentication
        
        with pytest.raises(Exception):  # Should raise some exception
            task = client.task.create(
                input_={
                    "method": "central",
                    "kwargs": {
                        "time_col": "nonexistent_time_column",
                        "outcome_col": variables_config["outcome_col"],
                        "expl_vars": variables_config["expl_vars"],
                        "organization_ids": variables_config["organization_ids"],
                    }
                },
                organizations=variables_config["organization_ids"],
            )
            
            # This should fail either during task creation or when waiting for results
            results = client.wait_for_results(task_id=task.get("id"))

    @pytest.mark.integration
    @pytest.mark.vantage6
    @pytest.mark.docker
    def test_empty_explanatory_variables(self, authentication, variables_config):
        """Test that empty explanatory variables list triggers appropriate error."""
        client = authentication
        
        with pytest.raises(Exception):  # Should raise UserInputError or similar
            task = client.task.create(
                input_={
                    "method": "central",
                    "kwargs": {
                        "time_col": variables_config["time_col"],
                        "outcome_col": variables_config["outcome_col"],
                        "expl_vars": [],  # Empty list
                        "organization_ids": variables_config["organization_ids"],
                    }
                },
                organizations=variables_config["organization_ids"],
            )
            
            results = client.wait_for_results(task_id=task.get("id"))

    @pytest.mark.unit
    def test_validate_input_valid(self, variables_config):
        """Test input validation with valid parameters."""
        from v6-cox-ph.miscellaneous import validate_coxph_input
        
        # This should not raise any exception
        validated = validate_coxph_input(
            time_col=variables_config["time_col"],
            outcome_col=variables_config["outcome_col"],
            expl_vars=variables_config["expl_vars"],
            organization_ids=variables_config["organization_ids"]
        )
        
        assert validated.time_col == variables_config["time_col"]
        assert validated.outcome_col == variables_config["outcome_col"]
        assert validated.expl_vars == variables_config["expl_vars"]
        assert validated.organization_ids == variables_config["organization_ids"]

    @pytest.mark.unit
    def test_validate_input_empty_expl_vars(self):
        """Test input validation with empty explanatory variables."""
        from v6-cox-ph.miscellaneous import validate_coxph_input
        from vantage6.algorithm.tools.exceptions import UserInputError
        
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
        from v6-cox-ph.miscellaneous import validate_coxph_input
        from vantage6.algorithm.tools.exceptions import UserInputError
        
        with pytest.raises(UserInputError):
            validate_coxph_input(
                time_col="",
                outcome_col="event",
                expl_vars=["age"],
                organization_ids=[1, 2]
            )

    @pytest.mark.unit
    def test_compute_derivatives_basic(self):
        """Test derivative computation with basic inputs."""
        from v6-cox-ph.coxph_logic import compute_derivatives
        import numpy as np
        import pandas as pd
        
        # Create test data
        summed_agg1 = np.array([10.0, 8.0, 6.0])
        summed_agg2 = np.array([[1.0, 2.0], [1.5, 2.5], [0.5, 1.5]])
        summed_agg3 = np.array([
            [[1.0, 2.0], [2.0, 4.0]],
            [[1.5, 2.5], [2.5, 4.5]],
            [[0.5, 1.5], [1.5, 3.0]]
        ])
        
        aggregated_time_events = pd.DataFrame({
            'time': [1.0, 2.0, 3.0],
            'freq': [2, 3, 1]
        })
        
        z_sum = pd.Series([1.0, 2.0])
        
        # This should not raise any exception
        primary, secondary = compute_derivatives(
            summed_agg1, summed_agg2, summed_agg3,
            aggregated_time_events, z_sum
        )
        
        assert primary.shape[0] == 2, "Primary derivative should have 2 elements"
        assert secondary.shape == (2, 2), "Secondary derivative should be 2x2 matrix"

    @pytest.mark.unit
    def test_format_results_dataframe(self):
        """Test results DataFrame formatting."""
        from v6-cox-ph.coxph_logic import format_results_dataframe
        import numpy as np
        
        results_data = {
            "Coef": [0.5, -0.3],
            "Exp(coef)": [1.6487, 0.7408],
            "SE": [0.1, 0.05],
            "lower_CI": [1.3, 0.6],
            "upper_CI": [2.0, 0.9],
            "Z": [5.0, -6.0],
            "p-value": [0.001, 0.0001]
        }
        expl_vars = ["age", "treatment"]
        
        df = format_results_dataframe(results_data, expl_vars)
        
        assert len(df) == 2, "DataFrame should have 2 rows"
        assert list(df.index) == expl_vars, "Index should be the explanatory variables"
        assert "Coef" in df.columns, "Should have Coef column"
        assert df.loc["age", "Coef"] == 0.5, "age coefficient should be 0.5"
