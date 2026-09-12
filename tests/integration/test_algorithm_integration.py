"""
Comprehensive Vantage6 integration testing for the v6-cox-ph algorithm.

These tests dispatch real tasks through the vantage6 network, retrieve results,
parse logs for errors, and validate the federated model output against a
centralised Cox-PH fit on the combined test data.
"""

import json
from io import StringIO
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from vantage6.algorithm.tools.exceptions import (
    AlgorithmError,
    CollectResultsError,
    PrivacyThresholdViolation,
    UserInputError,
)


@pytest.fixture
def test_methods():
    """
    Fixture providing method kwargs templates for different test scenarios.

    Parameters set to None are filled from test_configurations by the test methods.
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
def test_configurations():
    """
    Fixture providing test configurations for algorithm validation.

    Each configuration specifies real parameters for the Cox-PH algorithm.
    Configurations marked as expected_failure should raise the specified
    error types.
    """
    return {
        "standard_dataset": {
            "database_label": "coxph_test_data_1",
            "time_col": "time",
            "outcome_col": "event",
            "expl_vars": ["age", "treatment"],
            "organisation_subset": [1, 2, 3],
        },
        "standard_dataset_bad_actor": {
            "database_label": "coxph_test_data_1",
            "time_col": "time",
            "outcome_col": "event",
            "expl_vars": ["age", "treatment"],
            "organisation_subset": [1],
            "expected_failure": True,
            "failure_reason": (
                "Single organisation may not meet sample size threshold."
            ),
            "expected_error_type": [CollectResultsError, PrivacyThresholdViolation],
        },
        "standard_dataset_incorrect_input": {
            "database_label": "coxph_test_data_1",
            "time_col": "time",
            "outcome_col": "event",
            "expl_vars": ["NonExistentVariable"],
            "organisation_subset": [1, 2, 3],
            "expected_failure": True,
            "failure_reason": "Non-existent variable requested.",
            "expected_error_type": [UserInputError, CollectResultsError],
        },
        "rare_dataset": {
            "database_label": "coxph_test_data_3",
            "time_col": "time",
            "outcome_col": "event",
            "expl_vars": ["age", "treatment"],
            "organisation_subset": [1],
            "expected_failure": True,
            "failure_reason": "Dataset with insufficient event count.",
            "expected_error_type": [CollectResultsError, PrivacyThresholdViolation],
        },
    }


@pytest.mark.integration
class TestCoxPHAlgorithmIntegration:
    """
    Comprehensive integration tests for the Cox-PH algorithm.

    These tests dispatch real tasks through the vantage6 network and validate
    results empirically against a centralised fit on the combined test data.
    """

    @pytest.mark.parametrize("method", ["central"])
    @pytest.mark.parametrize(
        "config_name",
        [
            "standard_dataset",
            "standard_dataset_incorrect_input",
            "rare_dataset",
        ],
    )
    def test_algorithm_basic(
        self,
        authentication,
        algorithm_image_name,
        test_configurations,
        test_methods,
        method,
        config_name,
    ):
        """
        Test algorithm with basic configuration, including expected failures.

        Dispatches a real task to the vantage6 network and validates the
        result empirically for happy paths, or checks error types for failures.
        """
        client = authentication
        config = test_configurations[config_name]
        method_config = test_methods[method]

        kwargs = method_config["basic"].copy()
        kwargs["time_col"] = config["time_col"]
        kwargs["outcome_col"] = config["outcome_col"]
        kwargs["expl_vars"] = config["expl_vars"]

        task = client.task.create(
            collaboration=1,
            organizations=[1],
            name=f"Test {method} — {config_name}",
            image=algorithm_image_name,
            description=f"Integration test for {method} using {config_name}.",
            input_={"method": method, "kwargs": kwargs},
            databases=[{"label": config["database_label"]}],
        )

        if config.get("expected_failure", False):
            with pytest.raises(Exception) as exc_info:
                extract_coxph_result(client, task)
            verify_error_type(exc_info, config)
        else:
            result = extract_coxph_result(client, task)
            determine_model_acceptance(result, config["database_label"], kwargs)

    @pytest.mark.parametrize("method", ["central"])
    @pytest.mark.parametrize(
        "config_name",
        [
            "standard_dataset",
            "standard_dataset_bad_actor",
            "standard_dataset_incorrect_input",
            "rare_dataset",
        ],
    )
    def test_algorithm_organisation_selection(
        self,
        authentication,
        algorithm_image_name,
        test_configurations,
        test_methods,
        method,
        config_name,
    ):
        """
        Test algorithm with organisation selection.

        Tests running the algorithm on a subset of organisations, including
        privacy threshold violations when too few organisations are selected.
        """
        client = authentication
        config = test_configurations[config_name]
        method_config = test_methods[method]

        if "organisation_selection" not in method_config:
            pytest.skip(f"Organisation selection not supported for {method}")

        kwargs = method_config["organisation_selection"].copy()
        kwargs["time_col"] = config["time_col"]
        kwargs["outcome_col"] = config["outcome_col"]
        kwargs["expl_vars"] = config["expl_vars"]
        kwargs["organization_ids"] = config["organisation_subset"]

        task = client.task.create(
            collaboration=1,
            organizations=[1],
            name=f"Test {method} org selection — {config_name}",
            image=algorithm_image_name,
            description=(
                f"Integration test for {method} with organisation "
                f"selection using {config_name}."
            ),
            input_={"method": method, "kwargs": kwargs},
            databases=[{"label": config["database_label"]}],
        )

        if config.get("expected_failure", False):
            with pytest.raises(Exception) as exc_info:
                extract_coxph_result(client, task)
            verify_error_type(exc_info, config)
        else:
            result = extract_coxph_result(client, task)
            determine_model_acceptance(result, config["database_label"], kwargs)

    @pytest.mark.parametrize("method", ["central"])
    @pytest.mark.parametrize(
        "config_name",
        [
            "standard_dataset",
            "standard_dataset_bad_actor",
            "standard_dataset_incorrect_input",
            "rare_dataset",
        ],
    )
    def test_algorithm_parameter_galore(
        self,
        authentication,
        algorithm_image_name,
        test_configurations,
        test_methods,
        method,
        config_name,
    ):
        """
        Test algorithm with all parameters combined (parameter galore).

        Combines all parameters: time, outcome, expl_vars, and organisation IDs.
        """
        client = authentication
        config = test_configurations[config_name]
        method_config = test_methods[method]

        if "parameter_galore" not in method_config:
            pytest.skip(f"Parameter galore not supported for {method}")

        kwargs = method_config["parameter_galore"].copy()
        kwargs["time_col"] = config["time_col"]
        kwargs["outcome_col"] = config["outcome_col"]
        kwargs["expl_vars"] = config["expl_vars"]
        kwargs["organization_ids"] = config["organisation_subset"]

        task = client.task.create(
            collaboration=1,
            organizations=[1],
            name=f"Test {method} parameter galore — {config_name}",
            image=algorithm_image_name,
            description=(
                f"Integration test for {method} with all parameters "
                f"using {config_name}."
            ),
            input_={"method": method, "kwargs": kwargs},
            databases=[{"label": config["database_label"]}],
        )

        if config.get("expected_failure", False):
            with pytest.raises(Exception) as exc_info:
                extract_coxph_result(client, task)
            verify_error_type(exc_info, config)
        else:
            result = extract_coxph_result(client, task)
            determine_model_acceptance(result, config["database_label"], kwargs)


def extract_coxph_result(client, task) -> Dict[str, Any]:
    """
    Extract Cox-PH model results from the algorithm task result.

    Waits for results, checks the task log for errors, and returns the
    deserialised result dictionary.

    Parameters
    ----------
    client : Client
        Authenticated vantage6 client.
    task : dict
        Task object returned from task creation.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing model results (coefficients, p-values, AIC, etc.)

    Raises
    ------
    AlgorithmError
        If an error is found in the task log.
    """
    print("Waiting for results")
    task_id = task["id"]
    result = client.wait_for_results(task_id)

    # Check for errors in the log
    run_info = client.run.from_task(task_id)
    log = run_info["data"][0]["log"]

    if "Traceback" in log:
        print(f"Error found in task log: {log}")
        lines = log.split("\n")

        # Look for vantage6 exception lines first
        for line in lines:
            if line.strip().startswith("vantage6.algorithm.tools.exceptions."):
                error_class_line = line.strip()
                error_message = (
                    error_class_line.split(": ", 1)[1]
                    if ": " in error_class_line
                    else "Unknown error"
                )
                if "UserInputError" in error_class_line:
                    raise UserInputError(error_message)
                elif "CollectResultsError" in error_class_line:
                    raise CollectResultsError(error_message)
                elif "PrivacyThresholdViolation" in error_class_line:
                    raise PrivacyThresholdViolation(error_message)
                else:
                    raise AlgorithmError(
                        f"Unknown error type in log: {error_class_line}"
                    )

        # Look for error > lines
        error_lines = [line for line in lines if line.startswith("error >")]
        if error_lines:
            error_message = error_lines[-1].replace("error >", "").strip()
            if error_message and error_message != "None":
                raise AlgorithmError(f"Algorithm execution failed: {error_message}")

        # Traceback found but no recognised error pattern — raise generic error
        raise AlgorithmError(
            f"Traceback found in task log but no error type recognised: {log}"
        )

    assert result is not None, "Result should not be None"

    result = json.loads(result["data"][0]["result"])
    return result


def verify_error_type(exc_info, config: dict) -> None:
    """
    Verify that the raised exception matches the expected error type(s).

    Parameters
    ----------
    exc_info : ExceptionInfo
        The exception info from pytest.raises.
    config : dict
        The test configuration with expected_error_type.
    """
    expected_errors = config.get("expected_error_type")
    if expected_errors:
        if not isinstance(expected_errors, list):
            expected_errors = [expected_errors]

        error_matched = any(
            isinstance(exc_info.value, expected_error)
            for expected_error in expected_errors
        )
        assert error_matched, (
            f"Expected one of {[err.__name__ for err in expected_errors]} "
            f"but got {type(exc_info.value).__name__}"
        )

    print(f"Expected failure occurred for {config}: {exc_info.value}")


def determine_model_acceptance(
    federated_result: Dict[str, Any],
    database_label: str,
    kwargs: Dict[str, Any],
    tolerance: float = 1e-3,
) -> None:
    """
    Validate federated results against a centralised Cox-PH fit.

    Loads the test data, fits a centralised Cox-PH model using lifelines,
    and compares the federated coefficients against the centralised fit
    within the specified tolerance.

    Parameters
    ----------
    federated_result : Dict[str, Any]
        Results from the federated computation.
    database_label : str
        Label of the test dataset to validate against.
    kwargs : Dict[str, Any]
        Algorithm kwargs containing time_col, outcome_col, expl_vars.
    tolerance : float
        Numerical tolerance for coefficient comparison.

    Raises
    ------
    AssertionError
        If federated coefficients deviate from centralised fit beyond tolerance.
    """
    from lifelines import CoxPHFitter

    repo_root = Path(__file__).parent.parent.parent
    dataset_file = repo_root / "tests" / "data" / f"{database_label}.csv"

    assert dataset_file.exists(), f"Test dataset not found: {dataset_file}"

    df = pd.read_csv(dataset_file)

    time_col = kwargs["time_col"]
    outcome_col = kwargs["outcome_col"]
    expl_vars = kwargs["expl_vars"]

    # Fit centralised Cox-PH model on the combined data
    cph = CoxPHFitter()
    central_df = df[[time_col, outcome_col] + expl_vars].copy()
    central_df[outcome_col] = central_df[outcome_col].astype(bool)
    cph.fit(central_df, duration_col=time_col, event_col=outcome_col)

    # Extract federated coefficients from the result
    model_json = federated_result.get("model")
    assert model_json is not None, "Model results should not be None"

    fed_df = pd.read_json(StringIO(model_json))

    # Compare each coefficient
    for var in expl_vars:
        if var in cph.params_.index and var in fed_df.index:
            central_coef = cph.params_[var]
            fed_coef = fed_df.loc[var, "Coef"]
            assert abs(fed_coef - central_coef) <= tolerance, (
                f"Coefficient mismatch for {var}: "
                f"federated={fed_coef}, centralised={central_coef}, "
                f"tolerance={tolerance}"
            )
            print(
                f"Coefficient validation passed for {var}: "
                f"federated={fed_coef}, centralised={central_coef}"
            )

    # Validate that p-values are present and finite
    assert "p-value" in fed_df.columns, "p-value column should be present"
    for var in expl_vars:
        if var in fed_df.index:
            pval = fed_df.loc[var, "p-value"]
            assert np.isfinite(pval), f"p-value for {var} should be finite"

    # Validate AIC is present
    assert (
        "aic" in federated_result or federated_result.get("aic") is not None
    ), "AIC should be present in the result"

    print("All model acceptance checks passed")
