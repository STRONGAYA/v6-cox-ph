"""
Unit tests for Cox-PH mathematical logic.

Tests the core mathematical functions in coxph_logic.py against known
correct values computed from the test data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add the algorithm module to the path
algorithm_path = Path(__file__).parent.parent.parent / "v6-cox-ph"
sys.path.insert(0, str(algorithm_path))

from coxph_logic import (  # noqa: E402
    compute_derivatives,
    compute_model_results,
    format_results_dataframe,
)


@pytest.fixture
def sample_aggregated_time_events():
    """Create a small aggregated time events DataFrame."""
    return pd.DataFrame({"time": [5.0, 10.0, 15.0], "freq": [2, 3, 1]})


@pytest.fixture
def sample_z_sum():
    """Create a sample z_sum Series for 2 covariates."""
    return pd.Series([10.0, 20.0], index=["age", "treatment"])


@pytest.fixture
def sample_aggs():
    """Create sample aggregated values for 3 event times, 2 covariates."""
    summed_agg1 = np.array([5.0, 8.0, 3.0])
    summed_agg2 = np.array([[2.0, 4.0], [3.0, 6.0], [1.0, 2.0]])
    summed_agg3 = np.array(
        [
            [[1.0, 2.0], [2.0, 4.0]],
            [[1.5, 3.0], [3.0, 6.0]],
            [[0.5, 1.0], [1.0, 2.0]],
        ]
    )
    return summed_agg1, summed_agg2, summed_agg3


@pytest.mark.unit
class TestComputeDerivatives:
    """Tests for the compute_derivatives function."""

    def test_compute_derivatives_returns_arrays(self, sample_aggs, sample_aggregated_time_events, sample_z_sum):
        """Test that compute_derivatives returns numpy arrays."""
        agg1, agg2, agg3 = sample_aggs
        primary, secondary = compute_derivatives(agg1, agg2, agg3, sample_aggregated_time_events, sample_z_sum)
        assert isinstance(primary, np.ndarray)
        assert isinstance(secondary, np.ndarray)

    def test_compute_derivatives_shape(self, sample_aggs, sample_aggregated_time_events, sample_z_sum):
        """Test that derivatives have correct shapes."""
        agg1, agg2, agg3 = sample_aggs
        primary, secondary = compute_derivatives(agg1, agg2, agg3, sample_aggregated_time_events, sample_z_sum)
        assert primary.shape == (2,)
        assert secondary.shape == (2, 2)

    def test_compute_derivatives_known_values(self, sample_aggs, sample_aggregated_time_events, sample_z_sum):
        """Test derivatives against manually computed values."""
        agg1, agg2, agg3 = sample_aggs
        primary, secondary = compute_derivatives(agg1, agg2, agg3, sample_aggregated_time_events, sample_z_sum)

        # Manual computation for first event time (index 0, freq=2):
        # s1 = 2 * ([2, 4] / 5) = [0.8, 1.6]
        # first_part = [[1, 2], [2, 4]] / 5 = [[0.2, 0.4], [0.4, 0.8]]
        # numerator = outer([2,4], [2,4]) = [[4, 8], [8, 16]]
        # second_part = [[4, 8], [8, 16]] / 25 = [[0.16, 0.32], [0.32, 0.64]]
        # s2 = 2 * ([0.2-0.16, 0.4-0.32], [0.4-0.32, 0.8-0.64])
        #    = 2 * [[0.04, 0.08], [0.08, 0.16]] = [[0.08, 0.16], [0.16, 0.32]]
        # ... continuing for all 3 event times and summing
        np.testing.assert_allclose(primary, [7.741667, 15.483333], atol=1e-6)
        np.testing.assert_allclose(
            secondary,
            [[-0.276181, -0.552361], [-0.552361, -1.104722]],
            atol=1e-6,
        )

    def test_compute_derivatives_raises_on_nonpositive_s1(
        self, sample_aggs, sample_aggregated_time_events, sample_z_sum
    ):
        """A risk-set sum of zero, negative or NaN raises AlgorithmError."""
        from vantage6.algorithm.tools.exceptions import AlgorithmError

        _, agg2, agg3 = sample_aggs
        for bad in (0.0, -1.0, float("nan")):
            agg1 = np.array([bad, 8.0, 3.0])
            with pytest.raises(AlgorithmError):
                compute_derivatives(agg1, agg2, agg3, sample_aggregated_time_events, sample_z_sum)


@pytest.mark.unit
class TestComputeModelResults:
    """Tests for the compute_model_results function."""

    @pytest.fixture
    def model_fixture(self, sample_z_sum):
        """Common fixture for model results tests."""
        beta = np.array([0.5, -0.3])
        secondary = np.array([[-10.0, 2.0], [2.0, -8.0]])
        agg_time_events = pd.DataFrame({"time": [5.0, 10.0], "freq": [2, 3]})
        summed_agg1 = np.array([5.0, 8.0])
        result = compute_model_results(
            beta=beta,
            secondary_derivative=secondary,
            z_sum=sample_z_sum,
            aggregated_time_events=agg_time_events,
            summed_agg1=summed_agg1,
            expl_vars=["age", "treatment"],
        )
        return result, beta, secondary, agg_time_events, summed_agg1

    def test_compute_model_results_returns_dict(self, model_fixture):
        """Test that compute_model_results returns a properly structured dict."""
        result, *_ = model_fixture
        assert isinstance(result, dict)
        assert "results_data" in result
        assert "covariance" in result
        assert "standard_errors" in result
        assert "zvalues" in result
        assert "pvalues" in result
        assert "overall_p_value" in result
        assert "aic" in result
        assert "n_params" in result
        assert "warnings" in result

    def test_compute_model_results_n_params(self, model_fixture):
        """Test that n_params matches the number of coefficients."""
        result, *_ = model_fixture
        assert result["n_params"] == 2

    def test_compute_model_results_covariance(self, model_fixture):
        """Test the covariance matrix is the inverse of the negative Hessian."""
        result, beta, secondary, *_ = model_fixture
        expected_cov = np.linalg.inv(-secondary)
        np.testing.assert_allclose(result["covariance"], expected_cov)

    def test_compute_model_results_se(self, model_fixture):
        """Test standard errors are the square root of the covariance diagonal."""
        result, beta, secondary, *_ = model_fixture
        expected_se = np.sqrt(np.diag(np.linalg.inv(-secondary)))
        np.testing.assert_allclose(result["standard_errors"], expected_se)

    def test_compute_model_results_z_is_beta_over_se(self, model_fixture):
        """Regression test: Z must equal Coef / SE (not (exp(beta)-1)/SE)."""
        result, *_ = model_fixture
        se = result["standard_errors"]
        beta = np.array([0.5, -0.3])
        np.testing.assert_allclose(result["zvalues"], beta / se, atol=1e-9)

    def test_compute_model_results_z_values(self, model_fixture):
        """Test Z-values against manually computed values."""
        result, *_ = model_fixture
        np.testing.assert_allclose(result["zvalues"], [1.5411035, -0.82704293], atol=1e-6)

    def test_compute_model_results_p_values(self, model_fixture):
        """Test p-values are 2*Phi(-|Z|)."""
        from scipy.stats import norm

        result, *_ = model_fixture
        z = result["zvalues"]
        expected_p = 2 * norm.cdf(-np.abs(z))
        np.testing.assert_allclose(result["pvalues"], expected_p)

    def test_compute_model_results_confidence_intervals(self, model_fixture):
        """Test confidence intervals are exp(beta ± 1.96*SE)."""
        result, *_ = model_fixture
        beta = np.array([0.5, -0.3])
        se = result["standard_errors"]
        rd = result["results_data"]
        np.testing.assert_allclose(rd["lower_CI"], np.exp(beta - 1.96 * se), atol=1e-5)
        np.testing.assert_allclose(rd["upper_CI"], np.exp(beta + 1.96 * se), atol=1e-5)

    def test_compute_model_results_wald(self, model_fixture):
        """Test the overall Wald statistic and p-value."""
        result, beta, secondary, *_ = model_fixture
        from scipy.stats import chi2

        expected_wald = beta @ (-secondary) @ beta
        expected_p = float(chi2.sf(expected_wald, 2))
        np.testing.assert_allclose(result["overall_p_value"], expected_p)

    def test_compute_model_results_aic(self, model_fixture):
        """Test AIC against manually computed value."""
        result, beta, secondary, agg_time_events, summed_agg1 = model_fixture
        z_sum = pd.Series([10.0, 20.0])
        linear_part = np.dot(z_sum.values, beta)
        risk_set_part = 2 * np.log(5.0) + 3 * np.log(8.0)
        expected_ll = linear_part - risk_set_part
        expected_aic = -2 * expected_ll + 2 * 2
        assert result["aic"] is not None
        np.testing.assert_allclose(result["aic"], expected_aic, atol=1e-6)

    def test_compute_model_results_warnings_for_large_coef(self, sample_z_sum):
        """Test that warnings are generated for large coefficients."""
        beta = np.array([20.0, -0.3])  # Large coefficient
        secondary = np.array([[-10.0, 2.0], [2.0, -8.0]])
        agg_time_events = pd.DataFrame({"time": [5.0, 10.0], "freq": [2, 3]})
        summed_agg1 = np.array([5.0, 8.0])

        result = compute_model_results(
            beta=beta,
            secondary_derivative=secondary,
            z_sum=sample_z_sum,
            aggregated_time_events=agg_time_events,
            summed_agg1=summed_agg1,
            expl_vars=["age", "treatment"],
        )
        assert len(result["warnings"]) > 0
        assert "age" in result["warnings"][0]

    def test_compute_model_results_non_converged_warning(self, sample_z_sum):
        """Test that a non-convergence warning is appended."""
        beta = np.array([0.5, -0.3])
        secondary = np.array([[-10.0, 2.0], [2.0, -8.0]])
        agg_time_events = pd.DataFrame({"time": [5.0, 10.0], "freq": [2, 3]})
        summed_agg1 = np.array([5.0, 8.0])

        result = compute_model_results(
            beta=beta,
            secondary_derivative=secondary,
            z_sum=sample_z_sum,
            aggregated_time_events=agg_time_events,
            summed_agg1=summed_agg1,
            expl_vars=["age", "treatment"],
            converged=False,
            n_iterations=10,
        )
        warning_text = " ".join(result["warnings"])
        assert "did not converge" in warning_text
        assert "10" in warning_text

    def test_compute_model_results_converged_no_warning(self, model_fixture):
        """Test that no convergence warning is appended when converged."""
        result, *_ = model_fixture
        for w in result["warnings"]:
            assert "did not converge" not in w


@pytest.mark.unit
class TestFormatResultsDataframe:
    """Tests for the format_results_dataframe function."""

    def test_format_results_dataframe_index(self):
        """Test that the results DataFrame is indexed by variable name."""
        results_data = {
            "Coef": np.array([0.5, -0.3]),
            "Exp(coef)": np.array([1.65, 0.74]),
            "SE": np.array([0.1, 0.2]),
        }
        df = format_results_dataframe(results_data, ["age", "treatment"])
        assert df.index.tolist() == ["age", "treatment"]
        assert "Coef" in df.columns
        assert "Exp(coef)" in df.columns
        assert "SE" in df.columns

    def test_format_results_dataframe_values(self):
        """Test that the results DataFrame contains the correct values."""
        results_data = {
            "Coef": np.array([0.5, -0.3]),
            "Exp(coef)": np.array([1.65, 0.74]),
            "SE": np.array([0.1, 0.2]),
        }
        df = format_results_dataframe(results_data, ["age", "treatment"])
        assert df.loc["age", "Coef"] == 0.5
        assert df.loc["treatment", "Coef"] == -0.3
