"""
Unit tests for Cox-PH logic in v6-cox-ph.
"""

import pandas as pd
import numpy as np

from v6_cox_ph.coxph_logic import (
    compute_derivatives,
    format_results_dataframe
)


class TestComputeDerivatives:
    """Tests for compute_derivatives function."""

    def test_basic_derivatives(self):
        """Test derivative computation with basic inputs."""
        # Create test data
        summed_agg1 = np.array([10.0, 8.0, 6.0])
        summed_agg2 = np.array([[1.0, 2.0], [1.5, 2.5], [0.5, 1.5]])
        summed_agg3 = np.array([
            [[1.0, 2.0], [2.0, 4.0]],
            [[1.5, 2.5], [2.5, 4.5]],
            [[0.5, 1.5], [1.5, 3.0]]
        ])
        
        aggregated_time_events = pd.DataFrame({
            "time": [1.0, 2.0, 3.0],
            "freq": [2, 3, 1]
        })
        
        z_sum = pd.Series([1.0, 2.0])
        
        # This should not raise any exception
        primary, secondary = compute_derivatives(
            summed_agg1, summed_agg2, summed_agg3,
            aggregated_time_events, z_sum
        )
        
        assert primary.shape[0] == 2
        assert secondary.shape == (2, 2)


class TestFormatResultsDataframe:
    """Tests for format_results_dataframe function."""

    def test_format_results(self):
        """Test results DataFrame formatting."""
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
        
        assert len(df) == 2
        assert list(df.index) == expl_vars
        assert "Coef" in df.columns
        assert df.loc["age", "Coef"] == 0.5
