"""
Docker-free end-to-end pipeline tests for the federated Cox-PH algorithm.

These tests drive ``central`` in-process through ``MockAlgorithmClient`` and
compare the federated output against ``lifelines.CoxPHFitter`` fitted on the
pooled data of the included organisations. They require no Docker or vantage6
network.
"""

import sys
from io import StringIO
from pathlib import Path
from importlib import import_module

import numpy as np
import pandas as pd
import pytest

# Add the algorithm module and repo root to the path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "v6-cox-ph"))

from vantage6.algorithm.tools.mock_client import MockAlgorithmClient  # noqa: E402

# Import the central module through the (hyphenated) package so that its
# relative imports resolve correctly.
central_module = import_module("v6-cox-ph.central")  # noqa: E402


DATA_DIR = repo_root / "tests" / "data"


def _load_datasets():
    """Load the three test CSVs."""
    df1 = pd.read_csv(DATA_DIR / "coxph_test_data_1.csv")
    df2 = pd.read_csv(DATA_DIR / "coxph_test_data_2.csv")
    df3 = pd.read_csv(DATA_DIR / "coxph_test_data_3.csv")
    return df1, df2, df3


def _run_central(datasets, organization_ids):
    """Run ``central`` through the mock client and return the result dict."""
    client = MockAlgorithmClient(
        datasets=datasets,
        module="v6-cox-ph",
        organization_ids=organization_ids,
    )
    task = client.task.create(
        input_={
            "method": "central",
            "kwargs": {
                "time_col": "time",
                "outcome_col": "event",
                "expl_vars": ["age", "treatment"],
                "organization_ids": organization_ids,
            },
        },
        organizations=[organization_ids[0]],
    )
    results = client.wait_for_results(task_id=task["id"])
    return results[0]


def _lifelines_reference(df, expl_vars):
    """Fit a centralised Cox-PH model with lifelines and return key outputs."""
    from lifelines import CoxPHFitter

    central_df = df[["time", "event"] + expl_vars].copy()
    central_df["event"] = central_df["event"].astype(bool)
    cph = CoxPHFitter()
    cph.fit(central_df, duration_col="time", event_col="event")
    return {
        "coef": cph.params_.to_dict(),
        "se": cph.standard_errors_.to_dict(),
        "aic": float(cph.AIC_partial_),
        "z": {k: cph.params_[k] / cph.standard_errors_[k] for k in cph.params_.index},
    }


@pytest.fixture
def standard_result():
    """Run the federated algorithm on the three-node test data."""
    df1, df2, df3 = _load_datasets()
    datasets = [
        [{"database": df1, "db_type": "csv"}],
        [{"database": df2, "db_type": "csv"}],
        [{"database": df3, "db_type": "csv"}],
    ]
    return _run_central(datasets, [1, 2, 3])


@pytest.mark.unit
class TestFederatedPipelineVsLifelines:
    """Compare the federated result against a lifelines reference."""

    def test_excludes_small_node(self, standard_result):
        """Node 3 (<= 10 events) should be excluded; nodes 1 and 2 included."""
        assert 3 in standard_result["excluded_organizations"]
        assert sorted(standard_result["included_organizations"]) == [1, 2]

    def test_converged(self, standard_result):
        """The optimiser should report convergence within a few iterations."""
        assert standard_result["converged"] is True
        assert isinstance(standard_result["n_iterations"], int)
        assert standard_result["n_iterations"] <= 10

    def test_coefficients_match_lifelines(self, standard_result):
        """Federated coefficients within 2e-3 of lifelines on pooled data."""
        df1, df2, _ = _load_datasets()
        ref = _lifelines_reference(pd.concat([df1, df2]), ["age", "treatment"])
        model = pd.read_json(StringIO(standard_result["model"]))
        for var in ["age", "treatment"]:
            assert abs(model.loc[var, "Coef"] - ref["coef"][var]) <= 2e-3, (
                f"coef mismatch {var}: fed={model.loc[var, 'Coef']}, "
                f"ref={ref['coef'][var]}"
            )

    def test_se_match_lifelines(self, standard_result):
        """Federated SE within 1e-3 of lifelines."""
        df1, df2, _ = _load_datasets()
        ref = _lifelines_reference(pd.concat([df1, df2]), ["age", "treatment"])
        model = pd.read_json(StringIO(standard_result["model"]))
        for var in ["age", "treatment"]:
            assert abs(model.loc[var, "SE"] - ref["se"][var]) <= 1e-3, (
                f"SE mismatch {var}: fed={model.loc[var, 'SE']}, "
                f"ref={ref['se'][var]}"
            )

    def test_z_matches_beta_over_se(self, standard_result):
        """Regression: Z must equal Coef / SE (not the old (exp(beta)-1)/SE)."""
        model = pd.read_json(StringIO(standard_result["model"]))
        for var in ["age", "treatment"]:
            coef = model.loc[var, "Coef"]
            se = model.loc[var, "SE"]
            z = model.loc[var, "Z"]
            # Coef and SE are rounded to 5 decimals, so allow rounding slack;
            # the original bug produces a ~13 % error on treatment.
            assert abs(z - coef / se) <= 1e-3, (
                f"Z != Coef/SE for {var}: Z={z}, Coef/SE={coef / se}"
            )

    def test_aic_close_to_lifelines(self, standard_result):
        """AIC within 0.1 of the lifelines AIC."""
        df1, df2, _ = _load_datasets()
        ref = _lifelines_reference(pd.concat([df1, df2]), ["age", "treatment"])
        assert abs(standard_result["aic"] - ref["aic"]) <= 0.1, (
            f"AIC mismatch: fed={standard_result['aic']}, ref={ref['aic']}"
        )


@pytest.mark.unit
class TestFederatedEqualsPooled:
    """The federated run must equal a centralised run on the pooled data."""

    def test_coefficients_identical_to_pooled(self):
        """Federated (2 nodes) coefficients equal pooled single-node to 1e-8."""
        df1, df2, _ = _load_datasets()

        fed = _run_central(
            [
                [{"database": df1, "db_type": "csv"}],
                [{"database": df2, "db_type": "csv"}],
            ],
            [1, 2],
        )

        pooled_df = pd.concat([df1, df2], ignore_index=True)
        pooled = _run_central(
            [[{"database": pooled_df, "db_type": "csv"}]], [1]
        )

        fed_model = pd.read_json(StringIO(fed["model"]))
        pooled_model = pd.read_json(StringIO(pooled["model"]))
        for var in ["age", "treatment"]:
            np.testing.assert_allclose(
                fed_model.loc[var, "Coef"],
                pooled_model.loc[var, "Coef"],
                atol=1e-8,
                err_msg=f"coef differs from pooled for {var}",
            )


@pytest.mark.unit
class TestNonConvergence:
    """The output reports non-convergence when the epoch budget is exhausted."""

    def test_forced_non_convergence(self, monkeypatch):
        """With a single epoch the optimiser cannot converge."""
        monkeypatch.setattr(central_module, "MAX_ITERATIONS", 1)

        df1, df2, _ = _load_datasets()
        result = _run_central(
            [
                [{"database": df1, "db_type": "csv"}],
                [{"database": df2, "db_type": "csv"}],
            ],
            [1, 2],
        )

        assert result["converged"] is False
        assert result["n_iterations"] == 1
        warning_text = " ".join(result["warnings"])
        assert "did not converge" in warning_text
        # A model is still returned
        assert result["model"] is not None
