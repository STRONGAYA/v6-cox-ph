"""
Docker-free end-to-end pipeline tests for the federated Cox-PH algorithm.

These tests drive ``central`` in-process through ``MockAlgorithmClient`` and
compare the federated output against ``lifelines.CoxPHFitter`` fitted on the
pooled data of the included organisations. They require no Docker or vantage6
network.

Exactness tests run with ``COXPH_MIN_RISK_SET_CHANGE=1`` (guards disabled) so
the federated result matches lifelines closely. A separate test exercises the
default guards (``k=5``) and asserts the risk-set privacy property on per-node
``agg1`` at ``beta = 0``.
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
EXPL_VARS = ["age", "treatment"]


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
                "expl_vars": EXPL_VARS,
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
    }


@pytest.fixture
def guards_off(monkeypatch):
    """Disable the risk-set jump guard (k=1) and binning for exactness tests."""
    monkeypatch.setenv("COXPH_MIN_RISK_SET_CHANGE", "1")
    monkeypatch.delenv("COXPH_TIME_BIN_WIDTH", raising=False)


@pytest.fixture
def standard_result(guards_off):
    """Run the federated algorithm on the three-node test data (guards off)."""
    df1, df2, df3 = _load_datasets()
    datasets = [
        [{"database": df1, "db_type": "csv"}],
        [{"database": df2, "db_type": "csv"}],
        [{"database": df3, "db_type": "csv"}],
    ]
    return _run_central(datasets, [1, 2, 3])


@pytest.mark.unit
class TestFederatedPipelineVsLifelines:
    """Compare the federated result (guards off) against a lifelines reference."""

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
        ref = _lifelines_reference(pd.concat([df1, df2]), EXPL_VARS)
        model = pd.read_json(StringIO(standard_result["model"]))
        for var in EXPL_VARS:
            assert abs(model.loc[var, "Coef"] - ref["coef"][var]) <= 2e-3, (
                f"coef mismatch {var}: fed={model.loc[var, 'Coef']}, "
                f"ref={ref['coef'][var]}"
            )

    def test_se_match_lifelines(self, standard_result):
        """Federated SE within 1e-3 of lifelines."""
        df1, df2, _ = _load_datasets()
        ref = _lifelines_reference(pd.concat([df1, df2]), EXPL_VARS)
        model = pd.read_json(StringIO(standard_result["model"]))
        for var in EXPL_VARS:
            assert abs(model.loc[var, "SE"] - ref["se"][var]) <= 1e-3, (
                f"SE mismatch {var}: fed={model.loc[var, 'SE']}, "
                f"ref={ref['se'][var]}"
            )

    def test_z_matches_beta_over_se(self, standard_result):
        """Regression: Z must equal Coef / SE (not the old (exp(beta)-1)/SE)."""
        model = pd.read_json(StringIO(standard_result["model"]))
        for var in EXPL_VARS:
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
        ref = _lifelines_reference(pd.concat([df1, df2]), EXPL_VARS)
        assert abs(standard_result["aic"] - ref["aic"]) <= 0.1, (
            f"AIC mismatch: fed={standard_result['aic']}, ref={ref['aic']}"
        )


@pytest.mark.unit
class TestFederatedEqualsPooled:
    """The federated run must equal a centralised run on the pooled data."""

    def test_coefficients_identical_to_pooled(self, guards_off):
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
        for var in EXPL_VARS:
            np.testing.assert_allclose(
                fed_model.loc[var, "Coef"],
                pooled_model.loc[var, "Coef"],
                atol=1e-8,
                err_msg=f"coef differs from pooled for {var}",
            )


@pytest.mark.unit
class TestNonConvergence:
    """The output reports non-convergence when the epoch budget is exhausted."""

    def test_forced_non_convergence(self, monkeypatch, guards_off):
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


def _per_node_agg1_at_beta_zero(datasets, organization_ids):
    """Dispatch perform_iteration at beta=0 and return per-node agg1 lists.

    At beta=0, exp(beta . X) = 1, so agg1[t] equals the risk-set size |R(t)|.
    """
    client = MockAlgorithmClient(
        datasets=datasets,
        module="v6-cox-ph",
        organization_ids=organization_ids,
    )

    # Gather the global event-time grid from get_unique_event_times.
    task = client.task.create(
        input_={
            "method": "get_unique_event_times",
            "kwargs": {"time_col": "time", "outcome_col": "event"},
        },
        organizations=organization_ids,
    )
    time_results = client.wait_for_results(task_id=task["id"])
    grid = []
    for r in time_results:
        if "times" in r:
            t = pd.DataFrame.from_dict(r["times"])
            grid.extend(t["time"].tolist())
    grid = sorted(set(grid))

    # Dispatch perform_iteration at beta = 0.
    task = client.task.create(
        input_={
            "method": "perform_iteration",
            "kwargs": {
                "time_col": "time",
                "expl_vars": EXPL_VARS,
                "beta": [0.0, 0.0],
                "unique_time_events": grid,
            },
        },
        organizations=organization_ids,
    )
    iter_results = client.wait_for_results(task_id=task["id"])
    return [r["agg1"] for r in iter_results]


@pytest.mark.unit
class TestDefaultGuardsPrivacyProperty:
    """With default guards (k=5), shared risk-set aggregates are protected."""

    def test_agg1_changes_at_least_k(self):
        """Every positive decrease in per-node agg1 is >= k and min >= k."""
        k = 5
        df1, df2, _ = _load_datasets()
        datasets = [
            [{"database": df1, "db_type": "csv"}],
            [{"database": df2, "db_type": "csv"}],
        ]
        per_node_agg1 = _per_node_agg1_at_beta_zero(datasets, [1, 2])

        for node_idx, agg1 in enumerate(per_node_agg1):
            values = [float(v) for v in agg1]
            # Non-increasing
            for i in range(1, len(values)):
                assert values[i] <= values[i - 1] + 1e-9, (
                    f"node {node_idx}: agg1 not non-increasing at {i}"
                )
            # Every positive change (decrease) is >= k
            for i in range(1, len(values)):
                decrease = values[i - 1] - values[i]
                if decrease > 1e-9:
                    assert decrease >= k - 1e-9, (
                        f"node {node_idx}: decrease {decrease} < k={k} at {i}"
                    )
            # Smallest non-zero value is >= k
            non_zero = [v for v in values if v > 1e-9]
            assert non_zero, f"node {node_idx}: all agg1 are zero"
            assert min(non_zero) >= k - 1e-9, (
                f"node {node_idx}: min non-zero agg1 {min(non_zero)} < k={k}"
            )

    def test_coefficients_within_loose_bound(self):
        """With default guards, coefficients stay within 0.1 of lifelines."""
        df1, df2, _ = _load_datasets()
        datasets = [
            [{"database": df1, "db_type": "csv"}],
            [{"database": df2, "db_type": "csv"}],
        ]
        result = _run_central(datasets, [1, 2])
        ref = _lifelines_reference(pd.concat([df1, df2]), EXPL_VARS)
        model = pd.read_json(StringIO(result["model"]))
        for var in EXPL_VARS:
            assert abs(model.loc[var, "Coef"] - ref["coef"][var]) <= 0.1, (
                f"coef {var}: fed={model.loc[var, 'Coef']}, "
                f"ref={ref['coef'][var]}"
            )


@pytest.mark.unit
class TestTimeBinning:
    """With COXPH_TIME_BIN_WIDTH set, shared event times are on the grid."""

    def test_shared_times_on_grid(self, monkeypatch):
        """All shared event times are multiples of the bin width and it converges."""
        monkeypatch.setenv("COXPH_TIME_BIN_WIDTH", "10")
        monkeypatch.setenv("COXPH_MIN_RISK_SET_CHANGE", "1")

        df1, df2, _ = _load_datasets()
        datasets = [
            [{"database": df1, "db_type": "csv"}],
            [{"database": df2, "db_type": "csv"}],
        ]
        client = MockAlgorithmClient(
            datasets=datasets,
            module="v6-cox-ph",
            organization_ids=[1, 2],
        )
        task = client.task.create(
            input_={
                "method": "get_unique_event_times",
                "kwargs": {"time_col": "time", "outcome_col": "event"},
            },
            organizations=[1, 2],
        )
        results = client.wait_for_results(task_id=task["id"])
        shared_times = []
        for r in results:
            if "times" in r:
                t = pd.DataFrame.from_dict(r["times"])
                shared_times.extend(t["time"].tolist())
        for t in shared_times:
            assert t % 10 == 0 or np.isclose(t % 10, 0), (
                f"shared time {t} is not a multiple of 10"
            )

        # The full pipeline still converges.
        result = _run_central(datasets, [1, 2])
        assert result["converged"] is True
