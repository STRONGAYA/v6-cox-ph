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
from vantage6.algorithm.tools.exceptions import PrivacyThresholdViolation

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
    """Run the federated algorithm on nodes 1 and 2 (guards off).

    Node 3 has too few events; under the fail-closed threshold policy a run
    that includes it stops with ``PrivacyThresholdViolation``, so the
    happy-path fixtures select organisations 1 and 2 explicitly.
    """
    df1, df2, _ = _load_datasets()
    datasets = [
        [{"database": df1, "db_type": "csv"}],
        [{"database": df2, "db_type": "csv"}],
    ]
    return _run_central(datasets, [1, 2])


@pytest.mark.unit
class TestFederatedPipelineVsLifelines:
    """Compare the federated result (guards off) against a lifelines reference."""

    def test_run_including_below_threshold_node_fails(self, guards_off):
        """A run that includes node 3 (too few events) fails closed (D4)."""
        df1, df2, df3 = _load_datasets()
        datasets = [
            [{"database": df1, "db_type": "csv"}],
            [{"database": df2, "db_type": "csv"}],
            [{"database": df3, "db_type": "csv"}],
        ]
        with pytest.raises(PrivacyThresholdViolation):
            _run_central(datasets, [1, 2, 3])

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
                f"coef mismatch {var}: fed={model.loc[var, 'Coef']}, " f"ref={ref['coef'][var]}"
            )

    def test_se_match_lifelines(self, standard_result):
        """Federated SE within 1e-3 of lifelines."""
        df1, df2, _ = _load_datasets()
        ref = _lifelines_reference(pd.concat([df1, df2]), EXPL_VARS)
        model = pd.read_json(StringIO(standard_result["model"]))
        for var in EXPL_VARS:
            assert abs(model.loc[var, "SE"] - ref["se"][var]) <= 1e-3, (
                f"SE mismatch {var}: fed={model.loc[var, 'SE']}, " f"ref={ref['se'][var]}"
            )

    def test_z_matches_beta_over_se(self, standard_result):
        """Regression: Z must equal Coef / SE (not the old (exp(beta)-1)/SE)."""
        model = pd.read_json(StringIO(standard_result["model"]))
        for var in EXPL_VARS:
            coef = model.loc[var, "Coef"]
            se = model.loc[var, "SE"]
            z = model.loc[var, "Z"]
            assert abs(z - coef / se) <= 1e-9, f"Z != Coef/SE for {var}: Z={z}, Coef/SE={coef / se}"

    def test_aic_close_to_lifelines(self, standard_result):
        """AIC within 0.1 of the lifelines AIC."""
        df1, df2, _ = _load_datasets()
        ref = _lifelines_reference(pd.concat([df1, df2]), EXPL_VARS)
        assert (
            abs(standard_result["aic"] - ref["aic"]) <= 0.1
        ), f"AIC mismatch: fed={standard_result['aic']}, ref={ref['aic']}"


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
        pooled = _run_central([[{"database": pooled_df, "db_type": "csv"}]], [1])

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
        """With a single epoch the optimiser cannot converge.

        FR-A1: the reported Coef is beta_0 (all zeros) because the final
        Newton step is not applied. SE, Z and AIC are evaluated at the
        same beta_0.
        """
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
        assert "last evaluated beta" in warning_text
        # A model is still returned
        assert result["model"] is not None

        # FR-A1: Coef == 0 (beta_0 reported, not beta_1)
        model = pd.read_json(StringIO(result["model"]))
        for var in EXPL_VARS:
            assert (
                model.loc[var, "Coef"] == 0.0
            ), f"Non-converged Coef should be 0 (beta_0), got {model.loc[var, 'Coef']}"
            # Z = Coef / SE = 0 / SE = 0
            assert model.loc[var, "Z"] == 0.0
            # SE must be finite (computed from the Hessian at beta_0)
            assert np.isfinite(model.loc[var, "SE"])


@pytest.mark.unit
class TestSingularHessian:
    """A singular Hessian produces a graceful result, not a crash."""

    def test_collinear_covariate(self, monkeypatch, guards_off):
        """FR-A2: collinear covariate -> converged=False, NaN SE, JSON-serialisable."""
        df1, df2, _ = _load_datasets()
        df1 = df1.copy()
        df2 = df2.copy()
        df1["age2"] = df1["age"]
        df2["age2"] = df2["age"]
        collinear_vars = ["age", "age2"]

        client = MockAlgorithmClient(
            datasets=[
                [{"database": df1, "db_type": "csv"}],
                [{"database": df2, "db_type": "csv"}],
            ],
            module="v6-cox-ph",
            organization_ids=[1, 2],
        )
        task = client.task.create(
            input_={
                "method": "central",
                "kwargs": {
                    "time_col": "time",
                    "outcome_col": "event",
                    "expl_vars": collinear_vars,
                    "organization_ids": [1, 2],
                },
            },
            organizations=[1],
        )
        results = client.wait_for_results(task_id=task["id"])
        result = results[0]

        assert result["converged"] is False
        model = pd.read_json(StringIO(result["model"]))
        for var in collinear_vars:
            assert np.isnan(model.loc[var, "SE"]), f"SE for {var} should be NaN"
        assert result["model"] is not None
        # The singular cause must appear in the warnings list
        warning_text = " ".join(result["warnings"])
        assert "singular" in warning_text.lower(), f"Expected 'singular' in warnings, got: {warning_text}"


@pytest.mark.unit
class TestSubTaskValidation:
    """The validation helpers catch malformed sub-task results (FR-A3)."""

    def test_validate_iteration_result_bad_agg1_length(self):
        from importlib import import_module

        central = import_module("v6-cox-ph.central")
        with pytest.raises(Exception, match="agg1 has length"):
            central._validate_iteration_result(
                {"agg1": [1.0, 2.0], "agg2": {}, "agg3": []},
                n_times=5,
                n_covs=2,
                expl_vars=["age", "treatment"],
                org_id=1,
            )

    def test_validate_iteration_result_missing_key(self):
        from importlib import import_module

        central = import_module("v6-cox-ph.central")
        with pytest.raises(Exception, match="missing key"):
            central._validate_iteration_result(
                {"agg1": [1.0]},
                n_times=1,
                n_covs=1,
                expl_vars=["age"],
                org_id=1,
            )

    def test_validate_zsum_result_missing_sum(self):
        from importlib import import_module

        central = import_module("v6-cox-ph.central")
        with pytest.raises(Exception, match="missing 'sum'"):
            central._validate_zsum_result({}, ["age", "treatment"], "time", org_id=1)

    def test_validate_zsum_result_missing_variable(self):
        from importlib import import_module

        central = import_module("v6-cox-ph.central")
        with pytest.raises(Exception, match="missing variables"):
            central._validate_zsum_result(
                {"sum": {"age": 10.0}, "times": {"time": [1.0], "freq": [1]}},
                ["age", "treatment"],
                "time",
                org_id=1,
            )

    def test_validate_zsum_result_missing_times(self):
        from importlib import import_module

        central = import_module("v6-cox-ph.central")
        with pytest.raises(Exception, match="missing 'times'"):
            central._validate_zsum_result(
                {"sum": {"age": 10.0, "treatment": 5.0}},
                ["age", "treatment"],
                "time",
                org_id=1,
            )


@pytest.mark.unit
class TestNaNPolicy:
    """NaN rows are dropped identically in all partials (FR-B4)."""

    def test_nan_rows_dropped_consistently(self, monkeypatch, guards_off):
        """A NaN in age on an event row of node 1 is dropped in
        compute_summed_z (and thus excluded from event counts, z_sum and
        risk sets); the federated result matches lifelines on the NaN-free
        pooled data.

        Putting the NaN on an *event* row is critical: without the shared
        preparation step the event counts (freq) could include rows that
        z_sum and the risk sets exclude — a 10-20x tolerance violation.
        """
        df1, df2, _ = _load_datasets()
        df1 = df1.copy()
        # Introduce NaN in age on the first event row (row 1, event=1)
        event_rows = df1[df1["event"] == 1].index[:3].tolist()
        for idx in event_rows:
            df1.loc[idx, "age"] = np.nan

        datasets = [
            [{"database": df1, "db_type": "csv"}],
            [{"database": df2, "db_type": "csv"}],
        ]
        result = _run_central(datasets, [1, 2])
        assert result["converged"] is True

        # Build the NaN-free reference
        df1_clean = df1.dropna(subset=["time", "event", "age", "treatment"])
        ref = _lifelines_reference(pd.concat([df1_clean, df2]), EXPL_VARS)
        model = pd.read_json(StringIO(result["model"]))
        for var in EXPL_VARS:
            assert abs(model.loc[var, "Coef"] - ref["coef"][var]) <= 2e-3, (
                f"coef mismatch {var}: fed={model.loc[var, 'Coef']}, " f"ref={ref['coef'][var]}"
            )


def _per_node_agg1_at_beta_zero(datasets, organization_ids):
    """Dispatch perform_iteration at beta=0 and return per-node agg1 lists.

    At beta=0, exp(beta . X) = 1, so agg1[t] equals the risk-set size |R(t)|.
    """
    client = MockAlgorithmClient(
        datasets=datasets,
        module="v6-cox-ph",
        organization_ids=organization_ids,
    )

    # Gather the pooled event-time grid from compute_summed_z results.
    task = client.task.create(
        input_={
            "method": "compute_summed_z",
            "kwargs": {"time_col": "time", "outcome_col": "event", "expl_vars": EXPL_VARS},
        },
        organizations=organization_ids,
    )
    time_results = client.wait_for_results(task_id=task["id"])
    grid = []
    for r in time_results:
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

    def test_agg1_changes_at_least_k(self, monkeypatch):
        """Every positive decrease in per-node agg1 is >= k and min >= k."""
        monkeypatch.delenv("COXPH_MIN_RISK_SET_CHANGE", raising=False)
        monkeypatch.delenv("COXPH_TIME_BIN_WIDTH", raising=False)
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
                assert values[i] <= values[i - 1] + 1e-9, f"node {node_idx}: agg1 not non-increasing at {i}"
            # Every positive change (decrease) is >= k
            for i in range(1, len(values)):
                decrease = values[i - 1] - values[i]
                if decrease > 1e-9:
                    assert decrease >= k - 1e-9, f"node {node_idx}: decrease {decrease} < k={k} at {i}"
            # Smallest non-zero value is >= k
            non_zero = [v for v in values if v > 1e-9]
            assert non_zero, f"node {node_idx}: all agg1 are zero"
            assert min(non_zero) >= k - 1e-9, f"node {node_idx}: min non-zero agg1 {min(non_zero)} < k={k}"

    def test_coefficients_within_loose_bound(self, monkeypatch):
        """With default guards, coefficients stay within 0.1 of lifelines."""
        monkeypatch.delenv("COXPH_MIN_RISK_SET_CHANGE", raising=False)
        monkeypatch.delenv("COXPH_TIME_BIN_WIDTH", raising=False)
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
                f"coef {var}: fed={model.loc[var, 'Coef']}, " f"ref={ref['coef'][var]}"
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
                "method": "compute_summed_z",
                "kwargs": {
                    "time_col": "time",
                    "outcome_col": "event",
                    "expl_vars": EXPL_VARS,
                },
            },
            organizations=[1, 2],
        )
        results = client.wait_for_results(task_id=task["id"])
        shared_times = []
        for r in results:
            t = pd.DataFrame.from_dict(r["times"])
            shared_times.extend(t["time"].tolist())
        for t in shared_times:
            assert t % 10 == 0 or np.isclose(t % 10, 0), f"shared time {t} is not a multiple of 10"

        # The full pipeline still converges.
        result = _run_central(datasets, [1, 2])
        assert result["converged"] is True
