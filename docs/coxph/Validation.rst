Validation
==========

The algorithm is validated at three levels.

Numeric unit tests
-------------------

``tests/unit/test_coxph_logic.py`` asserts exact numbers for the core
mathematical functions: ``compute_derivatives`` returns the expected primary
and secondary derivatives on a small fixture; ``compute_model_results``
asserts the covariance matrix, standard errors, Z-values (``Z = beta / SE``),
p-values (``2 * Phi(-|Z|)``), confidence intervals, the overall Wald
statistic and the AIC against hand-computed values. A regression test
verifies ``Z == Coef / SE`` to catch the previous ``(exp(beta) - 1) / SE`` bug.

``tests/unit/test_privacy_guards.py`` covers settings parsing, the
sample-size threshold (rows and events, strict inequality), the parent-task
guard (parent absent / present / lookup failure / bad token / mock client),
iteration-input validation, time binning, tail censoring and the risk-set
jump guard.

Docker-free federated pipeline test
------------------------------------

``tests/unit/test_federated_pipeline.py`` drives ``central`` in-process
through ``MockAlgorithmClient`` on the three-node test data and compares
with ``lifelines.CoxPHFitter`` fitted on the pooled data of the selected
organisations. With the risk-set guards disabled
(``COXPH_MIN_RISK_SET_CHANGE=1``) it asserts:

- coefficients within ``2e-3`` of lifelines;
- standard errors within ``1e-3``;
- ``Z == Coef / SE`` (regression check);
- AIC within ``0.1``;
- ``converged is True`` and ``n_iterations <= 10``;
- a run that includes node 3 (too few events) fails with
  ``PrivacyThresholdViolation`` — organisations are selected explicitly;
- federated (2 nodes) equals pooled single-node coefficients to ``1e-8``;
- forced non-convergence (``epochs = 1``) reports ``converged=False`` with a
  warning.

With default guards (``k = 5``) it asserts the privacy property on per-node
``agg1`` at ``beta = 0``: the values are non-increasing, every positive
decrease is ``>= 5`` and the smallest non-zero value is ``>= 5``, while the
coefficients stay within ``0.1`` of lifelines. A binning test
(``COXPH_TIME_BIN_WIDTH=10``) checks that all shared event times are
multiples of 10 and the pipeline converges.

Risk-set performance
--------------------

``perform_iteration`` computes its risk-set aggregates with a vectorised
``searchsorted``/``bincount``/reverse-cumsum implementation
(``guarded_risk_set_aggregates``) instead of T x N boolean masks; the jump
guard is derived from the bucket counts alone. The previous mask
implementation is kept verbatim as the reference in
``tests/unit/reference_risk_sets.py`` and randomised equivalence tests
(ties, rows on and off grid points, ``k`` in {1, 3, 5}) assert agreement
to ``rtol=1e-12``.

Benchmark (``scripts/benchmark_risk_sets.py``, 100,000 rows x 10,000
event times x 10 covariates, ``k=5``):

=========================  ============  =============
Implementation             Time (s)      Peak memory
=========================  ============  =============
Mask-based (reference)     89.3          716.8 MB
Vectorised (current)        0.12          28.2 MB
=========================  ============  =============

Guard bias study
----------------

``scripts/bias_study.py`` (not collected by pytest) measures how much the
privacy guards perturb the coefficients: simulated Cox data (known true
coefficients, three nodes), sizes {200, 1k, 10k} per node, k in {1, 5, 10},
time binning off/on, 50 repetitions per configuration, parallelised per
configuration. It reports the mean absolute deviation from the k = 1 fit on
the same data, the deviation from the true coefficients and the 95 %
confidence-interval coverage.

Smoke-run results (2 repetitions, sizes 200 and 1000; the table below is
provisional until the full 50-repetition run is executed with
``python scripts/bias_study.py --reps 50``)::

    size   k  binning  mean|d vs k=1|  mean|d vs true|  95% CI coverage
     200   1  off/on            0.000            ~0.07              1.00
     200   5      off            0.011            0.07              1.00
     200  10      off            0.025            0.07              1.00
     200 5/10     on            ~0.04            0.06              1.00
    1000   5      off            0.002           0.033              ~0.5*
    1000  10      off            0.005           0.032              ~0.5*
    1000 5/10     on             0.04           0.027              1.00

  * coverage at 2 repetitions is noise; the full run gives the real rate.

The fixed-seed unit regression ``TestGuardBiasRegression`` bounds the
default-guards coefficient deviation at 0.05, derived from this study with
headroom; re-derive the bound after the full run rather than loosening it.

Run the unit suite without Docker::

    pytest -m unit
    # or
    pytest tests/unit -q

Integration tests
-----------------

``tests/integration/test_algorithm_integration.py`` dispatches real tasks
through a ``v6 dev`` network. ``determine_model_acceptance`` builds the
lifelines reference on the per-node row slices (the demo network partitions
each CSV across nodes by row index). The organisation-to-slice mapping is a
property of the network creation, so the reference is built on candidate
slice combinations and the federated coefficients must match at least one
candidate. The demo nodes run with ``COXPH_MIN_RISK_SET_CHANGE=1`` and
``SAMPLE_SIZE_THRESHOLD=5`` (see ``tests/data/
additional_vantage6_node_config.yaml``); the fail-closed threshold
behaviour is exercised through the ``coxph_test_data_3`` scenarios. The
assertions include:

- coefficients within ``0.05`` of the reference;
- SE within 10 % relative;
- ``Z == Coef / SE`` to ``1e-4``;
- ``p == 2 * norm.cdf(-|Z|)``;
- AIC finite;
- ``converged`` and ``n_iterations`` present.

A negative test submits each partial method directly (as a user task) and
expects a privacy error, proving the partials cannot be used as a query
oracle.

Run the integration suite (requires Docker and the vantage6 CLI)::

    pytest tests/integration

These tests skip locally when Docker or the vantage6 CLI is absent and fail
in continuous integration, so the CI run reports a real result.

lifelines as reference
----------------------

``lifelines`` uses Efron ties by default while this algorithm uses Breslow
ties, so small differences in coefficients, SE and AIC are expected. The
tolerances above accommodate this. To run a quick comparison locally::

    python3.10 -m pytest tests/unit/test_federated_pipeline.py -q
