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
