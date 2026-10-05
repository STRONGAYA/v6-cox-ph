# AGENTS.md — v6-cox-ph

A guide for coding agents (Junie, Mistral Vibe, Claude Code, Copilot, …)
working in this repository. Read this before touching node-side code.

## What this repository is

A federated Cox proportional hazards model (Breslow ties) for
[vantage6](https://vantage6.ai) 4.14–4.15 (the range pinned in
``pyproject.toml`` and tested in CI). One Docker image exposes a single
``central`` aggregator function and three ``partial`` functions that run on
data stations. The partials share only aggregated quantities with the
central aggregator; no row-level data leaves a node.

## Repository layout

```
v6-cox-ph/                 algorithm package (the hyphenated name is intentional)
  central.py               aggregator: dispatches sub-tasks, Newton-Raphson loop
  partial.py               node-side: get_unique_event_times, compute_summed_z,
                           perform_iteration (all decorated, all guarded)
  coxph_logic.py           pure math: derivatives, Newton step, model results
  privacy_guards.py        pure guards: parent-task check, thresholds, binning,
                           tail censoring, risk-set jump guard
  miscellaneous.py         pydantic input model, data-quality helpers
  __init__.py              exports central and the partials
algorithm_store.json       wire contract — function names, arguments, types
Dockerfile                 container image; entrypoint is
                           ``wrap_algorithm`` from vantage6.algorithm.tools.wrap
pyproject.toml             dependencies, dev/lint extras, pytest config
tests/
  unit/                    no-Docker tests (math, guards, mock pipeline)
  integration/             Docker + v6 dev network tests
  data/                    coxph_test_data_{1,2,3}.csv, node/store configs
  conftest.py              Docker/vantage6 fixtures, markers
docs/coxph/                Privacy.rst, Validation.rst, Usage.rst, etc.
.github/workflows/         test-suite.yml (test + lint + security), release.yaml
```

## How the algorithm executes

```
user task → central → get_unique_event_times → compute_summed_z
           → perform_iteration × N (Newton–Raphson)
```

- ``central`` is decorated with ``@algorithm_client`` and is the only function
  a user should call. It validates input, collects organisation IDs,
  dispatches sub-tasks, runs the Newton–Raphson loop, and returns the model.
- The three partials are decorated with ``@data(1)`` and ``@algorithm_client``.
  They run on each node and return aggregates. Their keyword arguments
  **must match** ``algorithm_store.json`` — that file is the wire contract.
  Each partial declares exactly one database there, matching ``@data(1)``.
- A node that fails the sample-size threshold is handled differently per
  partial, on purpose. ``get_unique_event_times`` returns the marker
  ``{"N-Threshold not met": org_id}`` so that ``central`` can exclude that
  organisation and retry (at most three rounds). ``compute_summed_z`` and
  ``perform_iteration`` raise ``PrivacyThresholdViolation``, which stops the
  whole analysis. That can still happen for a node that passed the first
  check, once rows with NaN in ``expl_vars`` are dropped. Do not "unify" the
  two without changing ``central`` to match.
- ``central`` builds the event-time grid and per-time event counts
  (``aggregated_time_events``) from the ``times`` returned by
  ``compute_summed_z``, **not** from ``get_unique_event_times``. Only
  ``compute_summed_z`` also drops rows with NaN in ``expl_vars``, so only its
  counts are consistent with ``z_sum`` and the risk sets.
  ``get_unique_event_times`` is used for node exclusion only.
- The Newton–Raphson loop tests ``max|step| <= 1e-6`` *before* applying the
  step, so the reported ``beta``, the Hessian and ``summed_agg1`` are all
  evaluated at the same ``beta``. Results include ``converged`` and
  ``n_iterations``.
- Results must be JSON-serialisable (they travel over the vantage6 wire).
- ``MAX_ITERATIONS`` (module-level in ``central.py``) controls the epoch
  budget; tests monkeypatch it to force non-convergence.

## Environment and commands

Python 3.10. Install in editable mode with the test (``dev``) and lint
(``lint``) extras::

    pip install -e .[dev,lint]

Unit tests (no Docker, run in a few seconds)::

    pytest -m unit
    # or
    pytest tests/unit -q

Integration tests (require Docker and the ``v6`` CLI). Missing
infrastructure makes them skip locally but fail when ``CI`` or
``REQUIRE_INTEGRATION_TESTS`` is set to ``1``/``true``/``yes``; set the latter
to force failures locally::

    pytest tests/integration
    REQUIRE_INTEGRATION_TESTS=1 pytest tests/integration

Lint and format (mirrors ``.github/workflows/test-suite.yml``; Black reads
``line-length = 120`` from ``[tool.black]`` in ``pyproject.toml``)::

    black --fast --check --diff --target-version py310 v6-cox-ph/ tests/
    flake8 v6-cox-ph/ tests/ --max-line-length=120 --extend-ignore=E203,W503
    ln -s v6-cox-ph v6_cox_ph
    mypy v6_cox_ph --ignore-missing-imports --follow-imports=silent
    rm v6_cox_ph

Build the Docker image::

    docker build -t v6-cox-ph:ci-test .

## Testing conventions

- Markers: ``unit``, ``integration``, ``slow``, ``vantage6``, ``docker``
  (``--strict-markers`` is on). Unit tests are tagged ``@pytest.mark.unit``.
- Assert numbers, not just finiteness. ``compute_derivatives``,
  ``compute_model_results``, the guards and the pipeline all have exact-value
  assertions.
- ``MockAlgorithmClient`` (from ``vantage6.algorithm.tools.mock_client``)
  runs ``central`` end-to-end in-process. The parent-task guard skips it by
  an explicit ``isinstance`` check, **not** because it has no
  ``_access_token``. Any other client without a token fails closed (see
  Privacy rules). Import the central module via
  ``import_module("v6-cox-ph.central")`` because the package name has a
  hyphen.
- ``lifelines.CoxPHFitter`` is the reference, but it uses Efron ties while
  this algorithm uses Breslow ties — expect small differences and use the
  tolerances already in the tests.
- Test data lives in ``tests/data/``. Node 3 has too few events and is the
  small-node exclusion case.
- Exactness tests disable the risk-set guards with
  ``COXPH_MIN_RISK_SET_CHANGE=1``; the default-guard test asserts the
  privacy property separately.

## Privacy rules (non-negotiable for node-side code)

- **Never return row-level data** from a partial. Only aggregates leave a
  node.
- **Every partial keeps its guards**, in order:
  ``ensure_spawned_by_central`` → ``load_privacy_settings``
  → ``validate_expl_vars`` (``compute_summed_z``, ``perform_iteration``)
  → ``drop_incomplete_rows`` → ``check_sample_size``
  → ``validate_iteration_input`` (``perform_iteration`` only)
  → ``prepare_time_column`` → (work; ``perform_iteration`` builds its risk
  sets with ``guarded_risk_set_masks``). ``get_unique_event_times`` also
  runs ``check_data_quality`` before ``drop_incomplete_rows``. Do not
  reorder or skip any of them.
- **No new partial** without a sample-size threshold, parent-task guard and
  documentation in ``Privacy.rst``.
- **Do not log data values or counts**; use ``info``/``warn`` for status
  only. The container log is returned to the researcher with the run, so a
  row or event count, especially a below-threshold one, is a leak. Name the
  configured threshold, never the actual count.
- **Do not loosen** thresholds or tolerances to make tests pass.
- Privacy settings are read from node ``algorithm_env`` via ``get_env_var``:
  ``SAMPLE_SIZE_THRESHOLD`` (default 10), ``COXPH_TIME_BIN_WIDTH`` (disabled
  by default), ``COXPH_MIN_RISK_SET_CHANGE`` (default 5, must be >= 1, and
  ``SAMPLE_SIZE_THRESHOLD + 1 >= COXPH_MIN_RISK_SET_CHANGE``).
- The parent-task guard fails closed: if the task lookup fails or the token
  is missing (and the client is not a ``MockAlgorithmClient``), raise
  ``AlgorithmError``, never proceed.
- Rows with NaN in ``time_col``, ``outcome_col`` or any ``expl_var`` are
  dropped before the sample-size threshold is checked.

## Coding conventions

- numpy/pandas vectorised maths; avoid per-row Python loops in the hot path.
- Pydantic input validation via ``CoxPHInput`` / ``validate_coxph_input``.
- vantage6 logging: ``info``/``warn``/``error`` from
  ``vantage6.algorithm.tools.util``.
- Raise vantage6 exception types: ``UserInputError``,
  ``PrivacyThresholdViolation``, ``PrivacyViolation``, ``AlgorithmError``.
- UK English in prose, comments, docstrings and log messages (organisation,
  behaviour, optimise). Identifiers keep vantage6's US spelling and must not
  be "corrected": ``organization_ids``, ``client.organization_id``,
  ``client.organization.list()``, ``included_organizations`` and
  ``excluded_organizations`` are part of the wire contract or the vantage6
  API.
- Black formatting, line length 120, target ``py310`` (configured in
  ``[tool.black]`` in ``pyproject.toml``); flake8 uses the same 120 limit.

## Branches, docs and releases

- ``standard-coxph`` — stable; the default branch and PR target. There is no
  ``main`` branch.
- ``phase1-algorithm-improvements`` — mathematical and privacy hardening of
  the standard algorithm (convergence, result validation, node-side guards).
- ``phase2-strong-aya`` — STRONG AYA guards and ``safe_log``; keep changes
  portable across branches.
- Update ``docs/coxph`` and ``algorithm_store.json`` whenever a signature or
  output field changes.
- CI lives in ``.github/workflows``. ``test-suite.yml`` (test, lint,
  security) runs on pushes and PRs to ``standard-coxph``, ``phase1-…``,
  ``phase2-…`` and ``main`` (kept in case the default branch is renamed).
  ``release.yaml`` runs when a git tag is pushed and publishes
  ``ghcr.io/<owner>/<repository>-<branch>:<tag>``, where ``<branch>`` is the
  branch that contains the tagged commit.

## Do not

- Do not change wire signatures without updating ``algorithm_store.json``
  **and** ``central`` in the same commit.
- Do not commit test data or local configuration with secrets.
- Do not add ``__init__.py`` author/license headers unless asked.
- Do not rename the ``v6-cox-ph`` package directory — the hyphen is part of
  the vantage6 module contract.
