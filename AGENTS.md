<!-- BEGIN block1 vantage6-general (copied verbatim; edit at the source, never here) -->
<!-- v6-agents v3 (2026-10-26). Block 1 of an algorithm repository's AGENTS.md.
     Copied verbatim from the conventions repository; do not edit here. -->

## 1. vantage6 general rules

Handbook: the ``vantage6-federated-algorithm`` skill. This block is the rule
sheet; it applies to every federated algorithm on vantage6 4.x.

### Structure
- Central–partial pattern. ``central`` orchestrates and never touches data;
  partials run on nodes and return **aggregates only**.
- ``central.py`` and ``partial.py`` are recipes; maths lives in dedicated
  modules; guards in a pure-function guard module; Pydantic models in
  ``miscellaneous.py``/``models.py``.
- Package directory ``v6-<algorithm-name>`` is hyphenated on purpose and
  matches ``pyproject.toml`` and the Dockerfile ``PKG_NAME``. Relative imports
  inside the package; tests import with
  ``importlib.import_module("v6-<name>.<module>")``.
- ``algorithm_store.json`` is the wire contract; partial kwargs must match it;
  one database per partial (``@data(1)``).

### Validation
- Pydantic v2 at the three boundaries: user input at ``central`` entry
  (``UserInputError``), kwargs at partial entry (``UserInputError``), partial
  results on reception (``AlgorithmError`` naming the organisation).
- Results are JSON-serialisable and include ``algorithm_version``.
- Raise vantage6 exception types only: ``UserInputError``,
  ``PrivacyThresholdViolation``, ``PrivacyViolation``, ``AlgorithmError``.

### Privacy (non-negotiable for node-side code)
- Only aggregates leave a node; return the minimum the aggregator uses.
- Parent-task guard **fails closed**: missing/undecodable token or failed
  lookup ⇒ ``AlgorithmError``; no parent ⇒ ``PrivacyViolation``; the only
  exemption is ``isinstance(client, MockAlgorithmClient)``.
- Drop rows with missing values in every analysed column **before** any
  threshold; thresholds count rows (and events where applicable).
- **Same rows in every partial**: one shared ``prepare_node_data`` with a
  single marked hook; a unit test asserts the guard order.
- **Never log data values or counts.** Name the threshold, not the number.
- Never loosen thresholds or tolerances to make tests pass.
- No new partial without guard, threshold, shared preparation and a
  ``Privacy.rst`` entry.
- Privacy settings come from node ``algorithm_env`` via ``get_env_var``,
  validated at load and documented with defaults and constraints.

### Testing
- Markers ``unit``/``integration``/``slow``/``vantage6``/``docker``,
  ``--strict-markers``.
- Unit: assert numbers; guards in every branch; both sides of every
  threshold; ``MockAlgorithmClient`` pipeline vs a centralised reference at
  tight tolerance; guard-order test; injected-missing-values test.
- Integration: real ``client.task.create``; full method × configuration
  matrix; empirical validation on the rows each node actually received;
  specific exception types asserted; thresholds just above/below;
  ``_skip_or_fail`` (skip locally, fail in CI).
- Exactness tests may disable estimate-perturbing guards via their env var;
  a separate test asserts the privacy property with defaults on.
- ``tests/conftest.py`` and ``tests/integration/test_vantage6_integration.py``
  are copies from the conventions repository's ``test-infra/``; change only
  dataset names and ``variables_config`` locally; fix the shared files at
  their source.

### CI, Docker, releases
- ``test-suite.yml``: ``test`` (3.10 and 3.12 × supported vantage6),
  ``lint`` (Black via ``[tool.black]``, flake8 120, mypy via symlink),
  ``security`` (Bandit ``-ll`` + pip-audit, able to fail), ``docker``
  (build + import smoke test); ``permissions: contents: read``; Dependabot.
- ``security-scan.yml``: weekly cron + dispatch; pip-audit on the lock
  file, Bandit, Trivy/Grype on the **released image**; fails on
  HIGH/CRITICAL with a fix; opens a ``security-scan`` issue; suppressions
  committed with justification and expiry.
- Dockerfile: Alpine two-stage, ``PYTHON_VERSION`` build arg (3.12),
  digest-pinned base, ``requirements-lock.txt`` as constraints, no ``USER``.
- Semver; a breaking wire-contract change (one that breaks an existing
  request or result) is a major bump and its commit message starts with
  ``BREAKING(wire):``; an additive change — a new optional argument, a
  widened argument type that keeps the old form valid, a new result key —
  is a minor bump; store/central/docs in the same commit. Releases by tag;
  image ``ghcr.io/<owner>/<repo>-<branch>:<tag>`` with the branch whose tip
  is the tag.

### Security audits
- Handbook: the ``vantage6-security-audit`` skill. An audit is
  **mandatory** before a major release and after any wire-contract change.
- Audit artefacts (reports, findings, mitigation plans, evidence) are
  **confidential until released**: they live in the audit vault — a private
  store outside every code repository — never in this repository, the image,
  issues, PRs, commit messages or the CHANGELOG.
- Belt and braces: ``.gitignore``/``.dockerignore`` ignore
  ``security-audit/``, ``*-audit/`` and ``F-[0-9][0-9][0-9]*.md``; CI
  fails if any tracked file contains the audit header marker defined in
  the ``vantage6-security-audit`` skill.
- Mitigations are developed on a private fork and merged at release; the
  *released* gate needs the four sign-offs in the audit skill.

### Upstream and fork
- Upstream improvements first, free of project code; project layer on top,
  stacked in git. Upstream → fork by ``git merge`` only, never by copying
  commits. Fork → upstream never. Repository-specific material at the end
  of files.

### House style
- UK English in prose; vantage6 identifiers keep US spelling
  (``organization_ids``). Black 120 / py310. Formatting-only commits
  separate. No headers in ``__init__.py`` unless asked.

### Do not
- Return, log or echo row-level values or counts.
- Reorder, skip or weaken the guard sequence.
- Change a wire signature without store, central and docs in one commit.
- Edit blocks 1–3 of this file locally; change them at their source.
- Copy upstream commits by hand.
- Commit real records or configuration with secrets.
<!-- END block1 vantage6-general -->

<!-- BEGIN block2 strong-aya-layer (copied verbatim; edit at the source, never here) -->
<!-- strong-aya-agents v2 (2026-10-03). Block 2 of a STRONG AYA algorithm
     repository's AGENTS.md. Copied verbatim from the conventions repository;
     do not edit here. Requires block 1 (vantage6 general rules) above it. -->

## 2. STRONG AYA layer rules

Handbook: the ``strong-aya-algorithm`` skill. Everything in block 1 still
applies; this block adds the STRONG AYA libraries, data standard and
placement rules.

### Libraries (infrastructure only — no algorithm maths)
- ``vantage6_strongaya_general.miscellaneous``: ``safe_log(level, msg)``
  (replaces ``info``/``warn``/``error``/``print`` everywhere),
  ``collect_organisation_ids``, ``check_partial_result_presence``,
  ``apply_data_stratification``, ``set_datatypes``, ``VariableDetails``,
  ``StratificationDetails``.
- ``vantage6_strongaya_general.privacy_measures``:
  ``mask_unnecessary_variables``, ``apply_sample_size_threshold`` — **not**
  from ``miscellaneous``.
- ``vantage6_strongaya_rdf.collect_sparql_data`` — **partials only**, never in
  ``central.py``; pattern ``if "endpoint" in df.columns: df =
  collect_sparql_data(variables_to_analyse, endpoint=df["endpoint"].iloc[0])``.
- Dependencies pinned to tags:
  ``vantage6-strongaya-general @ git+https://github.com/STRONGAYA/v6-tools-general.git@vX``,
  ``vantage6-strongaya-rdf @ git+https://github.com/STRONGAYA/v6-tools-rdf.git@vY``;
  regenerate ``requirements-lock.txt`` after a bump.

### Where the STRONG AYA steps go
Inside the algorithm's ``prepare_node_data`` hook, in this order, **in every
partial**:
1. RDF collect if ``endpoint`` present
2. ``mask_unnecessary_variables(df, variables_to_analyse)``
3. ``set_datatypes`` — only with ``Dict[str, VariableDetails]``; never a bare
   list
4. ``apply_data_stratification`` (if the algorithm supports it)
5. *(native validation, NaN drop and native threshold run here)*
6. ``apply_sample_size_threshold(client, df, variables_to_analyse)``

The STRONG AYA guards are **in addition to** the algorithm's native guards,
never instead of them. ``safe_log`` obeys the no-values-no-counts rule.

### Data standard
- Variables are addressed by STRONG AYA names and typed via
  ``VariableDetails`` (``numerical``/``categorical``, inliers); stratification
  via ``StratificationDetails``. Validate the ``variables_to_*`` structures
  with Pydantic at ``central`` entry.

### Tests
- Integration matrix includes organisation selection, stratification,
  inlier and parameter-galore scenarios where the algorithm supports them;
  narrow-stratification and single-record attacks must raise
  ``PrivacyThresholdViolation``/``CollectResultsError``.
- Test infrastructure comes from the conventions repository ``test-infra/``
  (block 1 rule); ``STRONGAYA/v6-descriptive-statistics`` is the worked
  example of the full method × configuration matrix.

### Do not
- Import ``vantage6_strongaya_rdf`` in ``central.py``.
- Call ``apply_sample_size_threshold`` in some partials but not others.
- Introduce ``safe_log`` or STRONG AYA dependencies into upstream
  (block 3) code.
<!-- END block2 strong-aya-layer -->

<!-- BEGIN block3 upstream AGENTS.md (merged verbatim; edit upstream, never here) -->
# AGENTS.md — v6-cox-ph

A guide for coding agents (Junie, Mistral Vibe, Claude Code, Copilot, …)
working in this repository. Read this before touching node-side code.

## What this repository is

A federated Cox proportional hazards model (Breslow ties) for
[vantage6](https://vantage6.ai) 4.14–4.15 (the range pinned in
``pyproject.toml`` and tested in CI). One Docker image exposes a single
``central`` aggregator function and two ``partial`` functions that run on
data stations. The partials share only aggregated quantities with the
central aggregator; no row-level data leaves a node.

## Scope

This repository is the **vanilla** Cox-PH algorithm. Project-specific
features — data filtering, recoding, project privacy tooling, other data
sources — do not belong here; they go in project branches or forks. Vanilla
never depends on project code, and project code never flows back.

## Building on this algorithm

Projects build on this algorithm in a fork or a stacked branch. To stay
compatible with that:

- Any step that removes rows runs **before** ``check_sample_size`` and
  identically in every partial: put project row selection at the marked
  ``select_rows`` hook inside the shared ``prepare_node_data``. That way
  every aggregate central combines comes from the same row set on each
  node, and rows the project removes still count against the threshold.
- Never weaken the vanilla guards: the parent-task guard, the sample-size
  threshold and the logging rules stay as they are.
- A **breaking** wire-contract change — one that breaks an existing request
  or result — is a **major** version bump, and the commit message starts
  with ``BREAKING(wire):``. An additive change — a new optional argument,
  a widened argument type that keeps the old form valid, a new result key
  — is a minor bump.

## Repository layout

```
v6-cox-ph/                 algorithm package (the hyphenated name is intentional)
  central.py               aggregator: dispatches sub-tasks, Newton-Raphson loop
  partial.py               node-side: compute_summed_z, perform_iteration
                           (both decorated, both guarded)
  coxph_logic.py           pure math: derivatives, Newton step, model results
  privacy_guards.py        pure guards: parent-task check, thresholds, binning,
                           tail censoring, vectorised risk-set aggregates with
                           the jump guard
  miscellaneous.py         pydantic models for the wire contract (input,
                           iteration input, partial results),
                           data-quality helpers
  __init__.py              exports central and the partials
algorithm_store.json       wire contract — function names, arguments, types
Dockerfile                 container image; entrypoint is
                           ``wrap_algorithm`` from vantage6.algorithm.tools.wrap
pyproject.toml             dependencies, dev/lint extras, pytest config
tests/
  unit/                    no-Docker tests (math, guards, mock pipeline;
                           reference_risk_sets.py holds the pre-vectorisation
                           mask implementation for equivalence tests)
  integration/             Docker + v6 dev network tests
  data/                    coxph_test_data_{1,2,3}.csv, node/store configs
  conftest.py              Docker/vantage6 fixtures, markers
scripts/                   benchmark_risk_sets.py, bias_study.py (not
                           collected by pytest)
docs/coxph/                Privacy.rst, Validation.rst, Usage.rst, etc.
.github/workflows/         test-suite.yml (test + lint + security + docker),
                           security-scan.yml (weekly), release.yaml
.github/dependabot.yml     pip, Docker and action updates
```

## How the algorithm executes

```
user task → central → compute_summed_z
           → perform_iteration × N (Newton–Raphson)
```

- ``central`` is decorated with ``@algorithm_client`` and is the only function
  a user should call. It validates input, collects organisation IDs,
  dispatches sub-tasks, runs the Newton–Raphson loop, and returns the model.
- The two partials are decorated with ``@data(1)`` and ``@algorithm_client``.
  They run on each node and return aggregates. Their keyword arguments
  **must match** ``algorithm_store.json`` — that file is the wire contract.
  Each partial declares exactly one database there, matching ``@data(1)``.
- A node that fails the sample-size threshold raises
  ``PrivacyThresholdViolation`` in every partial, which stops the whole
  analysis; there is no automatic exclusion and no retry round. Researchers
  select organisations explicitly via ``organization_ids``.
- ``central`` builds the event-time grid and per-time event counts
  (``aggregated_time_events``) from the ``times`` returned by
  ``compute_summed_z``. It drops rows with NaN in ``expl_vars``, so its
  counts are consistent with ``z_sum`` and the risk sets.
- The covariates are standardised before iterating: ``central`` derives the
  pooled covariate centre and scale over the event cases from ``z_sum`` and
  the sum of squares (both returned by ``compute_summed_z``, rounded to two
  significant figures), nodes fit on ``(x - centre) / scale`` and ``central``
  back-transforms the reported coefficients, standard errors and covariance.
  The partial likelihood is invariant to this affine transform; centring
  prevents ``exp`` overflow and scaling fixes the Hessian conditioning.
- ``compute_summed_z`` also returns the node's privacy settings
  (configuration, not data); the result summary based on them is built in a
  later step.
- ``expl_vars`` entries are plain column names or single-entry mappings of a
  column name to a categorical spec (``reference`` + ``categories``) or a
  scored spec (``scores``, optional ``missing``); the store type of
  ``expl_vars`` is ``json`` and a list of names stays valid. The model terms
  they expand to (one per plain or scored variable, one per non-reference
  category) replace the variable names in the result's ``model``, the order
  of ``covariance`` and the partials' ``sum``/``sum_squares`` keys.
- ``central`` optionally takes ``covariate_profiles`` (validated by the
  Pydantic input model: keys must be known explanatory variables) and
  returns ``baseline_cumulative_hazard`` and ``survival_curves`` computed
  centrally from the pooled event counts and risk-set sums — no new partial,
  no new data leaves a node.
- ``perform_iteration`` receives ``outcome_col`` and drops rows with a
  missing outcome, so every partial analyses the same rows; a node that
  passed ``compute_summed_z`` cannot fail there — its threshold check is
  defence in depth only.
- The Newton–Raphson loop tests ``max|step| <= 1e-6`` *before* applying the
  step, so the reported ``beta``, the Hessian and ``summed_agg1`` are all
  evaluated at the same ``beta`` — the last *accepted* evaluation. A
  secondary tolerance also declares convergence when the log-likelihood gain
  stays at rounding level for two consecutive accepted evaluations while the
  step stays below ``1e-3`` (coarse aggregate rounding can stall the step
  above the primary tolerance without the fit having moved). The
  log-likelihood is computed centrally each iteration from the aggregates
  already collected; when it decreases the Newton step is halved and the
  previous accepted state is kept (a decrease is only visible at the next
  round-trip). Results include ``converged``, ``n_iterations`` and
  ``convergence_tolerance`` (the tolerance that fired), plus
  ``log_likelihood``, ``log_likelihood_null``, ``lr_statistic``,
  ``lr_p_value``, ``n_events``, ``covariance`` and ``algorithm_version``.
- Results must be JSON-serialisable (they travel over the vantage6 wire).
- ``MAX_ITERATIONS`` (module-level in ``central.py``, 20) controls the epoch
  budget; tests monkeypatch it to force non-convergence.

## Environment and commands

Python 3.10 or later. The container image runs Python 3.12; the code
supports ``>= 3.10`` until 3.10 is dropped deliberately (then ``requires-python``
and Black's ``target-version`` move up together). Install in editable mode
with the test (``dev``) and lint (``lint``) extras::

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

Build the Docker image (Alpine two-stage; the base is pinned by digest and
kept current by Dependabot)::

    docker build -t v6-cox-ph:ci-test .

The runtime dependencies are pinned in ``requirements-lock.txt`` and
installed as constraints (``-c``), not hashes. Regenerate after a dependency
change with::

    uv pip compile pyproject.toml --python-version 3.12 \
        --python-platform x86_64-unknown-linux-musl -o requirements-lock.txt

The lock targets musl/Python 3.12 (what the image runs); the CI test matrix
deliberately runs glibc and unlocked to test the declared range.

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
- Test data lives in ``tests/data/``. Node 3 has too few events: a run that
  includes it fails closed with ``PrivacyThresholdViolation``.
- Exactness tests disable the risk-set guards with
  ``COXPH_MIN_RISK_SET_CHANGE=1``; the default-guard test asserts the
  privacy property separately.

## Privacy rules (non-negotiable for node-side code)

- **Never return row-level data** from a partial. Only aggregates leave a
  node. The result dicts of the partials are built from the Pydantic
  result models, so their keys and nesting are pinned in one place.
- **Every partial prepares its data through the shared**
  ``prepare_node_data(client, df, time_col, outcome_col, covariates, *,
  need_outcome)`` (in ``privacy_guards.py``; ``covariates`` is the resolved
  ``ResolvedCovariates`` from ``resolve_expl_vars``), which runs the guards
  in order: ``ensure_spawned_by_central`` → ``load_privacy_settings`` →
  ``encode_covariates`` (when covariates are given) →
  ``validate_expl_vars`` (when covariates are given) →
  ``drop_incomplete_rows`` → ``validate_survival_columns`` →
  ``select_rows`` (the marked hook for project row selection) →
  ``check_sample_size`` → ``check_category_sizes`` (only when the threshold
  was met) → the unlisted-code ``UserInputError`` (only when every threshold
  passed). Do not reorder or skip any of them; the unit tests assert the
  order. ``validate_iteration_input`` (``perform_iteration`` only) and
  ``prepare_time_column`` run afterwards.
  ``perform_iteration`` builds its aggregates with the vectorised
  ``guarded_risk_set_aggregates`` (``searchsorted``/``bincount``/reverse
  cumsum; the jump guard comes from the bucket counts) — the previous mask
  implementation lives on in ``tests/unit/reference_risk_sets.py`` as the
  equivalence-test reference. ``agg2`` travels as a plain (T x p) list.
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
  ``SAMPLE_SIZE_THRESHOLD + 1 >= COXPH_MIN_RISK_SET_CHANGE``). The hardening
  knobs are selected by the ``COXPH_PRIVACY_LEVEL`` preset (``minimal`` /
  ``standard`` / ``strict``) and can be overridden individually — an
  explicitly set variable beats the preset:
  ``COXPH_RECONSTRUCTION_ROW_FACTOR``, ``COXPH_MIN_TIME_EVENT_COUNT``,
  ``COXPH_AGGREGATE_SIG_FIGS`` and ``COXPH_MIN_RISK_SET_SIZE`` (the last two
  accept an empty string to switch the knob off). Central additionally reads
  ``COXPH_MAX_COVARIATES`` (default 20), ``COXPH_MAX_GRID_POINTS`` (default
  1000), ``COXPH_CURVE_TIME_SIG_FIGS`` (default 6; an empty string
  reports the exact curve times) and ``COXPH_MAX_HELD_EVENT_FRACTION``
  (default 0.5; central aborts a run whose held-event fraction exceeds it;
  an empty string disables the abort). The full semantics live in
  ``docs/coxph/Privacy.rst``.
- The parent-task guard fails closed: if the task lookup fails or the token
  is missing (and the client is not a ``MockAlgorithmClient``), raise
  ``AlgorithmError``, never proceed.
- Rows with NaN in ``time_col``, ``outcome_col`` or any model term are
  dropped before the sample-size threshold is checked.

## Coding conventions

- numpy/pandas vectorised maths; avoid per-row Python loops in the hot path.
- Pydantic validation of everything that crosses the wire: ``CoxPHInput``,
  ``IterationInput``, ``SummedZResult``, ``IterationResult``.
- vantage6 logging: ``info``/``warn``/``error`` from
  ``vantage6.algorithm.tools.util``.
- Raise vantage6 exception types: ``UserInputError``,
  ``PrivacyThresholdViolation``, ``PrivacyViolation``, ``AlgorithmError``.
- UK English in prose, comments, docstrings and log messages (organisation,
  behaviour, optimise). Identifiers keep vantage6's US spelling and must not
  be "corrected": ``organization_ids``, ``client.organization_id`` and
  ``client.organization.list()`` are part of the wire contract or the
  vantage6 API.
- Black formatting, line length 120, target ``py310`` (configured in
  ``[tool.black]`` in ``pyproject.toml``); flake8 uses the same 120 limit.

## Branches, docs and releases

- ``standard-coxph`` is the default branch and the PR target.
- Semantic versioning. A **breaking** wire-contract change (one that breaks
  an existing request or result — partial signatures, result keys,
  ``algorithm_store.json``) is a major bump and its commit message starts
  with ``BREAKING(wire):``. An additive change — a new optional argument,
  a widened argument type that keeps the old form valid, a new result key
  — is a minor bump.
- Releases are git tags. ``release.yaml`` publishes
  ``ghcr.io/<owner>/<repository>-<branch>:<tag>`` where ``<branch>`` is the
  branch whose **tip** is the tagged commit; it fails clearly when none or
  several branches match, and a ``workflow_dispatch`` input is the escape
  hatch for a tag that is no longer a branch tip.
- Update ``docs/coxph`` and ``algorithm_store.json`` whenever a signature or
  output field changes.
- CI lives in ``.github/workflows`` and runs with ``permissions:
  contents: read``:
  - ``test-suite.yml`` (on pushes to the default branch, every pull request
    and manual dispatch) runs the ``test`` matrix (Python 3.10/3.12 ×
    vantage6 4.14/4.15), ``lint``, ``security`` (Bandit ``-ll`` and
    ``pip-audit``; both must be able to fail — a finding is fixed or
    suppressed with a justified ``# nosec`` comment, never ``|| true``) and
    a ``docker`` build + import smoke-test job.
  - ``security-scan.yml`` runs weekly (Mondays): ``pip-audit`` on
    ``requirements-lock.txt`` (suppressions in ``.pip-audit-ignore``, each
    with a justification and an expiry date — currently the exact pins
    ``click``, ``cryptography`` and ``pyjwt`` imposed by
    ``vantage6-common``), Bandit and Trivy (HIGH/CRITICAL,
    ``ignore-unfixed``) on the latest released image; failures open a
    ``security-scan`` issue. Suppressions are never replaced by lowering a
    severity threshold.
  - ``.github/dependabot.yml`` keeps pip, Docker (base-image digest) and the
    pinned action SHAs current; actions are pinned by commit SHA.

## Do not

- Do not change wire signatures without updating ``algorithm_store.json``
  **and** ``central`` in the same commit.
- Do not commit test data or local configuration with secrets.
- Do not add ``__init__.py`` author/license headers unless asked.
- Do not rename the ``v6-cox-ph`` package directory — the hyphen is part of
  the vantage6 module contract.
<!-- END block3 upstream AGENTS.md -->

## 4. v6-cox-ph STRONG AYA specifics

### What this repository is

The STRONG AYA layer on top of the vanilla federated Cox-PH model
(``v6-cox-ph``). Branch ``phase2-strong-aya`` carries only the project
layer: the STRONG AYA libraries, the data-access and privacy steps at the
shared ``prepare_node_data`` hook, and this repository's conventions. The
algorithm itself is owned by the upstream vanilla repository
(``MaastrichtU-CDS/v6-coxph``); every algorithmic question is answered by
block 3 above, not here.

### STRONG AYA layer

- This branch supports vantage6 4.15 only, unlike block 3's 4.14–4.15
  range: ``vantage6-strongaya-general`` v1.0.5 pins
  ``vantage6-algorithm-tools==4.15`` exactly, so vantage6 4.14 can no
  longer be installed next to this package. ``pyproject.toml`` declares
  ``vantage6-algorithm-tools>=4.15,<4.16`` and the CI test matrix runs
  4.15 only.
- Dependencies (pinned to git tags; regenerate ``requirements-lock.txt``
  after a bump): ``vantage6-strongaya-general @
  git+https://github.com/STRONGAYA/v6-tools-general.git@v1.0.5`` and
  ``vantage6-strongaya-rdf @
  git+https://github.com/STRONGAYA/v6-tools-rdf.git@v1.1.0``.
- Logging goes through ``safe_log(level, message)`` — never vantage6's
  ``info``/``warn``/``error`` and never ``print``.
- Inside ``prepare_node_data`` (block 2 order): RDF collect when the node
  carries an ``endpoint`` column, then ``mask_unnecessary_variables`` before
  the native validation, and ``apply_sample_size_threshold`` after the
  native ``check_sample_size`` — only when the native threshold passed; a
  node below the native threshold raises ``PrivacyThresholdViolation``
  from the caller and the STRONG AYA threshold must not double-raise.
- Stratification (``variables_to_stratify``, STRONG AYA data standard) is
  supported: ``apply_data_stratification`` runs after masking and before
  every threshold, so the rows the filter removes count against them. The
  spec is validated with Pydantic at ``central`` entry
  (``CoxPHInput.variables_to_stratify``) and re-validated on the node;
  stratification variable names must be identifier-shaped or prefixed
  names such as ``ncit:C28421`` (they are interpolated between backticks
  into a dataframe query; never allow a backtick), and a missing
  stratification variable raises ``UserInputError`` instead of the library
  filter's silent no-op. ``normalise_stratification`` brings RDF terms and
  numeric text into the form of the request before the library filter
  runs; keep the filter itself the library's ``apply_data_stratification``.
- Times (STRONG AYA data standard) are numeric deltas, never dates:
  ``coerce_time_column`` converts the time column to floats right after
  masking (text from RDF data, ``ncit:C54031`` for a missing time), and
  ``coerce_covariates`` does the same for ``expl_vars``. A value that is not
  a number is dropped before the thresholds and fails the run with
  ``UserInputError`` only after them, like an unlisted outcome code.
  ``variables_to_analyse`` names each variable once (a covariate may also be
  a stratification variable; the RDF collect would otherwise query it twice
  and its merge would rename both copies).
- Outcome codes (STRONG AYA data standard): RDF data codes the outcome with
  ontology terms (``survival_status``, ``ncit:C25717``), so ``outcome_col``
  is either a column name (coded 0/1) or a mapping of one column name to
  its ``event``, ``censored`` and optional ``missing`` codes — no separate
  parameter. ``central`` forwards ``outcome_col`` unchanged; each partial
  splits it with ``resolve_outcome_col`` and ``recode_outcome`` maps the
  codes to 1, 0 and NaN after masking and before stratification. Codes
  compare through ``term_key`` (a prefixed name matches its full
  IRI). Missing and unlisted codes become NaN and are dropped before every
  threshold; an unlisted code then fails the run with
  ``UserInputError``, but only **after** the thresholds passed — raising it
  earlier would tell a researcher whether a below-threshold subpopulation
  holds a code. Never move that check before the thresholds and never echo
  the codes found.
- The STRONG AYA guards are in addition to the vanilla guards, never
  instead of them; ``perform_iteration`` runs them too, through the same
  shared ``prepare_node_data``.

### Branches, CI and releases

- Upstream → this branch by ``git merge`` only (upstream improvements land
  in the vanilla repository first, then merge down); project code never
  flows back to vanilla.
- CI triggers include ``phase2-strong-aya``; the image is
  ``ghcr.io/<owner>/v6-cox-ph-phase2-strong-aya:<tag>`` (branch suffix per
  the release workflow).
- The lock file is compiled against musl/Python 3.12 and includes the git
  dependencies; regenerate it after any dependency bump.
- Versions: tags are shared by every branch of this repository, so a
  release of this branch takes the next version number no other branch
  uses (phase1: 2.1.0, then 2.2.0 with categorical covariates; this
  branch: 2.3.0). An additive wire change — a new optional argument, an
  argument that also accepts a mapping while a plain name stays valid, a
  new result key — is a minor bump without ``BREAKING(wire):``; a change
  that breaks an existing request or result stays a major bump. Blocks 1
  and 3 still say "wire change ⇒ major"; that wording is to be refined at
  its source.

### Do not

- Import ``vantage6_strongaya_rdf`` or any STRONG AYA library in
  ``central.py``.
- Loosen the vanilla guards, thresholds or tolerances to make tests pass.
- Copy upstream commits by hand; merge them.
- Commit plan documents, canary data or configuration with secrets.
