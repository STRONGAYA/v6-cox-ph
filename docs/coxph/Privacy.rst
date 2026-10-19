Privacy
=======

Guards
------

The partial functions run on each data station and share only aggregated
quantities with the central aggregator. Several guards protect against the
main leakage channels of the federated design. All guards are implemented in
``v6-cox-ph/privacy_guards.py`` and applied in the same order in every
partial: parent-task check, settings load, explanatory-variable validation
(``compute_summed_z`` and ``perform_iteration``), NaN row dropping,
sample-size threshold, iteration-input validation (``perform_iteration``
only), time-column preparation (binning, tail censoring), and the risk-set
jump guard (``perform_iteration`` only). On top of these mandatory guards,
the node's privacy level adds the reconstruction row bound, small-count
suppression of the shared event counts and rounding of the released
aggregates (see Configuration below); two central-side resource caps bound
the size of the run (see Resource caps below).

Parent-task guard
~~~~~~~~~~~~~~~~~

Every partial calls ``ensure_spawned_by_central``. The container JWT carries
the running ``task_id``; the partial looks up that task and requires a
non-empty ``parent`` field. The vantage6 server only sets ``parent_id`` for
container-created sub-tasks, so a user task created directly (calling a
partial without going through ``central``) has ``parent = None`` and is
refused with a ``PrivacyViolation``. A failure to look up the task fails
closed with an ``AlgorithmError``. The guard is skipped (with an info log)
for the ``MockAlgorithmClient`` used in tests.

Sample-size threshold
~~~~~~~~~~~~~~~~~~~~~

Each node must have strictly more than ``SAMPLE_SIZE_THRESHOLD`` rows **and**
events (default 10). A node below the threshold raises
``PrivacyThresholdViolation`` and the whole analysis stops: there is no
automatic exclusion. Researchers select the organisations explicitly via
``organization_ids``. The failing node is identifiable from the run status —
the same information today's exclusion marker used to leak — which is an
accepted consequence of failing closed.

The threshold bounds *participation*, not *reconstruction*: a node whose
entire content enters a single released risk set can be reconstructed from
the aggregates even though it passes the threshold. Each node must
therefore also have at least ``factor * k`` rows, where ``k`` is
``COXPH_MIN_RISK_SET_CHANGE`` and ``factor`` comes from the privacy level
(10 at ``standard``, 20 at ``strict``; no bound at ``minimal``) or from
``COXPH_RECONSTRUCTION_ROW_FACTOR`` when that variable is set. As with the
sample-size threshold, the warning on failure names the configured bound,
never the actual row count.

Time binning
~~~~~~~~~~~~

When ``COXPH_TIME_BIN_WIDTH`` is set to a positive value, event times are
coarsened to a regular grid ``floor(t / w) * w`` before anything else, so
event times, ``z_sum`` and risk sets are all on the same grid. Binning is
disabled by default: it is the node administrator's choice, and it should
be enabled when exact event dates are sensitive (a coarse grid cannot be
reversed to an exact date).

Tail (administrative) censoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``k = COXPH_MIN_RISK_SET_CHANGE`` (default 5), rows with ``time > t_cut``
— where ``t_cut`` is the k-th largest time on the node — are administratively
censored (``time := t_cut`` and, where the outcome column is available,
``event := 0``). This guarantees ``|R(t)| >= k`` for every non-empty shared
risk set. Setting ``k <= 1`` disables both tail censoring and the jump guard
(used by exactness tests).

Risk-set jump guard
~~~~~~~~~~~~~~~~~~~

In ``perform_iteration``, walking the requested time grid, if moving from
``t_i`` to ``t_{i+1}`` would remove fewer than ``k`` (but more than zero)
individuals from the node's risk set, the node keeps using ``R(t_i)``
("hold"). Consecutive shared aggregates therefore differ by 0 or by at
least ``k`` individuals.

Small-count time suppression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``compute_summed_z`` shares per-time event counts, and central builds the
event-time grid for the whole run from them. A per-time count of 1
identifies a singleton: an event at that exact time belongs to a single
individual. Per-time event counts below the level's minimum
(``COXPH_MIN_TIME_EVENT_COUNT``: 2 at ``standard``, 5 at ``strict``) are
therefore merged into the *following* event time (small-cell suppression).
The events are tied at the later time, never dropped, so the total number
of events is unchanged. The suppression applies to the ``times`` payload of
``compute_summed_z`` and therefore to the event-time grid the whole run
uses, including the keys of the returned ``baseline_cumulative_hazard``
and ``survival_curves``.

With high-precision event times most per-time counts are singletons and
the grid coarsens substantially; the model stays valid but the estimates
shift. Combine the suppression with ``COXPH_TIME_BIN_WIDTH`` when the
remaining grid is still too fine.

Aggregate rounding
~~~~~~~~~~~~~~~~~~

The risk-set sums ``S0``/``S1``/``S2`` released by ``perform_iteration`` are
rounded to the configured number of significant figures
(``COXPH_AGGREGATE_SIG_FIGS``: 6 at ``standard``, 4 at ``strict``; an empty
string disables rounding). This is deterministic quantisation, not noise:
it defeats exact reconstruction of values from the aggregates, but the
values are recoverable to the rounding precision. The rounding is
consistent with the existing two-significant-figure rounding of the pooled
centre and scale. Lowering the number of significant figures trades
accuracy for privacy.

Resource caps
~~~~~~~~~~~~~

Two central-side caps bound the compute loop and the per-node, per-round
result payload, which grows with the grid length ``T`` and with the square
of the number of covariates ``p``:

- ``COXPH_MAX_COVARIATES`` (int, default 20): central rejects a model with
  more explanatory variables.
- ``COXPH_MAX_GRID_POINTS`` (int, default 1000): central rejects a run
  whose pooled event-time grid is longer. The error names the cap and the
  remedies — enable ``COXPH_TIME_BIN_WIDTH``, or select fewer
  organisations — but never the actual grid length.

Binning the event times and running single-organisation tasks reduce the
payload further.

Logging
~~~~~~~

The algorithm container log is returned to the server with the run and can
be read by the researcher. Partials therefore log status only: never data
values, and never counts. A failed sample-size check names the configured
threshold but not the number of rows or events the node actually has.
Validation errors are re-raised without the original Pydantic message —
the chained message used to embed the offending input — and unexpected
errors inside a partial surface as an ``AlgorithmError`` naming only the
exception type.

Configuration
~~~~~~~~~~~~~

All settings are read from node ``algorithm_env`` variables via
``get_env_var``:

=================================== =========== ============
Variable                            Type        Default
=================================== =========== ============
``SAMPLE_SIZE_THRESHOLD``           int         10
``COXPH_TIME_BIN_WIDTH``            float       (disabled)
``COXPH_MIN_RISK_SET_CHANGE``       int         5
``COXPH_PRIVACY_LEVEL``             string      ``standard``
``COXPH_RECONSTRUCTION_ROW_FACTOR`` int         (preset)
``COXPH_MIN_TIME_EVENT_COUNT``      int         (preset)
``COXPH_AGGREGATE_SIG_FIGS``        int         (preset)
``COXPH_MAX_COVARIATES``            int         20
``COXPH_MAX_GRID_POINTS``           int         1000
=================================== =========== ============

``COXPH_PRIVACY_LEVEL`` is the primary variable: the one variable a node
administrator sets to choose the hardening level. The level presets the
three hardening settings:

============ ========== ================ =========
Level        Row factor Min. time events Sig. figs
============ ========== ================ =========
``minimal``  off        off              off
``standard`` 10         2                6
``strict``   20         5                4
============ ========== ================ =========

``minimal`` applies only the mandatory guards (sample-size threshold,
``k``, tail censoring, jump guard); it is meant for validation studies, not
for sensitive data. Three advanced overrides each beat the level preset
when set: ``COXPH_RECONSTRUCTION_ROW_FACTOR`` (int, at least 1),
``COXPH_MIN_TIME_EVENT_COUNT`` (int, at least 1) and
``COXPH_AGGREGATE_SIG_FIGS`` (int, at least 1; an empty string disables
rounding). Precedence is: an explicitly set variable beats the level
preset, which beats the built-in default. A node administrator normally
sets nothing — the default ``standard`` level applies. The two caps are
central-side; see Resource caps above.

``COXPH_MIN_RISK_SET_CHANGE`` must be at least 1, and
``SAMPLE_SIZE_THRESHOLD + 1`` must be greater than or equal to it;
otherwise the node raises ``UserInputError`` at start-up. This prevents
a configuration where a node passes the sample-size threshold but has
too few rows for tail censoring to guarantee risk sets of size ``k``.

The parent-task guard fails closed: a client without a decodable
container token (that is not a ``MockAlgorithmClient``) raises
``AlgorithmError`` rather than silently skipping the check.

Rows with NaN in ``time_col``, ``outcome_col`` or any explanatory
variable are dropped before the sample-size threshold is checked, so
thresholds apply to the analysed rows. Explanatory variables are
validated on the node: each must be a column, must not overlap
``time_col``/``outcome_col``, and must be numeric. The input model
``CoxPHInput`` rejects duplicate explanatory variables, a ``time_col``
equal to ``outcome_col``, and explanatory variables naming the time or
outcome columns. Central validates ``organization_ids`` against the
collaboration and rejects unknown identifiers with a ``UserInputError``
before dispatching any sub-task.

Guard-effect summary
~~~~~~~~~~~~~~~~~~~~

The result carries a ``privacy_guards`` object: an aggregate summary of the
guards that were active (configuration, not data; there is no per-node
breakdown). It reports the maximum ``k`` any node used, whether time
binning was on, and the maximum minimum time event count across nodes. A
warning is appended when any of these is active, because the estimates
are then approximate.

Data sharing
------------

Each data station shares, with the central aggregator only:

- the sum of each explanatory variable over the event cases, the matching sum
  of squares (used to derive the pooled covariate spread), the node's privacy
  settings (configuration, not data), and per-time event counts computed
  from the same NaN-dropped, censored DataFrame (``compute_summed_z``). The
  event counts are used by the aggregator to build the event-time grid that
  is consistent with ``z_sum`` and the risk sets; the per-time event counts
  are small-count-suppressed before they leave the node (see above). The
  sum of squares is a new aggregate leaving the node; it enjoys the same
  sample-size threshold protection as ``z_sum``;
- per event time ``t``, the risk-set sums ``S0(t) = sum exp(beta . X)``,
  ``S1(t) = sum X * exp(beta . X)`` and ``S2(t) = sum X X^T exp(beta . X)``
  (``perform_iteration``), computed on the standardised covariates
  ``(x - centre) / scale``. The pooled centre and scale are shared with every
  node in every iteration, rounded to two significant figures: a node
  therefore learns a coarse pooled event-case covariate mean and standard
  deviation, and nothing about any other individual node. The released
  risk-set sums are rounded to the configured number of significant figures
  (see above).

Two directions of sharing exist and both must be assumed visible to the
researcher that ran ``central``:

- **Node → central.** The pooled event-time grid is sent to *every* node in
  every iteration (``unique_time_events``): every node therefore learns the
  pooled, binned event times of the collaboration. Bin times
  (``COXPH_TIME_BIN_WIDTH``) when that is too much detail.
- **Sub-task results.** The sub-tasks are created under the researcher's
  identity and their results are retrievable from the server (the vantage6
  permission checks operate at the server). Every partial must therefore
  treat its entire return value as visible to the researcher — the sample
  contents below are all aggregates over at least ``k`` individuals.
- **Survival curves.** ``baseline_cumulative_hazard`` and
  ``survival_curves`` are computed centrally from the pooled event counts
  and risk-set sums that the researcher already sees; no new partial runs
  and no new data leaves a node. What the curve reveals is, per grid time,
  the pooled event count over the risk-set sum — information the shared
  aggregates already contained. The tail censoring and the jump guard carry
  through: every non-empty risk set behind a curve step covers at least
  ``k`` individuals on every node.

Aggregated model coefficients are shared with the data stations by the
central aggregator during the iteration process.

Vulnerabilities to known attacks
--------------------------------

.. Differencing / reconstruction / DLG

.. _mitigated-warning:

** Reconstruction — mitigated. **
The partials no longer return row-level data or exact per-event-time sums
that allow record reconstruction by differencing. Per-event-time risk-set
sums are shared, but the jump guard and tail censoring ensure every shared,
non-empty risk set covers at least ``k`` individuals and consecutive
aggregates differ by 0 or by at least ``k``. **Residual risk:** an
aggregator that collects the released sums over the smallest shared risk
set across several ``beta`` values obtains an exact equation system over
that group's covariate values. The guards — a larger ``k``, the
reconstruction row bound, coarser event times and rounded aggregates —
raise the cost of solving that system but cannot eliminate it. Differential
privacy on the released aggregates, or secure aggregation, is the complete
fix and is planned for a future release.

** Differencing — mitigated. **
The risk-set jump guard prevents an aggregator from differencing consecutive
shared aggregates to isolate fewer than ``k`` records. **Residual risk:**
a data-station manager who changes the dataset between tasks could still
differencing across runs; data-station managers should not be allowed to
run tasks directly.

** Deep Leakage from Gradients (DLG) — mitigated. **
The shared ``S0/S1/S2`` aggregates are sums over risk sets of at least
``k`` individuals, not per-record gradients. DLG-style reconstruction from
these aggregates would recover groups rather than individuals. The central
aggregator should nonetheless be treated as a trusted party until
differential privacy or secure aggregation is in place (see the residual
risk under Reconstruction above).

** Generative Adversarial Networks (GAN) — low risk. **
Synthetic data can statistically reproduce the data underlying the model,
but without the sensitive attributes an adversary cannot assess the
authenticity of recovered records.

** Model Inversion — low risk. **
Model predictions can be used to infer an individual's outcome, but without
the sensitive information the adversary cannot verify the inference.

** Watermark Attack — to be determined. **

Choosing thresholds per analysis type
-------------------------------------

The guards scale with the analysis type, in three tiers:

- **Descriptive statistics** may run at the base sample-size threshold:
  every output covers all rows of the node.
- **Survival modelling**, such as this algorithm, requires the
  reconstruction row bound in addition: the released risk-set sums cover
  groups of rows, not all rows.
- **Gradient-iterative modelling** — regression with iterates, clustering,
  deep learning — needs differential privacy or secure aggregation from
  the start; a threshold alone is not a mitigation there.

``COXPH_PRIVACY_LEVEL`` tunes within this algorithm's tier. Which analyses
may run at which node sizes is collaboration policy, owned by the project
privacy officer. Note that fail-closed threshold failures identify the
refusing node at any tier.

For reference
-------------

- **Reconstruction**: an adversary tries to reconstruct the original dataset
  from shared model parameters.
- **Differencing**: an adversary infers information about a specific data
  point by comparing outputs with and without it.
- **Deep Leakage from Gradients (DLG)**: an adversary infers training data
  from shared gradient updates.
- **GAN**: synthetic data statistically similar to the original data.
- **Model Inversion**: inferring input data given model output.
- **Watermark Attack**: embedding a watermark in a model to identify the
  data it was trained on.
