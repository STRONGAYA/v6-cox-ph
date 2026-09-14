Privacy
=======

Guards
------

The partial functions run on each data station and share only aggregated
quantities with the central aggregator. Several guards protect against the
main leakage channels of the federated design. All guards are implemented in
``v6-cox-ph/privacy_guards.py`` and applied in the same order in every
partial: parent-task check, settings load, sample-size threshold, time-column
preparation (binning, tail censoring), and the risk-set jump guard
(``perform_iteration`` only).

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
events (default 10). ``get_unique_event_times`` returns
``{"N-Threshold not met": org_id}`` on failure (the central function then
excludes that organisation); ``compute_summed_z`` and ``perform_iteration``
raise ``PrivacyThresholdViolation``.

Time binning
~~~~~~~~~~~~

When ``COXPH_TIME_BIN_WIDTH`` is set to a positive value, event times are
coarsened to a regular grid ``floor(t / w) * w`` before anything else, so
event times, ``z_sum`` and risk sets are all on the same grid. Binning is
disabled by default.

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

Configuration
~~~~~~~~~~~~~

All settings are read from node ``algorithm_env`` variables via
``get_env_var``:

============================== =========== ===========
Variable                        Type        Default
============================== =========== ===========
``SAMPLE_SIZE_THRESHOLD``       int         10
``COXPH_TIME_BIN_WIDTH``        float       (disabled)
``COXPH_MIN_RISK_SET_CHANGE``   int         5
============================== =========== ===========

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
``time_col``/``outcome_col``, and must be numeric.

Data sharing
------------

Each data station shares, with the central aggregator only:

- the unique event times and their frequencies (``get_unique_event_times``);
- the sum of each explanatory variable over the event cases
  (``compute_summed_z``);
- per event time ``t``, the risk-set sums ``S0(t) = sum exp(beta . X)``,
  ``S1(t) = sum X * exp(beta . X)`` and ``S2(t) = sum X X^T exp(beta . X)``
  (``perform_iteration``).

Event times are not shared between data stations. The aggregated model
coefficients are shared with the data stations by the central aggregator
during the iteration process.

Vulnerabilities to known attacks
--------------------------------

.. Differencing / reconstruction / DLG

.. _mitigated-warning:

** Reconstruction — mitigated. **
The partials no longer return row-level data or exact per-event-time sums
that allow record reconstruction by differencing. Per-event-time risk-set
sums are shared, but the jump guard and tail censoring ensure every shared,
non-empty risk set covers at least ``k`` individuals and consecutive
aggregates differ by 0 or by at least ``k``. **Residual risk:** without
secure aggregation, an aggregator that collects sums over ``>= k``
individuals across several ``beta`` values still obtains equations about
those records; this cannot be eliminated by node-side guards alone.

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
aggregator should nonetheless be treated as a trusted party.

** Generative Adversarial Networks (GAN) — low risk. **
Synthetic data can statistically reproduce the data underlying the model,
but without the sensitive attributes an adversary cannot assess the
authenticity of recovered records.

** Model Inversion — low risk. **
Model predictions can be used to infer an individual's outcome, but without
the sensitive information the adversary cannot verify the inference.

** Watermark Attack — to be determined. **

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
