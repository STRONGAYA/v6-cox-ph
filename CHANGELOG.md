# Changelog

## 2.0.0-dev (unreleased)

BREAKING wire-contract changes (a major version bump; the commit messages
of the changes below start with ``BREAKING(wire):``):

- ``perform_iteration`` gains the ``outcome_col``, ``centre`` and ``scale``
  arguments. Nodes drop rows with a missing outcome and fit on the
  standardised covariates ``(x - centre) / scale``; the pooled covariate
  centre and spread over the event cases are shared with the nodes,
  rounded to two significant figures. The partial likelihood is invariant
  to this shared affine transform; centring prevents overflow from large
  offsets and scaling fixes the Hessian conditioning from large spreads.
  Central back-transforms the reported coefficients, standard errors and
  covariance.
- ``compute_summed_z`` additionally returns the sum of squares of the
  covariates over the event cases (needed for the pooled scale) and the
  node's privacy settings (configuration, not data).
- ``get_unique_event_times`` is removed. A node below the sample-size
  threshold fails the whole analysis with ``PrivacyThresholdViolation``
  instead of being excluded: there is no eligibility round and no retry
  loop, and a missing column raises ``UserInputError``. Researchers select
  organisations explicitly via ``organization_ids``.
- Every partial result carries the ``organization_id`` it was computed
  for, and ``central`` attributes errors by it.
- The ``model`` result field is a JSON object (previously a JSON string),
  and ``included_organizations`` / ``excluded_organizations`` are removed
  from the result.

Non-breaking additions in the same release:

- Survival curves: ``central`` optionally takes ``covariate_profiles``
  and returns ``baseline_cumulative_hazard`` (Breslow, at the pooled
  event-case covariate mean) and ``survival_curves`` per profile — all
  computed centrally from existing aggregates; no new partial.
- The optimiser tracks the partial log-likelihood centrally, halves the
  Newton step when it decreases (keeping the last accepted state, so the
  reported statistics always belong to one evaluated beta) and uses a
  20-round-trip budget. New result fields: ``log_likelihood``,
  ``log_likelihood_null``, ``lr_statistic``, ``lr_p_value``, ``n_events``,
  ``covariance``, ``algorithm_version``, ``baseline_cumulative_hazard``,
  ``survival_curves`` and the aggregate ``privacy_guards`` summary
  (``active``, ``max_min_risk_set_change``, ``time_binning``) with a
  warning when the guards were active.
- The risk-set aggregates are computed vectorised
  (``searchsorted``/``bincount``/reverse cumsum, the jump guard derived
  from the bucket counts) instead of with T x N boolean masks; benchmark
  at 100k rows x 10k event times x 10 covariates: 89.3 s / 717 MB before,
  0.12 s / 28 MB after. ``agg2`` travels as a plain (T x p) list.
- Input validation is enforced on the node: the time column must be
  numeric, finite and non-negative; the outcome column must be binary;
  violations raise ``UserInputError`` and stop the analysis. Every node
  also fails closed unless every dispatched organisation answered.

## 1.0.0

Initial release of the hardened federated Cox proportional hazards model
(Breslow ties) for vantage6 4.14–4.15.
