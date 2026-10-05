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

Input validation is now enforced on the node: the time column must be
numeric, finite and non-negative; the outcome column must be binary;
violations raise ``UserInputError`` and stop the analysis.

## 1.0.0

Initial release of the hardened federated Cox proportional hazards model
(Breslow ties) for vantage6 4.14–4.15.
