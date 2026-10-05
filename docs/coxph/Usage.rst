How to use
==========

Input arguments
----------------
The input arguments for the central function consist of:

- ``time_col`` (string): the name of the column containing the time data.
- ``outcome_col`` (string): the name of the column containing the outcome data
  (1 = event, 0 = censored).
- ``expl_vars`` (list): a list of explanatory variables (predictors) to be
  used in the computation.
- ``organization_ids`` (list): a list of organisation IDs that participate in
  the collaboration and you wish to run the algorithm on. When ``None`` all
  organisations are used.

Input rules
------------

Every node validates its data before sharing anything:

- ``time_col`` must exist and be numeric, finite and non-negative.
- ``outcome_col`` must exist and be binary: only 0 and 1 (booleans are
  allowed).
- Every ``expl_var`` must exist and be numeric (see below), and rows with
  missing values in any analysed column are dropped before the sample-size
  threshold is checked.

A column that violates these rules raises ``UserInputError`` and stops the
whole analysis; there is no silent fallback. Projects whose data uses other
codings (for example an outcome coded 1/2, or string labels) must recode
**before** calling the algorithm — recoding belongs in the project layer,
not in the vanilla algorithm.

Output fields
-------------

The central function returns a dictionary with:

- ``model``: a JSON object indexed by variable name with columns ``Coef``,
  ``Exp(coef)``, ``SE``, ``lower_CI``, ``upper_CI``, ``Z`` and ``p-value``.
  The coefficients are on the original covariates: nodes fit on the
  standardised ``(x - centre) / scale`` and the central function transforms
  the reported statistics back.
- ``overall_p_value``: the overall Wald test p-value.
- ``aic``: the Akaike Information Criterion.
- ``degrees_of_freedom``: the number of model parameters.
- ``warnings``: a list of human-readable warning strings.
- ``converged`` (bool): whether the Newton-Raphson optimiser converged.
- ``n_iterations`` (int): the number of iterations performed.

The Wald statistic is reported as ``Z = Coef / SE`` and the p-value as
``p = 2 * Phi(-|Z|)``. When ``converged`` is ``False`` a warning is appended
advising that the SE/p-values may be unreliable; statistics are reported
at the last evaluated beta (the Newton step is not applied when the
iteration budget is exhausted, so Coef, SE and AIC stay consistent).

All statistics in the model table are full-precision floats (no
rounding); presentation rounding is a client concern.

Node configuration (algorithm_env)
----------------------------------

Privacy guards are configured through node environment variables
(``algorithm_env``):

============================== =========== =========== =========================
Variable                        Type        Default      Description
============================== =========== =========== =========================
``SAMPLE_SIZE_THRESHOLD``       int         10          Minimum rows and events
                                                        (strictly greater).
``COXPH_TIME_BIN_WIDTH``        float       (disabled)  Event-time bin width. When
                                                        set, times are coarsened
                                                        to ``floor(t/w)*w``.
``COXPH_MIN_RISK_SET_CHANGE``   int         5           Minimum risk-set change
                                                        (``k``). Consecutive
                                                        shared aggregates differ
                                                        by 0 or ``>= k``. Set to
                                                        ``1`` to disable tail
                                                        censoring and the jump
                                                        guard.
============================== =========== =========== =========================

See ``docs/coxph/Privacy.rst`` for the privacy implications of these
settings.

Python client example
----------------------

To understand the information below, you should be familiar with the vantage6
framework.
If you are not, please read the `documentation <https://docs.vantage6.ai>`_
first, especially the part about the
`Python client <https://docs.vantage6.ai/en/main/user/pyclient.html>`_.

.. code-block:: python

  from vantage6.client import Client

  server = 'http://localhost'
  port = 5000
  api_path = '/api'
  private_key = None
  username = 'org_1-admin'
  password = 'password'

  # Create connection with the vantage6 server
  client = Client(server, port, api_path)
  client.setup_encryption(private_key)
  client.authenticate(username, password)

  # When set to None it will run the algorithm on all organizations in the specified collaboration
  organization_ids = None

  input_ = {
    'method': 'central',
    'master': True,
    'kwargs': {
        'time_col': 'overall_survival_in_days',
        'outcome_col': 'event_overall_survival',
        'expl_vars': ['clin_n_1', 'index_tumour_location_oropharynx'],
        'organization_ids': organization_ids,
    },
    'output_format': 'json'
  }

  my_task = client.task.create(
      collaboration=1,
      organizations=[1],
      name='Cox proportional hazards',
      description='Cox proportional hazards model',
      image='ghcr.io/maastrichtu-cds/v6-coxph:latest',
      input=input_,
      data_format='json'
  )

  task_id = my_task.get('id')
  results = client.wait_for_results(task_id)
