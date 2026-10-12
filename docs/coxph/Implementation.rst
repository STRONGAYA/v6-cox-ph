Implementation
==============

Overview
--------

Wire validation
---------------
Every payload that crosses a trust boundary as JSON with a schema known in
advance is validated against a Pydantic model defined in
``v6-cox-ph/miscellaneous.py``: the user input (``CoxPHInput``), the
payload central sends to ``perform_iteration`` (``IterationInput``) and the
partial results (``SummedZResult``, ``IterationResult``). The partials
build their result dicts from the result models, so the keys and nesting of
what travels over the wire are pinned in one place; central validates the
received results with the same models, attributing any error by the
``organization_id`` the result carries. ``algorithm_store.json`` remains
the user-facing wire contract: function names, arguments and types.

Central
--------
The central part is responsible for the orchestration and aggregation of the algorithm.

``central``
~~~~~~~~~~~~~~~~
The central part is responsible for the following tasks:

- Request the summed z-statistics for the explanatory variables (i.e. the predictors) and, with them, the per-time event counts in each data station.
- Orchestrate iterations in the data stations to retrieve intermediate model parameters.
- Pass the intermediate model parameters to the compute_derivatives function.
- Compute the aggregated model parameters.

``compute_derivatives``
~~~~~~~~~~~~~~~~~~~~~~~
This function computes the primary and secondary derivatives needed to compute the maximum likelihood estimates of the model parameters.
The function is called by the central part and is executed on the central aggregator.

Partials
--------
Partials are the computations that are executed on each node. The partials have access
to the data that is stored on the node. The partials are executed in parallel on each
node.

``compute_summed_z``
~~~~~~~~~~~~~~~~~~~~
This function computes the sum of the specified explanatory variables for the outcome events.

``perform_iteration``
~~~~~~~~~~~~~~~~~~~~~
This function performs an iteration of the algorithm, computing the necessary aggregates.
