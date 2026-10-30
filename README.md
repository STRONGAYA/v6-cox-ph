# v6-cox-ph — STRONG AYA

A federated Cox proportional hazards model (Breslow ties) for
[vantage6](https://vantage6.ai) 4.15, with the STRONG AYA layer. The model is
fitted by Newton–Raphson across the data stations: only aggregated quantities
leave a node, and every node checks a sample-size threshold and further
privacy guards before it shares anything.

On top of the vanilla algorithm, the STRONG AYA layer adds:

- **RDF data stations**: data is collected from the station's RDF endpoint
  with [v6-tools-rdf](https://github.com/STRONGAYA/v6-tools-rdf), and the
  request is checked against the STRONG AYA semantic map before anything is
  computed;
- **outcome codes**: `outcome_col` can map the outcome column to the ontology
  terms that mean an event, censored or missing;
- **stratification**: `variables_to_stratify` restricts the analysis to a
  subpopulation, and the sample-size thresholds apply to the stratified rows;
- **STRONG AYA logging (`safe_log`) and privacy measures** from
  [v6-tools-general](https://github.com/STRONGAYA/v6-tools-general).

CSV stations get no schema check; the rest applies to every station.

## Documentation

The full documentation (usage, privacy, implementation and validation) is in
the [repository wiki](../../wiki). The wiki is published from `docs/` on the
default branch, and every page states the release or development version it
describes. The documentation of this branch is in [`docs/coxph`](docs/coxph).

## Quick start

Call `central`; it dispatches the node-side partials itself and returns the
fitted model as JSON. On an RDF station every name and code must come from
the STRONG AYA semantic map:

```python
task = client.task.create(
    collaboration=1,
    organizations=[1],  # the organisation that runs central
    name="Cox proportional hazards",
    image="ghcr.io/strongaya/v6-cox-ph-<branch>:<tag>",
    input_={
        "method": "central",
        "kwargs": {
            "time_col": "ncit:C125201",  # overall survival in days
            "outcome_col": {
                "ncit:C25717": {  # survival status
                    "event": ["ncit:C99799", "ncit:C99798"],
                    "censored": ["ncit:C37987", "ncit:C141463"],
                    "missing": ["ncit:C54031"],
                }
            },
            "expl_vars": ["ncit:C156420"],  # age at diagnosis
            "variables_to_stratify": {"ncit:C28421": ["ncit:C16576"]},  # females only
            "organization_ids": [2, 3],  # the data organisations only
        },
    },
    databases=[{"label": "default"}],
)
result = client.wait_for_results(task_id=task["id"])
```

Select the data organisations explicitly in `organization_ids`: a run that
includes the data-less aggregator organisation fails closed. The *Usage* page
of the wiki describes every argument, the schema rules and the output fields.

## Node configuration

Data-station managers tune the privacy guards through the node's
`algorithm_env`, for example `SAMPLE_SIZE_THRESHOLD` (default 10) and the
`COXPH_PRIVACY_LEVEL` preset (`minimal`, `standard` or `strict`), and choose
the schema version their data is checked against (`USE_REMOTE_SCHEMA`,
`SCHEMA_URL`, `SCHEMA_TAG`). The *Usage* and *Privacy* pages of the wiki list
every setting with its default.

## Development

Python 3.10 or later (the image runs Python 3.12). The STRONG AYA libraries
are git dependencies pinned in `pyproject.toml`:

```bash
pip install -e .[dev,lint]
pytest -m unit                # no Docker needed
pytest tests/integration      # needs Docker and the v6 CLI; includes the RDF tests
```

The lint, type-check and Docker commands that CI runs are listed in
[`AGENTS.md`](AGENTS.md).

## Releases

Releases are git tags and follow semantic versioning; see
[`CHANGELOG.md`](CHANGELOG.md). A tag publishes the image
`ghcr.io/strongaya/v6-cox-ph-<branch>:<tag>`, where `<branch>` is the branch
whose tip is the tagged commit.

## Licence

Apache License 2.0; see [`LICENSE`](LICENSE).
