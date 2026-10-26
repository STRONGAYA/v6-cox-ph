"""
STRONG AYA RDF integration tests for the v6-cox-ph algorithm.

The nodes of the developer network read their data from a disposable GraphDB
that holds the RDF test data of v6-tools-rdf (100 patients, triplified and
annotated with Flyover): the outcome is coded with NCIt terms, times and ages
arrive as text, and a missing value is the missing-data term. The federated
fit is compared with the in-process pipeline on the same patients, read from
the store per patient (not through the RDF library), so the comparison also
catches values that the collection attributes to the wrong patient.

The scenarios cover the outcome codes on one node and on three nodes, the
stratification on a coded variable and on the covariate itself, and a
categorical covariate, and they exercise the refusals: an unlisted outcome
code, a category whose code no test patient has, a categorical variable
given as a plain name, and a stratification that leaves too few rows.
"""

import sys
from importlib import import_module
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
import requests
from vantage6.algorithm.tools.exceptions import (
    AlgorithmError,
    CollectResultsError,
    PrivacyThresholdViolation,
    UserInputError,
)

from tests.integration.test_algorithm_integration import extract_coxph_result, verify_error_type

# The in-process reference pipeline imports the algorithm module through the
# (hyphenated) package, the way the unit suite does.
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "v6-cox-ph"))

from vantage6.algorithm.tools.mock_client import MockAlgorithmClient  # noqa: E402

# Import the central module through the (hyphenated) package so that its
# relative imports resolve correctly.
central_module = import_module("v6-cox-ph.central")  # noqa: E402

NCIT = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"
TIME = "ncit:C125201"  # survival_event_time (days)
STATUS = "ncit:C25717"  # survival_status
SEX = "ncit:C28421"  # biological_sex
AGE = "ncit:C156420"  # age_at_initial_diagnosis
GENDER = "ncit:C158277"  # gender
OUTCOME_COL = {STATUS: {"event": ["ncit:C99799"], "censored": ["ncit:C37987"], "missing": ["ncit:C54031"]}}

# The settings of additional_vantage6_node_config.yaml, for the in-process reference.
NODE_ENV = {"COXPH_MIN_RISK_SET_CHANGE": "1", "COXPH_PRIVACY_LEVEL": "minimal", "SAMPLE_SIZE_THRESHOLD": "5"}

# The model terms of the RDF run, mapped to the reference frame's columns.
TERM_TO_REFERENCE = {AGE: "age", f"{SEX}[female]": "sex_female", f"{SEX}[intersex]": "sex_intersex"}

# The stratification variables, mapped to the reference frame's columns.
STRATIFICATION_TO_REFERENCE = {SEX: "sex", AGE: "age"}

# Every value of a variable per patient, keyed by the patient's row in the store.
PATIENT_VALUES = """
PREFIX dbo: <http://um-cds/ontologies/databaseontology/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ncit: <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#>
SELECT ?patient ?value ?term WHERE {
  ?patient dbo:has_column ?column .
  ?column rdf:type VARIABLE .
  ?column dbo:has_cell ?cell .
  ?cell dbo:has_value ?value .
  OPTIONAL {
    ?cell rdf:type ?term . ?term rdfs:subClassOf VARIABLE .
    FILTER(?term != VARIABLE && STRSTARTS(STR(?term), STR(ncit:)))
  }
}
"""

RDF_CONFIGURATIONS = {
    "rdf_outcome_codes": {"organization_ids": [1], "expl_vars": [AGE]},
    "rdf_three_nodes": {"organization_ids": [1, 2, 3], "expl_vars": [AGE]},
    "rdf_stratified_sex": {
        "organization_ids": [1],
        "expl_vars": [AGE],
        "variables_to_stratify": {SEX: ["ncit:C16576", "ncit:C20197"]},
    },
    "rdf_stratified_covariate_range": {
        "organization_ids": [1],
        "expl_vars": [AGE],
        "variables_to_stratify": {AGE: {"start": 20, "end": 70}},
    },
    "rdf_unlisted_outcome_code": {
        "organization_ids": [1],
        "expl_vars": [AGE],
        "outcome_col": {STATUS: {"event": ["ncit:C99799"], "censored": ["ncit:C141463"]}},
        "expected_failure": True,
        "expected_log": "holds codes that outcome_col does not list",
        "expected_error_type": [UserInputError, CollectResultsError, AlgorithmError],
    },
    "rdf_categorical_covariate": {
        "organization_ids": [1],
        # The test data holds male, female and intersex patients, so the
        # spec lists all three categories; the reference fit uses one 0/1
        # column per non-reference category with the missing-term rows
        # dropped.
        "expl_vars": [
            AGE,
            {
                SEX: {
                    "reference": "male",
                    "categories": {"male": ["ncit:C20197"], "female": ["ncit:C16576"], "intersex": ["ncit:C45908"]},
                }
            },
        ],
        "reference_expl_vars": ["age", "sex_female", "sex_intersex"],
    },
    "rdf_category_too_small": {
        "organization_ids": [1],
        # A category whose code no test patient has: the schema lists
        # transgender, non-binary and other for gender, and the test data
        # holds only male and female, so the empty category fails the
        # category threshold.
        "expl_vars": [
            AGE,
            {
                GENDER: {
                    "reference": "female",
                    "categories": {"female": ["ncit:C46110"], "male": ["ncit:C46109"], "transgender": ["ncit:C154421"]},
                }
            },
        ],
        "expected_failure": True,
        "expected_log": "does not exceed the configured sample-size threshold",
        "expected_error_type": [PrivacyThresholdViolation, CollectResultsError, AlgorithmError],
    },
    "rdf_categorical_as_plain_name": {
        "organization_ids": [1],
        # A categorical variable as a plain name is refused upfront, before
        # any threshold: the schema check names the rule.
        "expl_vars": [AGE, SEX],
        "expected_failure": True,
        "expected_log": "is categorical; give its categories and reference",
        "expected_error_type": [UserInputError, CollectResultsError, AlgorithmError],
    },
    "rdf_narrow_range": {
        "organization_ids": [1],
        "expl_vars": [AGE],
        "variables_to_stratify": {AGE: {"start": 30, "end": 32}},
        "expected_failure": True,
        "expected_log": "PrivacyThresholdViolation",
        "expected_error_type": [PrivacyThresholdViolation, CollectResultsError, AlgorithmError],
    },
}


def _patient_values(endpoint: str, variable: str) -> pd.Series:
    response = requests.post(
        endpoint,
        data={"query": PATIENT_VALUES.replace("VARIABLE", variable)},
        headers={"Accept": "text/csv"},
        timeout=60,
    )
    response.raise_for_status()
    rows = pd.read_csv(StringIO(response.text), dtype=str, keep_default_na=False)
    rows[variable] = rows["term"].where(rows["term"] != "", rows["value"])
    # A cell can reach the same term along two annotation paths; one value per patient remains.
    rows = rows[["patient", variable]].drop_duplicates()
    return rows.set_index("patient")[variable]


@pytest.fixture(scope="module")
def rdf_patients(rdf_store) -> pd.DataFrame:
    """The test patients as plain numbers, read from the store per patient."""
    values = pd.concat([_patient_values(rdf_store["endpoint"], v) for v in (TIME, STATUS, SEX, AGE)], axis=1)
    assert values.index.is_unique and len(values) == 100
    return pd.DataFrame(
        {
            "time": pd.to_numeric(values[TIME], errors="coerce"),
            "event": values[STATUS].map({f"{NCIT}C99799": 1.0, f"{NCIT}C37987": 0.0}),
            "age": pd.to_numeric(values[AGE], errors="coerce"),
            "sex": values[SEX].str.replace(NCIT, "ncit:"),
        }
    )


def _reference_rows(patients: pd.DataFrame, variables_to_stratify, reference_expl_vars) -> pd.DataFrame:
    """The rows the node analyses, as plain numbers, with the reference's columns.

    The stratification selects the subpopulation the node analyses, and the
    rows with a missing time, outcome or covariate are dropped, as the node
    drops them. A categorical covariate (the reference's "sex_female" and
    "sex_intersex" columns) adds one 0/1 column per non-reference category
    and drops the rows whose sex is the missing term.
    """
    rows = patients
    for variable, selection in (variables_to_stratify or {}).items():
        column = STRATIFICATION_TO_REFERENCE[variable]
        if isinstance(selection, dict):
            rows = rows[rows[column].between(selection["start"], selection["end"])]
        else:
            rows = rows[rows[column].isin(selection)]
    rows = rows.dropna(subset=["time", "event", "age"])
    out = pd.DataFrame({"time": rows["time"], "event": rows["event"], "age": rows["age"], "sex": rows["sex"]})
    if "sex_female" in reference_expl_vars:
        known = rows["sex"].isin(["ncit:C20197", "ncit:C16576", "ncit:C45908"])
        out = out[known]
        out["sex_female"] = (rows["sex"] == "ncit:C16576").astype(float)[known]
        out["sex_intersex"] = (rows["sex"] == "ncit:C45908").astype(float)[known]
    return out.reset_index(drop=True)


def _reference_fit(
    rows: pd.DataFrame, reference_expl_vars, n_nodes: int, monkeypatch, variables_to_stratify=None
) -> dict:
    """The in-process pipeline on numeric rows, with the nodes' settings."""
    for name, value in NODE_ENV.items():
        monkeypatch.setenv(name, value)
    organisations = list(range(1, n_nodes + 1))
    client = MockAlgorithmClient(
        datasets=[[{"database": rows.copy(), "db_type": "csv"}] for _ in organisations],
        module="v6-cox-ph",
        organization_ids=organisations,
    )
    kwargs = {
        "time_col": "time",
        "outcome_col": "event",
        "expl_vars": reference_expl_vars,
        "organization_ids": organisations,
    }
    if variables_to_stratify is not None:
        kwargs["variables_to_stratify"] = {
            STRATIFICATION_TO_REFERENCE[name]: selection for name, selection in variables_to_stratify.items()
        }
    task = client.task.create(input_={"method": "central", "kwargs": kwargs}, organizations=[1])
    return client.wait_for_results(task_id=task["id"])[0]


def _logs(client, task_id: int) -> str:
    """The logs of a task's runs and of its sub-tasks' runs."""
    task_ids = [task_id] + [task["id"] for task in client.task.list(parent=task_id)["data"]]
    return "\n".join(run.get("log") or "" for tid in task_ids for run in client.run.from_task(tid)["data"])


@pytest.mark.integration
class TestCoxPHRdfIntegration:
    @pytest.mark.parametrize("config_name", list(RDF_CONFIGURATIONS))
    def test_rdf(self, authentication, algorithm_image_name, rdf_patients, monkeypatch, config_name):
        """
        Fit on the RDF store and compare with the per-patient reference.

        Each node holds one endpoint row pointing at the same store, so every
        node analyses the same 100 patients; the reference runs the same
        number of nodes in-process on the numeric values.
        """
        client = authentication
        config = RDF_CONFIGURATIONS[config_name]
        kwargs = {
            "time_col": TIME,
            "outcome_col": config.get("outcome_col", OUTCOME_COL),
            "expl_vars": config["expl_vars"],
            "organization_ids": config["organization_ids"],
        }
        if "variables_to_stratify" in config:
            kwargs["variables_to_stratify"] = config["variables_to_stratify"]

        task = client.task.create(
            collaboration=1,
            organizations=[1],
            name=f"Test central RDF — {config_name}",
            image=algorithm_image_name,
            description=f"STRONG AYA RDF integration test {config_name}.",
            input_={"method": "central", "kwargs": kwargs},
            databases=[{"label": "rdf_store"}],
        )

        if config.get("expected_failure", False):
            with pytest.raises(Exception) as exc_info:
                extract_coxph_result(client, task)
            verify_error_type(exc_info, config)
            assert config["expected_log"] in _logs(client, task["id"])
            return

        result = extract_coxph_result(client, task)
        reference_expl_vars = config.get("reference_expl_vars", ["age"])
        rows = _reference_rows(rdf_patients, config.get("variables_to_stratify"), reference_expl_vars)
        reference = _reference_fit(
            rows,
            reference_expl_vars,
            len(config["organization_ids"]),
            monkeypatch,
            config.get("variables_to_stratify"),
        )

        expected_terms = [term for term, column in TERM_TO_REFERENCE.items() if column in reference_expl_vars]
        assert set(result["model"]) == set(expected_terms)
        for term in expected_terms:
            fitted = result["model"][term]
            expected = reference["model"][TERM_TO_REFERENCE[term]]
            assert abs(fitted["Coef"] - expected["Coef"]) <= 1e-6, (term, fitted, expected)
            assert abs(fitted["SE"] - expected["SE"]) <= 1e-6, (term, fitted, expected)
        assert result["n_events"] == reference["n_events"]
        assert result["converged"] is True
        assert result["provenance"]["input_arguments"]["outcome_col"] == kwargs["outcome_col"]
