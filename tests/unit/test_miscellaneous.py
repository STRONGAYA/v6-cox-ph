"""
Unit tests for miscellaneous Cox-PH functions.

Tests input validation via the Pydantic model, the code key, the
explanatory-variable resolution (plain, categorical and scored entries) and
the covariate-profile expansion.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from vantage6.algorithm.tools.exceptions import UserInputError

# Add the algorithm module to the path
algorithm_path = Path(__file__).parent.parent.parent / "v6-cox-ph"
sys.path.insert(0, str(algorithm_path))

from miscellaneous import (  # noqa: E402
    CoxPHInput,
    code_key,
    covariate_key,
    parse_expl_vars,
    parse_outcome_col,
    term_key,
    validate_coxph_input,
    validate_covariate_spec,
    validate_outcome_codes,
    validate_stratification_spec,
)

NCIT = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"
SEX_SPEC = {"reference": "M", "categories": {"M": ["M"], "F": ["F"]}}
STAGE_SPEC = {"scores": {"I": 1, "II": 2, "III": 3, "IV": 4}, "missing": ["unknown"]}


@pytest.mark.unit
class TestValidateCoxphInput:
    """Tests for the validate_coxph_input function."""

    def test_validate_input_valid(self):
        """Test input validation with valid parameters."""
        validated = validate_coxph_input(
            time_col="time",
            outcome_col="event",
            expl_vars=["age", "treatment"],
            organization_ids=[1, 2],
        )
        assert validated.time_col == "time"
        assert validated.outcome_col == "event"
        assert validated.expl_vars == ["age", "treatment"]
        assert validated.organization_ids == [1, 2]

    def test_validate_input_no_organisation_ids(self):
        """Test input validation with no organisation IDs."""
        validated = validate_coxph_input(
            time_col="time",
            outcome_col="event",
            expl_vars=["age"],
        )
        assert validated.organization_ids is None

    def test_validate_input_empty_expl_vars(self):
        """Test that empty explanatory variables raises UserInputError."""
        with pytest.raises(UserInputError):
            validate_coxph_input(
                time_col="time",
                outcome_col="event",
                expl_vars=[],
                organization_ids=[1, 2],
            )

    def test_validate_input_empty_time_col(self):
        """Test that empty time column name raises UserInputError."""
        with pytest.raises(UserInputError):
            validate_coxph_input(
                time_col="",
                outcome_col="event",
                expl_vars=["age"],
            )

    def test_validate_input_empty_outcome_col(self):
        """Test that empty outcome column name raises UserInputError."""
        with pytest.raises(UserInputError):
            validate_coxph_input(
                time_col="time",
                outcome_col="",
                expl_vars=["age"],
            )

    def test_validate_input_whitespace_column_name(self):
        """Test that whitespace-only column names are rejected."""
        with pytest.raises(UserInputError):
            validate_coxph_input(
                time_col="  ",
                outcome_col="event",
                expl_vars=["age"],
            )

    def test_validate_input_invalid_expl_var_name(self):
        """Test that invalid variable names raise UserInputError."""
        with pytest.raises(UserInputError):
            validate_coxph_input(
                time_col="time",
                outcome_col="event",
                expl_vars=["age", ""],
            )

    def test_validate_input_negative_org_id(self):
        """Test that negative organisation IDs raise UserInputError."""
        with pytest.raises(UserInputError):
            validate_coxph_input(
                time_col="time",
                outcome_col="event",
                expl_vars=["age"],
                organization_ids=[-1],
            )

    def test_validate_input_deduplicates_org_ids(self):
        """Test that duplicate organisation IDs are deduplicated."""
        validated = validate_coxph_input(
            time_col="time",
            outcome_col="event",
            expl_vars=["age"],
            organization_ids=[1, 1, 2],
        )
        assert sorted(validated.organization_ids) == [1, 2]

    def test_validate_input_strips_whitespace(self):
        """Test that whitespace in column names is stripped."""
        validated = validate_coxph_input(
            time_col="  time  ",
            outcome_col=" event ",
            expl_vars=["  age  "],
        )
        assert validated.time_col == "time"
        assert validated.outcome_col == "event"
        assert validated.expl_vars == ["age"]

    def test_validate_input_time_col_equals_outcome_col(self):
        """Test that time_col == outcome_col raises UserInputError."""
        with pytest.raises(UserInputError, match="different columns"):
            validate_coxph_input(
                time_col="time",
                outcome_col="time",
                expl_vars=["age"],
            )

    def test_validate_input_duplicate_expl_vars(self):
        """Test that duplicate explanatory variables raise UserInputError."""
        with pytest.raises(UserInputError, match="duplicate variables"):
            validate_coxph_input(
                time_col="time",
                outcome_col="event",
                expl_vars=["age", "treatment", "age"],
            )

    def test_validate_input_expl_var_names_time_col(self):
        """Test that an explanatory variable naming the time column is rejected."""
        with pytest.raises(UserInputError, match="time or outcome column"):
            validate_coxph_input(
                time_col="time",
                outcome_col="event",
                expl_vars=["time", "age"],
            )

    def test_validate_input_expl_var_names_outcome_col(self):
        """Test that an explanatory variable naming the outcome column is rejected."""
        with pytest.raises(UserInputError, match="time or outcome column"):
            validate_coxph_input(
                time_col="time",
                outcome_col="event",
                expl_vars=["age", "event"],
            )

    def test_model_rejects_semantic_violations_directly(self):
        """The model itself raises ValidationError for semantic violations."""
        with pytest.raises(ValidationError):
            CoxPHInput(time_col="time", outcome_col="time", expl_vars=["age"])
        with pytest.raises(ValidationError):
            CoxPHInput(time_col="time", outcome_col="event", expl_vars=["age", "age"])
        with pytest.raises(ValidationError):
            CoxPHInput(time_col="time", outcome_col="event", expl_vars=["event"])

    def test_wrapped_error_is_formatted_without_input(self):
        """The UserInputError wraps the one-line formatted validation message."""
        with pytest.raises(UserInputError) as excinfo:
            validate_coxph_input(
                time_col="time",
                outcome_col="event",
                expl_vars=["age"],
                organization_ids=[-1],
            )
        message = str(excinfo.value)
        assert message.startswith("Invalid Cox-PH input:")
        assert "https://errors.pydantic.dev" not in message


@pytest.mark.unit
class TestStratificationSpec:
    """Tests for the variables_to_stratify validation (STRONG AYA standard)."""

    def test_valid_value_list(self):
        spec = {"sex": ["F", "M"], "treatment": [1]}
        assert validate_stratification_spec(spec) == spec

    def test_valid_range_both_bounds(self):
        spec = {"age": {"start": 40, "end": 60}}
        assert validate_stratification_spec(spec) == spec

    def test_valid_range_single_bound(self):
        assert validate_stratification_spec({"age": {"start": 40}}) == {"age": {"start": 40}}
        assert validate_stratification_spec({"age": {"end": 60}}) == {"age": {"end": 60}}
        assert validate_stratification_spec({"age": {"start": None, "end": 60}}) == {"age": {"start": None, "end": 60}}

    @pytest.mark.parametrize("spec", [{}, "not-a-dict", [], {"sex": "F"}, {"sex": 1}])
    def test_non_mapping_rejected(self, spec):
        with pytest.raises(ValueError):
            validate_stratification_spec(spec)

    @pytest.mark.parametrize("name", ["ncit:C28421", "sct:419099009", "strongaya:n_follow_ups", "biological_sex"])
    def test_prefixed_name_accepted(self, name):
        """RDF variables are named by their class: prefixed names are accepted."""
        assert validate_stratification_spec({name: ["ncit:C16576"]}) == {name: ["ncit:C16576"]}

    @pytest.mark.parametrize(
        "name",
        [
            "age at diagnosis",
            "age`",
            "age>=40",
            "1age",
            "",
            "age;drop",
            "ncit:",
            ":C28421",
            "ncit:C28421:C1",
            "ncit:C28421`",
            "ncit: C28421",
            "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C28421",
        ],
    )
    def test_non_identifier_name_rejected(self, name):
        with pytest.raises(ValueError, match="identifier-shaped"):
            validate_stratification_spec({name: [1]})

    @pytest.mark.parametrize("selection", [[], ["F", None], ["F", True], ["F", ["nested"]], ["F", {"start": 1}]])
    def test_invalid_value_list_rejected(self, selection):
        with pytest.raises(ValueError):
            validate_stratification_spec({"sex": selection})

    @pytest.mark.parametrize(
        "selection",
        [
            {"start": 60, "end": 40},
            {"start": "40"},
            {"start": True},
            {"lower": 40},
            {"start": 40, "end": 60, "step": 5},
            {},
        ],
    )
    def test_invalid_range_rejected(self, selection):
        with pytest.raises(ValueError):
            validate_stratification_spec({"age": selection})

    def test_input_model_accepts_valid_stratification(self):
        validated = validate_coxph_input(
            time_col="time",
            outcome_col="event",
            expl_vars=["age", "treatment"],
            variables_to_stratify={"sex": ["F"], "age": {"start": 40, "end": 60}},
        )
        assert validated.variables_to_stratify == {"sex": ["F"], "age": {"start": 40, "end": 60}}

    def test_input_model_defaults_to_none(self):
        validated = validate_coxph_input(time_col="time", outcome_col="event", expl_vars=["age"])
        assert validated.variables_to_stratify is None

    def test_input_model_rejects_invalid_stratification(self):
        with pytest.raises(UserInputError):
            validate_coxph_input(
                time_col="time",
                outcome_col="event",
                expl_vars=["age"],
                variables_to_stratify={"age": {"start": 60, "end": 40}},
            )

    def test_wrapped_error_does_not_echo_values(self):
        """The formatted validation error never echoes the offending values."""
        with pytest.raises(UserInputError) as excinfo:
            validate_coxph_input(
                time_col="time",
                outcome_col="event",
                expl_vars=["age"],
                variables_to_stratify={"age": {"start": 61, "end": 40}},
            )
        message = str(excinfo.value)
        assert "61" not in message
        assert "40" not in message


@pytest.mark.unit
class TestTermKey:
    """Coded values compare through term_key (STRONG AYA data standard)."""

    def test_full_iri_and_prefixed_name_match(self):
        assert term_key(f"{NCIT}C37987") == "ncit:C37987"
        assert term_key("ncit:C37987") == "ncit:C37987"
        assert term_key(f"<{NCIT}C37987>") == "ncit:C37987"
        assert term_key(f"  {NCIT}C37987 ") == "ncit:C37987"

    def test_both_snomed_namespaces_compact_to_sct(self):
        assert term_key("http://snomed.info/id/419099009") == "sct:419099009"
        assert term_key("http://purl.bioontology.org/ontology/SNOMEDCT/419099009") == "sct:419099009"

    def test_unknown_namespace_is_compared_as_written(self):
        assert term_key("http://example.org/dead") == "http://example.org/dead"
        assert term_key("dead") == "dead"

    def test_numbers_and_numeric_strings_match(self):
        assert term_key(1) == term_key(1.0) == term_key("1") == term_key(" 1.0 ") == 1.0
        assert term_key(2) == 2.0
        assert term_key(True) == 1.0

    @pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "", "   ", "<>"])
    def test_missing_values_have_no_key(self, value):
        assert term_key(value) is None


@pytest.mark.unit
class TestOutcomeCodes:
    """Tests for the outcome codes in outcome_col (STRONG AYA standard)."""

    CODES = {
        "event": ["ncit:C99799", "ncit:C99798"],
        "censored": ["ncit:C37987", "ncit:C141463"],
        "missing": ["ncit:C54031"],
    }

    def test_plain_column_name(self):
        assert parse_outcome_col(" event ") == ("event", None)

    def test_column_name_with_codes(self):
        assert parse_outcome_col({"ncit:C25717": self.CODES}) == ("ncit:C25717", self.CODES)
        assert parse_outcome_col({" ncit:C25717 ": self.CODES}) == ("ncit:C25717", self.CODES)

    def test_missing_list_is_optional_and_may_be_empty(self):
        assert validate_outcome_codes({"event": [2], "censored": [1]}, "status") == {"event": [2], "censored": [1]}
        codes = {"event": ["dead"], "censored": ["alive"], "missing": []}
        assert validate_outcome_codes(codes, "status") == codes

    @pytest.mark.parametrize(
        "outcome_col",
        [
            "",
            "   ",
            None,
            1,
            [],
            ["event"],
            {},
            {"event": {"event": [1], "censored": [0]}, "status": {"event": [1], "censored": [0]}},
            {"": {"event": [1], "censored": [0]}},
        ],
    )
    def test_malformed_outcome_col_rejected(self, outcome_col):
        with pytest.raises(ValueError):
            parse_outcome_col(outcome_col)

    @pytest.mark.parametrize(
        "codes",
        [
            None,
            [],
            "ncit:C99799",
            ["ncit:C99799"],
            {"event": ["ncit:C99799"]},
            {"censored": ["ncit:C37987"]},
            {"event": [], "censored": ["ncit:C37987"]},
            {"event": ["ncit:C99799"], "censored": []},
            {"event": "ncit:C99799", "censored": ["ncit:C37987"]},
            {"event": ["ncit:C99799"], "censored": ["ncit:C37987"], "missing": "ncit:C54031"},
            {"event": ["ncit:C99799"], "censored": ["ncit:C37987"], "unknown": ["ncit:C54031"]},
        ],
    )
    def test_malformed_codes_rejected(self, codes):
        with pytest.raises(ValueError):
            parse_outcome_col({"ncit:C25717": codes})

    @pytest.mark.parametrize("code", [True, None, float("nan"), "", ["ncit:C99799"], {"code": "ncit:C99799"}])
    def test_invalid_code_rejected(self, code):
        with pytest.raises(ValueError):
            validate_outcome_codes({"event": ["ncit:C99799", code], "censored": ["ncit:C37987"]}, "ncit:C25717")

    def test_code_under_two_keys_rejected(self):
        with pytest.raises(ValueError, match="listed under both 'event' and 'censored'"):
            validate_outcome_codes({"event": ["ncit:C99799"], "censored": ["ncit:C99799"]}, "ncit:C25717")

    def test_code_under_two_keys_rejected_across_spellings(self):
        """A prefixed name and its full IRI are the same code."""
        with pytest.raises(ValueError, match="listed under both"):
            validate_outcome_codes(
                {"event": ["ncit:C99799"], "censored": ["ncit:C37987"], "missing": [f"{NCIT}C99799"]}, "ncit:C25717"
            )
        with pytest.raises(ValueError, match="listed under both"):
            validate_outcome_codes({"event": [1], "censored": ["1.0"]}, "status")

    def test_duplicate_code_under_one_key_accepted(self):
        codes = {"event": ["ncit:C99799", f"{NCIT}C99799"], "censored": ["ncit:C37987"]}
        assert validate_outcome_codes(codes, "ncit:C25717") == codes

    def test_input_model_accepts_codes(self):
        validated = validate_coxph_input(
            time_col="ncit:C125201", outcome_col={"ncit:C25717": self.CODES}, expl_vars=["age"]
        )
        assert validated.outcome_col == {"ncit:C25717": self.CODES}
        assert validated.outcome_name == "ncit:C25717"

    def test_input_model_keeps_plain_name(self):
        validated = validate_coxph_input(time_col="time", outcome_col=" event ", expl_vars=["age"])
        assert validated.outcome_col == "event"
        assert validated.outcome_name == "event"

    def test_input_model_rejects_invalid_codes(self):
        with pytest.raises(UserInputError, match="outcome_col"):
            validate_coxph_input(time_col="time", outcome_col={"status": {"event": ["ncit:C99799"]}}, expl_vars=["age"])

    def test_column_semantics_use_the_name(self):
        """The time and explanatory columns must differ from the outcome column's name."""
        with pytest.raises(UserInputError, match="must be different columns"):
            validate_coxph_input(time_col="status", outcome_col={"status": self.CODES}, expl_vars=["age"])
        with pytest.raises(UserInputError, match="must not name the time or outcome column"):
            validate_coxph_input(time_col="time", outcome_col={"status": self.CODES}, expl_vars=["status"])


@pytest.mark.unit
class TestCodeKey:
    """Tests for code_key, the single code-comparison extension point."""

    def test_missing_values_give_none(self):
        assert code_key(None) is None
        assert code_key(float("nan")) is None
        assert code_key(np.nan) is None
        assert code_key(pd.NA) is None
        assert code_key("") is None
        assert code_key("   ") is None

    def test_numbers_and_numeric_text_give_the_same_float(self):
        assert code_key(1) == 1.0
        assert code_key(1.0) == 1.0
        assert code_key("1") == 1.0
        assert code_key(" 1 ") == 1.0
        assert code_key("1e2") == 100.0
        assert code_key(1) == code_key("1") == code_key(1.0)

    def test_other_strings_are_compared_stripped(self):
        assert code_key("M") == "M"
        assert code_key(" M ") == "M"
        assert code_key("stage II") == "stage II"


@pytest.mark.unit
class TestCovariateKey:
    """Covariate codes compare through covariate_key (STRONG AYA layer).

    ``covariate_key`` applies ``term_key`` — a prefixed name matches its
    full IRI — and the STRONG AYA missing term is always missing for
    covariates.
    """

    def test_full_iri_and_prefixed_name_match(self):
        assert covariate_key(f"{NCIT}C16576") == "ncit:C16576"
        assert covariate_key("ncit:C16576") == "ncit:C16576"
        assert covariate_key(f"<{NCIT}C16576>") == "ncit:C16576"
        assert covariate_key(f"  {NCIT}C16576 ") == "ncit:C16576"

    def test_missing_term_is_always_missing(self):
        """ncit:C54031 is missing for covariates, as a prefixed name or a full IRI."""
        assert covariate_key("ncit:C54031") is None
        assert covariate_key(f"{NCIT}C54031") is None
        assert covariate_key(f"<{NCIT}C54031>") is None

    def test_numbers_and_numeric_strings_match(self):
        assert covariate_key(1) == covariate_key(1.0) == covariate_key("1") == covariate_key(" 1.0 ") == 1.0
        assert covariate_key(True) == 1.0

    def test_other_strings_are_compared_stripped(self):
        assert covariate_key("male") == "male"
        assert covariate_key(" male ") == "male"

    @pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "", "   ", "<>"])
    def test_missing_values_have_no_key(self, value):
        assert covariate_key(value) is None


@pytest.mark.unit
class TestValidateCovariateSpec:
    """Structural validation of one categorical or scored spec."""

    def test_valid_categorical(self):
        validate_covariate_spec(SEX_SPEC, "sex")

    def test_valid_scored(self):
        validate_covariate_spec(STAGE_SPEC, "stage")

    def test_spec_not_a_mapping(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            validate_covariate_spec(["M", "F"], "sex")

    def test_unknown_keys(self):
        with pytest.raises(ValueError, match="unknown spec keys"):
            validate_covariate_spec({**SEX_SPEC, "bogus": 1}, "sex")

    def test_both_categories_and_scores(self):
        with pytest.raises(ValueError, match="exactly one"):
            validate_covariate_spec({"categories": {"M": ["M"], "F": ["F"]}, "scores": {"M": 0, "F": 1}}, "sex")

    def test_reference_with_scores(self):
        with pytest.raises(ValueError, match="'reference' is only allowed"):
            validate_covariate_spec({"reference": "M", "scores": {"M": 0, "F": 1}}, "sex")

    def test_no_reference(self):
        with pytest.raises(ValueError, match="'reference' must be one of"):
            validate_covariate_spec({"categories": {"M": ["M"], "F": ["F"]}}, "sex")

    def test_reference_not_a_label(self):
        with pytest.raises(ValueError, match="'reference' must be one of"):
            validate_covariate_spec({"reference": "X", "categories": {"M": ["M"], "F": ["F"]}}, "sex")

    def test_one_category(self):
        with pytest.raises(ValueError, match="at least two labels"):
            validate_covariate_spec({"reference": "M", "categories": {"M": ["M"]}}, "sex")

    def test_empty_code_list(self):
        with pytest.raises(ValueError, match="at least one code"):
            validate_covariate_spec({"reference": "M", "categories": {"M": ["M"], "F": []}}, "sex")

    def test_bad_label(self):
        with pytest.raises(ValueError, match="category label"):
            validate_covariate_spec({"reference": "M", "categories": {"M": ["M"], "F!": ["F"]}}, "sex")

    def test_bool_code(self):
        with pytest.raises(ValueError, match="boolean code"):
            validate_covariate_spec({"reference": "M", "categories": {"M": [True], "F": ["F"]}}, "sex")

    def test_code_not_a_string_or_number(self):
        with pytest.raises(ValueError, match="not a string or a number"):
            validate_covariate_spec({"reference": "M", "categories": {"M": [["M"]], "F": ["F"]}}, "sex")

    def test_missing_not_a_list(self):
        with pytest.raises(ValueError, match="'missing' must be a list"):
            validate_covariate_spec({"scores": {"I": 1, "II": 2}, "missing": "unknown"}, "stage")

    def test_one_score(self):
        with pytest.raises(ValueError, match="at least two codes"):
            validate_covariate_spec({"scores": {"I": 1}}, "stage")

    def test_score_not_a_number(self):
        with pytest.raises(ValueError, match="must be a number"):
            validate_covariate_spec({"scores": {"I": "one", "II": 2}}, "stage")

    def test_score_not_finite(self):
        with pytest.raises(ValueError, match="must be finite"):
            validate_covariate_spec({"scores": {"I": float("inf"), "II": 2}}, "stage")
        with pytest.raises(ValueError, match="must be finite"):
            validate_covariate_spec({"scores": {"I": float("nan"), "II": 2}}, "stage")

    def test_bool_score(self):
        with pytest.raises(ValueError, match="must be a number"):
            validate_covariate_spec({"scores": {"I": True, "II": 2}}, "stage")


@pytest.mark.unit
class TestParseExplVars:
    """Tests for parse_expl_vars: resolution, model terms and refusals."""

    def test_plain_names(self):
        resolved = parse_expl_vars(["age", "treatment"])
        assert resolved.columns == ["age", "treatment"]
        assert resolved.terms == ["age", "treatment"]
        assert resolved.plain == ["age", "treatment"]
        assert resolved.categorical == []
        assert resolved.scored == []

    def test_categorical_entry(self):
        resolved = parse_expl_vars(["age", {"sex": SEX_SPEC}])
        assert resolved.columns == ["age", "sex"]
        assert resolved.terms == ["age", "sex[F]"]
        assert resolved.plain == ["age"]
        assert len(resolved.categorical) == 1
        entry = resolved.categorical[0]
        assert entry.name == "sex"
        assert entry.reference == "M"
        assert entry.categories == {"M": ["M"], "F": ["F"]}
        assert entry.missing == []

    def test_scored_entry(self):
        resolved = parse_expl_vars([{"stage": STAGE_SPEC}])
        assert resolved.columns == ["stage"]
        assert resolved.terms == ["stage"]
        assert resolved.scored[0].name == "stage"
        assert resolved.scored[0].scores == {"I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0}
        assert resolved.scored[0].missing == ["unknown"]

    def test_model_term_order(self):
        resolved = parse_expl_vars(["age", {"sex": SEX_SPEC}, {"stage": STAGE_SPEC}])
        assert resolved.terms == ["age", "sex[F]", "stage"]

    def test_non_reference_labels_in_listed_order(self):
        spec = {"reference": "M", "categories": {"M": ["M"], "F": ["F"], "X": ["X"]}}
        resolved = parse_expl_vars([{"sex": spec}])
        assert resolved.terms == ["sex[F]", "sex[X]"]

    def test_empty_expl_vars(self):
        with pytest.raises(ValueError, match="At least one"):
            parse_expl_vars([])

    def test_mapping_with_two_names(self):
        with pytest.raises(ValueError, match="exactly one name"):
            parse_expl_vars([{"sex": SEX_SPEC, "age": {}}])

    def test_duplicate_names(self):
        with pytest.raises(ValueError, match="duplicate variables"):
            parse_expl_vars(["age", "age"])
        with pytest.raises(ValueError, match="duplicate variables"):
            parse_expl_vars(["age", {"age": SEX_SPEC}])

    def test_term_name_clash(self):
        with pytest.raises(ValueError, match="clash"):
            parse_expl_vars(["sex[F]", {"sex": SEX_SPEC}])

    def test_code_listed_twice(self):
        with pytest.raises(ValueError, match="are the same"):
            parse_expl_vars([{"sex": {"reference": "M", "categories": {"M": ["M"], "F": ["F", "M"]}}}])

    def test_code_listed_in_category_and_missing(self):
        with pytest.raises(ValueError, match="are the same"):
            parse_expl_vars([{"sex": {**SEX_SPEC, "missing": ["M"]}}])

    def test_number_and_its_text_collide(self):
        with pytest.raises(ValueError, match="are the same"):
            parse_expl_vars([{"grp": {"reference": "0", "categories": {"0": [0, "0"], "1": [1]}}}])

    def test_missing_codes_are_ignored(self):
        """A listed code that is missing anyway (None, NaN, empty text) is ignored."""
        resolved = parse_expl_vars([{"sex": {"reference": "M", "categories": {"M": ["M", None, ""], "F": ["F"]}}}])
        assert resolved.terms == ["sex[F]"]


@pytest.mark.unit
class TestCoxPHInputCovariates:
    """The categorical/scored wire format through the input model."""

    def test_model_terms_property(self):
        validated = validate_coxph_input(
            time_col="time",
            outcome_col="event",
            expl_vars=["age", {"sex": SEX_SPEC}, {"stage": STAGE_SPEC}],
        )
        assert validated.model_terms == ["age", "sex[F]", "stage"]

    def test_model_terms_count_for_the_covariate_cap(self):
        """The cap counts model terms, not entries: 11 three-category entries give 22 terms."""
        expl_vars = [
            {f"v{i}": {"reference": "a", "categories": {"a": ["a"], "b": ["b"], "c": ["c"]}}} for i in range(11)
        ]
        validated = validate_coxph_input(time_col="time", outcome_col="event", expl_vars=expl_vars)
        assert len(validated.expl_vars) == 11
        assert len(validated.model_terms) == 22

    def test_spec_refusals_raise_user_input_error(self):
        bad_specs = [
            {"categories": {"M": ["M"], "F": ["F"]}},  # no reference
            {"reference": "X", "categories": {"M": ["M"], "F": ["F"]}},  # reference not a label
            {"reference": "M", "categories": {"M": ["M"]}},  # one category
            {"reference": "M", "categories": {"M": ["M"], "F": []}},  # empty code list
            {"reference": "M", "categories": {"M": ["M"], "F": ["F", "M"]}},  # code listed twice
            {"categories": {"M": ["M"], "F": ["F"]}, "scores": {"M": 0, "F": 1}},  # both
            {"reference": "M", "scores": {"M": 0, "F": 1}},  # reference with scores
            {"reference": "M", "categories": {"M": ["M"], "F!": ["F"]}},  # bad label
            {"reference": "M", "categories": {"M": [True], "F": ["F"]}},  # bool code
            {**SEX_SPEC, "bogus": 1},  # unknown key
        ]
        for spec in bad_specs:
            with pytest.raises(UserInputError):
                validate_coxph_input(time_col="time", outcome_col="event", expl_vars=[{"sex": spec}])

    def test_term_name_clash_raises(self):
        with pytest.raises(UserInputError, match="clash"):
            validate_coxph_input(time_col="time", outcome_col="event", expl_vars=["sex[F]", {"sex": SEX_SPEC}])

    def test_name_clashes_with_time_and_outcome_columns(self):
        with pytest.raises(UserInputError, match="time or outcome column"):
            validate_coxph_input(time_col="time", outcome_col="event", expl_vars=[{"time": SEX_SPEC}])
        with pytest.raises(UserInputError, match="time or outcome column"):
            validate_coxph_input(time_col="time", outcome_col="event", expl_vars=[{"event": SEX_SPEC}])

    def test_model_term_clashes_with_time_and_outcome_columns(self):
        # A categorical term that the node would encode into a column of
        # the time/outcome column's name is refused at central entry.
        with pytest.raises(UserInputError, match="Model terms must not equal"):
            validate_coxph_input(
                time_col="grp[rare]",
                outcome_col="event",
                expl_vars=[{"grp": {"reference": "base", "categories": {"base": ["base"], "rare": ["rare"]}}}],
            )
        with pytest.raises(UserInputError, match="Model terms must not equal"):
            validate_coxph_input(
                time_col="time",
                outcome_col="event[x]",
                expl_vars=[{"event": {"reference": "a", "categories": {"a": ["0"], "x": ["1"]}}}],
            )

    def test_duplicate_names_raise(self):
        with pytest.raises(UserInputError, match="duplicate variables"):
            validate_coxph_input(time_col="time", outcome_col="event", expl_vars=["age", {"age": SEX_SPEC}])

    def test_profiles_with_labels(self):
        validated = validate_coxph_input(
            time_col="time",
            outcome_col="event",
            expl_vars=["age", {"sex": SEX_SPEC}],
            covariate_profiles=[{"sex": "F"}, {"age": 70.0, "sex": "M"}],
        )
        assert validated.expanded_covariate_profiles == [{"sex[F]": 1.0}, {"age": 70.0, "sex[F]": 0.0}]

    def test_profile_reference_label_expands_to_zeros(self):
        validated = validate_coxph_input(
            time_col="time",
            outcome_col="event",
            expl_vars=[{"sex": SEX_SPEC}],
            covariate_profiles=[{"sex": "M"}],
        )
        assert validated.expanded_covariate_profiles == [{"sex[F]": 0.0}]

    def test_profile_unknown_label(self):
        with pytest.raises(UserInputError, match="one of its labels"):
            validate_coxph_input(
                time_col="time",
                outcome_col="event",
                expl_vars=[{"sex": SEX_SPEC}],
                covariate_profiles=[{"sex": "X"}],
            )

    def test_profile_unknown_variable(self):
        with pytest.raises(UserInputError, match="unknown variables"):
            validate_coxph_input(
                time_col="time",
                outcome_col="event",
                expl_vars=["age"],
                covariate_profiles=[{"nope": 1.0}],
            )

    def test_profile_plain_variable_takes_a_number(self):
        with pytest.raises(UserInputError, match="takes a number"):
            validate_coxph_input(
                time_col="time",
                outcome_col="event",
                expl_vars=["age"],
                covariate_profiles=[{"age": "old"}],
            )
