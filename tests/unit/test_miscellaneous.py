"""
Unit tests for miscellaneous Cox-PH functions.

Tests input validation via the Pydantic model.
"""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from vantage6.algorithm.tools.exceptions import UserInputError

# Add the algorithm module to the path
algorithm_path = Path(__file__).parent.parent.parent / "v6-cox-ph"
sys.path.insert(0, str(algorithm_path))

from miscellaneous import (  # noqa: E402
    CoxPHInput,
    term_key,
    parse_outcome_col,
    validate_coxph_input,
    validate_outcome_codes,
    validate_stratification_spec,
)

NCIT = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"


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
