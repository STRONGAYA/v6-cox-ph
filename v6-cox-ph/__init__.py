"""
v6-cox-ph: Federated Cox Proportional Hazards algorithm for STRONG AYA.

This package provides a federated implementation of the Cox Proportional Hazards
model following STRONG AYA conventions and data standards.
"""

from .central import central
from .partial import get_unique_event_times, compute_summed_z, perform_iteration
from .miscellaneous import (
    CoxPHInput,
    PartialResult,
    PrivacyThresholdConfig,
    validate_coxph_input,
    validate_partial_result,
    check_event_count,
    check_data_quality,
)
from .coxph_logic import (
    compute_derivatives,
    compute_model_results,
    format_results_dataframe,
)

__all__ = [
    "central",
    "get_unique_event_times",
    "compute_summed_z",
    "perform_iteration",
    "CoxPHInput",
    "PartialResult",
    "PrivacyThresholdConfig",
    "validate_coxph_input",
    "validate_partial_result",
    "check_event_count",
    "check_data_quality",
    "compute_derivatives",
    "compute_model_results",
    "format_results_dataframe",
]
