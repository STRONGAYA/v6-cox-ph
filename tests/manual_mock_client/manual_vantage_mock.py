"""
Run this script to test your algorithm locally (without building a Docker
image) using the mock client.

Run as:

    python tests/manual_mock_client/manual_vantage_mock.py

Make sure to do so in an environment where `vantage6-algorithm-tools` is
installed. This can be done by running:

    pip install vantage6-algorithm-tools
"""

import os

# Set sample size threshold for testing (mock client creates small working drafts)
os.environ["SAMPLE_SIZE_THRESHOLD"] = "5"

from vantage6.algorithm.tools.mock_client import MockAlgorithmClient
from pathlib import Path

# Get path of current directory
current_path = Path(__file__).parent

# Mock client for v6-cox-ph algorithm
# Note: Each organization gets its own dataset
client = MockAlgorithmClient(
    datasets=[
        # Data for the first organisation
        [
            {
                "database": current_path / "coxph_test_data_1.csv",
                "db_type": "csv",
                "input_data": {},
            }
        ],
        # Data for the second organisation
        [
            {
                "database": current_path / "coxph_test_data_2.csv",
                "db_type": "csv",
                "input_data": {},
            }
        ],
        # Data for the third organisation
        [
            {
                "database": current_path / "coxph_test_data_3.csv",
                "db_type": "csv",
                "input_data": {},
            }
        ],
    ],
    module="v6-cox-ph",
)

# List mock organisations
organisations = client.organization.list()
print("Mock organisations:")
print(organisations)
org_ids = [organisation["id"] for organisation in organisations]
print(f"\nOrganisation IDs: {org_ids}")

# Test 1: Run the central method on all organisations
print("\n" + "=" * 60)
print("Test 1: Central method with all organisations")
print("=" * 60)

try:
    central_task = client.task.create(
        input_={
            "method": "central",
            "kwargs": {
                "time_col": "time",
                "outcome_col": "event",
                "expl_vars": ["age", "treatment"],
                "organization_ids": org_ids,
            },
        },
        organizations=org_ids,
    )
    results = client.wait_for_results(central_task.get("id"))
    print("\nCentral method results:")
    print(results)
except Exception as e:
    print(f"\nError in Test 1: {type(e).__name__}: {e}")

# Test 2: Run with subset of organisations (just first two)
print("\n" + "=" * 60)
print("Test 2: Central method with subset of organisations")
print("=" * 60)

try:
    subset_task = client.task.create(
        input_={
            "method": "central",
            "kwargs": {
                "time_col": "time",
                "outcome_col": "event",
                "expl_vars": ["age"],  # Single covariate
                "organization_ids": org_ids[:2],  # First two organisations
            },
        },
        organizations=org_ids[:2],
    )
    subset_results = client.wait_for_results(subset_task.get("id"))
    print("\nSubset results:")
    print(subset_results)
except Exception as e:
    print(f"\nError in Test 2: {type(e).__name__}: {e}")

# Test 3: Run with single organisation (should work)
print("\n" + "=" * 60)
print("Test 3: Central method with single organisation")
print("=" * 60)

try:
    single_task = client.task.create(
        input_={
            "method": "central",
            "kwargs": {
                "time_col": "time",
                "outcome_col": "event",
                "expl_vars": ["age", "treatment"],
                "organization_ids": [org_ids[0]],
            },
        },
        organizations=[org_ids[0]],
    )
    single_results = client.wait_for_results(single_task.get("id"))
    print("\nSingle organisation results:")
    print(single_results)
except Exception as e:
    print(f"\nError in Test 3: {type(e).__name__}: {e}")

# Test 4: Test with invalid input (should fail gracefully)
print("\n" + "=" * 60)
print("Test 4: Invalid input (empty explanatory variables)")
print("=" * 60)

try:
    invalid_task = client.task.create(
        input_={
            "method": "central",
            "kwargs": {
                "time_col": "time",
                "outcome_col": "event",
                "expl_vars": [],  # Empty list - should fail
                "organization_ids": org_ids,
            },
        },
        organizations=org_ids,
    )
    invalid_results = client.wait_for_results(invalid_task.get("id"))
    print("\nInvalid input results (unexpected):")
    print(invalid_results)
except Exception as e:
    print(f"\nExpected error caught: {type(e).__name__}: {e}")

# Test 5: Test with non-existent column
print("\n" + "=" * 60)
print("Test 5: Invalid input (non-existent column)")
print("=" * 60)

try:
    nonexistent_task = client.task.create(
        input_={
            "method": "central",
            "kwargs": {
                "time_col": "nonexistent_column",
                "outcome_col": "event",
                "expl_vars": ["age"],
                "organization_ids": org_ids,
            },
        },
        organizations=org_ids,
    )
    nonexistent_results = client.wait_for_results(nonexistent_task.get("id"))
    print("\nNon-existent column results (unexpected):")
    print(nonexistent_results)
except Exception as e:
    print(f"\nExpected error caught: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("All manual mock client tests completed!")
print("=" * 60)
