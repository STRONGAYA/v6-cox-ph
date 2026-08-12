# v6-cox-ph

Federated Cox Proportional Hazards algorithm for STRONG AYA.

This algorithm implements the Cox Proportional Hazards model in a federated setting,
following the STRONG AYA convention and data standard for vantage6.

## Overview

The Cox Proportional Hazards model is a popular method for survival analysis that
investigates the relationship between the survival time of subjects and one or more
predictor variables. This federated implementation allows multiple organizations
to collaboratively fit a Cox-PH model without sharing their individual data.

## Features

- **Federated Computation**: Distributed computation across multiple nodes
- **Privacy Preservation**: Implements STRONG AYA privacy guards including:
  - Sample size thresholds
  - Variable masking
  - Data stratification support
- **RDF Data Support**: Can query data from RDF endpoints via v6-tools-rdf
- **Comprehensive Testing**: Full integration test suite with Docker

## Architecture

The algorithm follows the STRONG AYA central-partial pattern:

- **`central.py`**: Orchestrates the federated computation, coordinates subtasks
- **`partial.py`**: Executes on each node with access to local data
- **`coxph_logic.py`**: Contains the core mathematical computations
- **`miscellaneous.py`**: Input validation and helper functions

## Requirements

- Python >= 3.10
- vantage6 >= 4.14
- pandas >= 2.2.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- pydantic >= 2.0.0
- vantage6-strongaya-general >= 1.0.4
- vantage6-strongaya-rdf >= 1.0.2

## Installation

```bash
# Clone the repository
git clone https://github.com/STRONGAYA/v6-cox-ph.git
cd v6-cox-ph

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e .[dev]
```

## Usage

### Running the Algorithm

The algorithm can be run using the vantage6 client:

```python
from vantage6.client import Client

# Connect to the server
client = Client("http://your-server-url", 5000)
client.authenticate(username="your-username", password="your-password")

# Define algorithm parameters
input_data = {
    "method": "central",
    "kwargs": {
        "time_col": "overall_survival_in_days",
        "outcome_col": "event_overall_survival",
        "expl_vars": ["age", "treatment", "sex"],
        "organization_ids": [1, 2, 3]
    }
}

# Create and run a task
task = client.task.create(
    input_=input_data,
    organizations=[1, 2, 3],
    name="Cox-PH Analysis"
)

results = client.wait_for_results(task_id=task["id"])
```

### Input Parameters

- `time_col` (str, required): Name of the column containing survival time data
- `outcome_col` (str, required): Name of the column containing event indicators (1=event, 0=censored)
- `expl_vars` (list, required): List of explanatory variable names to include in the model
- `organization_ids` (list, optional): List of organization IDs to include. If None, all organizations are used.

### Output

The algorithm returns a dictionary containing:

- `included_organizations`: List of organization IDs that contributed to the results
- `excluded_organizations`: List of organization IDs excluded due to privacy thresholds
- `model`: JSON string of a DataFrame containing:
  - `Coef`: Regression coefficients
  - `Exp(coef)`: Exponentiated coefficients (hazard ratios)
  - `SE`: Standard errors
  - `lower_CI`: Lower bound of 95% confidence interval
  - `upper_CI`: Upper bound of 95% confidence interval
  - `Z`: Z-values
  - `p-value`: P-values
- `overall_p_value`: Overall model p-value from Wald test
- `aic`: Akaike Information Criterion
- `degrees_of_freedom`: Number of degrees of freedom
- `warnings`: List of warning messages (e.g., perfect prediction)

## Docker

### Building the Docker Image

```bash
# Build the image
docker build -t v6-cox-ph .

# Or with a specific package name
docker build --build-arg PKG_NAME=v6-cox-ph -t v6-cox-ph .
```

### Running with Docker

The Docker image can be pushed to a registry and used with vantage6:

```bash
# Tag and push to a registry
docker tag v6-cox-ph harbor2.vantage6.ai/algorithms/v6-cox-ph
docker push harbor2.vantage6.ai/algorithms/v6-cox-ph
```

## Testing

### Running Tests

The algorithm includes a comprehensive test suite:

```bash
# Run all tests
pytest

# Run only unit tests
pytest tests/unit/

# Run only integration tests (requires Docker)
pytest tests/integration/

# Run with coverage
pytest --cov=v6-cox-ph --cov-report=html
```

### Test Data

Test data is provided in the `tests/data/` directory:
- `coxph_test_data_1.csv`: Primary test dataset
- `coxph_test_data_2.csv`: Secondary test dataset
- `coxph_test_data_3.csv`: Small dataset for edge case testing

## Privacy and Security

The algorithm implements several privacy measures:

1. **Sample Size Thresholds**: Organizations with insufficient data are excluded
2. **Variable Masking**: Only necessary variables are processed
3. **Data Stratification**: Support for stratified analysis (coming soon)
4. **RDF Integration**: Secure data access from RDF endpoints

## Mathematical Details

The Cox Proportional Hazards model uses partial likelihood estimation with
Newton-Raphson optimization. The algorithm computes:

1. Unique event times across all organizations
2. Summed Z statistics (sum of covariates for event cases)
3. Iterative computation of aggregates for derivative calculation
4. Final model statistics including coefficients, standard errors, and p-values

## Contributing

Contributions are welcome! Please follow the STRONG AYA conventions:

1. Keep `central.py` and `partial.py` clean and readable
2. Place complex mathematical logic in separate modules
3. Use `safe_log()` for all logging
4. Validate all inputs with Pydantic models
5. Implement privacy guards for all partial functions

n
## Development

This algorithm was adapted to STRONG AYA conventions through a series of stacked pull requests:

### Stacked PRs for STRONG AYA Adaptation

1. **PR1: Initial STRONG AYA adaptation** - Package structure and basic dependencies
   - Restructured package from `coxph/` to `v6-cox-ph/`
   - Added `pyproject.toml` with STRONG AYA dependencies
   - Replaced logging with `safe_log` from v6-tools-general
   - Added Pydantic v2 models for input validation
   - Extracted mathematical logic to `coxph_logic.py` module

2. **PR2: Privacy and data handling improvements**
   - Enhanced partial.py with complete STRONG AYA data pipeline
   - Added RDF data collection support
   - Added comprehensive data quality checks
   - Added event count validation
   - Enhanced privacy configurations

3. **PR3: Comprehensive testing infrastructure**
   - Added conftest.py with Cox-PH specific fixtures
   - Added unit tests for all utility functions
   - Added integration tests for algorithm workflows
   - Enhanced test data and configurations

4. **PR4: Final refinements** - CI/CD and documentation
   - Added GitHub Actions workflow for testing
   - Enhanced documentation
   - Final code cleanup and optimization

### Contributing

To contribute to this algorithm:

1. **Follow STRONG AYA conventions**: Keep `central.py` and `partial.py` clean
2. **Use proper logging**: Always use `safe_log()` instead of `print()`
3. **Validate inputs**: Use Pydantic models for all external inputs
4. **Implement privacy guards**: All partial functions must include privacy checks
5. **Test thoroughly**: Add both unit and integration tests
## License

Apache 2.0

## References

- Cox, D. R. (1972). Regression models and life-tables. Journal of the Royal Statistical Society, Series B, 34(2), 187-220.
- Andersen, P. K., & Gill, R. D. (1982). Cox's regression model for counting processes: A large sample study. The Annals of Statistics, 10(4), 1100-1120.
