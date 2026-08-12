# Backport Analysis: v6-cox-ph STRONG AYA Adaptation

This document identifies which improvements made during the STRONG AYA adaptation can be backported to the base (non-STRONG AYA) repository to benefit the general vantage6 Cox-PH algorithm.

## 📋 Overview

The STRONG AYA adaptation involved 4 stacked PRs. While many changes are STRONG AYA-specific (RDF support, v6-tools dependencies, etc.), several improvements are **algorithm-agnostic** and would benefit the base repository.

---

## ✅ Backportable Improvements

### 1. **Code Architecture & Modularization** (PR1, PR2)

**Files Modified**: `central.py`, `partial.py`, `coxph_logic.py` (new)

**Changes**:
- **Extracted mathematical logic** from `central.py` into separate `coxph_logic.py` module
- **Cleaned up `central.py`** to be pure orchestration (removed inline math computations)
- **Improved function organization** with clear separation of concerns
- **Better variable naming** and code structure

**Benefits for Base Repository**:
- More maintainable code
- Easier to test individual components
- Clearer separation between orchestration and computation
- Better readability

**Backport Strategy**: 
- Extract `compute_derivatives()` and related math functions to new module
- Keep the same function signatures but remove STRONG AYA-specific logging
- Replace `safe_log()` calls with standard logging or print statements

---

### 2. **Input Validation** (PR1, PR2)

**Files Modified**: `miscellaneous.py` (new)

**Changes**:
- **Added Pydantic v2 models** for input validation (`CoxPHInput`, `PartialResult`)
- **Comprehensive validation** of column names, variable lists, organization IDs
- **Type hints** throughout the codebase
- **Better error messages** for invalid inputs

**Benefits for Base Repository**:
- Catches input errors early with clear messages
- Prevents runtime errors from malformed inputs
- Improves API contract clarity
- Better IDE support with type hints

**Backport Strategy**:
- Add Pydantic as a dependency (already widely used)
- Create validation functions that work with the existing code
- Can be optional/gradual adoption

---

### 3. **Data Quality Checks** (PR2)

**Files Modified**: `miscellaneous.py`

**Changes**:
- **`check_event_count()`** - Validates sufficient events for analysis
- **`check_data_quality()`** - Comprehensive data validation (missing columns, negative times, etc.)
- **Better error handling** for edge cases

**Benefits for Base Repository**:
- Prevents silent failures from bad data
- Better error messages for users
- More robust algorithm execution
- Identifies data issues early

**Backport Strategy**:
- Add as utility functions in existing code
- Can be called before main computation
- Non-breaking changes

---

### 4. **Testing Infrastructure** (PR3)

**Files Modified**: `tests/`, `conftest.py` (new)

**Changes**:
- **Comprehensive test data** (3 CSV datasets with various scenarios)
- **Unit tests** for utility functions
- **Integration test patterns** that can be adapted
- **Test fixtures** for common configurations

**Benefits for Base Repository**:
- Better test coverage
- Reusable test patterns
- Example datasets for testing
- CI/CD ready structure

**Backport Strategy**:
- Port test data and basic test structure
- Adapt tests to work with existing code
- Can be added incrementally

---

### 5. **Algorithm Improvements** (PR2, PR4)

**Files Modified**: `central.py`, `coxph_logic.py`

**Changes**:
- **Better convergence handling** with configurable thresholds
- **Improved numerical stability** checks
- **Better error handling** for edge cases (NaN, infinite values)
- **Wald test** for overall model significance
- **AIC computation** for model comparison
- **Confidence interval calculation**
- **Perfect prediction warnings**

**Benefits for Base Repository**:
- More robust algorithm
- Better statistical outputs
- Improved numerical stability
- More informative warnings

**Backport Strategy**:
- Port the mathematical improvements directly
- Keep existing logging approach
- Maintain backward compatibility

---

### 6. **Documentation** (PR4)

**Files Modified**: `README.md`

**Changes**:
- **Comprehensive usage examples**
- **Clear parameter descriptions**
- **Output format documentation**
- **Mathematical references**
- **Installation instructions**

**Benefits for Base Repository**:
- Better user experience
- Clearer API documentation
- Improved maintainability

**Backport Strategy**:
- Merge documentation improvements directly
- Update to match base repository's actual API

---

### 7. **Docker & CI/CD** (PR4)

**Files Modified**: `Dockerfile`, `.github/workflows/test-suite.yml`

**Changes**:
- **Modern Dockerfile** with Alpine base
- **GitHub Actions workflow** with test matrix
- **Linting configuration** (Black, Flake8, MyPy)
- **Security scanning** (Bandit, Safety)

**Benefits for Base Repository**:
- Better CI/CD pipeline
- Improved code quality
- Security scanning
- Consistent formatting

**Backport Strategy**:
- Adapt workflow to base repository's needs
- Use existing Docker approach if preferred
- Can be adopted piecemeal

---

## ❌ STRONG AYA-Specific Changes (Do NOT Backport)

These changes are specific to STRONG AYA and should NOT be backported:

### 1. **STRONG AYA Dependencies**
- `vantage6-strongaya-general` dependency
- `vantage6-strongaya-rdf` dependency
- Pydantic v2 (if base wants to keep v1)

### 2. **STRONG AYA Infrastructure**
- `safe_log()` usage (replace with standard logging)
- `collect_organisation_ids()` usage
- `check_partial_result_presence()` usage
- RDF data collection via `v6-tools-rdf`

### 3. **STRONG AYA Data Pipeline**
- `mask_unnecessary_variables()` calls
- `apply_sample_size_threshold()` calls
- `set_datatypes()` calls
- `apply_data_stratification()` calls

### 4. **Package Structure**
- Package name change from `coxph` to `v6_cox_ph`
- `pyproject.toml` vs `setup.py` (can be backported separately)

---

## 🎯 Recommended Backport Order

### Priority 1: High Impact, Low Risk
1. **Algorithm Improvements** (convergence, numerical stability, statistical outputs)
2. **Data Quality Checks** (prevents silent failures)
3. **Input Validation** (improves error handling)

### Priority 2: Medium Impact, Medium Risk
4. **Code Architecture** (modularization, separation of concerns)
5. **Testing Infrastructure** (test data, basic test patterns)

### Priority 3: Low Impact, Higher Risk
6. **Documentation** (can be done anytime)
7. **Docker & CI/CD** (requires infrastructure changes)

---

## 📝 Backport Implementation Plan

### For Each Backportable Change:

1. **Create a separate branch** in the base repository
2. **Cherry-pick or manually port** the specific changes
3. **Remove STRONG AYA-specific code**:
   - Replace `safe_log()` with `print()` or standard logging
   - Remove v6-tools imports
   - Remove RDF-specific code
4. **Test thoroughly** to ensure no regressions
5. **Update documentation** to reflect changes

### Example: Backporting Algorithm Improvements

```bash
# In base repository
git checkout -b improve-coxph-algorithm

# Manually copy improvements from:
# - coxph_logic.py (mathematical functions)
# - central.py (convergence handling, error checking)

# Remove STRONG AYA specifics:
# - Replace safe_log with print
# - Remove v6-tools imports
# - Keep the core algorithm logic

git commit -m "Improve Cox-PH algorithm: better convergence, numerical stability, statistical outputs"
```

---

## 🔄 Specific Code Examples for Backporting

### Example 1: Convergence Handling (from central.py)

**STRONG AYA version:**
```python
if math.isnan(delta):
    safe_log("warning", "Delta has turned into a NaN")
    break

if delta <= 0.000001:
    safe_log("info", "Betas have settled! Finished iterating!")
    break
```

**Base repository version:**
```python
if math.isnan(delta):
    print("Delta has turned into a NaN")
    break

if delta <= 0.000001:
    print("Betas have settled! Finished iterating!")
    break
```

### Example 2: Data Quality Check (from miscellaneous.py)

**STRONG AYA version:**
```python
def check_event_count(df: pd.DataFrame, outcome_col: str, min_events: int = None) -> bool:
    if min_events is None:
        min_events = PrivacyThresholdConfig.MIN_EVENT_COUNT
    event_count = df[df[outcome_col] == 1].shape[0]
    return event_count > min_events
```

**Base repository version:**
```python
def check_event_count(df, outcome_col, min_events=5):
    """Check if data has sufficient events for Cox-PH analysis."""
    event_count = df[df[outcome_col] == 1].shape[0]
    return event_count > min_events
```

### Example 3: Wald Test (from coxph_logic.py)

**STRONG AYA version:**
```python
degrees_of_freedom = len(beta)
wald_statistic = np.dot(beta, np.dot(-secondary_derivative, beta))
overall_p_value = chi2.sf(wald_statistic, degrees_of_freedom)
```

**Base repository version:**
```python
# Same code - no STRONG AYA dependencies
degrees_of_freedom = len(beta)
wald_statistic = np.dot(beta, np.dot(-secondary_derivative, beta))
overall_p_value = chi2.sf(wald_statistic, degrees_of_freedom)
```

---

## 📊 Summary Table

| Change | Backportable? | Priority | Risk | Notes |
|--------|---------------|----------|------|-------|
| Code modularization | ✅ Yes | High | Low | Extract math logic |
| Input validation | ✅ Yes | High | Low | Add Pydantic models |
| Data quality checks | ✅ Yes | High | Low | Prevent failures |
| Algorithm improvements | ✅ Yes | High | Low | Better stats, stability |
| Testing infrastructure | ✅ Yes | Medium | Medium | Test data, patterns |
| Documentation | ✅ Yes | Medium | Low | Improve README |
| CI/CD pipeline | ✅ Yes | Low | Medium | GitHub Actions |
| STRONG AYA dependencies | ❌ No | N/A | N/A | v6-tools-specific |
| STRONG AYA logging | ❌ No | N/A | N/A | safe_log usage |
| RDF data support | ❌ No | N/A | N/A | v6-tools-rdf |
| Privacy guards | ❌ No | N/A | N/A | STRONG AYA-specific |

---

## 🎯 Conclusion

**Approximately 60-70% of the improvements** made during the STRONG AYA adaptation can and should be backported to the base repository. These include:

- **Algorithm improvements** (better math, stability, outputs)
- **Code architecture** (modularization, separation of concerns)
- **Input validation** (error handling, type safety)
- **Data quality checks** (robustness, error prevention)
- **Testing infrastructure** (coverage, maintainability)
- **Documentation** (usability)

The remaining 30-40% are STRONG AYA-specific and should remain in the STRONG AYA fork.

**Recommendation**: Create separate PRs in the base repository for each backportable category, starting with the high-priority, low-risk changes (algorithm improvements and data quality checks).