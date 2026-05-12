# `test/test_generic.py` Coverage Summary

This file summarizes what `test/test_generic.py` currently tests.

## Scope

The test file focuses on the generic dual-bounds implementation:

- `dualbounds.generic.DualBounds`
- `dualbounds.generic.plug_in_no_covariates`
- distribution helpers such as `parse_dist` and `BatchedCategorical`
- regression and propensity model wiring from `dualbounds.dist_reg`
- generated regression data from `dualbounds.gen_data`

It uses both `unittest` style test classes and `pytest` markers. One test is marked `slow`.

## `TestGenericDualBounds`

### `test_var_ite_oracle`

Checks that `DualBounds.compute_dual_variables` returns objective values matching known analytical bounds for the variance of the individual treatment effect, `Var(Y(1)-Y(0))`.

The test uses known marginal distributions for `Y(0)` and `Y(1)`, including Gaussian, exponential, t, and Bernoulli cases. It compares the computed lower and upper bounds against simulation benchmarks based on monotone and countermonotone couplings.

### `test_gridsearch_loss`

Checks that the grid-search dual-variable computation has small average objective residuals.

It generates heavy-tailed continuous outcomes and evaluates several estimands:

- squared treatment effect, `(Y(0)-Y(1))^2`
- treatment-benefit indicator, `Y(0) <= Y(1)`
- joint threshold indicator, `(Y(0) <= 0) * (Y(1) <= 0)`

For each estimand, it verifies that average `objdiffs` are close to zero.

### `test_optimal_dualvars_consistency`

Checks that fitted optimal dual variables produce unbiased estimates of their own population objective values in a well-specified oracle setting.

The test uses discrete counterfactual distributions with `Y(1) = Y(0) + 1`, evaluates both unrestricted support and a monotonicity support restriction, and loops over dual strategies:

- `ot`
- `lp`
- `qp`
- `se`

It repeatedly resamples outcomes and treatments, recomputes realized dual variables, and verifies that AIPW and IPW estimates average to the stored objective values.

### `test_base_models_cts`

Checks that supported continuous-outcome model specifications can be used by `DualBounds.fit`.

It covers string model types and model objects:

- `ridge`
- `lasso`
- `elasticnet`
- `rf`
- `knn`
- `CtsDistReg`
- `QuantileDistReg`

It also checks heteroskedastic model options such as `none`, `lasso`, and `ols`, and verifies that the fitted internal model classes match the expected parsed model types.

### `test_base_models_discrete`

Checks that supported discrete or binary outcome models can be used by `DualBounds.fit`.

It generates Bernoulli outcome data and tests model specifications including:

- `ridge`
- `lasso`
- `elasticnet`
- `rf`
- `knn`
- `BinaryDistReg`

The test verifies that the fitted internal model class matches the expected discrete model implementation.

### `test_fit_propensity_scores`

Checks propensity-score estimation when `propensities=None`.

It tests propensity model specifications:

- `ridge`
- `lasso`
- `knn`
- `rf`
- `AdaBoostClassifier`

For each model, it fits `DualBounds` and verifies that the stored propensity model fit has the expected class.

### `test_from_pd`

Checks that `DualBounds` accepts pandas inputs without changing data shapes or values.

It compares initialization from pandas `Series` and `DataFrame` objects against initialization from NumPy arrays for outcome, treatment, covariates, and propensities.

It also checks a more mixed pandas case with:

- string-valued treatment labels
- missing values in covariates
- a string-valued discrete covariate column

The mixed case is mainly a no-error fit and summary check.

### `test_generic_model_selection_no_error`

Checks that passing multiple outcome models to `DualBounds` does not raise an error.

It fits a generic bound with a list containing a string model and a `CtsDistReg` object.

### `test_support_restriction_consistency`

Marked as `slow`.

Checks consistency of bounds under a monotonicity support restriction, `Y(0) <= Y(1)`.

The test compares oracle dual variables, fitted using the true counterfactual distributions, against empirical dual variables from a heavily trained ridge distribution model. It repeatedly resamples outcomes and treatments and verifies that empirical and oracle objective values and estimates agree within tolerance.

### `test_generic_dualbound_clustered_ses`

Checks clustered standard-error handling for `DualBounds`.

It wraps a `DualBounds.fit().summary()` workflow in a helper function and passes it to the shared `check_clustered_ses` test utility from `context.DBTest`.

## `TestPluginBounds`

### `test_plug_in_no_covariates`

Checks the no-covariate plug-in bounds estimator under nonconstant propensity scores.

The test compares:

- a naive estimator that does not receive propensities
- the real estimator that does receive propensities
- an oracle estimator that observes both potential outcomes

It verifies that the naive estimator is biased when propensities vary, and that the propensity-adjusted estimator matches the oracle benchmark within tolerance.

### `test_plug_in_ses`

Checks bootstrap standard errors from `plug_in_no_covariates`.

It computes estimated standard errors from one bootstrap run, then approximates oracle standard errors by repeatedly regenerating data and measuring the variation of the plug-in estimates. The test verifies that estimated and empirical standard errors are close within a ratio tolerance.

## Helpers

### `_make_adaboost_classifier`

Builds an `AdaBoostClassifier` in a way that works across scikit-learn versions. Older versions accept `algorithm='SAMME'`; newer versions removed that argument.

### Local helper functions

The file also defines small nested helpers inside individual tests:

- `gen_new_data(reps)` in `test_support_restriction_consistency`
- `gdb_se_function(data, clusters)` in `test_generic_dualbound_clustered_ses`

## Main Behaviors Covered

Overall, the file tests:

- oracle dual-bound objective values
- grid-search objective residuals
- consistency of realized dual variables with objective values
- continuous and discrete outcome model parsing and fitting
- propensity model parsing and fitting
- pandas input compatibility
- multiple-outcome-model selection
- monotonicity support restrictions
- clustered standard errors
- no-covariate plug-in bounds
- bootstrap standard errors for plug-in bounds

