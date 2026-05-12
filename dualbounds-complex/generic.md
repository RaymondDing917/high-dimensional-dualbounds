Code Inventory

Test functions in test_generic.py (line 38)

TestGenericDualBounds.test_var_ite_oracle

## test/test_generic.py :: test_var_ite_oracle

### Test covered
- `test_var_ite_oracle`: verifies that `DualBounds` computes correct lower and upper bounds for `Var(ITE) = Var(Y(1)-Y(0))` when the marginal distributions of `Y(0)` and `Y(1)` are known.

### Mathematical/statistical object validated
- The test sets `f(y0, y1, x) = (y0-y1)^2`, so `DualBounds` computes bounds on `E[(Y(0)-Y(1))^2]`.
- It then subtracts `(E[Y(1)]-E[Y(0)])^2` to obtain bounds on `Var(Y(1)-Y(0))`.
- The analytical benchmark is computed using one-dimensional monotone and countermonotone couplings:
  - lower bound: `Y1 = F1^{-1}(U)`, `Y0 = F0^{-1}(U)`;
  - upper bound: `Y1 = F1^{-1}(U)`, `Y0 = F0^{-1}(1-U)`.

the data storage of `discrete distribution`
 for each observation i
 vals[i]  = [0, 1] #support 
probs[i] = [1 - p_i, p_i]  #probability distribution 

`**it's a problem we need to redesign for vector Y**`



TestGenericDualBounds.test_gridsearch_loss

TestGenericDualBounds.test_optimal_dualvars_consistency

TestGenericDualBounds.test_base_models_cts

TestGenericDualBounds.test_base_models_discrete

TestGenericDualBounds.test_fit_propensity_scores

TestGenericDualBounds.test_from_pd

TestGenericDualBounds.test_generic_model_selection_no_error

TestGenericDualBounds.test_support_restriction_consistency

TestGenericDualBounds.test_generic_dualbound_clustered_ses

TestPluginBounds.test_plug_in_no_covariates

TestPluginBounds.test_plug_in_ses



Helper functions or fixtures

_make_adaboost_classifier()

Nested helper in test_support_restriction_consistency: gen_new_data(reps)

Nested helper in test_generic_dualbound_clustered_ses: gdb_se_function(data, clusters)

No pytest fixtures are defined in this file.



Imported project-level classes/functions

context

context.dualbounds as db

dualbounds.generic

dualbounds.utilities

dualbounds.gen_data

dualbounds.utilities.parse_dist

dualbounds.utilities.BatchedCategorical

Used through db or module aliases:

db.generic.DualBounds

generic.DualBounds

db.generic.plug_in_no_covariates

generic.plug_in_no_covariates

db.gen_data.gen_regression_data

gen_data.gen_regression_data

db.dist_reg.parse_model_type

db.dist_reg.CtsDistReg

db.dist_reg.QuantileDistReg

db.dist_reg.BinaryDistReg






Methods called on DualBounds-like objects

compute_dual_variables(...)

fit(...)

summary()

_compute_realized_dual_variables(...)

_compute_final_bounds(...)



Assert statements and checked variables

test_var_ite_oracle: checks vdb.objvals[:, 0] - mu2 equals analytical [lower, upper].

test_gridsearch_loss: checks gdb.objdiffs.mean(axis=1) is approximately [0, 0].

test_optimal_dualvars_consistency: checks estimates.mean(axis=0) equals gdb.objvals.mean(axis=1) for AIPW and IPW estimates.

test_base_models_cts: checks Ym = gdb.model_fits[0].model has expected model class.

test_base_models_cts: checks Ym2 = gdb.model_fits[0].sigma2_model has expected heteroskedastic model class.

test_base_models_cts: checks accessing gdb.model_fits[0].sigma2_model raises AttributeError when no sigma model is expected.

test_base_models_discrete: checks Ym = gdb.model_fits[0].model has expected classifier/discrete model class.

test_fit_propensity_scores: checks Wm = gdb.propensity_model_fits[0] has expected propensity model class.

test_from_pd: checks gdb1.X.shape == gdb2.X.shape.

test_from_pd: checks pandas-initialized gdb1.y, gdb1.W, gdb1.pis equal array-initialized gdb2.y, gdb2.W, gdb2.pis.

test_support_restriction_consistency: checks empirical dual objective values/estimates e_objs, e_ests are close to oracle values/estimates o_objs, o_ests.

test_plug_in_no_covariates: checks naive plug-in means naive_mu differ materially from oracle means oracle_mu when propensity scores are nonconstant.

test_plug_in_no_covariates: checks propensity-adjusted plug-in means est_mu match oracle means oracle_mu.

test_plug_in_ses: checks estimated bootstrap SEs hatses are within a factor of 1.2 of empirical repeated-sampling SEs ses.




Statistical Meaning Of The Printed Results

Most printed blocks are from .summary(). Each block reports bounds on a causal estimand of the form E[f(Y(0), Y(1), X)], where the joint distribution of the two potential outcomes is only partially identified.


Inference: the final lower and upper bound estimates, their standard errors, and confidence intervals. These are the robust/AIPW-style dual-bound estimates.

Outcome model: how well the fitted conditional outcome distribution predicts observed outcomes out of sample. For binary outcomes you see accuracy and likelihood; for continuous outcomes you see RMSE and MAE.

Treatment model: diagnostics for estimated propensity scores when they are learned, or for the supplied propensity score model.

Nonrobust plug-in bounds: direct model-based bounds without the full robustness correction. These are often narrower and lower-variance, but depend more heavily on correct outcome modeling.

Technical diagnostics: numerical optimization and stability diagnostics for the dual formulation.


The first repeated binary-outcome summaries correspond to test_base_models_discrete. Statistically, they show that several discrete outcome model choices produce nearly the same bound for P(Y(0) <= Y(1)): lower around 0.959, upper exactly 1. The exact upper bound of 1 means the fitted marginals allow a coupling where monotone improvement is always possible. The warning and NaN upper leverage happen because the upper dual problem is degenerate at the boundary.

The later binary summaries with upper bounds above 1, such as 1.005933 or 1.018000, are finite-sample inference artifacts: the point estimate and confidence interval are not clipped to the logical parameter range. Statistically, they still indicate the upper bound is essentially at the maximum possible value, while the lower bound varies with model flexibility.

The continuous-outcome summaries under “Fitting propensity scores” correspond to test_fit_propensity_scores. They estimate bounds for an event like Y(0) <= Y(1) using learned propensity models. The outcome model has strong predictive performance, with R^2 ≈ 0.735, but treatment models have weak or negative out-of-sample R^2, meaning treatment assignment is hard to predict from covariates. One bad propensity fit gives a very unstable lower estimate, around 0.303 with SE 0.470, illustrating how poor propensity estimates can inflate variance.

The single summary with equal lower and upper estimates around 0.570075 corresponds to a point-identified estimand, likely an average treatment-effect-style functional Y(1)-Y(0). Equal lower and upper bounds mean no partial-identification ambiguity remains for that functional.

The support-restriction summaries test monotonicity-style restrictions, e.g. Y(0) <= Y(1). The oracle and empirical summaries have wide intervals because n=11 is tiny, but the test checks that when the learned counterfactual distributions converge to the oracle distributions, the resulting dual objectives and estimates agree.

The final continuous summaries with lower around 2.83 and upper around 3.07 are clustered-SE diagnostics for E[|Y(1)-Y(0)|]. Statistically, they show a partially identified average absolute individual treatment effect: the marginal outcome models pin it down to an interval, and clustered standard errors quantify uncertainty when observations may be grouped.


# About the generic.py
## def _solve_single_instance

'ot': uses ot.lp.emd2(...) from Python Optimal Transport. Fastest. Solves the ordinary sharp lower/upper OT bound.
'lp': uses CVXPY to solve the same dual linear program explicitly. Slower, more general/debuggable.
'qp': solves a modified quadratic program. It still enforces the dual constraints, but penalizes unstable dual variables approximately.
'se': solves a modified convex program that maximizes something like “bound estimate minus 2 standard errors.” It is designed to improve finite-sample confidence bounds, often at the cost of less sharp point bounds.

This only happens after attempting 'ot' with support restrictions / numerical instability. If OT returns extremely large dual variables, the code falls back to 'se'

## def compute_dual_variables
# set parameter values
			fvals = self._apply_ot_fn(
				fn=self.f,
				y0=self.y0_vals[i],
				y1=self.y1_vals[i],
				x=self.X[i],
			)
# calculate
    maximize    nu0 @ probs0 + nu1 @ probs1
subject to  nu0[j] + nu1[k] <= fvals[j,k]

## def _compute_realized_dual_variables
