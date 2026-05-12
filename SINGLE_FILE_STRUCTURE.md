# `dual_bounds_single.py` Structure

This file is a reproducible experiment for dual bounds on vector-valued
discrete VarITE:

```text
VarITE_sq = E[||Y(1)-Y(0)||_2^2] - ||E[Y(1)-Y(0)]||_2^2
```

The default outcome support is the full binary support `{0,1}^10`, so
`support_size = 1024`.

## How to run

Smoke run:

```bash
/home/raymond/miniconda3/envs/duality/bin/python dual_bounds_single.py \
  --n-list 100 --reps 1 --model-type oracle \
  --dual-max-units 10 --oracle-max-units 10
```

Tests:

```bash
/home/raymond/miniconda3/envs/duality/bin/python dual_bounds_single.py test
```

Default experiment:

```bash
/home/raymond/miniconda3/envs/duality/bin/python dual_bounds_single.py
```

The default uses `n-list = 100 500 1000` with Monte Carlo repetitions
`50, 10, 5`, respectively. Exact OT on a 1024-atom support is still the main
runtime bottleneck, so use the smoke controls below while debugging.

## Sections

1. **Support, Costs, And Numerics**
   - `make_binary_support(dim_y)` builds the full `{0,1}^dim_y` support.
   - `build_vector_cost_matrix(...)` builds the squared Euclidean cost matrix.
   - `stable_softmax(...)` and probability validation guard numerical stability.

2. **DGP**
   - `generate_binary_vector_dgp(...)` generates `X`, `W`, observed `Y`, full
     potential outcomes `Y0/Y1`, atom indices, and true full-support PMFs.
   - Treatment assignment is randomized with known propensity `0.3` by default.

3. **Outcome PMF Estimation**
   - Oracle mode uses `probs0_true/probs1_true`.
   - Estimated mode fits separate multinomial logistic regressions by arm.
   - Missing support atoms are expanded back to 1024 classes and smoothed.

4. **Exact Discrete OT Duals**
   - Uses POT `ot.emd` for exact lower and upper OT bounds.
   - Upper bounds are computed as `-min_gamma <gamma, -C>`.
   - Dual potentials are used to build IPW/AIPW summands.

5. **Dual-Bound Estimator**
   - Supports simple split and 2-fold cross-fitting.
   - Computes first-term lower/upper estimates, vector ATE estimates, VarITE
     estimates, delta-method SEs, and confidence bounds.
   - Does not truncate negative finite-sample estimates.

6. **Experiment Loop**
   - Runs Monte Carlo repetitions over `--n-list`.
   - Records oracle benchmark fields, solver diagnostics, model diagnostics,
     runtime, coverage, and anticonservative flags.
   - Coverage and anticonservative flags are populated only when both oracle
     and dual computations are full-sample; smoke/subsampled runs leave them
     as `NaN`.
   - Writes results to `varite_discrete_experiment_results.csv` by default.

## Useful Runtime Controls

- `--oracle-max-units`: compute oracle sharp bounds on only the first units.
- `--dual-max-units`: evaluate the dual estimator on a random subset of units.
- `--skip-oracle true`: skip oracle benchmark fields.
- `--model-type oracle|estimated|both`: choose outcome model mode.
- `--crossfit true|false`: choose cross-fit or simple split.
- `--smoothing-epsilon`: smoothing for missing multinomial classes.
- `--progress-every`: print OT progress every K evaluated units.
- `--ot-max-iter`: pass a maximum iteration count to POT `emd2`.
- `--checkpoint-every-rep`: rewrite the CSV after each completed repetition.
