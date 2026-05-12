# High-dimensional discrete VarITE dual bounds

本仓库用于 high-dimensional discrete potential outcome / VarITE dual
bounds 实验。核心目标是研究高维离散潜在结果下 VarITE 的上下界估计。

估计目标为：

```text
VarITE_sq = E[||Y(1) - Y(0)||_2^2] - ||E[Y(1) - Y(0)]||_2^2
```

默认实验中，`Y(0)` 和 `Y(1)` 都是 `dim_y=10` 的二元向量，因此：

```text
Y ∈ {0,1}^10
support size = 2^10 = 1024
```

脚本会生成 synthetic data，估计 conditional outcome PMF，求解 dual
potentials，计算 VarITE lower/upper bounds，并把结果写入 CSV。

## 文件说明

- `dual_bounds_single_ot.py`
  - OT baseline 版本。
  - dual estimator 和 oracle benchmark 都使用 POT 的 exact discrete OT
    求解，即 `ot.emd2`。
  - 适合作为标准 OT dual-bounds 实验基准。

- `dual_bounds_single_se.py`
  - SE-adjusted 版本。
  - dual estimator 使用 CVXPY + CLARABEL 求解 standard-error-adjusted dual
    problem。
  - oracle benchmark 仍然保留 exact OT sharp-bound benchmark，因此可继续
    和 OT sharp bounds 对照。
  - 默认 `dim_y=10` 时计算量明显大于 OT 版本，建议先用小规模参数测试。

- `varite_discrete_experiment_results1.csv`
  - 已保存的实验结果。
  - `n=500`，`dim_y=10`，`support_size=1024`。
  - 包含 `model_type=oracle` 和 `model_type=estimated` 两类结果。

- `varite_discrete_experiment_results2.csv`
  - 已保存的实验结果。
  - `n=100`，`dim_y=10`，`support_size=1024`。
  - 包含 `model_type=oracle` 和 `model_type=estimated` 两类结果。

- `varite_discrete_experiment_results3.csv`
  - 已保存的实验结果。
  - `n=1000`，`dim_y=10`，`support_size=1024`。
  - 包含 `model_type=oracle` 和 `model_type=estimated` 两类结果。

## 运行环境和依赖

从代码导入项可以判断，运行脚本需要：

```text
Python >= 3.10
numpy
scipy
scikit-learn
POT
```

运行 `dual_bounds_single_se.py` 还需要：

```text
cvxpy
clarabel
```

安装示例：

```bash
python -m pip install numpy scipy scikit-learn POT cvxpy clarabel
```

精确版本号待补充。

## 如何运行

两个脚本都支持普通实验运行和内置测试：

```bash
python dual_bounds_single_ot.py test
python dual_bounds_single_se.py test
```

OT 版本 smoke run：

```bash
python dual_bounds_single_ot.py \
  --n-list 20 --reps 1 --model-type oracle \
  --dual-max-units 5 --oracle-max-units 5 \
  --output-csv "" --quiet true
```

SE 版本 smoke run：

```bash
python dual_bounds_single_se.py \
  --n-list 20 --reps 1 --model-type oracle \
  --dual-max-units 5 --oracle-max-units 5 \
  --output-csv "" --quiet true --dim-y 3
```

注意：SE 版本在默认 `dim_y=10`、support size 1024 下会为每个 evaluation
unit 解较大的 CVXPY dual problem，运行时间可能很长。调试时建议先降低
`--dim-y`，或使用 `--dual-max-units`、`--oracle-max-units` 限制计算规模。

保存新实验结果时，建议显式指定新的 CSV 文件名：

```bash
python dual_bounds_single_ot.py \
  --n-list 100 --reps 1 --model-type both \
  --output-csv varite_discrete_experiment_results_new.csv
```

如果只是测试程序，不希望写 CSV，可以使用：

```bash
--output-csv ""
```

## 常用参数

- `--n-list`：样本量列表。
- `--reps`：每个样本量重复实验次数。
- `--dim-y`：二元向量结果的维度，默认 `10`。
- `--model-type`：可选 `oracle`、`estimated`、`both`。
- `--crossfit`：是否使用 cross-fitting，默认 `true`。
- `--n-folds`：cross-fitting 折数，默认 `2`。
- `--dual-max-units`：限制 dual estimator 使用的 evaluation units 数量。
- `--oracle-max-units`：限制 oracle benchmark 使用的 units 数量。
- `--skip-oracle`：是否跳过 oracle benchmark。
- `--output-csv`：结果输出路径；设为 `""` 表示不写文件。
- `--quiet`：是否减少终端输出。

## 推荐使用流程

1. 安装依赖。
2. 先运行 `python dual_bounds_single_ot.py test`，确认 OT baseline 环境正常。
3. 用 `--output-csv ""` 跑一个小规模 OT smoke run。
4. 再运行 `python dual_bounds_single_se.py test`，确认 SE solver 环境正常。
5. 调试 SE 版本时先用小 `--dim-y` 或小 `--dual-max-units`，避免默认
   support size 1024 带来的长时间运行。
6. 正式保存新实验时使用新的 CSV 文件名，不要覆盖已有的
   `varite_discrete_experiment_results1.csv`、
   `varite_discrete_experiment_results2.csv` 和
   `varite_discrete_experiment_results3.csv`。
