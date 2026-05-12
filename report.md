# VarITE 离散向量实验结果对比报告

本文对比两组实验结果：

- 第一组：`varite_discrete_experiment_results1.csv`
- 第二组：`varite_discrete_experiment_results.csv`

两组实验使用同一个程序 `dual_bounds_single.py`，估计目标为

```text
VarITE_sq = E[||Y(1) - Y(0)||_2^2] - ||E[Y(1) - Y(0)]||_2^2
```

## 实验设计

DGP 中 `Y(0)` 和 `Y(1)` 都是 `dim_y` 维二元向量。两组 CSV 中 `dim_y=10`，因此

```text
Y ∈ {0,1}^10
support size = 2^10 = 1024
```

如果看 `(Y(0), Y(1))` 的 OT coupling 候选配对，则规模是

```text
1024 x 1024 = 1,048,576
```

其他主要设定在两组实验中保持一致：

| 参数 | 数值 |
|---|---:|
| `dim_y` | 10 |
| `support_size` | 1024 |
| `p_x` | 50 |
| `propensity` | 0.3 |
| `alpha_scale` | 0.5 |
| `beta_scale` | 0.3 |
| `crossfit` | True |
| `n_folds` | 2 |
| `aipw` | False |
| `smoothing_epsilon` | 1e-6 |
| `dual_computation_mode` | full |
| `oracle_computation_mode` | full |

`--progress-every 10` 只改变 OT 求解过程中的终端进度输出频率，不改变 DGP、样本划分、PMF 估计、OT 求解或 VarITE bounds。它最多会给运行时间带来很小的日志输出开销。

## 两组实验规模

| 组别 | CSV | rows | reps | n | n_eval | oracle_n_units |
|---|---|---:|---:|---:|---:|---:|
| 第一组 | `varite_discrete_experiment_results1.csv` | 50 | 25 | 500 | 500 | 500 |
| 第二组 | `varite_discrete_experiment_results.csv` | 100 | 50 | 100 | 100 | 100 |

第二组相对第一组的核心变化是样本量从 `n=500` 降为 `n=100`，重复次数从 25 增为 50。support 仍是 1024，没有变化。

## 主要结果

| 组别 | model | dual lower | dual upper | mean CI | oracle bounds | lower cover | upper cover | any anti |
|---|---|---:|---:|---|---|---:|---:|---:|
| 第一组 n=500 | oracle | 0.299 | 9.552 | [0.100, 9.959] | [0.361, 9.634] | 1.00 | 0.96 | 0.04 |
| 第一组 n=500 | estimated | -0.012 | 9.877 | [-0.235, 10.348] | [0.361, 9.634] | 1.00 | 1.00 | 0.00 |
| 第二组 n=100 | oracle | 0.118 | 9.295 | [-0.475, 10.257] | [0.365, 9.631] | 1.00 | 0.96 | 0.04 |
| 第二组 n=100 | estimated | -0.163 | 9.575 | [-0.847, 10.841] | [0.365, 9.631] | 1.00 | 0.98 | 0.02 |

这里的 `oracle bounds` 是在真实 conditional PMF 下计算的 sharp benchmark。`model=oracle` 的 `dual lower/upper` 不是 benchmark 本身，而是使用真实 PMF 的有限样本估计量，所以仍然有抽样误差。

## Support 稀疏性

| 组别 | observed arm0 | arm0 占理论 support | observed arm1 | arm1 占理论 support |
|---|---:|---:|---:|---:|
| 第一组 n=500 | 157.2 | 15.35% | 71.3 | 6.96% |
| 第二组 n=100 | 34.7 | 3.39% | 14.6 | 1.42% |

第二组的理论 support 仍是 1024，但实际训练中每个 treatment arm 能看到的 Y 原子大幅减少。尤其 treated arm 因为 `propensity=0.3`，每折训练样本更少，平均只观察到约 14.6 个不同原子。这说明第二组的 estimated multinomial PMF 面临更严重的离散类别稀疏问题。

## 估计误差与标准误

| 组别 | model | lower SE | upper SE | lower bias | upper bias | CI width |
|---|---|---:|---:|---:|---:|---:|
| 第一组 n=500 | oracle | 0.101 | 0.208 | -0.063 | -0.082 | 9.860 |
| 第一组 n=500 | estimated | 0.114 | 0.241 | -0.374 | 0.243 | 10.583 |
| 第二组 n=100 | oracle | 0.303 | 0.491 | -0.247 | -0.337 | 10.732 |
| 第二组 n=100 | estimated | 0.349 | 0.646 | -0.528 | -0.057 | 11.688 |

`bias` 是 `dual estimate - oracle benchmark` 的平均值。

从 `n=500` 降到 `n=100` 后：

1. `oracle` 估计的标准误显著上升。lower SE 从 0.101 增至 0.303，upper SE 从 0.208 增至 0.491。
2. `estimated` 估计的标准误也显著上升。lower SE 从 0.114 增至 0.349，upper SE 从 0.241 增至 0.646。
3. lower bound 估计整体更保守。第二组 estimated lower 平均为 -0.163，而 oracle benchmark lower 为 0.365，平均偏差为 -0.528。
4. 第二组的 CI 更宽，estimated 的平均 CI 宽度从 10.583 增至 11.688。

这些变化符合有限样本直觉：样本量越小，观测到的 Y support 越稀疏，IPW 型 summands 波动越大，delta-method 标准误也更大。

## Coverage 表现

第一组：

- `oracle`：lower coverage 100%，upper coverage 96%，有 1 次 upper bound 未覆盖。
- `estimated`：lower coverage 100%，upper coverage 100%，没有 anti-conservative case。

第二组：

- `oracle`：lower coverage 100%，upper coverage 96%，有 2 次 upper bound 未覆盖。
- `estimated`：lower coverage 100%，upper coverage 98%，有 1 次 upper bound 未覆盖。

第二组的 coverage 仍然总体较好，但因为 `n=100` 带来更大的有限样本波动，upper bound 出现了少量未覆盖。estimated 模式通常更保守，但第二组中仍有 1 次 upper CI 没有覆盖 oracle upper benchmark。

## Oracle benchmark 的稳定性

两组 oracle benchmark 非常接近：

| 组别 | oracle lower mean | oracle upper mean | true ATE norm sq |
|---|---:|---:|---:|
| 第一组 n=500 | 0.361 | 9.634 | 0.00134 |
| 第二组 n=100 | 0.365 | 9.631 | 0.00146 |

这说明两组实验的 DGP 本质相同，sharp bounds 本身没有发生结构性变化。第二组结果变差主要来自样本量减少，而不是 support 或 DGP 改变。

## 运行时间

| 组别 | model | OT solves | cache hits | runtime mean | runtime median |
|---|---|---:|---:|---:|---:|
| 第一组 n=500 | oracle | 0 | 500 | 0.047s | 0.044s |
| 第一组 n=500 | estimated | 500 | 0 | 323.5s | 201.4s |
| 第二组 n=100 | oracle | 0 | 100 | 0.013s | 0.013s |
| 第二组 n=100 | estimated | 100 | 0 | 83.0s | 39.6s |

`model=oracle` 行运行很快，是因为 oracle benchmark 已经先计算过，后续 estimator 命中 OT cache。`estimated` 行不能复用 oracle PMF 下的 OT 解，因此需要重新解 OT。第二组每次只评估 100 个 unit，所以 OT solves 从 500 降到 100，运行时间明显下降。

第二组 estimated 的平均运行时间高于其中位数，说明有少数重复运行时间异常长。CSV 中第二组 estimated 最大运行时间约为 1956.7 秒，主要由 OT 求解时间贡献。

## 结论

第二组实验没有改变 Y 的 support，仍然是 `{0,1}^10` 共 1024 个原子；真正改变的是样本量从 500 降到 100。这个变化导致实际观测到的 support 更稀疏、标准误显著增大、置信区间变宽，并使 upper bound 出现少量未覆盖。

从结果看，oracle benchmark 两组非常接近，说明 DGP 和 sharp bounds 稳定。第二组的主要问题不是识别边界本身，而是有限样本下对高维离散 support 的估计和 IPW 估计波动更大。
