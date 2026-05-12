# 当前程序与实验问题说明

本文针对 `dual_bounds_single.py` 和当前离散向量 VarITE 实验回答四个问题。

## 1. Oracle benchmark、oracle estimator、estimated estimator 的区别

你的理解大体正确，但需要把 `benchmark` 和 `model_type=oracle` 区分开。

当前目标是

```text
VarITE_sq = E[||Y(1) - Y(0)||_2^2] - ||E[Y(1) - Y(0)]||_2^2
```

因为只知道或估计条件边际分布 `P(Y(0)|X)` 和 `P(Y(1)|X)`，而不知道二者的联合分布，所以第一项

```text
E[||Y(1) - Y(0)||_2^2]
```

通过每个 `X_i` 下的 optimal transport 问题给出 sharp lower/upper bounds。

程序里有三层概念：

| 名称 | 程序位置 | 含义 |
|---|---|---|
| Oracle benchmark | `compute_oracle_sharp_bounds(...)` | 用 DGP 生成的真实 `probs0_true/probs1_true`，在全局 support `{0,1}^dim_y` 上逐个 `X_i` 解 OT，然后平均得到 sharp benchmark。CSV 中是 `oracle_lower_VarITE` 和 `oracle_upper_VarITE`。 |
| `model_type="oracle"` estimator | `estimate_varite_dual_bounds(..., model_type="oracle")` | estimator 不估计 nuisance，而是直接拿真实 `P(Y(w)|X)` 来构造 dual variables 和 IPW/AIPW summands。它仍然是有限样本估计量，会有抽样误差，因此不等于 oracle benchmark 本身。 |
| `model_type="estimated"` estimator | `estimate_varite_dual_bounds(..., model_type="estimated")` | 先在每个 treatment arm 内用 multinomial logistic regression 估计 `P(Y|X,W=w)`，再用估计出的 PMF 解 OT 和构造 dual-bound estimator。 |

所以更精确地说：

1. `benchmark` 是“真实条件边际分布下的 sharp bounds”，在当前程序里作为比较基准。
2. `oracle estimator` 是“估计器拿到了真实 `P(Y(w)|X)` 的版本”，用于分离 nuisance 估计误差和有限样本/IPW 波动。
3. `estimated estimator` 是现实可行版本：先估计 `P(Y(w)|X)`，再估计 dual bounds。

还有一个细节：当前 benchmark 是对本次 Monte Carlo 样本中的 `X_i` 做平均。如果把目标定义为无限总体里的 population bound，它是该 population bound 的 Monte Carlo 近似；如果把目标定义为 sample-X 条件下的 sharp bound，它就是对应的 oracle benchmark。

## 2. 只基于已观测 1.42% support 做 LP 的后果

严格说，理论 support 不是 1.42%。当前默认 `dim_y=10`，理论 support 是

```text
{0,1}^10, support size = 1024
```

`1.42%` 来自 `n=100` 实验中 treated arm 的平均 observed unique atoms：

```text
14.56 / 1024 = 1.421875%
```

control arm 的平均 observed support 比例约为

```text
34.68 / 1024 = 3.38671875%
```

如果只在这部分已观测 support 上解 LP，而不是在 1024 个全局原子上解 LP，会有以下后果。

### 2.1 计算会快很多，但目标变了

全局 LP 的 coupling 规模是

```text
1024 x 1024 = 1,048,576
```

如果粗略用 `n=100` 下的平均 observed support，则 LP 规模大约变成

```text
34.68 x 14.56 ≈ 505
```

这会显著加速 OT/LP，但它不再是原始问题的 sharp bound。原因是 DGP 的真实条件 PMF 是 full-support softmax，理论上每个 atom 都有正概率。样本中没观测到某个 atom，并不表示该 atom 的真实概率为 0。

### 2.2 会把“未观测”误当成“不可能”

只用 observed support 等价于对问题加入一个很强的额外假设：

```text
P(Y(w) = y | X) = 0 for all unobserved y
```

这个假设在当前 DGP 下是错的。当前 `probs0_true/probs1_true` 是 softmax，因此所有 1024 个 atom 都有正概率。

如果把概率重新归一化到 observed atoms 上，解出来的是“截断分布”的 OT bound，不是原始分布的 OT bound。这个 bound 不能再和 `oracle_lower_VarITE/oracle_upper_VarITE` 做同一目标下的 coverage 比较。

### 2.3 bounds 不一定只是更宽或更窄，而是可能偏向任意方向

restricted-support LP 不是全局 LP 的简单放松或收紧，因为边际分布本身也被改了。删除未观测 atoms 后，质量会被隐含地移到 observed atoms 上；lower bound 和 upper bound 都可能上移或下移。

因此后果不是“保守一点”这么简单，而是：

```text
得到的是另一个 estimand 的 bounds，可能对原始 full-support estimand 失去有效覆盖。
```

### 2.4 对 cross-fitting estimator 还会造成实现问题

当前 estimator 在 held-out fold 上评估 dual potentials。若 LP 只在 training fold 里观测到的 atoms 上求解，那么 held-out unit 的实际 `Y` atom 可能不在 training observed support 内。此时 dual potential `u[atom]` 或 `v[atom]` 没有定义。

当前程序避免了这个问题：即使某些类别没有在训练中出现，`fit_predict_multinomial_pmf(...)` 也会把预测概率扩展回完整的 1024 类，并加上 `smoothing_epsilon`。这样 OT dual potentials 总是在全局 support 上定义。

结论：只用 observed support 可以作为一个“截断 support 的快速敏感性实验”，但不能把它当作当前理论目标的 oracle benchmark 或有效 inference 版本。若要正式使用，需要在论文中明确改成 truncated-support estimand，并重新证明相应性质。

## 3. 当前 OT 中的 LP 是怎么解的

当前程序没有自己手写 LP simplex 或 interior-point 求解器，而是调用 POT 包，也就是 Python Optimal Transport。

程序开头导入：

```python
import ot
```

核心函数是 `solve_ot_bound_pair(...)`。对每个 evaluation unit，它拿到两组边际概率：

```text
probs0 = P(Y(0) = y_a | X_i)
probs1 = P(Y(1) = y_b | X_i)
```

以及 cost matrix：

```text
C[a,b] = ||y_b - y_a||_2^2
```

lower first-term bound 解的是 primal OT：

```text
min_gamma sum_{a,b} gamma[a,b] C[a,b]
s.t.      sum_b gamma[a,b] = probs0[a]
          sum_a gamma[a,b] = probs1[b]
          gamma[a,b] >= 0
```

代码调用：

```python
lower_cost, lower_log = ot.emd2(
    probs0,
    probs1,
    cost,
    log=True,
    return_matrix=False,
    numItermax=num_iter_max,
    numThreads=num_threads,
    check_marginals=False,
)
```

upper first-term bound 用

```text
max_gamma <gamma, C> = - min_gamma <gamma, -C>
```

所以代码对 `-cost` 再调用一次 `ot.emd2(...)`，然后取负号：

```python
upper_neg_cost, upper_log = ot.emd2(probs0, probs1, -cost, ...)
upper_value = -upper_neg_cost
```

`log=True` 会返回 dual potentials。对 lower bound，POT 返回的 `u, v` 满足

```text
u[a] + v[b] <= C[a,b]
```

对 upper bound，程序把 `-cost` 问题返回的 potentials 取负，使其变成满足

```text
phi[a] + psi[b] >= C[a,b]
```

这些 dual potentials 后续用于构造 IPW/AIPW summands。例如 observed unit 来自 arm 0 时，用 `lower_u0[atom] / (1 - pi)`；来自 arm 1 时，用 `lower_u1[atom] / pi`。

因此，当前代码自己做的是：

1. 构造全局 support 和 cost matrix。
2. 为每个 `X_i` 准备 `probs0/probs1`。
3. 调 POT 的 exact discrete OT solver `ot.emd2`。
4. 从 POT 返回值中读取 objective 和 dual potentials。
5. 用 dual potentials 构造 dual-bound estimator。
6. 用 cache 避免重复解相同 PMF/cost 的 OT 问题。

真正的 LP/OT 数值求解过程由 POT 完成。

## 4. cross-fitting 的作用

这里我把“overlit 交叉拟合”理解为“用于处理 overfitting 的 cross-fitting”。

当前程序里 `crossfit=True` 时，`_split_indices(...)` 会把样本随机分成 `n_folds` 份。默认 `n_folds=2`：

1. 第 1 折：用第 2 折训练 nuisance PMF，在第 1 折上评估 dual summands。
2. 第 2 折：用第 1 折训练 nuisance PMF，在第 2 折上评估 dual summands。
3. 最后把两折 evaluation summands 合并。

主要作用有三个。

### 4.1 降低 nuisance overfitting 对 moment estimator 的影响

`model_type="estimated"` 时，同一批数据如果既用于估计 `P(Y(w)|X)`，又用于评估 dual/IPW moment，容易产生 in-sample overfitting bias。Cross-fitting 让每个 observation 的 summand 都使用不含该 observation 的模型预测，近似 out-of-sample 评估。

这对灵活 nuisance 模型尤其重要，也是 semiparametric / double machine learning 中常用的做法。

### 4.2 提高样本利用率

如果 `crossfit=False`，程序只做一次 train/eval split。默认 `split_frac=0.5`，所以 `n=100` 时只有约 50 个 observations 用于最终 moment evaluation。

如果 `crossfit=True` 且不设置 `dual_max_units`，每个 observation 都会在某一折中作为 held-out evaluation unit 出现一次。因此 `n=100` 时 `n_eval=100`。

### 4.3 代价是每折训练 support 更稀疏

Cross-fitting 不是免费午餐。默认 2-fold 下，每个 nuisance model 只用另一半样本训练。对于 `n=100`、`propensity=0.3`，每折训练中 treated observations 大约只有 15 个左右，因此 observed treated support 非常稀疏，这正是 1.42% 的来源。

所以 cross-fitting 在当前实验中同时带来：

```text
更干净的 out-of-fold moment evaluation
更完整的 n_eval
更稀疏的每折 nuisance training support
```

`model_type="oracle"` 时，nuisance PMF 已知，不存在 nuisance overfitting；此时 cross-fitting 主要只是保持 estimator 流程一致，并决定哪些 units 被 evaluation。`model_type="estimated"` 时，cross-fitting 才是关键。

## 总结

1. 你的 Oracle/estimate 理解基本正确，但要区分 `oracle benchmark` 和 `model_type="oracle"` estimator。前者是真实 PMF 下的 sharp benchmark，后者是拿到真实 PMF 的有限样本 estimator。
2. `n=100` 下的 1.42% 是 treated arm observed support 占理论 1024 support 的比例，不是理论 support 本身。
3. 只用 observed support 解 LP 会快很多，但会改变 estimand，通常不能再作为原始 full-support causal bounds 的有效估计。
4. 当前 OT/LP 数值求解调用 POT 的 `ot.emd2`，程序本身负责封装 lower/upper、dual potentials、cache 和 estimator。
5. Cross-fitting 的核心作用是 out-of-fold nuisance prediction，减少 overfitting bias，并让所有 observations 都能参与 evaluation；但在小样本高维离散 support 下会让每折训练 support 更稀疏。
