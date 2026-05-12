# `dualbounds/gen_data.py` 说明

这个文件用于为测试、示例和方法验证生成合成数据。代码围绕因果推断和统计学习中的几类典型数据生成过程（data-generating process, DGP）展开：标准处理效应回归数据、带选择偏差的 Lee bounds 数据、以及工具变量（instrumental variables, IV）数据。

## 重要函数总览

| 函数 | 类型 | 作用 |
| --- | --- | --- |
| `heteroskedastic_scale(X, heterosked='constant')` | 辅助函数 | 根据协变量 `X` 生成异方差噪声尺度，并标准化使边际误差方差约为 1。 |
| `create_cov(p, covmethod='identity')` | 辅助函数 | 生成协变量的协方差矩阵。目前只支持单位矩阵。 |
| `_sample_norm_vector(dim, norm, sparsity=0)` | 辅助函数 | 随机生成指定范数和稀疏度的系数向量。用于构造回归系数、倾向得分系数等 DGP 参数。 |
| `gen_regression_data(...)` | 主要数据生成器 | 生成二元处理变量 `W`、协变量 `X`、观测结果 `y` 和潜在结果分布 `Y(0), Y(1)`。这是最基础的处理效应模拟数据。 |
| `gen_lee_bound_data(...)` | 主要数据生成器 | 在 `gen_regression_data` 的基础上加入二元选择变量 `S`，用于模拟样本选择、截尾或可观测性问题。 |
| `gen_iv_data(...)` | 主要数据生成器 | 生成工具变量 `Z`、内生处理变量 `W`、结果变量 `y` 和相关潜在结构，用于模拟 IV/LATE 场景。 |

### `heteroskedastic_scale`

该函数生成误差项的条件标准差尺度：

- `constant` 或 `none`：同方差，所有观测的误差尺度相同。
- `linear`：误差尺度与第一个协变量的绝对值相关。
- `norm`：误差尺度与协变量平方和相关，高范数样本噪声更大。
- `invnorm`：误差尺度与协变量平方和成反比。
- `exp_linear`：误差尺度按协变量的指数函数变化。

统计意义：它用于模拟异方差性，即 `Var(Y | X)` 随 `X` 改变。很多真实数据中，高风险、高收入或极端特征个体的结果波动更大，异方差模拟可以检验方法在非理想噪声结构下的稳健性。



从“研究设计”角度看，文件生成 3 种主要数据：

1. **标准处理效应回归数据**：由 `gen_regression_data` 生成。
2. **带选择偏差的数据**：由 `gen_lee_bound_data` 生成。
3. **工具变量数据**：由 `gen_iv_data` 生成。

从“变量类型”角度看，数据同时包含连续变量和离散变量：

| 变量 | 出现在哪些数据中 | 类型 | 统计含义 |
| --- | --- | --- | --- |
| `X` | 全部主要生成器 | 通常连续 | 个体协变量或特征。默认来自多元正态或椭圆分布，用于产生混杂、异质性和预测信息。 |
| `W` | 全部主要生成器 | 离散二元，0/1 | 处理、干预或暴露状态。 |
| `Z` | `gen_iv_data` | 离散二元，0/1 | 工具变量，影响处理 `W`，但在标准 IV 假设下不直接影响结果。 |
| `S` | `gen_lee_bound_data` | 离散二元，0/1 | 选择、入样、存活、响应或结果可观测指标。 |
| `y` | 全部主要生成器 | 取决于 `eps_dist` | 观测结果变量。默认连续；当 `eps_dist='bernoulli'` 时为二元离散结果。 |
| `Y0`, `Y1` | 函数内部生成 | 取决于 `eps_dist` | 潜在结果，分别表示未处理和处理状态下的结果。返回的是其分布对象和相关均值差。 |
| `pis` | `gen_regression_data`, `gen_lee_bound_data`, `gen_iv_data` | 连续概率 | 倾向得分或工具变量概率，即条件概率。 |
| `cates` | `gen_regression_data`, `gen_iv_data` | 连续数值 | 条件平均处理效应，表示 `E[Y(1)-Y(0) | X]` 或相应分布均值差。 |

### `y` 的连续与离散

`y` 的类型由 `eps_dist` 决定。`gen_data.py` 通过 `utilities.parse_dist` 解析分布：

- 连续结果：`gaussian`/`normal`、`t`/`tdist`、`cauchy`、`laplace`、`uniform`/`unif`、`expon`/`exponential`、`gamma`、`skewnorm`、`powerlaw`、`invchi2` 等。
- 离散结果：`bernoulli` 会生成二元结果。
- 退化结果：`constant` 会生成确定性结果，可视为无噪声或退化分布。

默认参数 `eps_dist='gaussian'`，因此默认的 `y` 是连续变量。

### at utilities 546 def parse_dist






`X` 默认是连续协变量。`lmda_dist` 控制椭圆分布中的随机半径或尺度，形式是 `X_i = lambda_i * N(0, Sigma)`。默认 `lmda_dist='constant'` 时近似多元正态；如果使用重尾或偏态的 `lambda_i`，则可以生成更复杂的协变量分布。

## 辅助函数的统计意义



### `create_cov`

目前只支持 `identity`，即协变量各维度独立且方差相同。统计上这对应最基础的标准化协变量设计。函数保留了扩展接口，未来可加入相关协变量结构。

### `_sample_norm_vector`

该函数生成回归系数向量，并控制：

- `norm`：信号强度。
- `sparsity`：稀疏程度，即有多少协变量真实系数为 0。

统计意义：模拟高维协变量，且信息分布在低维

## `gen_regression_data`

这是基础的处理效应数据生成器，主要生成：

- `X`：协变量。
- `W`：二元处理变量。
- `y`：观测结果。
- `pis`：真实倾向得分 `P(W=1 | X)`。
- `y0_dists`, `y1_dists`：潜在结果 `Y(0)` 和 `Y(1)` 的条件分布。
- `beta`, `beta_int`, `betaW`：结果模型、处理效应异质性和处理分配模型的真实系数。
- `cates`：条件平均处理效应。

其核心结构为：

```text
X_i = lambda_i * N(0, Sigma)
P(W_i = 1 | X_i) = logistic(X_i betaW)
Y_i(0) = X_i beta + epsilon_i(0)
Y_i(1) = X_i beta + tau + X_i beta_int + epsilon_i(1)
Y_i = W_i Y_i(1) + (1-W_i) Y_i(0)
```

统计意义：
无treatment X如何影响Y
mu = X @ beta    ##$\mu(X_i) = X_i^\top \beta.$  
sigmas = heteroskedastic_scale(X, heterosked=heterosked)

有treatment x如何影响y
cates = X @ beta_int + tau
	y1_dists = parse_dist(
		eps_dist, loc=mu+cates, scale=sigmas1
	)




## `gen_lee_bound_data`

该函数先调用 `gen_regression_data`，然后额外生成选择变量 `S`：

```text
P(S_i(0)=1 | X_i) = logistic(X_i betaS)
P(S_i(1)=1 | X_i) = logistic(X_i betaS + stau)
S_i = W_i S_i(1) + (1-W_i) S_i(0)
```

返回数据在基础回归数据之外增加：

- `s0_probs`：未处理状态下被选择的概率。
- `s1_probs`：处理状态下被选择的概率。
- `S`：实际观测到的选择指标。

统计意义：

- `S` 可表示就业、存活、响应问卷、结果是否可观测等二元选择过程。
- 当处理影响 `S` 时，简单比较观测到的 `y` 可能产生选择偏差。
- `stau` 是处理对选择概率的影响强度。若 `stau > 0`，处理提高选择概率；若 `stau < 0`，处理降低选择概率。
- 这类数据适合研究 Lee bounds：当只在 `S=1` 的样本中观察结果时，对处理效应进行部分识别和边界估计。

## `gen_iv_data`

该函数生成工具变量设计下的数据：

- `X`：协变量。
- `Z`：二元工具变量。
- `W`：由 `Z` 和 `X` 决定的二元处理变量。
- `y`：观测结果。
- `pis`：`P(Z=1 | X)`。
- `wprobs`：`P(W=1 | X, Z=0)` 和 `P(W=1 | X, Z=1)`。
- `ydists`：按工具变量和处理状态组织的潜在结果分布。
- `betaZ`, `betaW`：工具变量分配和处理分配的真实系数。

其核心结构为：

```text
P(Z_i = 1 | X_i) = logistic(X_i betaZ)
P(W_i = 1 | X_i, Z_i) = logistic(X_i betaW + tauZ * Z_i)
Y_i(0) = X_i beta + tau_conf * sign(U_i - 1/2) + epsilon_i(0)
Y_i(1) = Y_i(0) + tau + X_i beta_int + epsilon_i(1)
Y_i = W_i Y_i(1) + (1-W_i) Y_i(0)
```

统计意义：

- `Z` 是工具变量，目标是制造或模拟外生的处理变异。
- `tauZ` 控制工具变量对处理的影响强度，也就是第一阶段强度。`tauZ` 越大，工具变量越强。
- `betaZ_norm` 控制工具变量分配对协变量的依赖程度。
- `betaW_norm` 控制处理分配对协变量的依赖程度。
- `tau_conf` 引入未观测混杂 `U` 对结果的影响，使 `W` 和潜在结果之间可能存在内生性。这正是 IV 方法要处理的核心问题。
- 在 IV 语境中，常见目标不是全体平均处理效应，而是对服从者（compliers）的局部平均处理效应，即 LATE。

## 参数与统计场景对应关系

| 参数 | 影响对象 | 统计含义 |
| --- | --- | --- |
| `n` | 样本量 | 控制模拟数据规模。 |
| `p` | 协变量维度 | 控制低维或高维特征场景。 |
| `r2` | 结果模型信号 | 控制 `X` 对 `Y` 的解释力。 |
| `sparsity` | 系数稀疏性 | 模拟少数变量真正相关的高维稀疏场景。 |
| `interactions` | 处理效应异质性 | 是否允许 `Y(1)-Y(0)` 随 `X` 改变。 |
| `tau` | 平均处理效应 | 处理效应的基础水平。 |
| `tauv` | 潜在结果方差比例 | 控制处理组和对照组潜在结果方差不同。 |
| `betaW_norm` | 处理分配机制 | 控制混杂强度和倾向得分重叠程度。 |
| `betaZ_norm` | 工具变量分配机制 | 控制 `Z` 是否依赖协变量。 |
| `tauZ` | 第一阶段强度 | 控制工具变量对处理的影响强弱。 |
| `tau_conf` | 未观测混杂 | 控制 IV 数据中的内生性强度。 |
| `heterosked` | 噪声结构 | 控制同方差或异方差。 |
| `eps_dist` | 结果分布 | 控制连续、二元、重尾、偏态等结果类型。 |
| `lmda_dist` | 协变量分布 | 控制协变量的椭圆尺度、尾部厚度和非正态性。 |
| `dgp_seed` | DGP 参数随机性 | 固定真实系数和结构参数。 |
| `sample_seed` | 样本随机性 | 固定抽样误差和观测样本。 |

## 小结

`gen_data.py` 不是单纯的随机数生成脚本，而是一个面向因果推断方法测试的 DGP 集合。它能生成：

- 标准处理效应数据，用于检验协变量调整、倾向得分和处理效应估计。
- 带选择机制的数据，用于研究选择偏差和 Lee bounds。
- 工具变量数据，用于研究内生性、第一阶段强度和 LATE。

变量类型上，`X` 和概率类对象通常是连续的，`W`、`Z`、`S` 是二元离散变量，`y` 默认连续但可通过 `eps_dist='bernoulli'` 变为二元离散结果。这个设计使同一个文件可以覆盖连续结果、二元结果、同方差、异方差、混杂、选择偏差和内生性等多个重要统计场景。
