测试函数
  test/test_varbounds.py 里有 8 个测试函数：

  | 测试函数 | 位置 | 测试内容 |
  | --- | --- | --- |
  | test_varite_consistency | test/test_varbounds.py:22 | 测试 Var(ITE) 双重界估计是否接近解析界。 |
  | test_varite_delta_method_se | test/test_varbounds.py:53 | 测试 Var(ITE) 的 delta method 标准误是否正确。 |
  | test_no_error | test/test_varbounds.py:63 | 测试连续 gaussian 和离散 bernoulli 结果下 VarITEDualBounds 能否正常跑
  完。 |
  | test_varcate_delta_method_se | test/test_varbounds.py:80 | 测试 Var(CATE) 的 6 维 delta method 标准误。 |
  | test_no_discrete_error | test/test_varbounds.py:93 | 测试二元结果下 VarCATEDualBounds 能否正常运行。 |
  | test_varcate_consistency | test/test_varbounds.py:106 | 测试 Var(CATE) 估计是否接近真实 CATE 方差。实际只跑
  ('gaussian', 0.9) 和 ('expon', 0.5) 两组，因为 zip(['gaussian','expon'], [0.9,0.5,0.0]) 会截断。 |
  | test_varcate_cluster_bootstrap | test/test_varbounds.py:138 | 测试多个 VarCATE 模型的 bootstrap 聚合结果是否合理。
  |
  | test_calibrated_varcate_db | test/test_varbounds.py:180 | 测试校准版 CalibratedVarCATEDualBounds 在连续和二元结果下
  能否运行。 |

  测试的统计量
  主要测试两个因果统计量：

  1. Var(ITE)：个体处理效应方差
     数学对象是：

  Var(Y(1) - Y(0))

  对应实现：dualbounds/varite.py:71

  VarITEDualBounds 把目标函数固定为：

  f = lambda y0, y1, x: (y0-y1)**2

  位置：dualbounds/varite.py:80

  最后再用：

  E[(Y(1)-Y(0))^2] - ATE^2

  得到 Var(ITE)，标准误由 varite_delta_method_se 计算。

  2. Var(CATE)：条件平均处理效应方差
     数学对象是：

  Var(E[Y(1) - Y(0) | X])

  对应实现：dualbounds/varcate.py:76

  它先估计：

  CATE(X) = mu1(X) - mu0(X)

  然后用 6 个 AIPW 矩估计量计算 lower bound 和标准误。核心位置：dualbounds/varcate.py:141

  模块协作流程
  整体运行链路是：

  test_varbounds.py
    -> gen_data.py 生成模拟数据 X, W, y, pis, y0_dists, y1_dists, cates
    -> varite.py / varcate.py 构造目标统计量
    -> generic.py 的 DualBounds 提供通用 fit 流程
    -> dist_reg.py 交叉拟合 Y | X,W 的条件分布
    -> utilities.py 处理二元变量、bootstrap、delta method 辅助计算

  更具体地说：

  1. gen_data.gen_regression_data() 或 gen_lee_bound_data() 生成测试数据。
     这里产生 X、W、y、真实倾向得分 pis，以及 oracle 分布 y0_dists/y1_dists。
  2. 测试初始化 VarITEDualBounds 或 VarCATEDualBounds。
     它们都继承自 generic.DualBounds，所以输入处理、协变量处理、倾向得分处理都走通用逻辑：dualbounds/generic.py:188。
  3. 调用 .fit()。
     通用入口在：dualbounds/generic.py:1293

  fit()
    -> cross_fit()
    -> compute_dual_variables()
    -> _compute_final_bounds()

  4. 如果测试没有直接传入 y0_dists/y1_dists，generic.cross_fit() 会调用 dist_reg.cross_fit_predictions() 做交叉拟合。
     位置：dualbounds/dist_reg.py:456
  5. VarITE 会真正使用 dual bound 框架：
     VarITEDualBounds 设定 f=(y0-y1)^2，通过 generic.DualBounds 求上下界，再在 _compute_final_bounds() 中扣掉 ATE^2，得
     到 Var(ITE) 的上下界。
  6. VarCATE 更特殊：
     它不做一般 dual variable 求解，compute_dual_variables() 是空操作：dualbounds/varcate.py:107。
     它直接用估计的 mu1-mu0 作为 h(X)，构造 6 个 AIPW summands，然后用 varcate_delta_method_se() 得到 lower bound 和标
     准误。
  7. varcate_cluster_bootstrap() 使用 utilities.cluster_bootstrap_se() 聚合多个 outcome model 的 Var(CATE) 下界。
     位置：dualbounds/varcate.py:175

  总结：这个测试文件不是在测试普通预测误差，而是在测试两个因果异质性统计量：Var(ITE) 的上下界，以及 Var(CATE) 的下界、
  标准误和 bootstrap 聚合逻辑。


  varite.py 和 varcate.py 都是在 generic.DualBounds 的通用框架上，把“想估计的因果统计量”改写成可由观测数据、潜在结果分布和 AIPW/delta method 处理的形式。
## 本质上varite 就是把方差拆开，涉及到joint distribution的部分扔到generic.py 再设计专门的梯度下降法让整个方差逼近最优。
  VarITE
  varite.py 的目标是：

  Var(ITE) = Var(Y(1) - Y(0))

  其中：

  ITE = Y(1) - Y(0)

  展开后是：

  Var(Y(1)-Y(0))
  = E[(Y(1)-Y(0))^2] - {E[Y(1)-Y(0)]}^2

  所以它拆成两部分：

  第一部分：E[(Y(1)-Y(0))^2]
  第二部分：ATE^2 = {E[Y(1)] - E[Y(0)]}^2

  关键代码在 dualbounds/varite.py:80：

  kwargs['f'] = lambda y0, y1, x: (y0-y1)**2
  super().__init__(*args, **kwargs)

  这里把 generic.DualBounds 的目标函数 f 固定成：

  f(Y(0), Y(1), X) = (Y(0)-Y(1))^2

  也就是说，通用 dual bounds 框架先去估计：

  E[(Y(1)-Y(0))^2]

  但注意，Y(0) 和 Y(1) 不能同时观测，所以这个二阶矩一般不是点识别的，只能给出 lower/upper bounds。

  然后在 dualbounds/varite.py:89 的 _compute_final_bounds() 里，它把这个二阶矩界转换成 Var(ITE) 界：

  sbetas = summands[1-lower]
  skappa1s = self.W * (self.y - self.mu1) / self.pis + self.mu1
  skappa0s = (1-self.W) * (self.y - self.mu0) / (1-self.pis) + self.mu0
  hattheta, se = varite_delta_method_se(
      sbetas=sbetas, skappa1s=skappa1s, skappa0s=skappa0s
  )

  这里的三个量是：

  sbetas   -> E[(Y(1)-Y(0))^2] 的 AIPW summands
  skappa1s -> E[Y(1)] 的 AIPW summands
  skappa0s -> E[Y(0)] 的 AIPW summands

  varite_delta_method_se() 在 dualbounds/varite.py:51：

  hat_beta = sbetas.mean()
  hat_kappa1 = skappa1s.mean()
  hat_kappa0 = skappa0s.mean()
  ate = hat_kappa1 - hat_kappa0
  hattheta = hat_beta - ate**2

  也就是：

  hat_beta = 估计的 E[(Y(1)-Y(0))^2]
  hat_kappa1 = 估计的 E[Y(1)]
  hat_kappa0 = 估计的 E[Y(0)]
  hat_ATE = hat_kappa1 - hat_kappa0

  hat_VarITE = hat_beta - hat_ATE^2

  标准误用 delta method：

  grad = np.array([1, -2 * ate, 2 * ate])
  se = np.sqrt(grad @ hatSigma @ grad / len(sbetas))

  这是对函数：

  g(beta, kappa1, kappa0) = beta - (kappa1-kappa0)^2

  求梯度：

  dg/dbeta = 1
  dg/dkappa1 = -2(kappa1-kappa0)
  dg/dkappa0 = 2(kappa1-kappa0)

  所以 varite.py 的构造逻辑是：

  先让 DualBounds 处理 E[(Y1-Y0)^2] 的不可识别部分
  再用 AIPW 估计 E[Y1] 和 E[Y0]
  最后组合成 Var(Y1-Y0)

  VarCATE
  ## 根本不需要joint distribution 和dual bound 毫无关系。
  varcate.py 的目标是：

  Var(CATE) = Var(E[Y(1)-Y(0) | X])

  令：

  tau(X) = E[Y(1)-Y(0) | X]
         = mu1(X) - mu0(X)

  其中：

  mu1(X) = E[Y(1) | X]
  mu0(X) = E[Y(0) | X]

  如果我们知道真实的 tau(X)，那么：

  Var(CATE) = Var(tau(X))

  但实际中 tau(X) 是估计出来的。varcate.py 使用一个 lower-bound 形式，令：

  h(X) = 估计出来的 CATE

  然后目标统计量写成：

  2 * Cov(h(X), Y(1)-Y(0)) - Var(h(X))

  这个量满足一个重要性质：

  2 Cov(h, tau) - Var(h)
  = Var(tau) - E[(h - tau)^2] + {E[h-tau]}^2 的相关形式

  直观上，如果 h(X) 接近真实 tau(X)，这个下界就接近真实 Var(CATE)；如果 h(X) 很差，下界会变弱。

  核心矩函数在 dualbounds/varcate.py:13：

  def _moments2varcate(
      hxy1, hxy0, hx, y1, y0, shx2,
  ):
      return 2 * (
          hxy1 - hxy0 - hx * y1 + hx * y0
      ) - shx2 + hx**2

  这些符号对应：

  hxy1 = E[h(X)Y(1)]
  hxy0 = E[h(X)Y(0)]
  hx   = E[h(X)]
  y1   = E[Y(1)]
  y0   = E[Y(0)]
  shx2 = E[h(X)^2]

  展开一下：

  Cov(h, Y(1)-Y(0))
  = E[h(Y(1)-Y(0))] - E[h]E[Y(1)-Y(0)]
  = E[hY(1)] - E[hY(0)] - E[h]E[Y(1)] + E[h]E[Y(0)]

  所以：

  2 Cov(h, Y(1)-Y(0)) - Var(h)

  等于：

  2 * (hxy1 - hxy0 - hx*y1 + hx*y0)
  - (E[h^2] - E[h]^2)

  也就是代码里的：

  2 * (
      hxy1 - hxy0 - hx * y1 + hx * y0
  ) - shx2 + hx**2

  在 _compute_final_bounds() 中，先计算条件均值：

  dualbounds/varcate.py:141

  self._compute_cond_means()
  self.cates = self.mu1 - self.mu0

  也就是：

  h(X) = mu1(X) - mu0(X)

  然后构造 6 个 AIPW summands：

  self.shxy1 = self.W * self.cates * (self.y - self.mu1)
  self.shxy1 = self.shxy1 / self.pis + self.cates * self.mu1

  对应：

  E[h(X)Y(1)]

  self.shxy0 = (1 - self.W) * self.cates * (self.y - self.mu0)
  self.shxy0 = self.shxy0 / (1 - self.pis) + self.cates * self.mu0

  对应：

  E[h(X)Y(0)]

  self.shx = self.cates

  对应：

  E[h(X)]

  self.sy1 = self.W * (self.y - self.mu1) / self.pis + self.mu1

  对应：

  E[Y(1)]

  self.sy0 = (1 - self.W ) * (self.y - self.mu0)
  self.sy0 = self.sy0 / (1 - self.pis) + self.mu0

  对应：

  E[Y(0)]

  self.shx2 = self.cates**2

  对应：

  E[h(X)^2]

  最后调用：

  estimate, se = varcate_delta_method_se(
      shxy1=self.shxy1,
      shxy0=self.shxy0,
      shx=self.shx,
      sy1=self.sy1,
      sy0=self.sy0,
      shx2=self.shx2,
  )

  varcate_delta_method_se() 会先取 6 个 summands 的均值：

  mus = summands.mean(axis=0)
  hattheta = _moments2varcate(*tuple(list(mus)))

  也就是把 6 个样本矩代入 _moments2varcate()，得到：

  hat VarCATE lower bound

  然后用 delta method 计算标准误：

  hatSigma = np.cov(summands.T)
  grad = np.array([
      2,
      -2,
      2 * (mus[4] - mus[3] + mus[2]),
      - 2 * mus[2],
      2 * mus[2],
      -1,
  ])
  se = np.sqrt(grad @ hatSigma @ grad / len(shx))

  这个梯度对应函数：

  g(hxy1,hxy0,hx,y1,y0,hx2)
  = 2(hxy1 - hxy0 - hx*y1 + hx*y0) - hx2 + hx^2

  所以 varcate.py 的构造逻辑是：

  先估计 mu1(X), mu0(X)
  构造 h(X)=mu1(X)-mu0(X)
  再用 AIPW 构造 6 个可识别矩
  最后把 6 个矩组合成 Var(CATE) 的 lower bound

  两者差异
  VarITE 和 VarCATE 的核心区别是：

  | 项目 | varite.py | varcate.py |
  | --- | --- | --- |
  | 目标 | Var(Y(1)-Y(0)) | Var(E[Y(1)-Y(0)|X]) |
  | 是否涉及个体潜在结果联合分布 | 是，需要处理 Y(0),Y(1) 的 joint/coupling | 不直接需要 |
  | 是否用通用 dual solver | 是，目标函数是 (y0-y1)^2 | 基本不用，compute_dual_variables() 是空 |
  | 主要不可识别性 | Y(0) 和 Y(1) 不能同时观测，二者相关结构未知 | 主要依赖 CATE 函数估计质量 |
  | 输出 | lower 和 upper bounds | lower bound，upper 为 nan |
  | 标准误 | 3 维 delta method | 6 维 delta method |

  一句话总结：varite.py 是“先界定二阶个体效应，再扣掉 ATE 平方”；varcate.py 是“先估计条件处理效应函数，再用它构造一个可识别的 Var(CATE) 下界”。
  ## Var(Y(1)−Y(0))=Var(E[Y(1)−Y(0)∣X])+E[Var(Y(1)−Y(0)∣X)]
          ITE               CATE
