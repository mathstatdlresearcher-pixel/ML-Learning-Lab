# 实验一：线性回归算法实验报告说明

本项目实现 **OLS、岭回归（Ridge）、Lasso、LARS** 四类线性回归算法，在自构造模拟数据与 Boston 房价数据上完成：探索分析 → 预处理 → 建模 → 评价 → 可视化。

> **公式怎么看最清晰？**  
> 1. 推荐用浏览器打开：[docs/公式手册.html](docs/公式手册.html)（MathJax 渲染，公式一定正常）  
> 2. 在 Cursor / VS Code 中请开启 `Markdown › Math: Enabled`，正文里的公式块使用 `$$ ... $$`  
> 3. 正文叙述里的简单符号直接用 Unicode：α、β、ε、R²、ŷ、β̂，避免行内 LaTeX 兼容问题

---

## 目录

1. [实验目标](#1-实验目标)
2. [整体流程](#2-整体流程)
3. [算法原理与公式](#3-算法原理与公式)
4. [评价指标公式](#4-评价指标公式)
5. [模拟数据设计](#5-模拟数据设计)
6. [实验一：模拟数据 EDA（逐图解读）](#6-实验一模拟数据-eda逐图解读)
7. [实验二：四算法对比（逐图解读）](#7-实验二四算法对比逐图解读)
8. [实验三：LARS 稳定性（逐图解读）](#8-实验三lars-稳定性逐图解读)
9. [实验四：Boston EDA（逐图解读）](#9-实验四boston-eda逐图解读)
10. [实验五：Boston 建模（逐图解读）](#10-实验五boston-建模逐图解读)
11. [项目结构与运行](#11-项目结构与运行)
12. [结论要点](#12-结论要点)

---

## 1. 实验目标

| 编号 | 目标 |
|:---:|------|
| 1 | 掌握 OLS、Ridge、Lasso、LARS 的优化目标、解的形式与适用场景 |
| 2 | 构造无噪声 / 有噪声 / 无多重共线性 / 有多重共线性数据，对比算法 |
| 3 | 多次重复实验，分析 LARS 变量选择稳定性及选择前后效果 |
| 4 | 对 Boston 房价做 EDA、预处理、特征工程，并比较四种算法 |

---

## 2. 整体流程

```mermaid
flowchart TD
    A[配置随机种子与超参] --> B[生成模拟数据 y1~y4]
    B --> C[EDA: 相关 / VIF / 散点]
    C --> D[标准化 + 训练测试划分]
    D --> E[OLS / Ridge / Lasso / LARS]
    E --> F[计算 R² MSE MAE RMSE]
    F --> G[可视化与报告]
    G --> H[LARS 重复实验稳定性]
    A --> I[加载 Boston]
    I --> J[EDA + 预处理 + 特征筛选]
    J --> K[四种算法建模对比]
    K --> L[输出 figures / reports]
```

**逐步说明**

1. **数据**：生成模拟数据，或下载并缓存 Boston。  
2. **EDA**：分布、相关、VIF、散点，确认线性与共线性。  
3. **预处理**：特征标准化；Boston 去封顶、按相关度筛选。  
4. **建模**：同一划分拟合四模型；Ridge / Lasso / LARS 用交叉验证选超参。  
5. **评价**：测试集上计算 R²、MSE、MAE、RMSE。  
6. **稳定性**：多次随机划分，统计 LARS 入选频率与选择前后指标。  
7. **出图**：写入 `outputs/figures/`，数值写入 `outputs/reports/`。

---

## 3. 算法原理与公式

设训练集有 n 个样本、p 个特征。设计矩阵、响应与参数为：

$$
X \in \mathbb{R}^{n \times p},\quad
y \in \mathbb{R}^{n},\quad
\beta \in \mathbb{R}^{p}.
$$

线性模型：

$$
y = X\beta + \varepsilon.
$$

其中 ε 为误差项。预测值为：

$$
\hat{y} = X\hat{\beta}.
$$

---

### 3.1 普通最小二乘 OLS

**优化目标（残差平方和）：**

$$
\hat{\beta}_{\mathrm{OLS}}
=
\arg\min_{\beta}
\| y - X\beta \|_{2}^{2}
=
\arg\min_{\beta}
\sum_{i=1}^{n}
\bigl( y_i - x_i^{\top}\beta \bigr)^{2}.
$$

**正规方程与闭式解**（当 XᵀX 可逆）：

$$
X^{\top}X\,\hat{\beta} = X^{\top}y,
\qquad
\hat{\beta}_{\mathrm{OLS}}
=
\bigl( X^{\top}X \bigr)^{-1} X^{\top} y.
$$

| 优点 | 缺点 |
|------|------|
| 无偏（Gauss–Markov 条件下 BLUE） | 噪声大时方差大 |
| 可解释、计算快 | 多重共线性时 XᵀX 近奇异，系数不稳 |
| 无额外超参 | 高维（p≈n 或 p>n）易过拟合 |

---

### 3.2 岭回归 Ridge（L₂ 正则）

**优化目标：**

$$
\hat{\beta}_{\mathrm{Ridge}}
=
\arg\min_{\beta}
\| y - X\beta \|_{2}^{2}
+
\alpha \| \beta \|_{2}^{2},
\qquad \alpha>0.
$$

**闭式解：**

$$
\hat{\beta}_{\mathrm{Ridge}}
=
\bigl( X^{\top}X + \alpha I \bigr)^{-1} X^{\top} y.
$$

αI 使矩阵正定，缓解共线性。本实验用 **RidgeCV** 按验证 MSE 选 α。

| 优点 | 缺点 |
|------|------|
| 共线时更稳 | 只收缩、不稀疏 |
| 易优化 | 不能硬变量选择 |
| 病态问题下预测常更好 | 需调节 α |

---

### 3.3 Lasso（L₁ 正则）

**优化目标：**

$$
\hat{\beta}_{\mathrm{Lasso}}
=
\arg\min_{\beta}
\| y - X\beta \|_{2}^{2}
+
\alpha \| \beta \|_{1},
\qquad
\| \beta \|_{1}=\sum_{j=1}^{p}|\beta_j|.
$$

L₁ 约束常使部分系数精确为 0，从而实现变量选择。本实验用 **LassoCV** 选 α。

| 优点 | 缺点 |
|------|------|
| 稀疏、可解释 | 强共线时倾向随机留下一个相关变量 |
| 适合含无关特征 | α 过大欠拟合，过小接近 OLS |
| 高维可用 | 一般无闭式解 |

---

### 3.4 LARS（Least Angle Regression）

LARS 是前向分段线性路径算法：每一步沿与当前残差“夹角相等”的方向前进。

记活跃集为 𝒜，残差为：

$$
r = y - X\hat{\beta}.
$$

步骤概要：

1. 将与残差相关性最大的变量加入活跃集；  
2. 沿等角方向前进；  
3. 直到新变量追上，或达到停止条件。

等角方向满足：

$$
X_{\mathcal{A}}^{\top} u_{\mathcal{A}}
=
A_{\mathcal{A}}\,\mathbf{1}.
$$

本实验：先用 **LarsCV**，再对非零个数 k 做网格 CV，取验证误差更优者，避免过稀。

| 优点 | 缺点 |
|------|------|
| 适合变量选择 | 选择可能随划分抖动 |
| 路径可解释 | 强相关下不稳定 |
| 高维相对高效 | 需 CV 决定停止位置 |

---

### 3.5 特征标准化

$$
\tilde{x}_{ij}
=
\frac{x_{ij}-\mu_j}{\sigma_j},
\qquad
\mu_j=\frac{1}{n}\sum_{i=1}^{n}x_{ij},
\quad
\sigma_j=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_{ij}-\mu_j)^2}.
$$

标准化后正则对各特征公平。若 y 未标准化，理论斜率在标准化尺度下为：

$$
\beta_j^{\mathrm{scaled}} = \beta_j\,\sigma_j.
$$

---

## 4. 评价指标公式

设测试集真实值为 yᵢ，预测为 ŷᵢ，样本数 m，均值为 ȳ。

### 决定系数 R²

$$
R^{2}
=
1-
\frac{\sum_{i=1}^{m}(y_i-\hat{y}_i)^{2}}
{\sum_{i=1}^{m}(y_i-\bar{y})^{2}}.
$$

越接近 1 越好；可以为负。

### 均方误差 MSE

$$
\mathrm{MSE}
=
\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^{2}.
$$

### 均方根误差 RMSE

$$
\mathrm{RMSE}
=
\sqrt{\mathrm{MSE}}
=
\sqrt{\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^{2}}.
$$

与 y 同量纲，便于和噪声标准差对照。

### 平均绝对误差 MAE

$$
\mathrm{MAE}
=
\frac{1}{m}\sum_{i=1}^{m}\lvert y_i-\hat{y}_i\rvert.
$$

### 方差膨胀因子 VIF

对第 j 个特征，用其余特征回归得到 R²ⱼ，则：

$$
\mathrm{VIF}_j
=
\frac{1}{1-R_j^{2}}.
$$

经验：VIF < 5 较安全；5～10 警惕；> 10 共线严重。

---

## 5. 模拟数据设计

特征结构（尺度为 1，等价于原稿 100u 再除以 100）：

$$
\begin{aligned}
x_1,x_2,x_3 &\sim \mathrm{Uniform}(0,1),\\
x_4 &= 6.7x_1 + 4x_2 + 0.05u,\\
x_5 &= 2.8x_1 + 3.4x_4 + 0.01u.
\end{aligned}
$$

信号：

$$
\begin{aligned}
s_1 &= 0.5 + 2x_1 + 0.3x_2,\\
s_2 &= 1 + 2x_1 + 4x_2,\\
s_3 &= 5 + x_0 + 1.2x_1 + 3.5x_2,\\
s_4 &= 0.7x_1 + 1.1x_2 - 6.4x_3 + x_4 + x_5.
\end{aligned}
$$

噪声：

$$
y = s + \varepsilon,\qquad
\varepsilon\sim\mathcal{N}\bigl(0,(\rho\cdot\sigma_s)^2\bigr),\quad
\sigma_s=\mathrm{Std}(s).
$$

| 数据集 | ρ | 特征 | 意图 |
|--------|:---:|------|------|
| y1 | 0 | x₁, x₂ | 无噪声理想线性 |
| y2 | 0.25 | x₁, x₂ + 16 个无关特征 | 有噪声 + 变量选择 |
| y3 | 0.15 | x₀, x₁, x₂ | 无多重共线性 |
| y4 | 0.28 | x₁～x₅ + 16 个无关特征 | 多重共线性 |

默认：n = 80，测试比例 30%，重复 25 次。

---

## 6. 实验一：模拟数据 EDA（逐图解读）

脚本：`experiments/01_simulated_eda.py`  
目录：`outputs/figures/01_eda/`

### 6.1 特征相关矩阵

![corr_features](outputs/figures/01_eda/corr_features.png)

**解读：** x₄ 与 x₁、x₂，以及 x₅ 与 x₁、x₄ 应呈强相关，这是多重共线性来源；x₃ 相对独立。

### 6.2 VIF：无共线 vs 有共线

![vif_y3](outputs/figures/01_eda/vif_y3.png)

![vif_y4](outputs/figures/01_eda/vif_y4.png)

**解读：** y3 的 VIF 接近 1；y4 中 x₄、x₅ 等 VIF 极高（对数轴）。虚线对应经验阈值 5 与 10。

### 6.3 三维散点与二维线性关系

![y1_3d](outputs/figures/01_eda/y1_3d.png)

![y2_3d](outputs/figures/01_eda/y2_3d.png)

![y1_x1](outputs/figures/01_eda/y1_x1.png)

![y2_x1](outputs/figures/01_eda/y2_x1.png)

![y2_x2](outputs/figures/01_eda/y2_x2.png)

**解读：**

- y1 无噪声：点几乎落在平面/直线上。  
- y2 有噪声：仍有清晰上升趋势，点带散布。  
- x₂–y 斜率应大于 x₁–y（真值系数 4 > 2）。

### 6.4 y4 配对图

![y4_pairplot](outputs/figures/01_eda/y4_pairplot.png)

**解读：** 对角线为分布，非对角为两两散点，可同时看到共线结构与对 y₄ 的关系。

---

## 7. 实验二：四算法对比（逐图解读）

脚本：`experiments/02_model_comparison.py`  
目录：`outputs/figures/02_sim_models/`

### 7.1 指标柱状图

![y1_metrics](outputs/figures/02_sim_models/y1_no_noise_metrics.png)

![y2_metrics](outputs/figures/02_sim_models/y2_noisy_metrics.png)

![y3_metrics](outputs/figures/02_sim_models/y3_no_multicollinearity_metrics.png)

![y4_metrics](outputs/figures/02_sim_models/y4_multicollinearity_metrics.png)

**解读：**

| 数据 | 读图结论 |
|------|----------|
| y1 | 四模型 R²≈1，误差≈0 |
| y2 | Lasso/LARS 优于 OLS（筛掉无关特征） |
| y3 | 四模型接近且都很高 |
| y4 | Lasso/LARS（及常 Ridge）优于 OLS |

### 7.2 真实值 vs 预测值

![y2_pred](outputs/figures/02_sim_models/y2_noisy_pred_vs_true.png)

**解读：** 点越贴近对角线越好。

### 7.3 测试曲线

![y2_series](outputs/figures/02_sim_models/y2_noisy_series.png)

**解读：** 分面对比各模型与真实曲线的贴合。

### 7.4 残差诊断

![y2_resid](outputs/figures/02_sim_models/y2_noisy_residuals.png)

**解读：** 左图残差应在 0 附近随机散布；右图直方图近似对称。

### 7.5 系数对比（真值 vs 估计）

![y2_coef](outputs/figures/02_sim_models/y2_noisy_coef.png)

![y4_coef](outputs/figures/02_sim_models/y4_multicollinearity_coef.png)

**解读：** 深色为真值，青色为估计。y4 共线下 OLS 易偏离，Ridge 收缩，Lasso/LARS 更稀疏。

### 7.6 系数稳定性

![y2_stab](outputs/figures/02_sim_models/y2_noisy_coef_stability.png)

![y4_stab](outputs/figures/02_sim_models/y4_multicollinearity_coef_stability.png)

**解读：** 箱子越短越稳；共线下 OLS 通常最散。

### 7.7 Ridge 路径

![y2_ridge](outputs/figures/02_sim_models/y2_noisy_ridge_path.png)

**解读：** 横轴 α（对数），纵轴 CV-MSE；虚线为最优 α。

---

## 8. 实验三：LARS 稳定性（逐图解读）

脚本：`experiments/03_lars_stability.py`  
目录：`outputs/figures/03_lars_stability/`

每次划分：CV 选 LARS 的非零个数 k，再对比

$$
\text{全变量 OLS}
\quad\text{vs}\quad
\text{仅 LARS 入选变量后的 OLS}.
$$

### 8.1 入选矩阵与频率

![y2_lars](outputs/figures/03_lars_stability/y2_noisy_stability.png)

![y3_lars](outputs/figures/03_lars_stability/y3_no_multicollinearity_stability.png)

![y4_lars](outputs/figures/03_lars_stability/y4_multicollinearity_stability.png)

**解读：** 行间图案越一致越稳定；相关变量频率应高，`noise*` 应低。y4 中 x₁/x₄/x₅ 可能互换。

### 8.2 选择前后指标

![y2_ba](outputs/figures/03_lars_stability/y2_noisy_before_after.png)

![y4_ba](outputs/figures/03_lars_stability/y4_multicollinearity_before_after.png)

**解读：** 若选择后 R² 升、RMSE/MAE 降，说明剔除无关/冗余变量提升了泛化。

---

## 9. 实验四：Boston EDA（逐图解读）

脚本：`experiments/04_boston_eda.py`  
目录：`outputs/figures/04_boston_eda/`

| 符号 | 含义 |
|------|------|
| CRIM | 人均犯罪率 |
| RM | 平均房间数 |
| LSTAT | 低收入占比 |
| PTRATIO | 师生比 |
| MEDV | 房价中位数（目标） |

### 9.1 分布

![dist](outputs/figures/04_boston_eda/distributions.png)

**解读：** 看偏态、离散与二值特征。

### 9.2 相关

![corr](outputs/figures/04_boston_eda/corr_heatmap.png)

![corr_price](outputs/figures/04_boston_eda/corr_with_price.png)

**解读：** 通常 LSTAT 与 MEDV 负相关，RM 与 MEDV 正相关。

### 9.3 关键散点

![key](outputs/figures/04_boston_eda/key_scatter.png)

**解读：** 验证主要线性趋势是否成立。

### 9.4 封顶处理

![box](outputs/figures/04_boston_eda/price_boxplot.png)

**解读：** 原始 MEDV 在 50 处封顶；实验去掉 MEDV = 50 的样本。

预处理：缺失检查 → 去封顶 → 按 |corr| 筛选 → 标准化。

---

## 10. 实验五：Boston 建模（逐图解读）

脚本：`experiments/05_boston_modeling.py`  
目录：`outputs/figures/05_boston_models/`

### 10.1 指标对比

![metrics](outputs/figures/05_boston_models/metrics.png)

**解读：** 比较 R²、RMSE、MAE；结合重复实验看稳定性。

### 10.2 真实 vs 预测

![pvt](outputs/figures/05_boston_models/pred_vs_true.png)

**解读：** 点云应沿对角线；高端房价可能系统性低估。

### 10.3 测试曲线：总览 + 分模型

![series](outputs/figures/05_boston_models/series.png)

![series_ols](outputs/figures/05_boston_models/series_ols.png)

![series_ridge](outputs/figures/05_boston_models/series_ridge.png)

![series_lasso](outputs/figures/05_boston_models/series_lasso.png)

![series_lars](outputs/figures/05_boston_models/series_lars.png)

**解读：** `series.png` 为四分面总览；`series_*.png` 为各模型单独与真实值对比（阴影为偏差）。

### 10.4 残差

![resid](outputs/figures/05_boston_models/residuals_ols.png)

**解读：** 若呈喇叭形或弯曲，提示异方差或非线性。

### 10.5 系数

![c_ols](outputs/figures/05_boston_models/coef_ols.png)

![c_ridge](outputs/figures/05_boston_models/coef_ridge.png)

![c_lasso](outputs/figures/05_boston_models/coef_lasso.png)

![c_lars](outputs/figures/05_boston_models/coef_lars.png)

**解读：** 标准化后可比相对重要性；Lasso/LARS 接近 0 表示被剔除。

### 10.6 Ridge 路径

![ridge](outputs/figures/05_boston_models/ridge_path.png)

**解读：** 最优 α 在 CV-MSE 最低处。

---

## 11. 项目结构与运行

```text
linear_regression_lab/
├── main.py / config.py / requirements.txt / README.md
├── docs/公式手册.html          # 浏览器看公式（推荐）
├── data/  models/  utils/  experiments/
└── outputs/figures|reports|data/
```

```bash
pip install -r requirements.txt
python main.py
python main.py --only 01,02,03
python main.py --only 04,05
```

Windows 可双击 `run.bat`。

> `sklearn≥1.2` 已移除 `load_boston`，本项目自动下载并缓存到 `outputs/data/boston.csv`。

---

## 12. 结论要点

1. **y1**：无噪声，OLS 已近乎完美。  
2. **y2**：无关特征下 Lasso/LARS 测试误差更低。  
3. **y3**：无共线，四模型接近。  
4. **y4**：高 VIF；Ridge/Lasso/LARS 更稳。  
5. **LARS 稳定性**：相关变量频率高；共线下存在变量互换；选择后常降 RMSE。  
6. **Boston**：LSTAT、RM 等关键；分模型曲线便于细看局部差异。

---

## 公式速查

$$
\begin{aligned}
\hat{\beta}_{\mathrm{OLS}}
&=(X^{\top}X)^{-1}X^{\top}y,\\[6pt]
\hat{\beta}_{\mathrm{Ridge}}
&=(X^{\top}X+\alpha I)^{-1}X^{\top}y,\\[6pt]
\hat{\beta}_{\mathrm{Lasso}}
&=\arg\min_{\beta}\|y-X\beta\|_{2}^{2}+\alpha\|\beta\|_{1},\\[6pt]
R^{2}
&=1-\frac{\sum_i(y_i-\hat{y}_i)^{2}}{\sum_i(y_i-\bar{y})^{2}},\\[6pt]
\mathrm{MSE}
&=\frac{1}{m}\sum_i(y_i-\hat{y}_i)^{2},\quad
\mathrm{RMSE}=\sqrt{\mathrm{MSE}},\quad
\mathrm{MAE}=\frac{1}{m}\sum_i|y_i-\hat{y}_i|,\\[6pt]
\mathrm{VIF}_j
&=\frac{1}{1-R_j^{2}}.
\end{aligned}
$$

---

## 参考文献

1. 课程实验要求 PDF  
2. 参考 Notebook：`机器学习第二组.ipynb`  
3. Hastie, Tibshirani, Friedman. *The Elements of Statistical Learning*  
4. Efron et al. Least Angle Regression. *Annals of Statistics*, 2004  
5. scikit-learn：Linear Models  
