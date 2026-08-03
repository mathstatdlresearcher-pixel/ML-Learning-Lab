# SVM-SVR-Models

支持向量机（SVM）分类与支持向量回归（SVR）对比实验项目。  
数据全部在代码中下载 / 生成 / 划分，不依赖本地数据集文件；实验结果与可视化统一输出到 `outputs/`。

---

## 1. 项目简介

本项目覆盖三类实验：

| 实验 | 任务 | 数据 | 关注点 |
|------|------|------|--------|
| 线性 SVM | 二分类 | `make_blobs` 模拟数据 | 决策边界与支持向量 |
| 鸢尾花 SVM | 多分类 | sklearn Iris | 核函数对比、C/gamma/degree 敏感性 |
| SVR | 回归 | OpenML Abalone | 核函数对比、参数调优、预测效果 |

---

## 2. 目录结构

```text
SVM-SVR-Models/
├── run_all.py                 # 一键运行入口
├── requirements.txt
├── README.md
├── common/                    # 公共配置与工具
│   ├── config.py
│   └── utils.py
├── data_loader/               # 数据下载 / 生成 / 划分
│   └── prepare_data.py
├── eda/                       # 探索性数据分析
│   └── eda.py
├── models/                    # 模型实验
│   ├── linear_svm.py
│   ├── iris_svm.py
│   └── svr.py
└── outputs/
    ├── figures/               # 全部图片
    │   ├── eda/
    │   ├── linear_svm/
    │   ├── iris_svm/
    │   └── svr/
    └── results/               # 指标表（csv）
```

---

## 3. 环境与运行

使用 Anaconda Python：

```bash
cd SVM-SVR-Models
D:\anaconda3\python.exe -m pip install -r requirements.txt
```

一键运行（推荐）：

```bash
D:\anaconda3\python.exe run_all.py --quick-svr
```

分模块运行：

```bash
D:\anaconda3\python.exe -m data_loader.prepare_data
D:\anaconda3\python.exe -m eda.eda
D:\anaconda3\python.exe -m models.linear_svm
D:\anaconda3\python.exe -m models.iris_svm --no-grid
D:\anaconda3\python.exe -m models.svr --quick
```

常用参数：

- `--quick-svr` / `--quick`：跳过 SVR 完整网格搜索  
- `--no-grid`：跳过鸢尾花 RBF 网格搜索  
- `--skip-eda` / `--skip-linear` / `--skip-iris` / `--skip-svr`：跳过对应阶段  

---

## 4. 数据说明

| 数据 | 获取方式 | 规模 | 用途 |
|------|----------|------|------|
| Iris | `sklearn.datasets.load_iris` | 150×4，3 类 | 多核 SVM 分类 |
| Abalone | `fetch_openml("abalone")` | 4177 样本，9 特征 → 预测年龄 | SVR 回归 |
| Blobs | `make_blobs` | 200 样本，2 维，2 类 | 线性 SVM 演示 |

划分策略：

- Iris：`train:test = 60%:40%`，分层抽样，`random_state=28`
- Abalone：`train:test = 80%:20%`，标准化后训练，`random_state=42`
- Blobs：`train:test = 70%:30%`

---

## 5. 实验结果总览

以下结果来自当前仓库已跑通实验（`outputs/results/`）。

### 5.1 线性 SVM（Blobs）

| 指标 | 训练集 | 测试集 |
|------|--------|--------|
| Accuracy | 1.000 | 1.000 |
| F1-macro | 1.000 | 1.000 |
| 支持向量数 | 2 | — |

结论：模拟数据近似线性可分，线性核可完美分开两类，支持向量很少，间隔清晰。

### 5.2 鸢尾花核函数对比（测试准确率）

| 特征组合 | linear | rbf | poly | sigmoid |
|----------|--------|-----|------|---------|
| 花瓣（petal） | 0.950 | **0.967** | 0.950 | 0.000 |
| 花萼（sepal） | **0.833** | 0.817 | 0.783 | 0.233 |
| 全部 4 特征 | **0.983** | 0.950 | 0.950 | 0.067 |

结论：

1. **花瓣特征**整体优于花萼，类别可分性更强。  
2. **linear / rbf / poly** 表现接近且优秀；**sigmoid** 在该设定下几乎失效。  
3. 使用全部 4 特征时，**linear** 测试准确率最高（0.983）。

### 5.3 RBF 参数敏感性（花萼二维）

| 参数 | 取值 | 训练准确率 | 测试准确率 |
|------|------|------------|------------|
| C | 0.3 / 1 / 10 | 0.811 / 0.811 / 0.833 | 0.833 / 0.833 / 0.833 |
| C | 1000 | 0.844 | **0.800（下降）** |
| gamma | 0.1 / 1 | 0.756 / 0.811 | 0.800 / **0.833** |
| gamma | 10 / 100 | 0.833 / 0.967 | 0.783 / **0.617（过拟合）** |

结论：C、gamma 过大都会抬高训练集、伤害测试集；中等取值更稳。

### 5.4 Poly 参数敏感性（花萼二维）

| 参数 | 取值 | 测试准确率 |
|------|------|------------|
| degree | 1 / 2 / 5 / 10 | 0.817 / **0.833** / 0.783 / 0.783 |
| C | 0.1 / 0.5 / 10 / 1000 | 0.817 / **0.833** / 0.833 / 0.833 |

结论：degree=2 较合适；度数过高边界更复杂，测试效果未必更好。

### 5.5 SVR（鲍鱼年龄）

| 模型 | MAE | RMSE | MAPE | R² |
|------|-----|------|------|----|
| linear（默认） | 1.562 | 2.256 | 0.150 | 0.530 |
| poly（默认） | 1.561 | 2.342 | 0.146 | 0.493 |
| **rbf（默认）** | **1.521** | **2.238** | **0.144** | **0.537** |
| rbf（C=8, γ=1） | 1.563 | 2.318 | 0.151 | 0.504 |
| rbf（原 notebook 参考 C=19, γ≈3.16） | 1.740 | 2.500 | 0.171 | 0.423 |

调参观察：

- gamma 扫描：`log10(γ)=0`（γ=1）CV 最优（约 0.503）  
- C 扫描：`C=8` CV 最优（约 0.556）  

结论：默认 RBF 在本划分与标准化设定下综合最优；过大的 C/γ 反而变差。

---

## 6. 图片说明（怎么理解每一张图）

图片均位于 `outputs/figures/`。

### 6.1 EDA：`outputs/figures/eda/`

| 图片 | 怎么看 |
|------|--------|
| `iris_class_distribution.png` | 三类样本数量柱状图。应接近均衡（各 50），否则后续准确率要结合类别不平衡解释。 |
| `iris_feature_hist.png` | 四个特征按类别的直方图。若某特征三类峰值分离明显，说明该特征分类价值高（花瓣通常更分离）。 |
| `iris_pairplot.png` | 特征两两散点 + 对角分布。点团重叠少 → 易分；versicolor 与 virginica 常有重叠。 |
| `iris_corr.png` | 特征相关热力图。花瓣长度/宽度通常高相关；高相关说明存在冗余，但不妨碍 SVM。 |
| `abalone_age_hist.png` | 年龄分布。若右偏，说明高龄样本少，回归对尾部可能误差更大。 |
| `abalone_feature_boxplot.png` | 连续特征箱线。可看量纲差异、异常值；这也是为什么 SVR 前要标准化。 |
| `abalone_corr.png` | 特征与年龄相关。颜色越深相关越强；壳重等尺寸特征通常与年龄正相关。 |
| `abalone_scatter_top.png` | 与年龄最相关的若干特征散点。点云趋势向上说明正线性关系，离散大说明噪声/非线性。 |
| `abalone_train_test_age.png` | 训练/测试年龄分布对比。形状应接近，否则评估会有偏。 |
| `blobs_scatter.png` | 模拟二分类散点。两类簇分离越清，越适合线性 SVM。 |

### 6.2 线性 SVM：`outputs/figures/linear_svm/`

| 图片 | 怎么看 |
|------|--------|
| `decision_boundary.png` | 黑色线是决策边界；绿圈是支持向量。边界应落在两类中间；支持向量通常靠近边界。本实验两类可分，边界干净。 |

### 6.3 鸢尾花 SVM：`outputs/figures/iris_svm/`

| 图片 | 怎么看 |
|------|--------|
| `kernel_compare_petal.png` / `sepal.png` / `all4.png` | 左：训练/测试准确率；右：训练耗时。关注测试准确率谁高、训练与测试是否差距过大（过拟合）。 |
| `decision_boundary_petal.png` | 花瓣二维空间中 4 种核的决策区域。背景色=预测类别，点=真实样本，空心圈=测试点。区域平滑且与真实点一致更好。 |
| `decision_boundary_sepal.png` | 同上，但基于花萼。通常重叠更多，边界更“挤”。 |
| `rbf_C_sensitivity.png` | 固定 γ，改变 C。C 增大 → 更贴合训练数据；若测试不升反降，说明过拟合。 |
| `rbf_C_boundary.png` | 不同 C 的决策边界形态。C 大时边界更曲折、更贴点。 |
| `rbf_gamma_sensitivity.png` | 固定 C，改变 γ。γ 大 → 影响范围变局部，边界更碎。 |
| `rbf_gamma_boundary.png` | γ=100 时常见“岛屿状”边界，训练集很高、测试集崩溃，是典型过拟合图示。 |
| `poly_degree_sensitivity.png` | 多项式次数影响。degree 升高模型更复杂。 |
| `poly_degree_boundary.png` | 高次多项式边界弯曲更强；不一定换来更高测试准确率。 |
| `poly_C_sensitivity.png` / `poly_C_boundary.png` | 多项式核下惩罚系数 C 的影响，读法同 RBF 的 C。 |

### 6.4 SVR：`outputs/figures/svr/`

| 图片 | 怎么看 |
|------|--------|
| `rbf_gamma_cv.png` | 横轴 `log10(γ)`，纵轴 5 折 CV。峰值对应较优 γ；右侧下滑说明 γ 过大泛化变差。 |
| `rbf_C_cv.png` | 横轴 C，纵轴 CV。先升后缓降/平台，峰值附近即可。 |
| `rbf_grid_heatmap.png`（完整模式才有） | C×γ 网格热力图。颜色越亮 CV 越好，可找联合最优区域。 |
| `pred_vs_true.png` | 测试集真实年龄 vs 预测年龄随样本序号变化。两条线走势接近说明拟合好；局部偏离大对应难点样本。 |
| `residuals.png` | 残差=真实−预测。理想应绕 y=0 随机散布；若漏斗形/弯曲趋势，说明仍有未建模结构。 |

---

## 7. 主要结论

1. **线性 SVM** 在可分模拟数据上可得到完美分类，适合理解最大间隔思想。  
2. **鸢尾花分类**：花瓣特征信息量更大；linear/rbf/poly 明显优于 sigmoid。  
3. **参数敏感性**：C、gamma、degree 过大易过拟合，需用验证/测试集约束。  
4. **SVR**：默认 RBF 综合最优（R²≈0.537，RMSE≈2.24）；盲目加大 C/γ 未必提升。  

---

## 8. 复现检查清单

- [ ] 使用 Anaconda：`D:\anaconda3\python.exe`  
- [ ] 安装依赖：`pip install -r requirements.txt`  
- [ ] 运行：`python run_all.py --quick-svr`  
- [ ] 检查 `outputs/figures/` 是否生成图片  
- [ ] 检查 `outputs/results/` 是否生成指标 csv  

---

## 9. 依赖

见 `requirements.txt`：`numpy` / `pandas` / `matplotlib` / `seaborn` / `scikit-learn`。
