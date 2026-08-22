# Ensemble-Learning-Adaboost-RF

集成学习实验：在 **Breast Cancer** 上对比不同基学习器的 AdaBoost，并与 Random Forest、Lars 分析关键特征；在 **Boston 房价** 上考察 Random Forest 参数，以及单棵决策树 vs AdaBoost vs Random Forest。

全部代码为 Python 脚本，结果与可视化统一写入 `ensemble_lab/outputs/`。

---

## 1. 项目简介

| 部分 | 任务 | 数据 | 关注点 |
|------|------|------|--------|
| 二 | 二分类 | `datasets/breast_cancer.csv` | EDA、预处理、AdaBoost 基学习器对比、关键特征 |
| 三 | 回归 | `datasets/boston.csv` | EDA、预处理、RF 参数影响、单模型 vs 集成 |

实验入口：`ensemble_lab/main.py`（8 个模块，编号 01–08）。

---

## 2. 目录结构

```text
Ensemble-Learning-Adaboost-RF/
├── README.md
├── datasets/
│   ├── breast_cancer.csv          # 569 × 30，diagnosis 为标签
│   └── boston.csv                 # 506 × 13，target/MEDV 为房价
├── 代码/                          # 原始 notebook / html（参考）
└── ensemble_lab/                  # 可运行实验工程
    ├── main.py                    # 一键入口
    ├── run.bat                    # 调用 Pytorch 环境
    ├── common/                    # 路径、主题、指标、绘图
    ├── data_loader/               # 读 CSV、清洗、划分、标准化
    ├── experiments/               # 01–08 实验脚本
    └── outputs/
        ├── data/                  # 划分后的训练集 / 测试集
        ├── figures/               # 全部图片（按实验分子目录）
        ├── results/               # 指标表 csv / json
        └── models/                # 拟合后的模型
```

---

## 3. 环境与运行

使用已有环境 **`D:\Anaconda3\envs\Pytorch`**，不额外装包。当前实验用到：

- Python 3.10
- scikit-learn 1.7
- pandas / numpy / matplotlib / seaborn / joblib

一键运行（推荐）：

```bat
cd /d E:\机器学习项目文件夹\Ensemble-Learning-Adaboost-RF\ensemble_lab
run.bat
```

或：

```bat
D:\Anaconda3\envs\Pytorch\python.exe -u main.py
```

常用参数：

```bat
D:\Anaconda3\envs\Pytorch\python.exe -u main.py --only 01,05
D:\Anaconda3\envs\Pytorch\python.exe -u main.py --quick
```

- `--only 01,03`：只跑指定编号  
- `--quick`：缩小网格，便于试跑  

说明：当前环境下 `GridSearchCV(n_jobs=-1)` 的 loky 多进程不稳定，代码里固定 `n_jobs=1`。完整网格约 9 分钟。

---

## 4. 数据集

数据已放在 `datasets/`，由 `ensemble_lab/common/config.py` 读取。

### 4.1 Breast Cancer

- 样本 569，特征 30，无缺失、无重复。
- 原始 `diagnosis`：`0=恶性，1=良性`（与 sklearn 一致）。
- 建模时改为 **`1=恶性，0=良性`**，使 Precision / Recall / F1 关注阳性类（恶性）。
- 类别：恶性 212（37.3%），良性 357（62.7%）。
- 划分：分层 8:2，训练集 455、测试集 114。

### 4.2 Boston 房价

- 样本 506，特征 13，无缺失。
- 目标列 `target` 统一为 `MEDV`（房价中位数，单位千美元）。
- 均值约 22.53，中位数 21.2；**16 条样本房价封顶在 50**，默认保留。
- 与房价相关最强：`LSTAT`（−0.74）、`RM`（+0.70）、`PTRATIO`（−0.51）。
- 划分：8:2，随机种子 42。

预处理要点：列名去空格；乳腺癌做温和 IQR 截尾；`StandardScaler` 只拟合训练集，供 LR / SVM / Lars 使用，树模型用原始尺度。

---

## 5. 实验模块

| 编号 | 脚本 | 内容 |
|:---:|------|------|
| 01 | `experiments/01_breast_eda.py` | 类别分布、特征直方图、相关热力图、按诊断箱线图 |
| 02 | `experiments/02_breast_preprocess.py` | 标签重编码、分层划分、标准化前后对比 |
| 03 | `experiments/03_breast_adaboost.py` | 决策树 / 逻辑回归 / SVM 为基学习器，网格搜索 AdaBoost |
| 04 | `experiments/04_breast_features.py` | AdaBoost(DT)、Random Forest、Lars 特征重要性对比 |
| 05 | `experiments/05_boston_eda.py` | 房价分布、相关热力图、关键散点、临河箱线图 |
| 06 | `experiments/06_boston_preprocess.py` | 列名统一、划分、标准化对照 |
| 07 | `experiments/07_boston_rf_params.py` | 逐参数扫描 + RF 网格搜索（R2 / MSE / MAE） |
| 08 | `experiments/08_boston_ensemble.py` | 单棵决策树 vs AdaBoost vs Random Forest |

评价指标：

- 分类：Precision、Recall、F1、AUC（另记录 Accuracy）
- 回归：R²、MSE、MAE（另记录 RMSE）

---

## 6. 实验结论（完整网格，测试集）

随机种子 42，测试比例 0.2，5 折交叉验证。数字来自 `ensemble_lab/outputs/results/`。

### 二、Breast Cancer

**1–2. 探索与预处理**

569 条、30 个特征；无缺失、无重复。标签改为 **1=恶性，0=良性**；分层 8:2 划分（训练 455 / 测试 114）。树模型用原始尺度，LR / SVM / Lars 用训练集拟合的 `StandardScaler`。恶性 212（37.3%），良性 357（62.7%）。

图：`outputs/figures/01_breast_eda/`、`02_breast_preprocess/`。

**3. 三种基学习器的 AdaBoost（网格搜索，测试集）**

| 基学习器 | 最优参数 | Precision | Recall | F1 | AUC |
|----------|----------|----------:|-------:|---:|----:|
| 决策树 | max_depth=3, lr=0.5, n=200 | 1.000 | 0.929 | 0.963 | 0.991 |
| 逻辑回归 | C=10, lr=0.5, n=40 | 0.975 | 0.929 | 0.951 | 0.993 |
| 线性 SVM | C=1, lr=0.5, n=20 | 1.000 | 0.952 | 0.976 | 0.994 |

三种基学习器的 AUC 都很接近；SVM 在 Precision / Recall / F1 上略好，逻辑回归交叉验证 AUC 最高（0.996）。决策树基学习器训练集指标为 1，测试仍高，但过拟合倾向更明显。

图：`outputs/figures/03_breast_adaboost/`。表：`outputs/results/breast_adaboost_compare.csv`。

**4. 关键特征（AdaBoost-DT、RF、Lars 交集）**

三者 Top10 交集：**凹点-均值、凹点-最值、周长-最值**。综合排名还包含半径 / 面积的均值与最值、纹理-最值。也就是肿瘤大小（半径 / 周长 / 面积）和形态不规则（凹点）是影响 Breast Cancer 分类的主要因素。

图：`outputs/figures/04_breast_features/`。明细：`outputs/results/breast_key_factors.json`。

---

### 三、Boston 房价

**1–2. 探索与预处理**

506 条、13 个特征；无缺失。目标列为房价中位数 `MEDV`；**16 条房价封顶在 50，默认保留**；8:2 划分。与房价相关最强的是低收入比例 `LSTAT`（−0.74）和平均房间数 `RM`（+0.70）。

图：`outputs/figures/05_boston_eda/`、`06_boston_preprocess/`。

**3. Random Forest 参数对模型性能的影响**

最优参数：`n_estimators=100, max_depth=14, max_features=0.5, min_samples_split=2`。

测试集：**R²=0.887，MSE=8.26，MAE=1.97**。树变多、深度加大时测试误差先降后稳；`min_samples_leaf` 过大则欠拟合。

图：`outputs/figures/07_boston_rf_params/`。

**4. 单棵树 vs 集成（测试集）**

| 模型 | R² | MSE | MAE | 训练R²−测试R² |
|------|---:|----:|----:|-------------:|
| 决策树 | 0.738 | 19.19 | 2.78 | 0.107 |
| AdaBoost | 0.897 | 7.52 | 2.03 | 0.096 |
| 随机森林 | 0.887 | 8.26 | 1.97 | 0.094 |

集成比单棵树明显更准，过拟合间隙也更小；AdaBoost 测试 R² 最高，随机森林 MAE 更稳。相对单模型，集成学习降低方差、提高泛化，增加基学习器数量后测试 R² 明显高于单棵决策树。

图：`outputs/figures/08_boston_ensemble/`。表：`outputs/results/boston_ensemble_compare.csv`。

---

## 7. 输出对照

### 图片 `ensemble_lab/outputs/figures/`

| 目录 | 主要内容 |
|------|----------|
| `01_breast_eda/` | 类别平衡、分布、相关、Top 特征箱线图 / pairplot |
| `02_breast_preprocess/` | 标准化前后、划分后类别比例 |
| `03_breast_adaboost/` | 三模型指标、ROC、混淆矩阵、网格热力图 |
| `04_breast_features/` | AdaBoost / RF / Lars 重要性及并列对比 |
| `05_boston_eda/` | 房价分布、相关、Top3 散点、临河箱线图 |
| `06_boston_preprocess/` | 标准化前后、划分后 y 分布 |
| `07_boston_rf_params/` | 五个参数的 R²/MSE/MAE 曲线、树数量学习曲线 |
| `08_boston_ensemble/` | 训练/测试柱状图、过拟合间隙、预测 vs 真实、残差、特征重要性、集成规模曲线 |

### 表格 `ensemble_lab/outputs/results/`

| 文件 | 内容 |
|------|------|
| `breast_adaboost_compare.csv` | 三种基学习器测试 / 训练指标 |
| `breast_adaboost_*_cv.csv` / `*_best.json` | 各基学习器网格细节与最优参数 |
| `breast_feature_importance.csv` | 三模型特征重要性 |
| `breast_key_factors.json` | Top 特征与交集 |
| `boston_rf_param_sweeps.csv` | RF 单因素扫描 |
| `boston_rf_best.json` / `boston_rf_gridcv.csv` | RF 网格最优 |
| `boston_ensemble_compare.csv` | 决策树 / AdaBoost / RF 对比 |
| `boston_ensemble_best_params.json` | 三个模型的最优超参 |
