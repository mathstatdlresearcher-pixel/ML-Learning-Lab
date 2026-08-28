from __future__ import annotations

import numpy as np
import pandas as pd

from titanic_ml.config.settings import ROOT


def fmt_md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    idx_name = df.index.name or "模型"
    header = "| " + idx_name + " | " + " | ".join(map(str, cols)) + " |"
    sep = "| --- | " + " | ".join(["---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in cols]) + " |"
    lines = [header, sep]
    for i, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append(f"{v:{floatfmt}}")
            else:
                cells.append(str(v))
        lines.append("| " + str(i) + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown_report(payload: dict, val_df, test_df, train_df, cv_df, rank_val, rank_test):
    bp_lines = []
    for name, params in payload["best_params"].items():
        items = ", ".join(f"`{k}`={v}" for k, v in params.items())
        bp_lines.append(f"- **{name}**：{items}")

    fill_lines = []
    for col, info in payload["fill_log"].items():
        fill_lines.append(f"- `{col}`：{info}")

    md = f"""# Titanic 生存预测实验报告

本报告汇总探索性分析、预处理、特征工程，以及 **决策树 / 逻辑回归 / AdaBoost / 随机森林 / GBDT / XGBoost** 的网格搜索与评价指标。

复现命令：`python run.py`（代码按文件夹拆分，见 `titanic_ml/`）。

## 1. 为什么官方 test.csv 不能直接算 Precision / Recall / F1 / AUC

Kaggle 文件 `titanic-dataset/test.csv` **没有 `Survived` 列**（418 条，仅特征）。没有真实标签时，任何分类指标都无法计算，只能输出预测概率与 0/1。

因此采用如下划分（全部随机种子 `{payload['split']['random_state']}`，分层抽样）：

| 数据集 | 来源 | 样本数 | 用途 |
| --- | --- | ---: | --- |
| 训练集 | `train.csv` 的 80% 再按 80/20 切开后的训练部分 | {payload['split']['train']} | 拟合 + 5 折网格搜索 |
| 验证集 | 上述 80% 中的 20% | {payload['split']['val']} | 调参后对照、选主模型 |
| **带标签测试集** | `train.csv` 预先留出的 20% | {payload['split']['labeled_test']} | **最终 Precision / Recall / F1 / AUC** |
| Kaggle 测试集 | `test.csv` | {payload['split']['kaggle_test']} | 无标签，只保存预测 |

也就是：先从 `train.csv` 划出 20% 作为**带标签测试集**（不参与训练与网格搜索），剩余 80% 再按 80%/20% 划分为训练集与验证集。

---

## 2. 探索性分析

全量 `train.csv` 共 {payload['eda']['n_rows']} 条，生存率 **{payload['eda']['survival_rate']:.2%}**。

### 2.1 缺失

| 字段 | 缺失比例 | 处理 |
| --- | ---: | --- |
| Cabin | {payload['eda']['missing'].get('Cabin', 0):.2%} | 超过 50%，删除原列；保留 `HasCabin` / `CabinDeck` |
| Age | {payload['eda']['missing'].get('Age', 0):.2%} | 连续型，拉格朗日插值 |
| Embarked | {payload['eda']['missing'].get('Embarked', 0):.2%} | 文字型，众数填充 |

![缺失](figures/01_missing_ratio.png)

### 2.2 类别与交叉

女性生存率 **{payload['sex_survival']['female']:.1%}**，男性 **{payload['sex_survival']['male']:.1%}**；一等舱 **{payload['pclass_survival']['1']:.1%}**，二等 **{payload['pclass_survival']['2']:.1%}**，三等 **{payload['pclass_survival']['3']:.1%}**。

![类别](figures/02_survival_by_cat.png)
![性别舱位](figures/07_sex_pclass_heatmap.png)
![堆叠](figures/17_stacked_pclass_sex.png)

### 2.3 连续变量与称谓

![年龄票价](figures/03_age_fare_dist.png)
![箱线港口](figures/04_box_embarked.png)
![小提琴](figures/16_violin_box.png)
![相关](figures/06_corr_heatmap.png)
![称谓家庭](figures/05_title_family.png)
![家庭分箱](figures/18_family_bin.png)
![甲板](figures/19_cabin_deck.png)
![插值分箱](figures/14_age_after_impute_bin.png)
![年龄箱生存](figures/26_agebin_survival.png)

---

## 3. 预处理与特征工程

规则：缺失 >50% 删列；文字缺失用**训练集众数**；连续缺失用**拉格朗日插值**（邻域 k=5，Age 裁剪到 [0,80]）。分箱与哑变量边界只在训练集上估计。

实际填充记录：

{chr(10).join(fill_lines) if fill_lines else "- （划分后训练集上无额外需填列，或已在特征生成后处理）"}

哑变量列：`{', '.join(payload['dummy_cols'])}`

最终特征维数：**{payload['n_features']}**。列名见 `outputs/run_summary.json`。

---

## 4. 网格搜索最优参数

5 折 `StratifiedKFold`（`shuffle=True, random_state=2`），`refit=roc_auc`，同时记录 CV 上的 Precision / Recall / F1 / Accuracy。

{chr(10).join(bp_lines)}

完整靠前组合见 `outputs/gridsearch_top.csv`。

### 4.1 最优参数的交叉验证多指标

{fmt_md_table(cv_df)}

![CV 多指标](figures/27_cv_multi_metrics.png)

---

## 5. 验证集指标（选模型用，不偷看测试集）

{fmt_md_table(val_df)}

按验证集 F1 选出的主模型：**{payload['winner_by_val_f1']}**。

![验证柱状](figures/val_metrics_bar.png)
![验证热力](figures/val_metrics_heatmap.png)
![验证 ROC](figures/val_roc.png)
![验证 PR](figures/val_pr.png)
![验证混淆](figures/val_confusion.png)
![验证雷达](figures/val_radar.png)

验证集指标名次（1 为最好）：

{fmt_md_table(rank_val, floatfmt="d")}

---

## 6. 带标签测试集指标（最终评价）

以下才是作业要求的 **Precision、Recall、F1、AUC** 在测试集上的结果。

{fmt_md_table(test_df)}

按测试集 F1 排序最优：**{payload['best_on_test_f1']}**（仅作对照；正式选型以验证集为准）。

![测试柱状](figures/test_metrics_bar.png)
![测试热力](figures/test_metrics_heatmap.png)
![测试 ROC](figures/test_roc.png)
![测试 PR](figures/test_pr.png)
![测试混淆](figures/test_confusion.png)
![测试雷达](figures/test_radar.png)
![三阶段 AUC](figures/21_train_val_test_auc.png)
![CV vs 验证 vs 测试](figures/15_cv_val_test_auc.png)

测试集指标名次：

{fmt_md_table(rank_test, floatfmt="d")}

### 6.1 训练集（拟合优度，不作选型）

{fmt_md_table(train_df)}

---

## 7. 方法对比要点

- **决策树**：可解释、网格含深度与叶节点；容易过拟合（训练 AUC 明显高于测试）。
- **逻辑回归**：线性基线，L1/L2 与 `C`、类别权重网格；特征已标准化。适合对照「非线性集成是否值得」。
- **AdaBoost**：加权弱分类器，Recall 往往较高。
- **随机森林**：袋装降低方差，验证/测试 AUC 通常稳定。
- **GBDT**：梯度提升，对表格式竞赛很强。
- **XGBoost**：二阶近似 + 列采样，需看训练/测试差距判断过拟合。

![校准](figures/22_calibration_test.png)
![一致率](figures/23_pred_agreement_test.png)
![重要性](figures/12_feature_importance.png)
![置换重要性](figures/24_permutation_importance.png)
![学习曲线](figures/25_learning_curves.png)

---

## 8. Kaggle 无标签测试集预测

无法计算分类指标。各模型预测见：

- `outputs/kaggle_predictions_all_models.csv`
- 验证集 F1 最优模型单独文件：`outputs/test_predictions.csv`（模型 = {payload['winner_by_val_f1']}）

![Kaggle 预测分布](figures/28_kaggle_pred_dist.png)

Kaggle 测试集上主模型预测生存比例：{payload['kaggle_pred_rate']:.1%}。

---

## 9. 产出清单

| 路径 | 说明 |
| --- | --- |
| `run.py` | 一键复现入口 |
| `titanic_ml/` | 分文件夹源码（数据 / 模型 / 可视化 / 报告） |
| `figures/` | EDA 与模型对比图（热力图格子内均有数值，左侧标签横排） |
| `outputs/val_metrics.csv` | 验证集指标 |
| `outputs/test_metrics.csv` | **带标签测试集指标** |
| `outputs/train_metrics.csv` | 训练集指标 |
| `outputs/cv_multi_metrics.csv` | 网格最优参数的 CV 多指标 |
| `outputs/gridsearch_top.csv` | 各模型网格搜索前列 |
| `outputs/metric_ranks_val.csv` / `metric_ranks_test.csv` | 名次 |
| `outputs/run_summary.json` | 参数与划分汇总 |
| `outputs/kaggle_predictions_all_models.csv` | 六模型对 test.csv 的预测 |
| `实验结果报告.md` | 本文件 |
"""
    (ROOT / "实验结果报告.md").write_text(md, encoding="utf-8")
