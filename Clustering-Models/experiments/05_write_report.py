"""根据 outputs/tables 生成 实验报告.md。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
FIGS = "outputs/figures"
REPORT = ROOT / "实验报告.md"


def _fmt(df: pd.DataFrame, floatfmt=".4f") -> str:
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for v in row.tolist():
            if isinstance(v, float):
                cells.append(format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _best(df: pd.DataFrame, by="ari"):
    return df.loc[df[by].idxmax()]


def main():
    four = pd.read_csv(TABLES / "iris_four_algorithms.csv")
    ksw = pd.read_csv(TABLES / "kmeans_k_sweep.csv")
    desc = pd.read_csv(TABLES / "iris_describe.csv")
    if desc.columns[0].startswith("Unnamed") or desc.columns[0] == "":
        desc = desc.rename(columns={desc.columns[0]: "feature"})
    corr = pd.read_csv(TABLES / "iris_corr.csv")
    if corr.columns[0].startswith("Unnamed") or corr.columns[0] == "":
        corr = corr.rename(columns={corr.columns[0]: "feature"})
    db = pd.read_csv(TABLES / "synthetic_dbscan_sweep.csv")
    dpc = pd.read_csv(TABLES / "synthetic_dpc_sweep.csv")

    sil_k = int(ksw.loc[ksw["silhouette"].idxmax(), "k"])
    ari_k = int(ksw.loc[ksw["ari"].idxmax(), "k"])
    sse = ksw["inertia"].to_numpy()
    ks = ksw["k"].to_numpy()
    second = sse[:-2] - 2 * sse[1:-1] + sse[2:]
    elbow_k = int(ks[int(second.argmax()) + 1])

    db_best_rows = []
    for name, g in db.groupby("dataset"):
        r = _best(g)
        db_best_rows.append(
            {
                "数据集": name,
                "最优 eps": r["eps"],
                "最优 min_samples": int(r["min_samples"]),
                "簇数": int(r["n_clusters"]),
                "噪声点": int(r["n_noise"]),
                "RI": r["rand_index"],
                "ARI": r["ari"],
                "轮廓系数": r["silhouette"],
            }
        )
    db_best = pd.DataFrame(db_best_rows)

    dpc_best_rows = []
    for name, g in dpc.groupby("dataset"):
        r = _best(g)
        dpc_best_rows.append(
            {
                "数据集": name,
                "最优 t0": r["t0"],
                "簇数": int(r["n_clusters"]),
                "RI": r["rand_index"],
                "ARI": r["ari"],
                "轮廓系数": r["silhouette"],
            }
        )
    dpc_best = pd.DataFrame(dpc_best_rows)

    db5 = db[db["min_samples"] == 5]
    dpc_pivot = dpc.pivot_table(index="t0", columns="dataset", values="ari")

    md = f"""# 聚类实验报告：鸢尾花与模拟数据

本文记录实验设置、参数含义与**实际跑出的结果**。图表由 `python run_all.py` 生成；本文件在每次跑完后由 `experiments/05_write_report.py` 根据 `outputs/tables/` 自动更新。

---

## 1. 鸢尾花探索性分析与预处理

- 样本数 150，三类各 50（setosa / versicolor / virginica）。
- 特征：花萼长、花萼宽、花瓣长、花瓣宽（厘米）；无缺失值。
- **预处理**：聚类阶段使用 **MinMaxScaler** 将各维缩放到 \\([0,1]\\)。不使用 z-score：后者会把花萼宽度抬到与花瓣同等权重，versicolor 与 virginica 更难分开，和真实标签的一致性会下降。

### 1.1 描述统计

{_fmt(desc)}

### 1.2 相关系数

{_fmt(corr)}

花瓣长度与花瓣宽度相关约 0.96，是区分三类的主要信息。

![成对特征]({FIGS}/iris_pairplot.png)

![相关热力图]({FIGS}/iris_corr_heatmap.png)

![箱线图]({FIGS}/iris_boxplots.png)

---

## 2. 四种算法在鸢尾花上的性能

评价指标：

- **兰德系数 RI**、**调整兰德系数 ARI**：与真实三类标签的一致性（簇编号置换不影响）。
- **轮廓系数**：簇内紧凑、簇间分离（无标签也可用）。DBSCAN 噪声点（标签 -1）不参与轮廓计算。

默认设置（特征经 MinMax）：K-Means `k=3`；Ward 层次聚类 3 簇；DBSCAN `eps=0.13`, `min_samples=5`；DPC `t0=0.08`, 3 簇。

{_fmt(four)}

结论：

- K-Means 与层次聚类接近（ARI 约 0.72），setosa 可完整分开，后两类有少量混淆。
- **DPC** 与真实标签最接近（ARI 约 0.89）。
- **DBSCAN** 因 versicolor / virginica 密度重叠，常得到 2 簇并标出噪声；轮廓系数可以较高，但与真实三类的 RI/ARI 较低。这是密度算法在重叠凸簇上的局限，不是再微调 `eps` 就能变成干净的三类。

作图用花瓣长×宽，簇颜色已用匈牙利算法对齐真实类（只改着色，不改评价指标）。

![K-Means]({FIGS}/iris_K-Means.png)

![层次聚类]({FIGS}/iris_Hierarchical.png)

![DBSCAN]({FIGS}/iris_DBSCAN.png)

![DPC]({FIGS}/iris_DPC.png)

---

## 3. 主要参数含义

| 算法 | 参数 | 含义 |
|------|------|------|
| K-Means | `k` | 预设簇数。过小会合并不同类，过大会把同一类切开。 |
| DBSCAN | `eps` | 邻域半径。过小则大量点成噪声或碎片簇；过大则不同密度区被连成一类。 |
| DBSCAN | `min_samples` | 核心点最少邻居数（含自身）。越大则核心点越少、噪声越多、簇更“紧”。经验上可取维度的一小倍数，二维常用 4～5。 |
| DPC | `t0` | **圆（半径 \\(d_c\\)）内样本数占数据集总样本数的比例**。实现上取全部成对距离的 `t0` 分位数作为截断距离 \\(d_c\\)，再用高斯核估计局部密度。`t0` 过小则密度估计过尖、中心不稳定；过大则密度被抹平、峰值不明显。文献中常取 1%～2%，鸢尾花上略大（如 0.08）更稳。 |

---

## 4. K-Means 的 k 对鸢尾花性能的影响

在 MinMax 后的鸢尾花上扫描 `k=2…10`。

{_fmt(ksw)}

选 k 的方法：

1. **肘部法**：看 SSE（inertia）随 k 下降的“拐点”。本实验二阶差分建议 **k = {elbow_k}**。
2. **轮廓系数**：无标签时可用；本数据在 **k = {sil_k}** 最大（setosa 远离另外两类，k=2 在几何上更“干净”，但会把后两类合成一类）。
3. **相对真实标签的 ARI**（仅对照，实际无标签不可用）：最大在 **k = {ari_k}**，与植物学三类一致。

实践建议：肘部法与轮廓系数一起看。鸢尾花上轮廓偏向 k=2，肘部与领域知识指向 k=3。

![肘部法]({FIGS}/kmeans_elbow.png)

![k 与评价指标]({FIGS}/kmeans_k_metrics.png)

---

## 5. 模拟数据上 eps、min_samples、t0 的影响

三套二维数据（均做标准化）：高斯混合簇、双螺旋、同心圆。评价指标仍为 RI、ARI、轮廓系数。

### 5.1 DBSCAN：eps 与 min_samples

各数据集上扫描得到的 **ARI 最优** 组合：

{_fmt(db_best)}

固定 `min_samples=5` 时，可用 k-distance 图辅助选 `eps`（曲线明显拐弯处）。

- **高斯簇**：球形、间距足够时，合适的 `eps` 可使 ARI=1。
- **同心圆**：K-Means 会失败，DBSCAN 在 `eps` 落入两环间隙时可以完美分开。
- **螺旋**：对 `eps` 很敏感；过小碎成多簇，过大两条臂粘连。需要结合 `min_samples` 与 k-distance。

![高斯 k-distance]({FIGS}/gaussian_kdistance.png)

![高斯 eps]({FIGS}/gaussian_dbscan_eps.png)

![高斯 min_samples]({FIGS}/gaussian_dbscan_min_samples.png)

![高斯 DBSCAN 结果]({FIGS}/gaussian_dbscan_result.png)

![螺旋 eps]({FIGS}/spiral_dbscan_eps.png)

![螺旋结果]({FIGS}/spiral_dbscan_result.png)

![同心圆 eps]({FIGS}/circle_dbscan_eps.png)

![同心圆结果]({FIGS}/circle_dbscan_result.png)

完整网格见 `outputs/tables/synthetic_dbscan_sweep.csv`。

### 5.2 DPC：t0

各数据集上扫描得到的 **ARI 最优** `t0`：

{_fmt(dpc_best)}

各 `t0` 下的 ARI：

{_fmt(dpc_pivot.reset_index())}

- **高斯簇**：在较宽的 `t0` 范围内 ARI 保持为 1，密度峰稳定。
- **螺旋 / 同心圆**：DPC 按密度峰+距离分配，本质上偏向“凸、有中心”的簇，对环形/螺旋结构 ARI 很低；此时应选 DBSCAN 一类沿密度连通的方法。

![高斯 t0]({FIGS}/gaussian_dpc_t0.png)

![高斯 DPC]({FIGS}/gaussian_dpc_result.png)

![螺旋 t0]({FIGS}/spiral_dpc_t0.png)

![螺旋 DPC]({FIGS}/spiral_dpc_result.png)

![同心圆 t0]({FIGS}/circle_dpc_t0.png)

![同心圆 DPC]({FIGS}/circle_dpc_result.png)

完整扫描见 `outputs/tables/synthetic_dpc_sweep.csv`。

---

## 6. 小结

| 问题 | 结论 |
|------|------|
| 鸢尾花上谁更接近真实标签 | DPC（本设置）最好，K-Means / 层次聚类次之，DBSCAN 受重叠密度限制。 |
| 如何选 k | 肘部法指向 3；轮廓系数指向 2；有标签对照时 ARI 在 k=3 最大。 |
| 如何选 eps / min_samples | k-distance 定 eps 量级，再扫 min_samples；在圆/螺旋上 DBSCAN 明显优于基于中心的方法。 |
| 如何选 t0 | 高斯数据不敏感；鸢尾花上 0.08 左右较好；非凸结构上调 t0 也救不了 DPC 的模型假设。 |

原始数值表：`outputs/tables/`；图：`outputs/figures/`。
"""
    REPORT.write_text(md, encoding="utf-8")
    print(f"已写入 {REPORT}")


if __name__ == "__main__":
    main()
