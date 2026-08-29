# Clustering-Models

鸢尾花与模拟数据上的聚类实验：探索性分析、四种算法对比，以及 K-Means / DBSCAN / DPC 的参数影响。

## 环境

```bash
pip install -r requirements.txt
python run_all.py
```

也可分步运行 `experiments/` 下脚本。跑完后会根据表格自动生成 **[实验报告.md](实验报告.md)**（含指标与插图）。

## 目录

```
src/                 数据、DPC 实现、评价指标、绘图
experiments/         各实验脚本
outputs/figures/     图
outputs/tables/      表
```

## 实验内容

1. 鸢尾花探索性分析与预处理（聚类用 **MinMax**，不用 z-score：后者会抬高萼片宽度、拉低与真实三类的一致性）。
2. 四种算法（K-Means、层次聚类、DBSCAN、DPC）在鸢尾花上用**兰德系数**与**轮廓系数**对比。散点图颜色已对齐真实类。
3. 参数说明：K-Means 的 `k`；DBSCAN 的 `eps`、`min_samples`；DPC 的 `t0`（截断圆内样本数占总样本数的比例，用于确定 `dc`）。
4. 在鸢尾花上扫描 `k`，结合肘部法与轮廓系数选 `k`。
5. 在高斯簇、螺旋、同心圆数据上扫描 `eps`、`min_samples`、`t0`。

## 算法与默认设置

| 算法 | 主要参数 | 鸢尾花默认 |
|------|----------|------------|
| K-Means | `k` 簇数 | 3 |
| 层次聚类（Ward） | 簇数 | 3 |
| DBSCAN | `eps` 邻域半径；`min_samples` 核心点最少邻居 | 0.13 / 5（MinMax 后） |
| DPC | `t0` → 距离分位数作为 `dc` | 0.08，簇数 3 |

## 评价指标

- **兰德系数 RI**、**调整兰德系数 ARI**：与真实标签的一致性（有标签时）。
- **轮廓系数**：簇内紧凑、簇间分离（无标签也可用）。DBSCAN 的噪声点（标签 -1）在计算轮廓时被排除。
