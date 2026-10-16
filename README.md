# 推荐序列预测任务：基于历史选择预测未来偏好顺序

这个项目实现一个“推荐序列预测”实验：

> 给定用户过去 `history_hours` 小时内选择/购买过的商品序列，预测用户未来 `future_hours` 小时内可能选择的商品 Top-K 顺序。

数据集选用 **UCI Online Retail**：

- 数据集页面：<https://archive.ics.uci.edu/dataset/352/online+retail>
- 数据集 DOI：<https://doi.org/10.24432/C5BW33>
- 数据规模：541,909 条交易记录
- 时间范围：2010-12-01 到 2011-12-09
- 数据类型：multivariate、sequential、time-series
- 许可：CC BY 4.0

UCI 页面说明该数据集包含一个英国在线零售商在 2010-12-01 到 2011-12-09 之间的所有交易记录，字段包含 `InvoiceNo`、`StockCode`、`Description`、`Quantity`、`InvoiceDate`、`UnitPrice`、`CustomerID`、`Country`。

## 1. 这个任务中的“历史”和“未来”

在推荐序列预测中，“历史”和“未来”按时间切分：

```text
某个用户在过去 72 小时内选择过：
商品 A -> 商品 B -> 商品 C

模型需要预测：
未来 24 小时内这个用户最可能选择哪些商品，以及它们的推荐顺序。
```

在本项目中：

```text
历史 = anchor_time 前 history_hours 小时内的用户商品序列
未来 = anchor_time 后 future_hours 小时内用户真实购买/选择的商品集合
```

例如：

```bash
--history-hours 168 --future-hours 24
```

表示：

```text
用过去 7 天的商品选择历史，预测未来 1 天的商品偏好。
```

## 2. 它和普通分类/回归有什么不同

这个任务不是预测一个数值，也不是只预测一个类别，而是预测一个 **Top-K 商品列表**。

例如模型输出：

```text
Top-5 推荐：商品 85123A, 商品 22423, 商品 85099B, 商品 20725, 商品 47566
```

真实未来购买：

```text
商品 22423, 商品 20725
```

那么这个推荐结果命中了 2 个商品。

## 3. 本项目实现的模型

代码支持 3 种模型：

| 模型 | 思路 | 适合性 |
|---|---|---|
| MLP | 将历史商品序列 embedding 后展平，再预测未来商品分布 | 简单基线，训练快 |
| LSTM | 带门控记忆单元的序列模型，按时间顺序读取商品序列 | 适合建模中短期序列兴趣 |
| SASRec | 基于 self-attention 的序列推荐模型 | 更适合长序列和“哪些历史行为更重要”的建模 |

其中 `SASRec` 是本项目里最接近现代推荐系统序列模型的方案。它会给商品序列加入位置编码，并使用因果注意力 mask，保证模型只能利用历史商品，不能偷看未来商品。

## 4. 本项目实现的优化器

代码支持 4 种优化器：

| 优化器 | 特点 |
|---|---|
| SGD | 经典随机梯度下降，带 momentum |
| Adam | 自适应学习率，深度学习中常用 |
| RMSprop | 对非平稳目标较稳 |
| Adagrad | 对稀疏特征较友好 |

本项目将训练脚本拆分为三个模型专用文件，每个文件内置 4 种优化器（SGD / Adam / RMSprop / Adagrad）的对比：

| 训练文件 | 模型 | 默认优化器配置 |
|---|---|---|
| `train_mlp.py` | MLP | 4 种优化器各跑一遍 |
| `train_lstm.py` | LSTM | 4 种优化器各跑一遍 |
| `train_sasrec.py` | SASRec | 4 种优化器各跑一遍 |

公共的数据加载、预处理、评估指标、可视化逻辑放在 `recommender_utils.py` 中，三个训练文件共享。

优化器定义在 `recommender_utils.py` 的 `build_optimizer()`：

```python
def build_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
```

如果要新增优化器，例如 `AdamW`，就在 `recommender_utils.py` 的 `build_optimizer()` 中增加：

```python
if name == "adamw":
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
```

## 5. 安装依赖

```bash
pip install -r requirements.txt
```

如果没有 GPU，CPU 也可以运行。第一次运行会自动下载 UCI Online Retail Excel 文件，约 22.6 MB。

如果网络下载失败，可以手动下载：

<https://archive.ics.uci.edu/dataset/352/online+retail>

然后把文件放到：

```text
data/Online Retail.xlsx
```

## 6. 运行实验

### 6.1 跑单个模型（自动对比 4 种优化器）

```bash
python train_mlp.py
python train_lstm.py
python train_sasrec.py
```

每个命令会依次用 **SGD、Adam、RMSprop、Adagrad** 训练同一模型，并生成对比图表。

### 6.2 只跑部分优化器

如果只想对比 Adam 和 RMSprop：

```bash
python train_mlp.py --optimizers adam,rmsprop
```

### 6.3 快速演示（小样本）

```bash
python train_mlp.py --max-samples 2000 --epochs 10
python train_lstm.py --max-samples 2000 --epochs 10
python train_sasrec.py --max-samples 2000 --epochs 10
```

### 6.4 各模型推荐超参数

三个脚本已经为对应模型配置了较合适的默认超参数。关键区别如下：

| 超参数 | MLP | LSTM | SASRec |
|---|---|---|---|
| `--embedding-dim` | 64 | 64 | 64 |
| `--hidden-dim` | 128 | 128 | 64 |
| `--num-layers` | 2（FC） | 2 | 2（Attention） |
| `--dropout` | 0.2 | 0.3 | 0.2 |
| `--max-seq-len` | 30 | 50 | 50 |
| `--lr` | 1e-3 | 1e-3 | 5e-4 |
| `--batch-size` | 256 | 256 | 128 |
| `--epochs` | 50 | 50 | 80 |
| `--patience` | 6 | 6 | 10 |

SASRec 因为 Transformer 收敛更慢，设置了更多的 epoch 和更宽松的 early stopping patience。

注意：`--embedding-dim` 必须能被 `--num-heads` 整除（SASRec 默认 `--num-heads 4`）。

## 7. 输出文件

默认输出到：

```text
outputs/
```

会生成：

| 文件 | 作用 |
|---|---|
| `metrics.csv` | 所有模型/优化器的评估指标 |
| `history_*.csv` | 每次实验的训练/验证损失 |
| `recommendations_*.csv` | 测试集样本的 Top-K 推荐结果 |
| `metrics_recall_ndcg.png` | Recall@K 和 NDCG@K 对比柱状图 |
| `metrics_precision_hit_rate.png` | Precision@K 和 HitRate@K 对比柱状图 |
| `loss_curves.png` | 不同实验的验证损失曲线 |

## 8. 评估指标

推荐任务不太适合用普通分类准确率，因为模型输出的是一个排序列表。这里使用 Top-K 推荐指标：

| 指标 | 含义 | 越大/越小 |
|---|---|---|
| Precision@K | 推荐的 K 个商品中，有多少比例被用户真实选择 | 越大越好 |
| Recall@K | 用户真实选择的商品中，有多少比例被 Top-K 命中 | 越大越好 |
| HitRate@K | Top-K 是否至少命中一个真实商品 | 越大越好 |
| NDCG@K | 命中的商品排得越靠前，分数越高 | 越大越好 |
| MAP@K | 综合考虑多个命中商品的排序质量 | 越大越好 |

如果你想要类似“准确率”的表达，推荐任务里最接近的是：

```text
HitRate@K：推荐列表里是否至少有一个商品是用户未来真实选择的商品。
```

## 9. 任务建模方式

每个训练样本长这样：

```text
输入 X：
用户在历史窗口内的商品序列
[item_12, item_87, item_12, item_304, ...]

目标 y：
用户在未来窗口内真实选择的商品集合
{item_87, item_431}
```

模型输出所有候选商品的分数：

```text
score(item_1), score(item_2), ..., score(item_N)
```

然后取分数最高的 Top-K 作为推荐结果。

