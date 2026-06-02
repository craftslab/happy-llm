# happy-llm

Scaled dot-product attention 实现（`attention.py`），附带单元测试。

本仓库代码参考 [Happy-LLM 教程](https://datawhalechina.github.io/happy-llm)（Datawhale 开源大模型实战教程）。

## 核心运算：Q @ K^T

`attention.py` 中以下代码计算 **Query 与 Key 的点积相似度矩阵**（缩放前的 attention scores）：

```python
raw_scores = torch.matmul(query, key.transpose(-2, -1))
```

### 张量形状

| 张量 | 形状 | 含义 |
|------|------|------|
| `query` | `(..., seq_q, d_k)` | seq_q 个 query 向量，每个 d_k 维 |
| `key` | `(..., seq_k, d_k)` | seq_k 个 key 向量，每个 d_k 维 |
| `raw_scores` | `(..., seq_q, seq_k)` | 每个 query 对每个 key 的相似度分数 |

其中 `...` 表示 batch 等前缀维度；`matmul` 只对最后两维做矩阵乘法，前面维度按广播规则对齐。

### `key.transpose(-2, -1)`

交换倒数第二维和最后一维，等价于对 key 做矩阵转置 **K^T**：

```
key:           (..., seq_k, d_k)
key.transpose: (..., d_k, seq_k)
```

### `torch.matmul` 运算

```
query          @  key.transpose(-2, -1)
(..., seq_q, d_k) @ (..., d_k, seq_k)
=                (..., seq_q, seq_k)
```

对 query 位置 `i` 与 key 位置 `j`，结果为两个向量的点积：

```
scores[i, j] = sum(query[i, t] * key[j, t] for t in range(d_k))
```

`scores[i, j]` 越大，表示第 i 个 query 与第 j 个 key 越相关；后续经 softmax 归一化后，会对第 j 个 value 赋予更高权重。

### 数值示例

```python
import torch

# seq_q=2, seq_k=3, d_k=2
query = torch.tensor([[1.0, 0.0],   # q0
                      [0.0, 1.0]])  # q1
# 形状 (2, 2)：2 个 query，每个 2 维

key = torch.tensor([[1.0, 0.0],   # k0
                    [1.0, 1.0],   # k1
                    [0.0, 1.0]])  # k2
# 形状 (3, 2)：3 个 key，每个 2 维（每行一个 key 向量）

key_t = key.transpose(-2, -1)
# 形状 (2, 3)：交换 -2 维（seq_k=3）与 -1 维（d_k=2）
# tensor([[1., 1., 0.],   ← 所有 key 的第 0 维：k0[0], k1[0], k2[0]
#         [0., 1., 1.]])  ← 所有 key 的第 1 维：k0[1], k1[1], k2[1]
# 转置后每个 key 变成一列，便于与 query 的每一行做点积

scores = torch.matmul(query, key_t)
# (2, 2) @ (2, 3) = (2, 3)
# tensor([[1., 1., 0.],
#         [0., 1., 1.]])
```

转置前后对比：

```
原始 key (3, 2)，按行读：          转置后 key_t (2, 3)，按列读：
        d=0  d=1                          k0  k1  k2
k0       1    0                    d=0    1    1    0
k1       1    1                    d=1    0    1    1
k2       0    1
```

含义：

- `scores[0, 0] = 1`：q0 与 k0 完全对齐
- `scores[0, 1] = 1`：q0 与 k1 也有重叠
- `scores[1, 2] = 1`：q1 与 k2 对齐

### 在 attention 流程中的位置

```
query (..., seq_q, d_k)
key   (..., seq_k, d_k)
         ↓
   Q @ K^T  →  raw_scores (..., seq_q, seq_k)
         ↓
   / sqrt(d_k)  →  scaled scores
         ↓
   softmax  →  attention weights
         ↓
   weights @ value  →  output
```

## 环境准备

```bash
# 创建并激活虚拟环境（可选，项目内已有 .venv 可跳过创建）
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

依赖：`torch>=2.0.0`、`pytest>=7.0.0`。

## 运行测试

在项目根目录执行：

```bash
# 运行全部测试
pytest

# 显示详细输出
pytest -v

# 只运行 attention 相关测试
pytest test_attention.py -v

# 运行单个用例
pytest test_attention.py::test_manual_computation -v
```

## 测试用例说明

| 用例 | 说明 |
|------|------|
| `test_output_shapes` | 验证输出与注意力权重的形状 |
| `test_attention_weights_sum_to_one` | 验证 softmax 后每行权重和为 1 |
| `test_manual_computation` | 简单数值场景下与手算结果一致 |
| `test_output_is_weighted_sum_of_values` | 验证 `output = weights @ value` |
| `test_with_dropout` | 验证训练模式下 dropout 会改变注意力权重 |

## 调试日志

`attention()` 使用 Python `logging` 输出 DEBUG 级别日志（形状、统计量、dropout 状态等）。查看日志示例：

```python
import logging
import torch
from attention import attention

logging.basicConfig(level=logging.DEBUG, format="%(name)s | %(levelname)s | %(message)s")

q = torch.randn(1, 3, 4)
k = torch.randn(1, 4, 4)
v = torch.randn(1, 4, 6)
attention(q, k, v)
```

运行测试时若需同时看到日志，可加上 pytest 的 log 参数：

```bash
pytest test_attention.py -v --log-cli-level=DEBUG
```
