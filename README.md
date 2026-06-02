# happy-llm

Scaled dot-product attention 实现（`attention.py`），附带单元测试。

本仓库代码参考 [Happy-LLM 教程](https://datawhalechina.github.io/happy-llm)（Datawhale 开源大模型实战教程）。

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
