# happy-llm

本仓库代码参考 [Happy-LLM 教程](https://datawhalechina.github.io/happy-llm)。

## 项目结构

| 文件 | 说明 |
|------|------|
| `attention.py` | 单头 scaled dot-product attention 函数 |
| `multihead_attention.py` | `MultiHeadAttention` 模块与 `ModelArgs` 配置 |
| `encoder.py` | `LayerNorm`、`MLP`、`EncoderLayer` 与 `Encoder` 模块 |
| `decoder.py` | `DecoderLayer` 与 `Decoder` 模块 |
| `transformer.py` | `PositionalEncoding` 与完整 `Transformer` 模型 |
| `test_attention.py` | `attention()` 单元测试 |
| `test_multihead_attention.py` | `MultiHeadAttention` 单元测试 |
| `test_encoder.py` | `EncoderLayer`、`Encoder` 及相关模块单元测试 |
| `test_decoder.py` | `DecoderLayer`、`Decoder` 单元测试 |
| `test_transformer.py` | `Transformer` 与 `PositionalEncoding` 单元测试 |

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

## 多头注意力：MultiHeadAttention

`multihead_attention.py` 在单头 attention 基础上，将 Q/K/V 拆成多个头并行计算，再拼接并投影回模型维度。

### 基本用法

```python
import torch
from multihead_attention import ModelArgs, MultiHeadAttention

args = ModelArgs(dim=64, n_heads=8, dropout=0.1, max_seq_len=512)
mha = MultiHeadAttention(args, is_causal=True)  # is_causal=True 启用因果 mask

x = torch.randn(2, 16, 64)  # (batch, seq_len, dim)
out = mha(x, x, x)          # 自注意力，输出形状 (2, 16, 64)
```

### 配置参数（ModelArgs）

| 参数 | 说明 |
|------|------|
| `dim` | 模型隐藏维度，必须能被 `n_heads` 整除 |
| `n_heads` | 注意力头数 |
| `n_embd` | 输入嵌入维度，默认等于 `dim` |
| `n_layer` | Encoder / Decoder 堆叠层数，默认 `1` |
| `vocab_size` | 词表大小，构建 `Transformer` 时必填 |
| `block_size` | 最大序列长度，用于位置编码与输入校验 |
| `dropout` | attention 与残差 dropout 概率 |
| `max_seq_len` | 因果 mask 的最大序列长度 |

### 前向流程

```
输入 q, k, v  (B, T, dim)
      ↓
Linear 投影 Wq, Wk, Wv
      ↓
拆分为多头  (B, n_heads, T, head_dim)
      ↓
Q @ K^T / sqrt(head_dim)  →  softmax  →  dropout
      ↓
weights @ V
      ↓
拼接多头  (B, T, dim)
      ↓
Linear 投影 Wo  →  resid_dropout  →  输出
```

其中 `head_dim = dim // n_heads`。以下分步说明 `multihead_attention.py` 中的关键实现。

### Q/K/V/Wo 线性投影

`__init__` 中定义四个线性层，将输入投影为 Q/K/V，并在多头拼接后再映射回模型维度：

```python
self.wq = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
self.wk = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
self.wv = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)
```

| 层 | 作用 | 输入 → 输出 |
|----|------|-------------|
| `wq` | 投影为 Query | `(n_embd)` → `(n_heads × head_dim)` |
| `wk` | 投影为 Key | `(n_embd)` → `(n_heads × head_dim)` |
| `wv` | 投影为 Value | `(n_embd)` → `(n_heads × head_dim)` |
| `wo` | 融合多头输出 | `(n_heads × head_dim)` → `(dim)` |

数值示例（`dim=8`, `n_heads=2`, `head_dim=4`）：

```python
x.shape = (1, 3, 8)    # 3 个 token，每个 8 维

xq = wq(x)             # (1, 3, 8)
# 最后一维 8 = 2 个头 × 每头 4 维，可拆为：
#   头0: xq[..., 0:4]
#   头1: xq[..., 4:8]
```

`wq/wk/wv` 用一个大矩阵 `(n_embd, n_heads × head_dim)` 等价于多个头的投影矩阵拼接，一次矩阵乘法即可算出所有头的 Q/K/V。`wo` 则在多头 attention 完成后，将拼接结果再线性变换回 `dim` 维。

### 拆分为多头：`view` + `transpose`

投影后的 `xq/xk/xv` 形状为 `(B, T, dim)`，需拆成 `(B, n_heads, T, head_dim)` 以便每个头独立计算 attention：

```python
xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
xk = xk.view(bsz, seqlen_k, self.n_heads, self.head_dim).transpose(1, 2)
xv = xv.view(bsz, seqlen_k, self.n_heads, self.head_dim).transpose(1, 2)
```

以 `B=1, T=3, n_heads=2, head_dim=2, dim=4` 为例：

| 步骤 | 形状 | 含义 |
|------|------|------|
| `wq(x)` 后 | `(1, 3, 4)` | 3 个 token，每个 4 维（2 头拼在一起） |
| `.view(1, 3, 2, 2)` | `(1, 3, 2, 2)` | 每个 token 拆成 2 个头，每头 2 维 |
| `.transpose(1, 2)` | `(1, 2, 3, 2)` | 按头分组，每头含 3 个 token 向量 |

`view` 把 `[a0, a1, b0, b1]` 拆成头0 `[a0, a1]` 与头1 `[b0, b1]`；`transpose(1, 2)` 从「先 token 后头」变为「先头后 token」，使每个头形成独立的 `(T, head_dim)` 矩阵，便于后续 `matmul`：

```python
raw_scores = torch.matmul(xq, xk.transpose(2, 3))
# (1, 2, 3, 2) @ (1, 2, 2, 3) = (1, 2, 3, 3)
#  每个头独立计算 3×3 的 attention 分数
```

可运行示例：

```python
import torch

B, T, n_heads, head_dim = 1, 3, 2, 2
xq = torch.arange(B * T * n_heads * head_dim, dtype=torch.float).view(B, T, -1)
xq = xq.view(B, T, n_heads, head_dim).transpose(1, 2)

print(xq.shape)       # torch.Size([1, 2, 3, 2])
print("头0:\n", xq[0, 0])   # token0~2 在头0的向量
print("头1:\n", xq[0, 1])   # token0~2 在头1的向量
```

### 因果 mask（Causal Mask）

当 `is_causal=True` 时，在 `__init__` 中构造上三角 mask，防止 token  attend 到未来位置：

```python
mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
mask = torch.triu(mask, diagonal=1)
self.register_buffer("mask", mask)
```

`max_seq_len=4` 时，mask 矩阵（去掉前两维 broadcast 维）为：

```
        key: 0    1    2    3
query 0    0  -inf -inf -inf
      1    0    0  -inf -inf
      2    0    0    0  -inf
      3    0    0    0    0
```

- `mask[i, j] = 0`：query 位置 `i` 可以 attend 到 key 位置 `j`（`j ≤ i`）
- `mask[i, j] = -inf`：禁止 attend 到未来 token（`j > i`）

前向时将 mask 加到 scores 上，再 softmax；`-inf` 位置的权重变为 0：

```python
scores = scores + self.mask[:, :, :seqlen, :seqlen_k]
```

示例：softmax 前 `scores[0] = [2.0, 1.0, 0.5]`，加上 mask `[0, -inf, -inf]` 后，softmax 结果约为 `[1.0, 0, 0]`，即 token 0 只能看自己。

`register_buffer` 将 mask 注册为模型缓冲区：不参与训练，但会随模型保存/加载并自动迁移到 GPU。

### 拼接多头、输出投影与 resid_dropout

每个头算完 `weights @ V` 后得到 `head_output`，形状为 `(B, n_heads, T, head_dim)`。随后三行代码完成输出收尾：

```python
output = head_output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)  # 拼接多头
output = self.wo(output)                                                  # 输出投影
output = self.resid_dropout(output)                                       # 残差 dropout
```

#### 拼接多头：`transpose` + `contiguous` + `view`

以 `B=1, T=3, n_heads=2, head_dim=2, dim=4` 为例，`head_output.shape = (1, 2, 3, 2)`：

```
头0: token0 [1, 2]   token1 [3, 4]   token2 [5, 6]
头1: token0 [7, 8]   token1 [9, 10]  token2 [11, 12]
```

| 步骤 | 形状 | 含义 |
|------|------|------|
| `head_output` | `(1, 2, 3, 2)` | 先按头、再按 token |
| `.transpose(1, 2)` | `(1, 3, 2, 2)` | 先按 token、再按头，便于拼接 |
| `.contiguous()` | `(1, 3, 2, 2)` | 保证内存连续（`transpose` 后 `view` 需要） |
| `.view(1, 3, -1)` | `(1, 3, 4)` | 每个 token 的 4 维 = 2 头 × 每头 2 维 |

拼接结果：

```
token0: [1, 2, 7, 8]
token1: [3, 4, 9, 10]
token2: [5, 6, 11, 12]
```

#### 输出投影 `Wo`

`wo = Linear(n_heads × head_dim, dim)`，在上例中为 `Linear(4, 4)`。拼接只是把各头向量首尾相接，头与头之间尚未混合；`Wo` 对每个 token 的 `dim` 维向量做线性变换，学习如何融合各头信息：

```
拼接: [头0 的 2 维 | 头1 的 2 维]  →  Wo  →  融合后的 4 维输出
```

形状不变：`(B, T, dim) → (B, T, dim)`。与 attention 前的 `Wq/Wk/Wv`（输入 → 多头）对应，`Wo` 负责 attention 后的多头 → 模型维度。

#### 残差 dropout

`resid_dropout` 作用于模块最终输出。在 Transformer 中通常加在残差连接之前：

```
x → MultiHeadAttention → resid_dropout → + x → 下一层
```

| Dropout | 作用对象 | 目的 |
|---------|----------|------|
| `attn_dropout` | attention 权重 | 正则化「看哪里」 |
| `resid_dropout` | 模块输出 | 正则化「传什么给下一层」 |

训练模式（`model.train()`）下随机置零部分元素；推理模式（`model.eval()`）下 dropout 关闭，输出不变。

#### 后半段流程小结

```
head_output          (B, n_heads, T, head_dim)   各头 attention 输出
      ↓ transpose + contiguous + view
output (concat)      (B, T, dim)                 按 token 拼接所有头
      ↓ Wo
output (after Wo)    (B, T, dim)                 线性融合多头
      ↓ resid_dropout
final output         (B, T, dim)                 训练时随机丢弃部分值
```

## Encoder 层：EncoderLayer

`encoder.py` 在 `MultiHeadAttention` 之上，组合 **LayerNorm**、**双向自注意力** 与 **前馈网络（MLP）**，构成 Transformer Encoder 的一个子层。采用 Pre-LayerNorm 结构：先归一化，再进入子模块，最后与输入做残差相加。

### 基本用法

```python
import torch
from encoder import EncoderLayer
from multihead_attention import ModelArgs

args = ModelArgs(dim=64, n_heads=8, dropout=0.1, max_seq_len=512)
layer = EncoderLayer(args)

x = torch.randn(2, 16, 64)  # (batch, seq_len, dim)
out = layer(x)                # 输出形状 (2, 16, 64)
```

### 模块组成

| 模块 | 说明 |
|------|------|
| `LayerNorm` | 对最后一维做均值/方差归一化，带可学习仿射参数 `a_2`、`b_2` |
| `MLP` | 两层线性变换 + ReLU + Dropout：`Linear → ReLU → Linear → Dropout` |
| `EncoderLayer` | 两个 LayerNorm + 一个 `MultiHeadAttention(is_causal=False)` + 一个 MLP |
| `Encoder` | N 个 `EncoderLayer` 堆叠 + 栈顶 LayerNorm |

### 前向流程

```
输入 x  (B, T, dim)
      ↓
LayerNorm  →  MultiHeadAttention（双向，is_causal=False）  →  + x  →  h
      ↓
LayerNorm  →  MLP  →  + h  →  输出
```

对应代码：

```python
norm_x = self.attention_norm(x)
h = x + self.attention.forward(norm_x, norm_x, norm_x)
out = h + self.feed_forward.forward(self.fnn_norm(h))
```

与 Decoder 的区别：Encoder 使用 `is_causal=False`，每个 token 可以 attend 到序列中所有位置（包括「未来」token），适合理解整句上下文。

### LayerNorm

对每个 token 在特征维度上独立归一化：

```python
mean = x.mean(-1, keepdim=True)
std = x.std(-1, keepdim=True)
return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

输入 `(B, T, dim)` 时，`mean` / `std` 形状为 `(B, T, 1)`，沿最后一维广播。

### MLP

```python
return self.dropout(self.w2(F.relu(self.w1(x))))
```

| 层 | 作用 |
|----|------|
| `w1` | `dim → hidden_dim` |
| ReLU | 非线性激活 |
| `w2` | `hidden_dim → dim` |
| `dropout` | 训练时正则化，推理时关闭 |

`EncoderLayer` 中 `hidden_dim` 与 `dim` 相同（`MLP(args.dim, args.dim, args.dropout)`）。

### 残差连接

Encoder 子层有两处残差：

```
x ─────────────────────────────┐
│                              ↓ (+)
└→ LayerNorm → Attention ──────┘ → h
h ─────────────────────────────┐
│                              ↓ (+)
└→ LayerNorm → MLP ────────────┘ → out
```

残差让底层信息直接传到高层，缓解深层网络训练困难。

## Encoder 块：Encoder

`Encoder` 由 **N 个 `EncoderLayer`** 堆叠而成，最后经一层 **LayerNorm** 输出，对应 Transformer 原文中的 Encoder 栈。

### 基本用法

```python
import torch
from encoder import Encoder
from multihead_attention import ModelArgs

args = ModelArgs(dim=64, n_heads=8, n_layer=6, dropout=0.1, max_seq_len=512)
encoder = Encoder(args)

x = torch.randn(2, 16, 64)  # (batch, seq_len, dim)
out = encoder(x)              # 输出形状 (2, 16, 64)
```

### 前向流程

```
输入 x  (B, T, dim)
      ↓
EncoderLayer × N（每层含 Attention + MLP + 残差）
      ↓
LayerNorm
      ↓
输出  (B, T, dim)
```

对应代码：

```python
for layer in self.layers:
    x = layer(x)
return self.norm(x)
```

| 子模块 | 说明 |
|--------|------|
| `layers` | `ModuleList`，长度为 `args.n_layer` |
| `norm` | 栈顶 LayerNorm，对最终 hidden states 规范化 |

## Decoder 层：DecoderLayer

`decoder.py` 实现 Transformer Decoder 子层，在 Encoder 输出的基础上，对目标序列做 **掩码自注意力**、**交叉注意力** 与 **前馈网络**，同样采用 Pre-LayerNorm + 残差结构。

### 基本用法

```python
import torch
from decoder import DecoderLayer
from multihead_attention import ModelArgs

args = ModelArgs(dim=64, n_heads=8, dropout=0.1, max_seq_len=512)
layer = DecoderLayer(args)

x = torch.randn(2, 16, 64)       # decoder 输入 (batch, dec_seq, dim)
enc_out = torch.randn(2, 20, 64) # encoder 输出 (batch, enc_seq, dim)
out = layer(x, enc_out)          # 输出形状 (2, 16, 64)
```

### 模块组成

| 子模块 | 说明 |
|--------|------|
| `mask_attention` | 掩码自注意力，`is_causal=True`，禁止 attend 未来 token |
| `attention` | 交叉注意力，`is_causal=False`；q 来自 decoder，k/v 来自 `enc_out` |
| `feed_forward` | 与 Encoder 相同的两层 MLP |
| `attention_norm_1/2`、`ffn_norm` | 三个 Pre-LayerNorm |

### 前向流程

```
decoder 输入 x  (B, T_dec, dim)          encoder 输出 enc_out  (B, T_enc, dim)
      ↓                                              ↓
LayerNorm → Masked Self-Attention → + x
      ↓
LayerNorm → Cross-Attention(q=x, k=v=enc_out) → + x  →  h
      ↓
LayerNorm → MLP → + h  →  输出
```

对应代码：

```python
norm_x = self.attention_norm_1(x)
x = x + self.mask_attention.forward(norm_x, norm_x, norm_x)
norm_x = self.attention_norm_2(x)
h = x + self.attention.forward(norm_x, enc_out, enc_out)
out = h + self.feed_forward.forward(self.ffn_norm(h))
```

与 Encoder 的区别：

| 对比项 | Encoder | Decoder |
|--------|---------|---------|
| 自注意力 | 双向（`is_causal=False`） | 因果掩码（`is_causal=True`） |
| 交叉注意力 | 无 | q 来自 decoder，k/v 来自 encoder |
| LayerNorm 数量 | 2 | 3 |

## Decoder 块：Decoder

`Decoder` 由 **N 个 `DecoderLayer`** 堆叠而成，最后经一层 **LayerNorm** 输出。

### 基本用法

```python
import torch
from decoder import Decoder
from multihead_attention import ModelArgs

args = ModelArgs(dim=64, n_heads=8, n_layer=6, dropout=0.1, max_seq_len=512)
decoder = Decoder(args)

x = torch.randn(2, 16, 64)
enc_out = torch.randn(2, 20, 64)
out = decoder(x, enc_out)  # 输出形状 (2, 16, 64)
```

### 前向流程

```
输入 x, enc_out
      ↓
DecoderLayer × N（每层接收相同的 enc_out）
      ↓
LayerNorm
      ↓
输出  (B, T_dec, dim)
```

对应代码：

```python
for layer in self.layers:
    x = layer(x, enc_out)
return self.norm(x)
```

## Transformer 模型

`transformer.py` 将词嵌入、位置编码、Encoder、Decoder 与语言模型头组装为完整的 Transformer，对应 Happy-LLM 教程中的端到端结构。

### 基本用法

```python
import torch
from multihead_attention import ModelArgs
from transformer import Transformer

args = ModelArgs(
    dim=64,
    n_heads=8,
    n_layer=2,
    vocab_size=5000,
    block_size=128,
    dropout=0.1,
    max_seq_len=128,
)
model = Transformer(args)

idx = torch.randint(0, args.vocab_size, (2, 16))  # token ids (batch, seq)
logits, loss = model(idx)                           # 推理：logits (2, 1, vocab_size)

targets = torch.randint(0, args.vocab_size, (2, 16))
logits, loss = model(idx, targets)                  # 训练：logits (2, 16, vocab_size)，loss 标量
```

### 模块组成

| 子模块 | 说明 |
|--------|------|
| `wte` | 词嵌入 `Embedding(vocab_size, n_embd)` |
| `wpe` | 正弦/余弦位置编码，加到 token embedding 上 |
| `drop` | Embedding 后的 Dropout |
| `encoder` | N 层 Encoder 栈 |
| `decoder` | N 层 Decoder 栈，cross-attend 到 encoder 输出 |
| `lm_head` | 线性投影到词表维度 |

### 前向流程

```
idx (B, T)
      ↓
Token Embedding + Positional Encoding + Dropout
      ↓
Encoder → enc_out
      ↓
Decoder(x, enc_out)
      ↓
lm_head → logits
      ↓
（若提供 targets）cross_entropy → loss
```

推理时只取序列最后一个位置的 hidden state 计算 logits，形状为 `(B, 1, vocab_size)`；训练时对每个位置计算 logits，形状为 `(B, T, vocab_size)`。

### PositionalEncoding

对序列每个位置生成固定的 sin/cos 编码，与 token embedding 相加：

```python
x = x + self.pe[:, : x.size(1)]
```

`block_size` 决定预计算位置编码的最大长度；输入序列长度 `T` 必须满足 `T <= block_size`。

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

# 只运行多头注意力相关测试
pytest test_multihead_attention.py -v

# 只运行 Encoder 相关测试
pytest test_encoder.py -v

# 只运行 Decoder 相关测试
pytest test_decoder.py -v

# 只运行 Transformer 相关测试
pytest test_transformer.py -v

# 运行单个用例
pytest test_attention.py::test_manual_computation -v
pytest test_multihead_attention.py::test_causal_mask_blocks_future_tokens -v
pytest test_encoder.py::test_attention_is_non_causal -v
pytest test_encoder.py::test_encoder_output_shape -v
pytest test_decoder.py::test_mask_attention_blocks_future_tokens -v
pytest test_decoder.py::test_cross_attention_uses_encoder_output -v
pytest test_transformer.py::test_transformer_inference_logits_shape -v
```

## 测试用例说明

### `test_attention.py`

| 用例 | 说明 |
|------|------|
| `test_output_shapes` | 验证输出与注意力权重的形状 |
| `test_attention_weights_sum_to_one` | 验证 softmax 后每行权重和为 1 |
| `test_manual_computation` | 简单数值场景下与手算结果一致 |
| `test_output_is_weighted_sum_of_values` | 验证 `output = weights @ value` |
| `test_with_dropout` | 验证训练模式下 dropout 会改变注意力权重 |

### `test_multihead_attention.py`

| 用例 | 说明 |
|------|------|
| `test_output_shape` | 验证输出形状为 `(batch, seq, dim)` |
| `test_invalid_head_count` | `dim` 不能被 `n_heads` 整除时抛出断言 |
| `test_causal_mask_blocks_future_tokens` | 因果 mask 下，修改未来 token 不影响过去位置输出 |
| `test_non_causal_uses_future_tokens` | 非因果模式下，未来 token 会影响输出 |
| `test_cross_attention_accepts_different_qkv` | q/k/v 序列长度不同时仍可前向 |
| `test_gradient_flow` | 反向传播梯度正常 |

### `test_encoder.py`

| 用例 | 说明 |
|------|------|
| `test_output_shape` | 验证输出形状为 `(batch, seq, dim)` |
| `test_attention_is_non_causal` | Encoder 使用双向注意力，修改未来 token 会影响过去位置 |
| `test_has_pre_layer_norm_submodules` | 验证 Pre-LayerNorm 子模块与 `is_causal=False` |
| `test_residual_connection_changes_output` | 残差连接使输出不同于输入 |
| `test_gradient_flow` | 梯度能流到 attention、MLP 与 LayerNorm |
| `test_layer_norm_normalizes_last_dimension` | `LayerNorm` 在最后一维做归一化 |
| `test_mlp_output_shape` | `MLP` 输出形状与输入一致 |
| `test_mlp_dropout_changes_output_in_train_mode` | 训练模式下 MLP dropout 生效 |
| `test_invalid_head_count` | `dim` 不能被 `n_heads` 整除时抛出断言 |
| `test_encoder_output_shape` | `Encoder` 输出形状为 `(batch, seq, dim)` |
| `test_encoder_has_n_layers` | `Encoder` 包含 `n_layer` 个 `EncoderLayer` 与最终 `norm` |
| `test_encoder_applies_final_layer_norm` | 栈顶 LayerNorm 对最后一维做归一化 |
| `test_encoder_stacked_layers_differ_from_single_layer` | 多层堆叠与单层输出不同 |
| `test_encoder_gradient_flow` | 梯度能流到各层与栈顶 LayerNorm |

### `test_decoder.py`

| 用例 | 说明 |
|------|------|
| `test_decoder_layer_output_shape` | 输出形状为 `(batch, dec_seq, dim)` |
| `test_decoder_layer_has_expected_submodules` | 三个 LayerNorm、因果/非因果注意力子模块 |
| `test_mask_attention_blocks_future_tokens` | 掩码自注意力下，修改未来 token 不影响过去位置 |
| `test_cross_attention_uses_encoder_output` | 修改 `enc_out` 会改变 decoder 输出 |
| `test_cross_attention_accepts_different_seq_lengths` | decoder 与 encoder 序列长度可不同 |
| `test_decoder_layer_residual_changes_output` | 残差连接使输出不同于输入 |
| `test_decoder_layer_gradient_flow` | 梯度能流到 x、`enc_out` 与各子模块 |
| `test_invalid_head_count` | `dim` 不能被 `n_heads` 整除时抛出断言 |
| `test_decoder_output_shape` | `Decoder` 输出形状为 `(batch, dec_seq, dim)` |
| `test_decoder_has_n_layers` | `Decoder` 包含 `n_layer` 个 `DecoderLayer` 与最终 `norm` |
| `test_decoder_applies_final_layer_norm` | 栈顶 LayerNorm 对最后一维做归一化 |
| `test_decoder_stacked_layers_differ_from_single_layer` | 多层堆叠与单层输出不同 |
| `test_decoder_gradient_flow` | 梯度能流到各层与栈顶 LayerNorm |

### `test_transformer.py`

| 用例 | 说明 |
|------|------|
| `test_transformer_inference_logits_shape` | 推理时 logits 形状为 `(batch, 1, vocab_size)` |
| `test_transformer_training_logits_and_loss` | 训练时 logits 为 `(batch, seq, vocab_size)`，loss 为标量 |
| `test_transformer_exceeds_block_size_raises` | 序列长度超过 `block_size` 时抛出断言 |
| `test_transformer_requires_vocab_size` | 缺少 `vocab_size` 时无法构建模型 |
| `test_transformer_requires_block_size` | 缺少 `block_size` 时无法构建模型 |
| `test_get_num_params` | 统计参数量，支持排除 embedding |
| `test_positional_encoding_preserves_shape` | 位置编码不改变张量形状 |
| `test_positional_encoding_changes_values` | 位置编码会改变 embedding 数值 |
| `test_loss_ignores_masked_targets` | 部分 `targets=-1` 时仍能得到有限 loss，且与全量 targets 不同 |
| `test_transformer_gradient_flow` | 训练模式下反向传播梯度正常 |

## 调试日志

`attention()` 与 `MultiHeadAttention.forward()` 均使用 Python `logging` 输出 DEBUG 级别日志（张量形状、统计量、dropout / mask 状态等）。`MultiHeadAttention` 复用 `attention.py` 中的 `_tensor_summary` 格式化张量信息，日志风格与单头 attention 一致。

### `attention()`

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

日志覆盖：`query` / `key` / `value` → `raw_scores` → `scaled_scores` → softmax 权重 → dropout（可选）→ 输出。

### `MultiHeadAttention`

```python
import logging
import torch
from multihead_attention import ModelArgs, MultiHeadAttention

logging.basicConfig(level=logging.DEBUG, format="%(name)s | %(levelname)s | %(message)s")

args = ModelArgs(dim=64, n_heads=8, dropout=0.1, max_seq_len=512)
mha = MultiHeadAttention(args, is_causal=True)

x = torch.randn(2, 16, 64)
mha(x, x, x)
```

日志覆盖各前向阶段：

| 阶段 | 日志内容 |
|------|----------|
| 输入 | `q` / `k` / `v` 形状与统计量；`n_heads`、`head_dim`、`is_causal`、`training` |
| 线性投影 | `xq` / `xk` / `xv`（Wq / Wk / Wv 之后） |
| 拆头 | 多头张量 `(B, n_heads, T, head_dim)` |
| Attention | `raw_scores` → `scaled_scores` → 因果 mask（若启用）→ softmax 权重 |
| Dropout | `attn_dropout` 是否修改权重 |
| 输出 | `weights @ V` → 拼接多头 → `Wo` → `resid_dropout` |

### 运行测试时查看日志

```bash
# 单头 attention
pytest test_attention.py -v --log-cli-level=DEBUG

# 多头 attention
pytest test_multihead_attention.py -v --log-cli-level=DEBUG

# 全部测试
pytest -v --log-cli-level=DEBUG
```

## 开源训练与强化学习框架推荐

手写 Attention / MultiHeadAttention / Encoder / Decoder / Transformer 之后，若要进入模型训练、微调与对齐阶段，可按下面分类选用开源框架。

### 通用深度学习训练

| 框架 | 特点 | 适合场景 |
|------|------|----------|
| [PyTorch](https://pytorch.org/) | 生态最大、论文复现多、调试直观 | 本仓库已使用，继续用它最自然 |
| [Lightning](https://lightning.ai/) | 封装训练循环、分布式、日志 | 小模型实验，减少手写 train/eval 循环 |
| [Accelerate](https://github.com/huggingface/accelerate) | Hugging Face 出品，几乎零侵入 | 单机多卡 / 多机，改动现有代码少 |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | ZeRO 显存优化 | 7B+ 全量或微调，显存紧张时 |
| [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) | PyTorch 原生分片 | 不想引入 DeepSpeed 时的原生方案 |

建议：基础阶段继续 PyTorch；需要多卡或更大模型时，叠加 **Accelerate** 或 **DeepSpeed**。

### 大模型训练 / 微调（SFT、LoRA 等）

| 框架 | 特点 | 适合场景 |
|------|------|----------|
| [Transformers](https://github.com/huggingface/transformers) | 模型、Tokenizer、Trainer 一体 | 加载预训练模型、做 SFT 的第一选择 |
| [TRL](https://github.com/huggingface/trl) | HF 官方 RLHF 库，与 Transformers 无缝 | SFT → DPO / PPO / GRPO 一条龙 |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | 配置化微调，中文文档友好 | 快速 LoRA / QLoRA 微调，少写代码 |
| [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) | YAML 配置驱动 | 社区常用微调方案，可复现性好 |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | NVIDIA 预训练栈 | 从零预训练 / 超大规模 |
| [Colossal-AI](https://github.com/hpcaitech/ColossalAI) | 国产，易用的大模型训练 | 预训练 + 微调，对国内环境友好 |

推荐学习路径：

```
手写 Attention（本仓库） → Transformers 加载模型 → LLaMA-Factory 做 LoRA → TRL 做 DPO / RLHF
```

### 强化学习框架

#### 通用 RL（游戏、机器人、控制）

| 框架 | 特点 | 适合场景 |
|------|------|----------|
| [Stable-Baselines3](https://github.com/DRL-RM/stable-baselines3) | API 简单、文档好 | PPO / A2C / DQN 等经典算法入门 |
| [CleanRL](https://github.com/vwxyzjn/cleanrl) | 单文件实现，易读易改 | 理解 RL 算法原理（类似本仓库手写 attention） |
| [RLlib](https://docs.ray.io/en/latest/rllib/index.html) | Ray 生态，分布式强 | 大规模并行 RL |
| [Tianshou](https://github.com/thu-ml/tianshou) | 国产、PyTorch 原生 | 研究型实验，模块化好 |

#### LLM 对齐 / RLHF

| 框架 | 特点 | 适合场景 |
|------|------|----------|
| [TRL](https://github.com/huggingface/trl) | PPO、DPO、GRPO、KTO 等 | **LLM RLHF 首选**，与 HF 生态一致 |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | 完整 RLHF 流水线（Reward Model + PPO） | 跑完整 RLHF，而非仅 DPO |
| [verl](https://github.com/volcengine/verl) | 字节开源，HybridFlow 架构 | 大规模 RLHF / GRPO |
| [NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner) | NVIDIA 对齐工具 | 企业级，与 NeMo 预训练栈配合 |

RLHF 常见流程：

```
SFT 监督微调
      ↓
Reward Model / 偏好数据
      ↓
PPO 或 DPO / GRPO
      ↓
对齐后的模型
```

- **DPO / GRPO**：实现简单、显存友好，目前主流（DeepSeek、Qwen 等用 GRPO 做推理增强）。
- **PPO + Reward Model**：经典 RLHF，工程更重，适合 OpenRLHF / verl。

### 按学习阶段选型

| 阶段 | 推荐组合 |
|------|----------|
| 现在（手写 Attention / MHA / Encoder / Decoder / Transformer） | PyTorch + pytest（本仓库） |
| 加载并微调小 LLM | Transformers + PEFT + LLaMA-Factory |
| 理解 RL 算法 | CleanRL 或 Stable-Baselines3 |
| LLM 对齐（DPO / GRPO） | TRL |
| 完整 RLHF / 大规模 GRPO | OpenRLHF 或 verl |
| 从零预训练 | Megatron-LM / DeepSpeed + 自建数据管线 |

### 推荐学习仓库

| 仓库 | 说明 |
|------|------|
| [nanoGPT](https://github.com/karpathy/nanoGPT) | 最小 GPT 训练，理念与本仓库接近 |
| [minGPT](https://github.com/karpathy/minGPT) | 更精简的 GPT 实现 |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | 微调上手最快 |
| [TRL examples](https://github.com/huggingface/trl/tree/main/examples) | DPO / PPO 官方示例 |
