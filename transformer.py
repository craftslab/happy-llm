"""Full Transformer model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder import Decoder
from encoder import Encoder
from multihead_attention import ModelArgs


class PositionalEncoding(nn.Module):
    """位置编码模块"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        pe = torch.zeros(args.block_size, args.n_embd)
        position = torch.arange(0, args.block_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, args.n_embd, 2) * -(math.log(10000.0) / args.n_embd)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return x


class Transformer(nn.Module):
    """整体模型"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size is not None
        assert args.block_size is not None
        self.args = args
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(args.vocab_size, args.n_embd),
                wpe=PositionalEncoding(args),
                drop=nn.Dropout(args.dropout),
                encoder=Encoder(args),
                decoder=Decoder(args),
            )
        )
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        self.apply(self._init_weights)

    def get_num_params(self, non_embedding: bool = False) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, t = idx.size()
        assert t <= self.args.block_size, (
            f"不能计算该序列，该序列长度为 {t}, 最大序列长度只有 {self.args.block_size}"
        )

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(tok_emb)
        x = self.transformer.drop(pos_emb)
        enc_out = self.transformer.encoder(x)
        x = self.transformer.decoder(x, enc_out)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
