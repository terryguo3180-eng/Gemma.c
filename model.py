from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# Requires Python 3.12+. Remove the 'type' keyword if your version is lower than that
# I personally like to write it this way :D
type Tensor = torch.Tensor

def cast_as[T](template: T, value) -> T:
    """
    Casting `value` to the type of `template`.
    This only tricks IDEs & type checkers, no runtime effect.
    """
    return value


FP16_MAX = torch.finfo(torch.float16).max

@dataclass
class GemmaConfig:
    embed_dim: int
    n_layers: int
    attn_local_layers: list[bool]
    max_seq_len: int
    sliding_window: int
    vocab_size: int
    local_theta: float
    global_theta: float
    eps: float
    n_heads: int
    head_dim: int
    n_kv_heads: int
    q_pre_attn_scalar: int
    use_qk_norm: bool
    attn_softcapping: int
    mlp_hidden_size: int
    pre_ffwd_norm: bool
    post_ffwd_norm: bool
    logit_softcapping: int

    dtype: torch.dtype
    quant: bool

    def __post_init__(self):
        assert self.n_heads % self.n_kv_heads == 0  # GQA
        assert self.head_dim % 2 == 0  # So that RoPE could work out fine


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, quant: bool, dtype: torch.dtype):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((vocab_size, embed_dim), dtype=torch.int8),
                requires_grad=False,
            )
            # Per-tensor quantization
            self.weight_scaler = nn.Parameter(
                torch.empty(vocab_size, dtype=dtype), requires_grad=False
            )
        else:
            self.weight = nn.Parameter(
                torch.rand((vocab_size, embed_dim), dtype=dtype) * (2*vocab_size**-0.5) - 1
            )
        self.quant = quant
        self.dtype = dtype

    def forward(self, x: Tensor):
        if self.quant:
            assert not self.training
            return (self.weight[x] * self.weight_scaler[x].unsqueeze(-1)).to(self.dtype)
        else:
            return self.weight[x]

    def quantize(self):
        self.quant = True

        weight = self.weight
        abs_max = weight.abs().amax(1, keepdim=True).clamp_min(1e-8)  # (vocab_size, 1)
        scales = abs_max / 127.0  # (vocab_size, 1)
        quantized = torch.round(weight / scales).clamp(-127, 127).to(torch.int8)

        self.weight.requires_grad = False
        self.weight.data = quantized
        self.weight_scaler = nn.Parameter(scales.squeeze(1).to(self.dtype), requires_grad=False)

    # Pylance doesn't prompt the signature of forward() when calling nn.Module objects
    # This is my way to get around this, basically appending this line of code for every
    # class, and use model.call(...) instead of model(...)
    call = cast_as(forward, nn.Module.__call__)


class Linear(nn.Module):
    def __init__(self, fan_in: int, fan_out: int, quant: bool, dtype: torch.dtype):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((fan_in, fan_out), dtype=torch.int8),
                requires_grad=False,
            )
            # Per-channel quantiation
            self.weight_scaler = nn.Parameter(
                torch.empty(fan_out, dtype=dtype), requires_grad=False
            )
        else:
            self.weight = nn.Parameter(
                torch.rand((fan_in, fan_out), dtype=dtype) * (2*fan_in**-0.5) - 1
            )
        self.quant = quant
        self.dtype = dtype

    def forward(self, x: Tensor):
        if self.quant:
            assert not self.training
            orig_shape = x.shape
            x2d = x.reshape(-1, orig_shape[-1])
            
            x_fp32 = x2d.float()
            x_abs_max = x_fp32.abs().amax(-1, keepdim=True).clamp_min(1e-8)
            x_scale = x_abs_max / 127.0  # float32
            
            x_q = torch.round(x_fp32 / x_scale).clamp(-127, 127).to(torch.int8)
            out = torch._int_mm(x_q, self.weight)  # type: ignore  # int32
            
            out = out.float() * x_scale * self.weight_scaler.float()
            
            return out.reshape(*orig_shape[:-1], -1).to(self.dtype)
        else:
            return x @ self.weight

    def quantize(self):
        self.quant = True

        weight = self.weight
        abs_max = weight.abs().amax(0, keepdim=True).clamp_min(1e-8)  # (fan_out,)
        scales = abs_max / 127.0  # (fan_out,)
        quantized = torch.round(weight / scales).clamp(-127, 127).to(torch.int8)

        self.weight.requires_grad = False
        self.weight.data = quantized
        self.weight_scaler = nn.Parameter(scales.to(self.dtype), requires_grad=False)

    call = cast_as(forward, nn.Module.__call__)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, config: GemmaConfig):
        super().__init__()
        self.eps = config.eps

        self.weight = nn.Parameter(torch.zeros(dim))  # It's added by 1 in the forward pass

    def forward(self, x: Tensor):
        x_fp32 = x.float()
        norm = x_fp32 * torch.rsqrt(x_fp32.square().mean(-1, True) + self.eps)
        out = norm * (self.weight.float() + 1)
        return out.to(x.dtype)
        
    call = cast_as(forward, nn.Module.__call__)


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float, scale: int = 1):
    freqs = (theta * torch.ones(dim // 2))**(-2 * torch.arange(dim // 2).float() / dim)
    freqs /= scale
    freqs = torch.outer(torch.arange(max_seq_len), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor):
    B, H, T, C = x.size()
    return torch.cat(torch.split(torch.view_as_real(
        freqs_cis * torch.view_as_complex(torch.stack(x.chunk(2, -1), -1).float())
    ), 1, -1), -2).reshape(B, H, T, C).to(x.dtype)


class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.n_kv_heads = config.n_kv_heads
        self.q_pre_attn_scalar = config.q_pre_attn_scalar
        self.use_qk_norm = config.use_qk_norm
        self.attn_softcaping = config.attn_softcapping

        self.n_qs_per_kv = self.n_heads // self.n_kv_heads
        self.q_size = self.head_dim * self.n_heads
        self.kv_size = self.head_dim * self.n_kv_heads
        self.scaling = self.q_pre_attn_scalar**-0.5

        self.q_proj = Linear(self.embed_dim, self.q_size, config.quant, config.dtype)
        self.k_proj = Linear(self.embed_dim, self.kv_size, config.quant, config.dtype)
        self.v_proj = Linear(self.embed_dim, self.kv_size, config.quant, config.dtype)
        self.o_proj = Linear(self.q_size, self.embed_dim, config.quant, config.dtype)

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, config)
            self.k_norm = RMSNorm(self.head_dim, config)

    def forward(
        self,
        x: Tensor,  # (B, T, C)
        positions: Tensor,  # (T,)
        freqs_cis: Tensor,  # (max_seq_len, head_dim // 2)
        mask: Tensor,  # (seq_len, seq_len)
        kv_cache: Tensor | None = None,  # (B, 2, n_kv_heads, seq_len, head_dim)
    ):
        B, T, _ = x.size()

        xq = self.q_proj.call(x)  # (B, T, q_size)
        xk = self.k_proj.call(x)  # (B, T, kv_size)
        xv = self.v_proj.call(x)  # (B, T, kv_size)

        xq = xq.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        xk = xk.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, n_kv_heads, T, head_dim)
        xv = xv.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, n_kv_heads, T, head_dim)

        if self.use_qk_norm:
            xq = self.q_norm.call(xq)
            xk = self.k_norm.call(xk)

        xq = apply_rotary_emb(xq, freqs_cis[positions])
        xk = apply_rotary_emb(xk, freqs_cis[positions])

        if self.training or kv_cache is None:
            key = xk
            val = xv
        else:
            key = kv_cache[:, 0]  # (B, n_kv_heads, seq_len, head_dim)
            val = kv_cache[:, 1]  # (B, n_kv_heads, seq_len, head_dim)

            key.index_copy_(2, positions, xk)
            val.index_copy_(2, positions, xv)

        if self.n_kv_heads != self.n_heads:
            key = key.repeat_interleave(self.n_qs_per_kv, 1)  # (B, n_heads, seq_len, head_dim)
            val = val.repeat_interleave(self.n_qs_per_kv, 1)  # (B, n_heads, seq_len, head_dim)

        scores = xq @ key.mT * self.scaling  # (B, n_heads, T, seq_len)
        if self.attn_softcaping != 0:
            scores = (scores / self.attn_softcaping).tanh() * self.attn_softcaping
        scores = scores + mask[positions]
        
        att = scores.softmax(-1)
        out = (att @ val).transpose(1, 2)  # (B, T, n_heads, head_dim)
        out = out.reshape(B, T, self.n_heads * self.head_dim)  # (B, T, n_heads * head_dim)
        out = self.o_proj.call(out)  # (B, T, C)
        return out
        
    call = cast_as(forward, nn.Module.__call__)


class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.mlp_hidden_size = config.mlp_hidden_size

        self.up_proj = Linear(config.embed_dim, self.mlp_hidden_size, config.quant, config.dtype)
        self.gate_proj = Linear(config.embed_dim, self.mlp_hidden_size, config.quant, config.dtype)
        self.down_proj = Linear(self.mlp_hidden_size, config.embed_dim, config.quant, config.dtype)

    def forward(self, x: Tensor):
        gate = self.gate_proj.call(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj.call(x)
        fuse = gate * up
        out = self.down_proj.call(fuse)
        return out
        
    call = cast_as(forward, nn.Module.__call__)


class GemmaDecoderBlock(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.attn = GemmaAttention(config)
        self.ffwd = GemmaMLP(config)

        self.norm1 = RMSNorm(config.embed_dim, config)
        self.norm2 = RMSNorm(config.embed_dim, config)

        self.pre_ffwd_norm = config.pre_ffwd_norm
        self.post_ffwd_norm = config.post_ffwd_norm
        
        if self.pre_ffwd_norm:
            self.norm3 = RMSNorm(config.embed_dim, config)
        if self.post_ffwd_norm:
            self.norm4 = RMSNorm(config.embed_dim, config)

    def forward(
        self,
        x: Tensor,  # (B, T, C)
        positions: Tensor,  # (T,)
        freqs_cis: Tensor,  # (T, head_dim // 2)
        mask: Tensor,  # (T, T)
        kv_cache: Tensor | None = None,  # (B, 2, n_kv_heads, max_seq_len, head_dim)
    ):
        resid = x
        x = self.norm1.call(x)
        x = self.attn.call(x, positions, freqs_cis, mask, kv_cache)
        x = self.norm2.call(x)
        x = (x + resid).clamp(-FP16_MAX, FP16_MAX)
        resid = x
        if self.pre_ffwd_norm:
            x = self.norm3.call(x)
        x = self.ffwd.call(x)
        if self.post_ffwd_norm:
            x = self.norm4.call(x)
        x = (x + resid).clamp(-FP16_MAX, FP16_MAX)
        return x
        
    call = cast_as(forward, nn.Module.__call__)


class GemmaModel(nn.Module):
    local_freqs_cis: Tensor
    global_freqs_cis: Tensor

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config

        self.embedding = Embedding(config.vocab_size, config.embed_dim, config.quant, config.dtype)
        self.layers = nn.ModuleList([GemmaDecoderBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.embed_dim, config)

        for name, theta in [
            ("local_freqs_cis", config.local_theta),
            ("global_freqs_cis", config.global_theta),
        ]:
            self.register_buffer(
                name, precompute_freqs_cis(config.head_dim, config.max_seq_len * 2, theta)
            )

    def forward(
        self,
        tokens: Tensor,  # (B, T)
        positions: Tensor,  # (T,)
        seq_len: int,
        global_mask: Tensor,
        local_mask: Tensor,
        kv_cache_layers: list[Tensor] | None = None,
    ) -> Tensor:
        config = self.config

        B = tokens.size(0)
        x = self.embedding.call(tokens)
        x = x * config.embed_dim**0.5

        assert seq_len <= config.max_seq_len

        if self.training or kv_cache_layers is None:
            get_extra_arg = lambda _: None
        else:
            if not kv_cache_layers:
                kv_cache_layers.extend([torch.zeros(
                    B, 2, config.n_kv_heads, seq_len, config.head_dim,
                    dtype=config.dtype, device=tokens.device, requires_grad=False,
                ) for _ in self.layers])
            get_extra_arg = lambda i: kv_cache_layers[i]

        for i, layer in enumerate(self.layers):
            local_sliding = config.attn_local_layers[i % len(config.attn_local_layers)]
            freqs_cis = self.local_freqs_cis if local_sliding else self.global_freqs_cis
            mask = local_mask if local_sliding else global_mask
            x = layer(x, positions, freqs_cis, mask, get_extra_arg(i))

        x = self.final_norm.call(x)

        if config.quant:
            logits = (
                x.float() @ (self.embedding.weight.T * self.embedding.weight_scaler).float()
            ).to(config.dtype)  # (B, T, vocab_size)
        else:
            logits = x @ self.embedding.weight.T  # (B, T, vocab_size)

        if config.logit_softcapping != 0:
            logits = (logits / config.logit_softcapping).tanh() * config.logit_softcapping

        return logits

    def quantize(self):
        # This only provides a speedup on cuda devices
        self.config.quant = True

        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "quantize") and callable(module.quantize):
                module.quantize()

    call = cast_as(forward, nn.Module.__call__)
