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
class GemmaVisionConfig:
    n_layers: int
    image_size: int
    patch_size: int
    hidden_dim: int
    n_heads: int
    mlp_hidden_size: int
    eps: float

    def __post_init__(self):
        assert self.hidden_dim % self.n_heads == 0


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

    tokens_per_image: int
    vision_config: GemmaVisionConfig | None

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
            x_scale = x_abs_max / 127.0
            x_q = torch.round(x_fp32 / x_scale).clamp(-127, 127).to(torch.int8)
            # torch._int_mm is extremely slow on my CPU, and I can't find a better substitution
            # in the PyTorch ecosystem. You would need to write C++ extensions to really boost this up
            out = torch._int_mm(x_q, self.weight)  # type: ignore
            out = (out.float() * x_scale * self.weight_scaler.float()).to(self.dtype)
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
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps

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

        qd = (config.quant, config.dtype)
        self.q_proj = Linear(self.embed_dim, self.q_size, *qd)
        self.k_proj = Linear(self.embed_dim, self.kv_size, *qd)
        self.v_proj = Linear(self.embed_dim, self.kv_size, *qd)
        self.o_proj = Linear(self.q_size, self.embed_dim, *qd)

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, config.eps)
            self.k_norm = RMSNorm(self.head_dim, config.eps)

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

        qd = (config.quant, config.dtype)
        self.up_proj = Linear(config.embed_dim, self.mlp_hidden_size, *qd)
        self.gate_proj = Linear(config.embed_dim, self.mlp_hidden_size, *qd)
        self.down_proj = Linear(self.mlp_hidden_size, config.embed_dim, *qd)

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

        self.norm1 = RMSNorm(config.embed_dim, config.eps)
        self.norm2 = RMSNorm(config.embed_dim, config.eps)

        self.pre_ffwd_norm = config.pre_ffwd_norm
        self.post_ffwd_norm = config.post_ffwd_norm
        
        if self.pre_ffwd_norm:
            self.norm3 = RMSNorm(config.embed_dim, config.eps)
        if self.post_ffwd_norm:
            self.norm4 = RMSNorm(config.embed_dim, config.eps)

    def forward(
        self,
        x: Tensor,  # (B, T, C)
        positions: Tensor,  # (T,)
        freqs_cis: Tensor,  # (T, head_dim // 2)
        mask: Tensor,  # (T, T)
        kv_cache: Tensor | None = None,  # (B, 2, n_kv_heads, max_seq_len, head_dim)
    ):
        resid = x
        print("x:")
        print(x)
        x = self.norm1.call(x)
        print("norm1(x):")
        print(x)
        x = self.attn.call(x, positions, freqs_cis, mask, kv_cache)
        print("attn(x):")
        print(x)
        x = self.norm2.call(x)
        print("norm2.weight.data:")
        print(self.norm2.weight.data)
        print("norm2(x):")
        print(x)
        x = x + resid
        print("x + resid:")
        print(x)
        resid = x
        if self.pre_ffwd_norm:
            x = self.norm3.call(x)
            print("norm3(x):")
            print(x)
        x = self.ffwd.call(x)
        print("ffwd(x):")
        print(x)
        if self.post_ffwd_norm:
            x = self.norm4.call(x)
            print("norm4(x):")
            print(x)
        x = x + resid
        print("x + resid:")
        print(x)
        return x
        
    call = cast_as(forward, nn.Module.__call__)


class SiglipAttention(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        assert config.vision_config is not None
        vconf = config.vision_config

        self.n_heads = vconf.n_heads
        self.head_dim = vconf.hidden_dim // self.n_heads

        qd = (config.quant, config.dtype)
        self.q_proj = Linear(vconf.hidden_dim, vconf.hidden_dim, *qd)
        self.k_proj = Linear(vconf.hidden_dim, vconf.hidden_dim, *qd)
        self.v_proj = Linear(vconf.hidden_dim, vconf.hidden_dim, *qd)
        self.o_proj = Linear(vconf.hidden_dim, vconf.hidden_dim, *qd)

        # Siglip uses biases here
        self.q_bias = nn.Parameter(torch.empty(vconf.hidden_dim, dtype=config.dtype))
        self.k_bias = nn.Parameter(torch.empty(vconf.hidden_dim, dtype=config.dtype))
        self.v_bias = nn.Parameter(torch.empty(vconf.hidden_dim, dtype=config.dtype))
        self.o_bias = nn.Parameter(torch.empty(vconf.hidden_dim, dtype=config.dtype))

        self.use_flash_att = hasattr(F, "scaled_dot_product_attention")
        if not self.use_flash_att:
            print("Warning: using slow attention. Consider update to PyTorch 2.3+")

    def forward(self, x: Tensor) -> Tensor:
        # Typical attention from "Attention is All You Need"
        B, T, C = x.size()

        # (B, T, n_heads * head_dim)
        xq = self.q_proj.call(x) + self.q_bias
        xk = self.k_proj.call(x) + self.k_bias
        xv = self.v_proj.call(x) + self.v_bias

        # (B, n_heads, T, head_dim)
        xq = xq.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_flash_att:
            attn_output = F.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            # (B, n_heads, T, head_dim)
            out = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        else:
            att = (xq @ xk.mT * self.head_dim**-0.5).softmax(-1)  # (B, n_heads, T, T)
            out = (att @ xv).transpose(1, 2).contiguous().view(B, T, C)

        x = self.o_proj.call(out) + self.o_bias  # (B, T, C)

        return x

    call = cast_as(forward, nn.Module.__call__)


class SiglipMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        assert config.vision_config is not None
        vconf = config.vision_config

        qd = (config.quant, config.dtype)
        self.fc1 = Linear(vconf.hidden_dim, vconf.mlp_hidden_size, *qd)
        self.fc2 = Linear(vconf.mlp_hidden_size, vconf.hidden_dim, *qd)

        self.fc1_bias = nn.Parameter(torch.empty(vconf.mlp_hidden_size, dtype=config.dtype))
        self.fc2_bias = nn.Parameter(torch.empty(vconf.hidden_dim, dtype=config.dtype))

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1.call(x) + self.fc1_bias
        x = F.gelu(x, approximate="tanh")
        x = self.fc2.call(x) + self.fc2_bias
        return x

    call = cast_as(forward, nn.Module.__call__)


class SiglipEncoderBlock(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        assert config.vision_config is not None
        vconf = config.vision_config

        self.norm1 = nn.LayerNorm(vconf.hidden_dim, vconf.eps)
        self.attn = SiglipAttention(config)
        self.norm2 = nn.LayerNorm(vconf.hidden_dim, vconf.eps)
        self.ffwd = SiglipMLP(config)
    
    def forward(self, x: Tensor) -> Tensor:
        resid = x
        x = self.attn.call(self.norm1(x))
        x += resid
        resid = x
        x = self.ffwd.call(self.norm2(x))
        x += resid
        return x

    call = cast_as(forward, nn.Module.__call__)


class SiglipVisionEncoder(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        assert config.vision_config is not None
        vconf = config.vision_config

        self.image_size = vconf.image_size
        self.patch_size = vconf.patch_size
        self.hidden_dim = vconf.hidden_dim
        self.tokens_per_image = config.tokens_per_image

        self.patch_emb = nn.Conv2d(
            3, self.hidden_dim,
            (self.patch_size, self.patch_size),
            (self.patch_size, self.patch_size),
        )
        num_patches = (self.image_size // self.patch_size) ** 2
        self.pos_emb = Embedding(num_patches, self.hidden_dim, config.quant, config.dtype)
        self.layers = nn.ModuleList([SiglipEncoderBlock(config) for _ in range(vconf.n_layers)])

        self.patches_per_image = self.image_size // self.patch_size
        tokens_per_side = round(self.tokens_per_image**0.5)
        self.kernal_size = self.patches_per_image // tokens_per_side

        self.post_layernorm = nn.LayerNorm(vconf.hidden_dim, vconf.eps)
        self.avg_pool = nn.AvgPool2d(self.kernal_size, self.kernal_size)
        self.proj = Linear(self.hidden_dim, config.embed_dim, config.quant, config.dtype)
        self.norm = RMSNorm(vconf.hidden_dim, vconf.eps)

    def forward(self, pixel_values: Tensor) -> Tensor:
        # pixel_values: (B, 3, image_size, image_size)
        x = self.patch_emb(pixel_values)  # (B, C, ppi, ppi)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = x + self.pos_emb.call(torch.arange(x.size(1), device=x.device))

        for layer in self.layers:
            assert isinstance(layer, SiglipEncoderBlock)
            x = layer.call(x)

        x = self.post_layernorm(x)

        B, N, C = x.size()
        x = self.avg_pool(
            x.mT.contiguous().view(B, C, self.patches_per_image, self.patches_per_image)
        )
        x = x.view(B, C, self.tokens_per_image).mT
        x = self.norm.call(x)
        x = self.proj.call(x)
        return x

    call = cast_as(forward, nn.Module.__call__)


class GemmaModel(nn.Module):
    local_freqs_cis: Tensor
    global_freqs_cis: Tensor

    def __init__(
        self, config: GemmaConfig,
        image_start_token: int, image_end_token: int, image_soft_token: int,
    ):
        super().__init__()
        self.config = config
        if config.vision_config is not None:
            self.multimodal = True
            self.vision_encoder = SiglipVisionEncoder(config)
        else:
            self.multimodal = False

        self.embedding = Embedding(config.vocab_size, config.embed_dim, config.quant, config.dtype)
        self.layers = nn.ModuleList([GemmaDecoderBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.embed_dim, config.eps)

        for name, theta in [
            ("local_freqs_cis", config.local_theta),
            ("global_freqs_cis", config.global_theta),
        ]:
            self.register_buffer(
                name, precompute_freqs_cis(config.head_dim, config.max_seq_len * 2, theta)
            )

        self.image_start_token = image_start_token
        self.image_end_token = image_end_token
        self.image_soft_token = image_soft_token

    def get_embeddings(
        self,
        tokens: Tensor,  # (B, T)
        pixel_values: Tensor,  # (n_images, 3, H, W)
    ):
        B, T = tokens.size()
        x = self.embedding.call(tokens)  # (B, T, C)

        image_embeddings = (
            self.vision_encoder.call(pixel_values)
            if pixel_values.size() != torch.Size([0]) else torch.tensor([])
        )  # (n_images, N, C)

        # Insert all the image embedding vectors into x
        # for loop over pytorch tensor, wow :D
        for b in range(B):
            if self.config.vision_config is None:
                break
            if pixel_values.size() == torch.Size([0]):
                break

            image_count = 0
            current_row = 0
            inside_image = False

            for t in range(T):
                tok = tokens[b, t]
                if tok == self.image_end_token and inside_image:
                    image_count += 1
                    inside_image = False

                elif tok == self.image_start_token:
                    inside_image = True
                    current_row = 0

                elif tok == self.image_soft_token and inside_image:
                    x[b, t] = image_embeddings[image_count, current_row]
                    current_row += 1

        # I explained the reason for this in gemma.c
        x = x * 1.0  # it used to be config.embed_dim**0.5
        return x

    def forward(
        self,
        emb: Tensor,  # (B, T, C)
        positions: Tensor,  # (T,)
        seq_len: int,
        global_mask: Tensor,
        local_mask: Tensor,
        kv_cache_layers: list[Tensor] | None = None,
    ) -> Tensor:
        config = self.config

        x = emb
        B = x.size(0)

        assert seq_len <= config.max_seq_len

        if self.training or kv_cache_layers is None:
            get_extra_arg = lambda _: None
        else:
            if not kv_cache_layers:
                kv_cache_layers.extend([torch.zeros(
                    B, 2, config.n_kv_heads, seq_len, config.head_dim,
                    dtype=config.dtype, device=x.device, requires_grad=False,
                ) for _ in self.layers])
            get_extra_arg = lambda i: kv_cache_layers[i]

        for i, layer in enumerate(self.layers):
            local_sliding = config.attn_local_layers[i % len(config.attn_local_layers)]
            freqs_cis = self.local_freqs_cis if local_sliding else self.global_freqs_cis
            mask = local_mask if local_sliding else global_mask
            x = layer(x, positions, freqs_cis, mask, get_extra_arg(i))
            break

        x = self.final_norm.call(x)

        if config.quant:
            x2d = x.reshape(-1, x.size(-1))
            x_fp32 = x2d.float()
            x_abs_max = x_fp32.abs().amax(-1, keepdim=True).clamp_min(1e-8)
            x_scale = x_abs_max / 127.0
            x_q = torch.round(x_fp32 / x_scale).clamp(-127, 127).to(torch.int8)

            weight_t = self.embedding.weight.T.contiguous()  # (embed_dim, vocab_size), int8
            out = torch._int_mm(x_q, weight_t)  # type: ignore  # int32, (B*T, vocab_size)
            logits = (
                out.float() * x_scale * self.embedding.weight_scaler.float()
            ).reshape(*x.shape[:-1], -1).to(config.dtype)
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
