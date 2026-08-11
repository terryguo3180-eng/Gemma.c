import math
import struct

from typing import TYPE_CHECKING

import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GemmaForCausalLM, Gemma2ForCausalLM, Gemma3ForCausalLM
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer, Gemma3Attention

if TYPE_CHECKING:
    from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer as Gemma1DecoderLayer
    from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer

from model import Embedding, GemmaConfig, GemmaDecoderBlock, GemmaModel, Linear, Tensor
from tokenizer import GemmaTokenizer


def export_hf(
    model_path: str,
    export_path: str,
    dtype: torch.dtype = torch.float16,
    quant: bool = False,
):
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True,
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
    assert isinstance(hf_model, (
        GemmaForCausalLM, Gemma2ForCausalLM, Gemma3ForCausalLM
    ))

    config = hf_model.config

    with open(export_path, "wb") as f:
        # Dump the model config
        f.write(struct.pack("> ccc HHHHH II fffff", *(
            # char (1 byte)
            bytes([config.num_hidden_layers]),
            bytes([config.num_attention_heads]),
            bytes([config.num_key_value_heads]),
            # unsigned short (2 bytes)
            config.head_dim,
            config.hidden_size,  # embed_dim
            config.intermediate_size,
            config.query_pre_attn_scalar,
            config.sliding_window or 0,
            # unsigned int (4 bytes)
            config.max_position_embeddings,
            config.vocab_size,
            # float (4 bytes)
            config.rope_parameters.get("sliding_attention", {}).get("rope_theta", 10000.0)
                if isinstance(hf_model, Gemma3ForCausalLM) and config.rope_parameters is not None
                else 10000.0,  # local_theta
            config.rope_parameters.get("full_attention", {}).get("rope_theta", 10000.0)
                if isinstance(hf_model, Gemma3ForCausalLM) and config.rope_parameters is not None
                else 10000.0,  # global_theta
            config.rms_norm_eps,
            config.attn_logit_softcapping or 0.0,
            config.final_logit_softcapping or 0.0,
        )))

        # attn_local_layers
        attn_local_layers = (
            [lt == "sliding_attention" for lt in config.layer_types]
            if isinstance(hf_model, (Gemma2ForCausalLM, Gemma3ForCausalLM))
                and config.layer_types is not None
            else [False]
        )
        n_layer_bytes = (len(attn_local_layers) + 7) // 8
        layer_bytes = bytearray((len(attn_local_layers) + 7) // 8)

        for i, bit in enumerate(attn_local_layers):
            if bit:
                layer_bytes[i // 8] |= (1 << (7 - i % 8))

        f.write(n_layer_bytes.to_bytes())
        f.write(layer_bytes)

        # Additional flags
        extra_flags = 0
        for bit in [
            isinstance(hf_model, Gemma3ForCausalLM),  # use_qk_norm
            isinstance(hf_model, (Gemma2ForCausalLM, Gemma3ForCausalLM)),  # pre_ffwd_norm
            isinstance(hf_model, (Gemma2ForCausalLM, Gemma3ForCausalLM)),  # post_ffwd_norm
            quant,
        ]:
            extra_flags = (extra_flags << 1) | bit
        f.write(extra_flags.to_bytes())

        def write_str(s: str):
            str_bytes = s.encode("utf-8")
            f.write(len(str_bytes).to_bytes())
            f.write(str_bytes)

        # dtype
        write_str(str(dtype).removeprefix("torch."))

        # Dump the tokenizer (vocab & merges)
        vocab: dict[str, int] = hf_tokenizer._vocab
        merges: list[list[str]] = hf_tokenizer._merges

        for tokstr in vocab.keys():
            write_str(tokstr.replace("▁", " "))

        f.write(len(merges).to_bytes(4))
        for lhs, rhs in merges:
            write_str(lhs.replace("▁", " "))
            write_str(rhs.replace("▁", " "))

        # Dump the weights
        def write_tensor(tensor: Tensor):
            f.write(tensor.contiguous().numpy().tobytes())

        emb_weight = hf_model.model.embed_tokens.weight.data.to(dtype)
        if quant:
            emb = Embedding(emb_weight.size(0), emb_weight.size(1), True, dtype)
            emb.weight.data = emb_weight
            emb.quantize()
            write_tensor(emb.weight_scaler.data)
            emb_weight = emb.weight.data
        write_tensor(emb_weight)

        for i in range(config.num_hidden_layers):
            layer: (
                Gemma1DecoderLayer | Gemma2DecoderLayer | Gemma3DecoderLayer
            ) = hf_model.model.layers[i]  # type: ignore

            hf_attn = layer.self_attn
            q_proj = hf_attn.q_proj.weight.data.to(dtype)
            k_proj = hf_attn.k_proj.weight.data.to(dtype)
            v_proj = hf_attn.v_proj.weight.data.to(dtype)
            o_proj = hf_attn.o_proj.weight.data.to(dtype)

            if quant:
                q_linear = Linear(q_proj.size(1), emb_weight.size(0), True, dtype)
                k_linear = Linear(k_proj.size(1), emb_weight.size(0), True, dtype)
                v_linear = Linear(v_proj.size(1), emb_weight.size(0), True, dtype)
                o_linear = Linear(o_proj.size(1), emb_weight.size(0), True, dtype)

                for mod, wei in (
                    (q_linear, q_proj),
                    (k_linear, k_proj),
                    (v_linear, v_proj),
                    (o_linear, o_proj),
                ):
                    mod.weight.data = wei.T
                    mod.quantize()
                    write_tensor(mod.weight_scaler.data)

                q_proj = q_linear.weight.data.T
                k_proj = k_linear.weight.data.T
                v_proj = v_linear.weight.data.T
                o_proj = o_linear.weight.data.T

            for wei in (q_proj, k_proj, v_proj, o_proj):
                write_tensor(wei)

            if isinstance(hf_attn, Gemma3Attention):  # use_qk_norm
                q_norm = hf_attn.q_norm.weight.data.to(dtype)
                k_norm = hf_attn.k_norm.weight.data.to(dtype)
                write_tensor(q_norm)
                write_tensor(k_norm)

            hf_ffwd = layer.mlp
            up_proj = hf_ffwd.up_proj.weight.data.to(dtype)
            gate_proj = hf_ffwd.gate_proj.weight.data.to(dtype)
            down_proj = hf_ffwd.down_proj.weight.data.to(dtype)

            if quant:
                up_linear = Linear(up_proj.size(1), up_proj.size(0), True, dtype)
                gate_linear = Linear(gate_proj.size(1), gate_proj.size(0), True, dtype)
                down_linear = Linear(down_proj.size(1), down_proj.size(0), True, dtype)

                for mod, wei in (
                    (up_linear, up_proj),
                    (gate_linear, gate_proj),
                    (down_linear, down_proj),
                ):
                    mod.weight.data = wei.T
                    mod.quantize()
                    write_tensor(mod.weight_scaler.data)

                up_proj = up_linear.weight.data.T
                gate_proj = gate_linear.weight.data.T
                down_proj = down_linear.weight.data.T

            for wei in (up_proj, gate_proj, down_proj):
                write_tensor(wei)

            norm1 = layer.input_layernorm.weight.data.to(dtype)
            norm2 = layer.post_attention_layernorm.weight.data.to(dtype)
            write_tensor(norm1)
            write_tensor(norm2)

            if isinstance(layer, Gemma3DecoderLayer):
                norm3 = layer.pre_feedforward_layernorm.weight.data.to(dtype)
                norm4 = layer.post_feedforward_layernorm.weight.data.to(dtype)
                write_tensor(norm3)
                write_tensor(norm4)

        final_norm = hf_model.model.norm.weight.data.to(dtype)
        write_tensor(final_norm)


def load_bin(path: str) -> tuple[GemmaModel, GemmaTokenizer]:
    with open(path, "rb") as f:
        # Build config
        fmt = "> ccc HHHHH II fffff"
        (
            n_layers, n_heads, n_kv_heads,
            head_dim, embed_dim, mlp_hidden_size,
            q_pre_attn_scalar, sliding_window,
            max_seq_len, vocab_size,
            local_theta, global_theta, eps,
            attn_softcapping, logit_softcapping,
        ) = struct.unpack(fmt, f.read(struct.calcsize(fmt)))

        n_layers = n_layers[0]
        n_heads = n_heads[0]
        n_kv_heads = n_kv_heads[0]

        def read_str():
            str_len = f.read(1)[0]
            return f.read(str_len).decode("utf-8")
            
        # attn_local_layers
        bits = 0
        local_layers_bytes = f.read(1)[0]
        for _ in range(local_layers_bytes):
            bits = (bits << 8) | f.read(1)[0]
        attn_local_layers = [bool((bits >> i) & 1) for i in range(
            local_layers_bytes * 8 - n_layers, local_layers_bytes * 8,
        )]
        attn_local_layers.reverse()
        
        # Additional flags
        extra_flags = f.read(1)[0]
        use_qk_norm = extra_flags & 8 == 8
        pre_ffwd_norm = extra_flags & 4 == 4
        post_ffwd_norm = extra_flags & 2 == 2
        quant = extra_flags & 1 == 1

        # dtype
        dtype_str = read_str()
        dtype = getattr(torch, dtype_str)
        dtype_np = getattr(np, dtype_str)

        config = GemmaConfig(
            embed_dim=embed_dim,
            n_layers=n_layers,
            attn_local_layers=attn_local_layers,
            max_seq_len=max_seq_len,
            sliding_window=sliding_window,
            vocab_size=vocab_size,
            local_theta=local_theta,
            global_theta=global_theta,
            eps=eps,
            n_heads=n_heads,
            head_dim=head_dim,
            n_kv_heads=n_kv_heads,
            q_pre_attn_scalar=q_pre_attn_scalar,
            use_qk_norm=use_qk_norm,
            attn_softcapping=attn_softcapping,
            mlp_hidden_size=mlp_hidden_size,
            pre_ffwd_norm=pre_ffwd_norm,
            post_ffwd_norm=post_ffwd_norm,
            logit_softcapping=logit_softcapping,
            dtype=dtype,
            quant=quant,
        )
        model = GemmaModel(config)

        # Build tokenizer
        vocab = {read_str(): i for i in range(vocab_size)}
        n_merges = int.from_bytes(f.read(4))
        merges = [(read_str(), read_str()) for _ in range(n_merges)]
        tokenizer = GemmaTokenizer(vocab, merges)

        # Build weights
        def read_tensor(shape: tuple, dtype=dtype_np) -> Tensor:
            numel = math.prod(shape)
            return torch.from_numpy(np.fromfile(f, dtype, numel)).view(shape)

        dtype_q = np.int8 if quant else dtype_np

        if quant:
            model.embedding.weight_scaler.data = read_tensor((vocab_size,))
        model.embedding.weight.data = read_tensor((vocab_size, embed_dim), dtype_q)

        for i in range(n_layers):
            layer: GemmaDecoderBlock = model.layers[i]  # type: ignore

            if quant:
                layer.attn.q_proj.weight_scaler.data = read_tensor((head_dim * n_heads,))
                layer.attn.k_proj.weight_scaler.data = read_tensor((head_dim * n_kv_heads,))
                layer.attn.v_proj.weight_scaler.data = read_tensor((head_dim * n_kv_heads,))
                layer.attn.o_proj.weight_scaler.data = read_tensor((embed_dim,))

            layer.attn.q_proj.weight.data = read_tensor((head_dim * n_heads, embed_dim), dtype_q).T
            layer.attn.k_proj.weight.data = read_tensor((head_dim * n_kv_heads, embed_dim), dtype_q).T
            layer.attn.v_proj.weight.data = read_tensor((head_dim * n_kv_heads, embed_dim), dtype_q).T
            layer.attn.o_proj.weight.data = read_tensor((embed_dim, head_dim * n_heads), dtype_q).T

            if use_qk_norm:
                layer.attn.q_norm.weight.data = read_tensor((head_dim,))
                layer.attn.k_norm.weight.data = read_tensor((head_dim,))

            if quant:
                layer.ffwd.up_proj.weight_scaler.data = read_tensor((mlp_hidden_size,))
                layer.ffwd.gate_proj.weight_scaler.data = read_tensor((mlp_hidden_size,))
                layer.ffwd.down_proj.weight_scaler.data = read_tensor((embed_dim,))

            layer.ffwd.up_proj.weight.data = read_tensor((mlp_hidden_size, embed_dim), dtype_q).T
            layer.ffwd.gate_proj.weight.data = read_tensor((mlp_hidden_size, embed_dim), dtype_q).T
            layer.ffwd.down_proj.weight.data = read_tensor((embed_dim, mlp_hidden_size), dtype_q).T

            layer.norm1.weight.data = read_tensor((embed_dim,))
            layer.norm2.weight.data = read_tensor((embed_dim,))

            if pre_ffwd_norm:
                layer.norm3.weight.data = read_tensor((embed_dim,))
            if post_ffwd_norm:
                layer.norm4.weight.data = read_tensor((embed_dim,))

        model.final_norm.weight.data = read_tensor((embed_dim,))

        return model, tokenizer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export Hugging Face Gemma models to custom .bin format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export.py -m ./Models/gemma-3-270m -o gemma-3-270m-pt.bin
  python export.py -m ./Models/gemma-3-270m-it -o gemma-3-270m-it.bin --quantize
  python export.py -m ./Models/gemma-3-1b-pt -o gemma-3-1b-pt.bin -d float32
        """
    )
    
    parser.add_argument(
        "-m", "--model-path",
        required=True,
        help="Path to the Hugging Face model id / local directory"
    )
    
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output binary file path"
    )
    
    parser.add_argument(
        "-d", "--dtype",
        choices=["float16", "float32", "bfloat16"],
        default="float16",
        help="Data type for model weights (default: float16)"
    )
    
    parser.add_argument(
        "-q", "--quantize",
        action="store_true",
        help="Enable 8-bit quantization"
    )
    
    args = parser.parse_args()
    
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    export_hf(args.model_path, args.output, dtype, args.quantize)
    print(f"Successfully exported to {args.output}")
