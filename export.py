import math
import struct

import numpy as np
import torch

from model import Embedding, GemmaConfig, GemmaDecoderBlock, GemmaModel, Linear
from tokenizer import GemmaTokenizer


def export_hf(
    model_path: str,
    export_path: str,
    dtype: torch.dtype = torch.float16,
    quant: bool = False,
    cache_dir: str | None = None,
):
    import gc
    import os
    import json

    from accelerate import init_empty_weights
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    from transformers import AutoConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import GemmaForCausalLM, \
        Gemma2ForCausalLM, Gemma3ForCausalLM, \
        Gemma3ForConditionalGeneration
    
    from tqdm import tqdm

    GemmaLLMs = (GemmaForCausalLM, Gemma2ForCausalLM, Gemma3ForCausalLM)
    GemmaVLMs = (Gemma3ForConditionalGeneration,)
    Gemma3Models = (Gemma3ForCausalLM, Gemma3ForConditionalGeneration)

    from collections import OrderedDict

    class LazyStateDict:
        """
        Reads individual tensors from a (possibly sharded) safetensors checkpoint
        on demand, without ever materializing the full model in memory. Keeps at most
        `max_open_handles` shard files mmap'd at once, evicting the least-recently-used
        one, otherwise every shard touched so far stays resident until the very end,
        which for a 12B model means the process can end up holding nearly the entire
        checkpoint in memory simultaneously by the time it reaches the last shard.
        """

        def __init__(self, model_dir: str, max_open_handles: int = 1):
            index_path = os.path.join(model_dir, "model.safetensors.index.json")
            if os.path.exists(index_path):
                with open(index_path) as fp:
                    weight_map = json.load(fp)["weight_map"]
            else:
                single_file = os.path.join(model_dir, "model.safetensors")
                if not os.path.exists(single_file):
                    raise FileNotFoundError(
                        f"No safetensors checkpoint found in {model_dir}. "
                        "This export path only supports safetensors checkpoints."
                    )
                with safe_open(single_file, framework="pt") as fp:
                    weight_map = {k: "model.safetensors" for k in fp.keys()}

            self.model_dir = model_dir
            self.weight_map = weight_map
            self.max_open_handles = max_open_handles
            self._handles: OrderedDict[str, safe_open] = OrderedDict()

        def _handle(self, filename: str):
            if filename in self._handles:
                self._handles.move_to_end(filename)
                return self._handles[filename]

            handle = safe_open(
                os.path.join(self.model_dir, filename), framework="pt", device="cpu"
            )
            self._handles[filename] = handle

            # Evict the least-recently-used shard(s) so we never hold more than
            # `max_open_handles` mmap'd files at once.
            while len(self._handles) > self.max_open_handles:
                old_name, old_handle = self._handles.popitem(last=False)
                del old_handle
                gc.collect()

            return handle

        def get(self, name: str) -> torch.Tensor:
            filename = self.weight_map.get(name)
            if filename is None:
                raise KeyError(f"Tensor '{name}' not found in checkpoint")
            # .clone() is mandatory here: get_tensor() can return a zero-copy view
            # directly backed by the mmap region. Without cloning, the "materialized"
            # tensor stays tied to the underlying file mapping, and if the caller's
            # target dtype happens to match the stored dtype, a later `.to(dtype)`
            # becomes a no-op that returns the very same mmap-backed tensor, silently
            # keeping the whole shard's pages pinned in the working set.
            return self._handle(filename).get_tensor(name).clone()
        
    def find_prefix(weight_map: dict, suffix: str) -> str:
        """
        Detect the checkpoint's actual key prefix for the text/decoder submodule
        by looking for a unique tensor name ending in `suffix`, instead of assuming
        a fixed module hierarchy (which can change across transformers versions).
        """
        candidates = {k[: -len(suffix)] for k in weight_map if k.endswith(suffix)}
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one tensor ending with {suffix!r} in the checkpoint, "
                f"found {len(candidates)}: {sorted(candidates)}"
            )
        return candidates.pop()

    # Resolve model_path (local dir or hub repo id) to a local directory containing
    # the config/tokenizer/safetensors files, without loading any weights yet.
    if os.path.isdir(model_path):
        local_dir = model_path
    else:
        local_dir = snapshot_download(
            model_path,
            cache_dir=cache_dir,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*"],
        )

    hf_config = AutoConfig.from_pretrained(local_dir)
    hf_tokenizer = AutoTokenizer.from_pretrained(local_dir)

    # Only used to classify the architecture (isinstance checks). meta device means
    # this costs ~0 memory and ~0 time regardless of model size.
    with init_empty_weights():
        hf_model = AutoModelForCausalLM.from_config(hf_config)
    assert isinstance(hf_model, GemmaLLMs + GemmaVLMs)

    if isinstance(hf_model, GemmaLLMs):
        config = hf_model.config
    elif isinstance(hf_model, GemmaVLMs):
        config = hf_model.config.text_config
        assert not isinstance(config, dict)
        assert config is not None
    else:
        assert False

    is_gemma3 = isinstance(hf_model, Gemma3Models)
    # use_qk_norm / pre_ffwd_norm / post_ffwd_norm are uniform across all layers for a
    # given model family, so a single global flag (rather than a per-layer isinstance
    # check against a live module) is enough and avoids depending on module identity.
    has_qk_norm = is_gemma3
    has_pre_post_ffwd_norm = isinstance(hf_model, (Gemma2ForCausalLM, *Gemma3Models))

    weights = LazyStateDict(local_dir)
    # Detect the checkpoint's actual prefix for the decoder stack (e.g. "model." for
    # plain Gemma LLMs, or "language_model.model." for Gemma3 VLM checkpoints)
    text_prefix = find_prefix(weights.weight_map, "embed_tokens.weight")

    def get(local_name: str) -> torch.Tensor:
        return weights.get(text_prefix + local_name).to(dtype)

    # Can't use config.vocab_size here, sometimes people set vocab_size to the next power
    # of 2 to keep GPUs busy, leaving a bunch of unused tokens. We just simply discard
    # those
    vocab_len = len(hf_tokenizer._vocab)

    # An annoying issue when dealing with fp16 inference. Sometimes the activations
    # get extremely large at runtime that they could overflow the fp16 limit, my first
    # attempt is just to clamp the activations to the valid range, it worked okay for
    # 1B and 4B models, but the 12B model was somewhat broken and made mistakes that it
    # shouldn't make (e.g. having problem with apostrophes or simple spelling). It
    # turned out that these massive outliers carry important variance information,
    # and naively clamping them would significantly impact the model's performance.
    # The fix is to scale down the embedding weight & post norm weights, the math
    # works the same in this way (since every pre-norm just scales them back up), but
    # it ensures all the activations to fit inside the fp16 range.
    # According to this PR: https://github.com/huggingface/transformers/pull/37226,
    # a nice value of this activation scaler is 1/sqrt(hidden_size), which is what I'm
    # using right here.
    act_scale = 1 / math.sqrt(config.hidden_size)

    with open(export_path, "wb") as f:
        # Dump the model config
        f.write(struct.pack("> ccc HHHHH II fffff", *(
            bytes([config.num_hidden_layers]),
            bytes([config.num_attention_heads]),
            bytes([config.num_key_value_heads]),
            config.head_dim,
            config.hidden_size,
            config.intermediate_size,
            config.query_pre_attn_scalar,
            config.sliding_window or 0,
            config.max_position_embeddings,
            vocab_len,
            config.rope_parameters.get("sliding_attention", {}).get("rope_theta", 10000.0)
                if isinstance(hf_model, Gemma3Models) and config.rope_parameters is not None
                else 10000.0,
            config.rope_parameters.get("full_attention", {}).get("rope_theta", 10000.0)
                if isinstance(hf_model, Gemma3Models) and config.rope_parameters is not None
                else 10000.0,
            config.rms_norm_eps,
            config.attn_logit_softcapping or 0.0,
            config.final_logit_softcapping or 0.0,
        )))

        # attn_local_layers
        # I'm using bit encoding here. Idk why, I regret doing this...
        attn_local_layers = (
            [lt == "sliding_attention" for lt in config.layer_types]
            if isinstance(hf_model, (Gemma2ForCausalLM, *Gemma3Models))
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
        extra_flags = (
            (int(has_qk_norm) << 3) |
            (int(has_pre_post_ffwd_norm) << 2) |
            (int(has_pre_post_ffwd_norm) << 1) |
            int(quant)
        )
        f.write(extra_flags.to_bytes())

        def write_str(s: str):
            str_bytes = s.encode("utf-8")
            f.write(len(str_bytes).to_bytes())
            f.write(str_bytes)

        # Write dtype
        write_str(str(dtype).removeprefix("torch."))

        # Dump the tokenizer (vocab & merges)
        vocab: dict[str, int] = hf_tokenizer._vocab
        merges: list[list[str]] = hf_tokenizer._merges

        for tokstr, _ in sorted(vocab.items(), key=lambda kv: kv[1]):
            write_str(tokstr.replace("▁", " "))

        f.write(len(merges).to_bytes(4))
        for lhs, rhs in merges:
            write_str(lhs.replace("▁", " "))
            write_str(rhs.replace("▁", " "))

        def write_tensor(tensor: torch.Tensor):
            arr = tensor.contiguous().numpy()
            arr.tofile(f)

        # Get the total amount of weight tensors for progress bar display
        total = config.num_hidden_layers * 7
        if quant:
            total *= 2  # Quantization scalers
        total += config.num_hidden_layers * 2
        if has_qk_norm:
            total += config.num_hidden_layers * 2  # q norm & k norm
        if has_pre_post_ffwd_norm:
            total += config.num_hidden_layers * 2  # pre norm & post norm
        total += 2  # embedding & final norm
        pbar = tqdm(total=total, desc="Writing weights")
        pbar.refresh()

        # NOTE: Do NOT multiply emb_weight by act_scale
        # Embedding is weight-tied with lm_head. Scaling it would scale the logits
        # and is equivalent to a large temperature change, which destroys sampling.
        emb_weight = get("embed_tokens.weight")[:vocab_len, :]
        if quant:
            # Bit of hacky monkey patch here, can't really think of a better way to do this
            emb = Embedding(emb_weight.size(0), emb_weight.size(1), True, dtype)
            emb.weight.data = emb_weight
            emb.quantize()
            emb_weight = emb.weight.data
        write_tensor(emb_weight)
        if quant:
            write_tensor(emb.weight_scaler.data)  # type: ignore
        pbar.update()
        del emb_weight
        if quant:
            del emb  # type: ignore

        # Write all the weights across all the layers
        for i in range(config.num_hidden_layers):
            # Attention weights
            q_proj = get(f"layers.{i}.self_attn.q_proj.weight")
            k_proj = get(f"layers.{i}.self_attn.k_proj.weight")
            v_proj = get(f"layers.{i}.self_attn.v_proj.weight")
            o_proj = get(f"layers.{i}.self_attn.o_proj.weight")

            # Weight scalers for quantization
            weight_scalers: list = [None, None, None, None]
            if quant:
                # Export transposed matrices
                q_linear = Linear(q_proj.size(1), q_proj.size(0), True, dtype)
                k_linear = Linear(k_proj.size(1), k_proj.size(0), True, dtype)
                v_linear = Linear(v_proj.size(1), v_proj.size(0), True, dtype)
                o_linear = Linear(o_proj.size(1), o_proj.size(0), True, dtype)

                for j, (mod, wei) in enumerate((
                    (q_linear, q_proj), (k_linear, k_proj),
                    (v_linear, v_proj), (o_linear, o_proj)
                )):
                    mod.weight.data = wei.T
                    mod.quantize()
                    pbar.update()
                    weight_scalers[j] = mod.weight_scaler.data

                q_proj = q_linear.weight.data.T
                k_proj = k_linear.weight.data.T
                v_proj = v_linear.weight.data.T
                o_proj = o_linear.weight.data.T
                del q_linear, k_linear, v_linear, o_linear

            # Write all the weights and weight scalers
            for wei, sca in zip((q_proj, k_proj, v_proj, o_proj), weight_scalers):
                write_tensor(wei)
                if sca is not None:
                    write_tensor(sca)
                pbar.update()
            del weight_scalers, q_proj, k_proj, v_proj, o_proj

            # q&k RMSNorms
            if has_qk_norm:
                q_norm = get(f"layers.{i}.self_attn.q_norm.weight")
                k_norm = get(f"layers.{i}.self_attn.k_norm.weight")
                write_tensor(q_norm)
                pbar.update()
                write_tensor(k_norm)
                pbar.update()
                del q_norm, k_norm

            # Feedforward MLP weights
            up_proj = get(f"layers.{i}.mlp.up_proj.weight")
            gate_proj = get(f"layers.{i}.mlp.gate_proj.weight")
            down_proj = get(f"layers.{i}.mlp.down_proj.weight")

            # MLP weight scalers
            weight_scalers = [None, None, None]
            if quant:
                up_linear = Linear(up_proj.size(1), up_proj.size(0), True, dtype)
                gate_linear = Linear(gate_proj.size(1), gate_proj.size(0), True, dtype)
                down_linear = Linear(down_proj.size(1), down_proj.size(0), True, dtype)

                for j, (mod, wei) in enumerate((
                    (up_linear, up_proj),
                    (gate_linear, gate_proj),
                    (down_linear, down_proj),
                )):
                    mod.weight.data = wei.T
                    mod.quantize()
                    pbar.update()
                    weight_scalers[j] = mod.weight_scaler.data

                up_proj = up_linear.weight.data.T
                gate_proj = gate_linear.weight.data.T
                down_proj = down_linear.weight.data.T
                del up_linear, gate_linear, down_linear

            # Write weights & scalers
            for wei, sca in zip((up_proj, gate_proj, down_proj), weight_scalers):
                write_tensor(wei)
                if sca is not None:
                    write_tensor(sca)
                pbar.update()
            del weight_scalers, up_proj, gate_proj, down_proj

            # Attention pre-norm
            norm1 = get(f"layers.{i}.input_layernorm.weight")
            # Attention post-norm
            norm2 = get(f"layers.{i}.post_attention_layernorm.weight")
            write_tensor(norm1)
            pbar.update()

            # Scale down the post norm, be careful with the +1 / -1 here, since Gemma
            # uses (weight + 1) as the actual coefficients
            norm2 += 1.0
            norm2 *= act_scale
            norm2 -= 1.0

            write_tensor(norm2)
            pbar.update()
            del norm1, norm2

            if has_pre_post_ffwd_norm:
                # Feedforward pre-norm
                norm3 = get(f"layers.{i}.pre_feedforward_layernorm.weight")
                # Feedforward post-norm
                norm4 = get(f"layers.{i}.post_feedforward_layernorm.weight")
                write_tensor(norm3)
                pbar.update()

                # Scale down the post norm, same way as norm2
                norm4 += 1.0
                norm4 *= act_scale
                norm4 -= 1.0

                write_tensor(norm4)
                pbar.update()
                del norm3, norm4

            gc.collect()

        final_norm = get("norm.weight")
        write_tensor(final_norm)
        pbar.update()


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
        local_layers_bytes = f.read(1)[0]
        layer_bytes = f.read(local_layers_bytes)
        attn_local_layers = []
        for i in range(n_layers):
            byte_index = i // 8
            bit_index = 7 - (i % 8)
            attn_local_layers.append(bool(layer_bytes[byte_index] & (1 << bit_index)))
        
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
        def read_tensor(shape: tuple, dtype=dtype_np) -> torch.Tensor:
            numel = math.prod(shape)
            return torch.from_numpy(np.fromfile(f, dtype, numel)).view(shape)

        dtype_q = np.int8 if quant else dtype_np

        model.embedding.weight.data = read_tensor((vocab_size, embed_dim), dtype_q)
        if quant:
            model.embedding.weight_scaler.data = read_tensor((vocab_size,))

        for i in range(n_layers):
            layer: GemmaDecoderBlock = model.layers[i]  # type: ignore

            layer.attn.q_proj.weight.data = read_tensor((head_dim * n_heads, embed_dim), dtype_q).T
            if quant:
                layer.attn.q_proj.weight_scaler.data = read_tensor((head_dim * n_heads,))
            layer.attn.k_proj.weight.data = read_tensor((head_dim * n_kv_heads, embed_dim), dtype_q).T
            if quant:
                layer.attn.k_proj.weight_scaler.data = read_tensor((head_dim * n_kv_heads,))
            layer.attn.v_proj.weight.data = read_tensor((head_dim * n_kv_heads, embed_dim), dtype_q).T
            if quant:
                layer.attn.v_proj.weight_scaler.data = read_tensor((head_dim * n_kv_heads,))
            layer.attn.o_proj.weight.data = read_tensor((embed_dim, head_dim * n_heads), dtype_q).T
            if quant:
                layer.attn.o_proj.weight_scaler.data = read_tensor((embed_dim,))

            if use_qk_norm:
                layer.attn.q_norm.weight.data = read_tensor((head_dim,))
                layer.attn.k_norm.weight.data = read_tensor((head_dim,))

            layer.ffwd.up_proj.weight.data = read_tensor((mlp_hidden_size, embed_dim), dtype_q).T
            if quant:
                layer.ffwd.up_proj.weight_scaler.data = read_tensor((mlp_hidden_size,))
            layer.ffwd.gate_proj.weight.data = read_tensor((mlp_hidden_size, embed_dim), dtype_q).T
            if quant:
                layer.ffwd.gate_proj.weight_scaler.data = read_tensor((mlp_hidden_size,))
            layer.ffwd.down_proj.weight.data = read_tensor((embed_dim, mlp_hidden_size), dtype_q).T
            if quant:
                layer.ffwd.down_proj.weight_scaler.data = read_tensor((embed_dim,))

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
    
    parser = argparse.ArgumentParser(description="Export Hugging Face Gemma models to custom .bin format")
    parser.add_argument(
        "modelfile",
        help="Path to the Hugging Face model id / local directory",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output binary file path",
    )
    parser.add_argument(
        "-d", "--dtype",
        choices=["float16", "float32", "bfloat16"],
        default="float16",
        help="Data type for model weights (default: float16)",
    )
    parser.add_argument(
        "-q", "--quantize",
        action="store_true",
        help="Enable 8-bit quantization",
    )
    parser.add_argument(
        "-c", "--cache-path",
        type=str,
        help="Hugging Face cache directory",
    )
    
    args = parser.parse_args()
    
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    export_hf(args.modelfile, args.output, dtype, args.quantize, args.cache_path)
    print(f"Successfully exported to {args.output}")
