import math
import struct

import numpy as np
import torch

from model import (
    Embedding,
    GemmaConfig,
    GemmaDecoderBlock,
    GemmaModel,
    GemmaVisionConfig,
    Linear,
    SiglipEncoderBlock,
)
from tokenizer import GemmaTokenizer


def export_hf(
    model_path: str,
    export_path: str,
    dtype: torch.dtype = torch.float16,
    quant: bool = False,
    cache_dir: str | None = None,
):
    # Pretty messy code, have to deal with all kinds of memory issues, Hugging Face
    # interfaces, and activation scalers due to float16 weirdness.
    # I do not want to deal with activation scalers ever again in my life. ever.

    import gc
    import os
    import json

    from accelerate import init_empty_weights
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    from transformers import AutoConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import (
        GemmaForCausalLM,
        Gemma2ForCausalLM,
        Gemma3ForCausalLM,
        Gemma3ForConditionalGeneration,
    )
    from transformers import SiglipVisionConfig
    
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
        vconfig = None
    elif isinstance(hf_model, GemmaVLMs):
        config = hf_model.config.text_config
        vconfig = hf_model.config.vision_config
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
    # Detect the checkpoint's actual prefix for the text decoder stack (e.g. "model."
    # for plain Gemma LLMs, or "language_model.model." for Gemma3 VLM checkpoints)
    text_prefix = find_prefix(weights.weight_map, "embed_tokens.weight")
    # Same for the vision tower
    vision_prefix = find_prefix(weights.weight_map, "embeddings.patch_embedding.weight")
    # And multi-modal projector
    mmproj_prefix = find_prefix(weights.weight_map, "mm_soft_emb_norm.weight")

    def get(local_name: str, prefix=text_prefix) -> torch.Tensor: 
        return weights.get(prefix + local_name).to(dtype)

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

    is_multimodal = isinstance(hf_model, GemmaVLMs)
    tokens_per_image = hf_model.config.mm_tokens_per_image if is_multimodal else 0

    with open(export_path, "wb") as f:
        # Dump the model config
        f.write(struct.pack("> ccc HHHHHH II fffff", *(
            bytes([config.num_hidden_layers]),
            bytes([config.num_attention_heads]),
            bytes([config.num_key_value_heads]),
            config.head_dim,
            config.hidden_size,
            config.intermediate_size,
            config.query_pre_attn_scalar,
            config.sliding_window or 0,
            tokens_per_image,
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
            (int(is_multimodal) << 4) |
            (int(has_qk_norm) << 3) |
            (int(has_pre_post_ffwd_norm) << 2) |
            (int(has_pre_post_ffwd_norm) << 1) |
            int(quant)
        )
        f.write(extra_flags.to_bytes())

        if is_multimodal:
            v_act_scale = 1 / math.sqrt(vconfig.hidden_size)  # type: ignore  # activation scale for SigLIP
        else:
            v_act_scale = 0.0  # To make Pylance happy

        if is_multimodal:
            # Dump the vision model config
            assert isinstance(vconfig, SiglipVisionConfig)
            f.write(struct.pack("> cc HHHH ff", *(
                bytes([vconfig.num_hidden_layers]),
                bytes([vconfig.num_attention_heads]),
                vconfig.intermediate_size,
                vconfig.hidden_size,
                vconfig.image_size,
                vconfig.patch_size,
                vconfig.layer_norm_eps * (v_act_scale ** 2)
            )))

        def write_str(s: str):
            str_bytes = s.encode("utf-8")
            f.write(len(str_bytes).to_bytes())
            f.write(str_bytes)

        # Write dtype
        write_str(str(dtype).removeprefix("torch."))

        # Dump the tokenizer (vocab & merges)
        vocab: dict[str, int] = hf_tokenizer._vocab
        merges: list[list[str]] = hf_tokenizer._merges
        if is_multimodal:
            vocab["<image_soft_token>"] = vocab_len  # Placeholder for image tokens

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
        # Such a terrible idea :(
        total = config.num_hidden_layers * 7  # q & k & v & o & gate & up & down
        if quant:
            total *= 2  # Quantization scalers
        total += config.num_hidden_layers * 2
        if has_qk_norm:
            total += config.num_hidden_layers * 2  # q norm & k norm
        if has_pre_post_ffwd_norm:
            total += config.num_hidden_layers * 2  # pre norm & post norm
        total += 2  # embedding & final norm
        if is_multimodal:
            assert isinstance(vconfig, SiglipVisionConfig)
            total += 3  # siglip patch embedding (weight & bias) & position embedding
            if quant:
                total += 2  # Quantization scalers for embeddings
            total += vconfig.num_hidden_layers * 16  # q & k & v & o & fc1 & fc2 & two layernorms (weight & bias)
            if quant:
                total += vconfig.num_hidden_layers * 6  # Quantization scalers
            total += 2  # mm proj norm & mm proj weight
            if quant:
                total += 1  # Quantization scalers for mm proj

        pbar = tqdm(total=total, desc="Writing weights")
        pbar.refresh()

        def write_emb(name, prefix=text_prefix, t=False, trunc=None, scale=None):
            emb_weight = get(name, prefix)
            if trunc:
                emb_weight = emb_weight[:trunc, :]
            if t:
                emb_weight = emb_weight.T
            if quant:
                # Bit of hacky monkey patch here, can't really think of a better way to do this
                emb = Embedding(emb_weight.size(0), emb_weight.size(1), True, dtype)
                emb.weight.data = emb_weight
                if scale is not None:
                    emb.weight.data *= scale
                emb.quantize()
                emb_weight = emb.weight.data
            elif scale is not None:
                emb_weight *= scale
            write_tensor(emb_weight)
            if quant:
                write_tensor(emb.weight_scaler.data)  # type: ignore
            pbar.update()
            del emb_weight
            if quant:
                del emb  # type: ignore

        # NOTE: Do NOT multiply emb_weight by act_scale
        # Embedding is weight-tied with lm_head. Scaling it would scale the logits
        # and is equivalent to a large temperature change, which destroys sampling.
        write_emb("embed_tokens.weight", trunc=vocab_len)

        def write_linears(*names, prefix=text_prefix, scale_last: float | None = None):
            weights = [get(n, prefix) for n in names]
            if scale_last is not None:
                weights[-1] = weights[-1] * scale_last

            weight_scalers: list = [None for _ in names]
            
            # Weight scalers for quantization
            if quant:
                # Export transposed matrices
                for i, wei in enumerate(weights):
                    w_linear = Linear(wei.size(1), wei.size(0), True, dtype)
                    w_linear.weight.data = wei.T
                    w_linear.quantize()
                    pbar.update()
                    weight_scalers[i] = w_linear.weight_scaler.data
                    weights[i] = w_linear.weight.data.T
                    del w_linear

            for wei, sca in zip(weights, weight_scalers):
                write_tensor(wei)
                if sca is not None:
                    write_tensor(sca)
                pbar.update()

            del weights, weight_scalers

        # Write all the weights across all the layers
        for i in range(config.num_hidden_layers):
            # Attention weights
            write_linears(
                f"layers.{i}.self_attn.q_proj.weight",
                f"layers.{i}.self_attn.k_proj.weight",
                f"layers.{i}.self_attn.v_proj.weight",
                f"layers.{i}.self_attn.o_proj.weight",
            )
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
            write_linears(
                f"layers.{i}.mlp.up_proj.weight",
                f"layers.{i}.mlp.gate_proj.weight",
                f"layers.{i}.mlp.down_proj.weight",
            )

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

        # Export weights in the SigLIP vision encoder
        if is_multimodal:
            assert isinstance(vconfig, SiglipVisionConfig)

            # Patch embedding
            patch_emb_w = get("embeddings.patch_embedding.weight", vision_prefix)
            patch_emb_w *= v_act_scale
            write_tensor(patch_emb_w)
            pbar.update()
            patch_emb_b = get("embeddings.patch_embedding.bias", vision_prefix)
            patch_emb_b *= v_act_scale
            write_tensor(patch_emb_b)
            pbar.update()

            # Position embedding
            write_emb("embeddings.position_embedding.weight", vision_prefix, scale=v_act_scale)
            pbar.update()

            # Write all the layers of the ViT
            for i in range(vconfig.num_hidden_layers):
                # First layernorm
                norm1_w = get(f"encoder.layers.{i}.layer_norm1.weight", vision_prefix)
                norm1_b = get(f"encoder.layers.{i}.layer_norm1.bias", vision_prefix)
                write_tensor(norm1_w)
                pbar.update()
                write_tensor(norm1_b)
                pbar.update()
                del norm1_w, norm1_b

                # Attention weights
                write_linears(
                    f"encoder.layers.{i}.self_attn.q_proj.weight",
                    f"encoder.layers.{i}.self_attn.k_proj.weight",
                    f"encoder.layers.{i}.self_attn.v_proj.weight",
                    f"encoder.layers.{i}.self_attn.out_proj.weight",
                    prefix=vision_prefix, scale_last=v_act_scale,
                )
                q_proj_b = get(f"encoder.layers.{i}.self_attn.q_proj.bias", vision_prefix)
                k_proj_b = get(f"encoder.layers.{i}.self_attn.k_proj.bias", vision_prefix)
                v_proj_b = get(f"encoder.layers.{i}.self_attn.v_proj.bias", vision_prefix)
                o_proj_b = get(f"encoder.layers.{i}.self_attn.out_proj.bias", vision_prefix)
                o_proj_b *= v_act_scale

                for bias in (q_proj_b, k_proj_b, v_proj_b, o_proj_b):
                    write_tensor(bias)
                    pbar.update()

                del q_proj_b, k_proj_b, v_proj_b, o_proj_b

                # Second layernorm
                norm2_w = get(f"encoder.layers.{i}.layer_norm2.weight", vision_prefix)
                norm2_b = get(f"encoder.layers.{i}.layer_norm2.bias", vision_prefix)
                for wb in (norm2_w, norm2_b):
                    write_tensor(wb)
                    pbar.update()
                del norm2_w, norm2_b

                # MLP
                write_linears(
                    f"encoder.layers.{i}.mlp.fc1.weight",
                    f"encoder.layers.{i}.mlp.fc2.weight",
                    prefix=vision_prefix, scale_last=v_act_scale,
                )
                fc1_b = get(f"encoder.layers.{i}.mlp.fc1.bias", vision_prefix)
                fc2_b = get(f"encoder.layers.{i}.mlp.fc2.bias", vision_prefix)
                fc2_b *= v_act_scale

                for bias in (fc1_b, fc2_b):
                    write_tensor(bias)
                    pbar.update()

                del fc1_b, fc2_b
                gc.collect()

            # Post-layernorm
            post_ln_w = get("post_layernorm.weight", vision_prefix)
            post_ln_b = get("post_layernorm.bias", vision_prefix)
            write_tensor(post_ln_w)
            pbar.update()
            write_tensor(post_ln_b)
            pbar.update()
            del post_ln_w, post_ln_b

            # Muti-modal projector
            mm_norm = get("mm_soft_emb_norm.weight", mmproj_prefix)
            # DO NOT multiply by v_act_scale here, this would break the pretrained scale
            write_tensor(mm_norm)
            pbar.update()
            del mm_norm

            # t=True cuz we are quantizing per-channel 
            # scale=act_scale because we want to map back to the scale of text embeddings
            write_emb("mm_input_projection_weight", prefix=mmproj_prefix, t=True, scale=act_scale)


def load_bin(path: str) -> tuple[GemmaModel, GemmaTokenizer]:
    with open(path, "rb") as f:
        # Build config
        fmt = "> ccc HHHHHH II fffff"; (
            n_layers, n_heads, n_kv_heads,
            head_dim, embed_dim, mlp_hidden_size, q_pre_attn_scalar, sliding_window, tokens_per_image,
            max_seq_len, vocab_size,
            local_theta, global_theta, eps, attn_softcapping, logit_softcapping,
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
        is_multimodal = extra_flags & 16 == 16
        use_qk_norm = extra_flags & 8 == 8
        pre_ffwd_norm = extra_flags & 4 == 4
        post_ffwd_norm = extra_flags & 2 == 2
        quant = extra_flags & 1 == 1

        if is_multimodal:
            # Just for comparative verification. It is incredibly slow to run the 4B vision
            # model in this crappy PyTorch implementation :'D
            fmt = "> cc HHHH f"; (
                v_n_layers, v_n_heads, v_mlp_hidden_size, v_hidden_dim,
                v_image_size, v_patch_size, v_eps
            ) = struct.unpack(fmt, f.read(struct.calcsize(fmt)))

            v_n_layers = v_n_layers[0]
            v_n_heads = v_n_heads[0]

            vision_config = GemmaVisionConfig(
                n_layers=v_n_layers,
                image_size=v_image_size,
                patch_size=v_patch_size,
                hidden_dim=v_hidden_dim,
                n_heads=v_n_heads,
                mlp_hidden_size=v_mlp_hidden_size,
                eps=v_eps,
            )
        else:
            vision_config = None
            v_image_size = 0

        # dtype
        dtype_str = read_str()
        dtype = getattr(torch, dtype_str)
        dtype_np = getattr(np, dtype_str)

        # Model config
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
            tokens_per_image=tokens_per_image,
            vision_config=vision_config,
            dtype=dtype,
            quant=quant,
        )

        # Build tokenizer
        vocab_len = vocab_size + 1 if is_multimodal else vocab_size  # +1 for <image_soft_token>
        vocab = {read_str(): i for i in range(vocab_len)}
        n_merges = int.from_bytes(f.read(4))
        merges = [(read_str(), read_str()) for _ in range(n_merges)]
        tokenizer = GemmaTokenizer(vocab, merges, tokens_per_image, v_image_size)

        # Build model
        model = GemmaModel(
            config,
            tokenizer.vocab["<start_of_image>"],
            tokenizer.vocab["<end_of_image>"],
            tokenizer.vocab["<image_soft_token>"],
        )

        # Build weights
        def read_tensor(shape: tuple, dtype=dtype_np) -> torch.Tensor:
            numel = math.prod(shape)
            return torch.from_numpy(np.fromfile(f, dtype, numel)).view(shape)

        dtype_q = np.int8 if quant else dtype_np

        model.embedding.weight.data = read_tensor((vocab_size, embed_dim), dtype_q)
        if quant:
            model.embedding.weight_scaler.data = read_tensor((vocab_size,))

        if is_multimodal:
            # Insert an extra row for <image_soft_token> to the embedding table
            extra = torch.zeros(1, embed_dim, dtype=model.embedding.weight.dtype)
            model.embedding.weight.data = torch.cat([model.embedding.weight.data, extra], dim=0)
            if quant:
                extra_scale = torch.ones(1, dtype=model.embedding.weight_scaler.dtype)
                model.embedding.weight_scaler.data = torch.cat(
                    [model.embedding.weight_scaler.data, extra_scale], dim=0
                )

        # Boring... *sigh*
        for i in range(n_layers):
            layer: GemmaDecoderBlock = model.layers[i]  # type: ignore

            # We need to .T back the (already transposed) linear weights cuz that's how Linear works in model.py
            # No need to do this in gemma.c though
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

        # And there's more :D
        if is_multimodal:
            assert vision_config is not None
            ve = model.vision_encoder
            v_hidden_dim = vision_config.hidden_dim
            v_mlp_hidden_size = vision_config.mlp_hidden_size

            ve.patch_emb.weight.data = read_tensor(
                (v_hidden_dim, 3, vision_config.patch_size, vision_config.patch_size), dtype_np
            )
            ve.patch_emb.bias.data = read_tensor((v_hidden_dim,))  # type: ignore

            # Position embedding
            num_patches = (vision_config.image_size // vision_config.patch_size)**2
            ve.pos_emb.weight.data = read_tensor((num_patches, v_hidden_dim), dtype_q)
            if quant:
                ve.pos_emb.weight_scaler.data = read_tensor((num_patches,))

            for i in range(vision_config.n_layers):
                # Use a different name here to avoid type conflit in IDEs
                vlayer: SiglipEncoderBlock = ve.layers[i]  # type: ignore

                vlayer.norm1.weight.data = read_tensor((v_hidden_dim,))
                vlayer.norm1.bias.data = read_tensor((v_hidden_dim,))

                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    proj: Linear = getattr(vlayer.attn, name)
                    proj.weight.data = read_tensor((v_hidden_dim, v_hidden_dim), dtype_q).T
                    if quant:
                        proj.weight_scaler.data = read_tensor((v_hidden_dim,))

                vlayer.attn.q_bias.data = read_tensor((v_hidden_dim,))
                vlayer.attn.k_bias.data = read_tensor((v_hidden_dim,))
                vlayer.attn.v_bias.data = read_tensor((v_hidden_dim,))
                vlayer.attn.o_bias.data = read_tensor((v_hidden_dim,))

                vlayer.norm2.weight.data = read_tensor((v_hidden_dim,))
                vlayer.norm2.bias.data = read_tensor((v_hidden_dim,))

                vlayer.ffwd.fc1.weight.data = read_tensor((v_mlp_hidden_size, v_hidden_dim), dtype_q).T
                if quant:
                    vlayer.ffwd.fc1.weight_scaler.data = read_tensor((v_mlp_hidden_size,))
                vlayer.ffwd.fc2.weight.data = read_tensor((v_hidden_dim, v_mlp_hidden_size), dtype_q).T
                if quant:
                    vlayer.ffwd.fc2.weight_scaler.data = read_tensor((v_hidden_dim,))

                vlayer.ffwd.fc1_bias.data = read_tensor((v_mlp_hidden_size,))
                vlayer.ffwd.fc2_bias.data = read_tensor((v_hidden_dim,))

            ve.post_layernorm.weight.data = read_tensor((v_hidden_dim,))
            ve.post_layernorm.bias.data = read_tensor((v_hidden_dim,))

            ve.norm.weight.data = read_tensor((v_hidden_dim,))
            ve.proj.weight.data = read_tensor((embed_dim, v_hidden_dim), dtype_q).T
            if quant:
                ve.proj.weight_scaler.data = read_tensor((embed_dim,))

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
