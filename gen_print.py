# This file is pretty much vibe-coded

import json
import pyperclip


def collect_all_labels(data: dict) -> list[str]:
    labels = []
    for sinfo in data.get("structs", {}).values():
        for field in sinfo.get("fields", []):
            labels.append(field["name"])
    labels.extend(["Weights", "KV Cache (tok)", "Gemma Buffer", "SigLIP Buffer"])
    return labels


def gen_print_code(field: dict, width: int, prefix: str = "conf") -> str:
    name = field["name"]
    ftype = field["type"]
    fmt = field.get("fmt")

    if ftype == "int":
        return (
            f'    printf("  %-*s: %d\\n", {width}, "{name}", '
            f"{prefix}->{name});"
        )
    if ftype == "float":
        fstr = fmt or "%f"
        return (
            f'    printf("  %-*s: {fstr}\\n", {width}, "{name}", '
            f"{prefix}->{name});"
        )
    if ftype == "bool":
        return (
            f'    printf("  %-*s: %s\\n", {width}, "{name}", '
            f'{prefix}->{name} ? "true" : "false");'
        )
    if ftype == "bool_array":
        len_expr = field["len_expr"]
        return f"""\
    printf("  %-*s: ", {width}, "{name}");
    for (int i = 0; i < {len_expr}; i++) {{
        printf("%d", {prefix}->{name}[i] ? 1 : 0);
        if ((i + 1) % {37 - width} == 0 && i + 1 < {len_expr}) {{ printf("\\n  {' ' * width}  "); }}
    }}
    printf("\\n");"""
    raise ValueError(f"Unknown type: {ftype}")


def generate_memory_code(rules: dict, width: int) -> str:
    lines: list[str] = [
        "    size_t total_bytes = 0;",
        "",
    ]

    # Embedding
    emb = rules.get("embedding", {})
    if emb:
        dims = " * ".join(emb["dims"])
        quant_cond = emb.get("quantized", "0")
        scales = emb.get("scales", "0")
        lines += [
            "    // Embedding",
            f"    if (!({quant_cond})) {{",
            f"        total_bytes += {dims} * sizeof(_Float16);",
            "    } else {",
            f"        total_bytes += {dims} * sizeof(int8_t);",
            f"        total_bytes += {scales} * sizeof(_Float16);  // scales",
            "    }",
            "",
        ]

    # Layers
    layers = rules.get("layers", {})
    if layers:
        lines += [
            "    // Each layer",
            "    for (int l = 0; l < conf->n_layers; l++) {",
            "        int q_size  = conf->n_heads * conf->head_dim;",
            "        int kv_size = conf->n_kv_heads * conf->head_dim;",
            "",
        ]

        lin_dims = layers.get("linear_dims", [])
        scale_dims = layers.get("scale_dims", [])
        if lin_dims:
            params_sum = "\n            + ".join(" * ".join(ds) for ds in lin_dims)
            lines += [
                f"        int layer_params = {params_sum};",
                "",
                "        if (!conf->quant) {",
                "            total_bytes += layer_params * sizeof(_Float16);",
                "        } else {",
                "            total_bytes += layer_params * sizeof(int8_t);",
            ]
            if scale_dims:
                scale_sum = "\n                + ".join(scale_dims)
                lines.append(
                    f"            total_bytes += (\n"
                    f"                {scale_sum}\n"
                    f"            ) * sizeof(_Float16);"
                    "  // scales"
                )
            lines += [
                "        }",
                "",
            ]

        for nw in layers.get("norm_weights", []):
            lines.append(
                f"        total_bytes += conf->embed_dim * sizeof(_Float16);"
                f"  // {nw}"
            )

        opt_norms = layers.get("optional_norms", {})
        for key, expr in opt_norms.items():
            if expr != "0":
                lines.extend([
                    f"        if ({expr}) {{",
                    f"            total_bytes += {expr} * sizeof(_Float16);  // {key}",
                    f"        }}",
                ])

        lines += [
            "    }",
            "",
        ]

    # Final norm
    fn = rules.get("final_norm", {})
    if fn:
        dims = " * ".join(fn["dims"])
        lines += [
            "    // Final norm",
            f"    total_bytes += {dims} * sizeof(_Float16);",
            "",
        ]

    lines += [
        f'    printf("  %-*s: %.2f GB\\n", {width}, "Weights",'
        f" total_bytes / (1024.0 * 1024.0 * 1024.0));",
        "",
        "    // KV Cache per token",
        "    int kv_size = conf->n_kv_heads * conf->head_dim;",
        "    size_t kv_cache_bytes ="
        " conf->n_layers * 2 * kv_size * sizeof(_Float16);",
        f'    printf("  %-*s: %.2f KB\\n", {width}, "KV Cache (tok)",'
        f" kv_cache_bytes / 1024.0);",
    ]

    return "\n".join(lines)


def generate_buffer_code(buf_desc: dict, width: int, is_siglip: bool = False) -> str:
    lines: list[str] = []

    if is_siglip:
        enabled = buf_desc.get("enabled", "1")
        lines += [
            f"    if ({enabled}) {{",
            "        SigLIPConfig *vconf = model->vision_enc->config;",
            "",
        ]
        for var, expr in buf_desc.get("local_dims", {}).items():
            lines.append(f"        int {var} = {expr};")
        lines += [
            "",
            "        size_t siglip_bytes = 0;",
            "",
        ]
        indent = "        "
        var_name = "siglip_bytes"
        label = "SigLIP Buffer"
    else:
        lines += [
            "    // GemmaBuffer",
            "",
        ]
        for var, expr in buf_desc.get("common_dims", {}).items():
            lines.append(f"    int {var} = {expr};")

        # Break the long mult_expr across lines for readability
        mult_expr = buf_desc.get("mult_expr", "1")
        lines += [
            f"    int mult = {mult_expr};",
            "",
            "    size_t buf_bytes = 0;",
            "",
        ]
        indent = "    "
        var_name = "buf_bytes"
        label = "Gemma Buffer"

    for mem in buf_desc.get("members", []):
        dims = " * ".join(mem["dims"])
        typ = mem["type"]
        cond = mem.get("cond")
        comment = mem.get("name", "")

        add_line = f"{indent}{var_name} += {dims} * sizeof({typ});  // {comment}"
        if cond:
            lines += [
                f"{indent}if ({cond}) {{",
                f"    {add_line}",
                f"{indent}}}",
            ]
        else:
            lines.append(add_line)

    lines += [
        "",
        f'{indent}printf("  %-*s: %.2f MB\\n", {width}, "{label}",'
        f" {var_name} / (1024.0 * 1024.0));",
    ]

    if is_siglip:
        lines.append("    }")

    return "\n".join(lines)


def main() -> None:
    with open("model_desc.json", "r") as f:
        data = json.load(f)

    labels = collect_all_labels(data)
    max_len = max(len(label) for label in labels) if labels else 20
    width = max_len + 2

    structs = data.get("structs", {})
    config_fields = structs.get("GemmaConfig", {}).get("fields", [])
    tokenizer_fields = structs.get("GemmaTokenizer", {}).get("fields", [])

    config_lines = [gen_print_code(f, width, "conf") for f in config_fields]
    tokenizer_lines = [gen_print_code(f, width, "tok") for f in tokenizer_fields]

    mem_code = generate_memory_code(data.get("memory_rules", {}), width)
    buffers = data.get("buffers", {})
    gemma_code = generate_buffer_code(buffers.get("gemma", {}), width, is_siglip=False)
    siglip_code = generate_buffer_code(buffers.get("siglip", {}), width, is_siglip=True)

    # Build the final C function with clear section separation
    parts: list[str] = [
        "void print_model_config(GemmaModel *model, int seqlen, bool enable_mm) {",
        "    (void)model;",
        "    (void)seqlen;",
        "    (void)enable_mm;",
        "#ifdef DEBUG",
        "    GemmaConfig *conf = model->config;",
        "    GemmaTokenizer *tok = model->tokenizer;",
        "",
        '    printf("\\n========== Model Configuration ==========\\n");',
        '    printf("Architecture:\\n");',
        "",
    ]

    parts.extend(config_lines)
    parts += [
        "",
        '    printf("\\nTokenizer:\\n");',
        "",
    ]
    parts.extend(tokenizer_lines)
    parts += [
        "",
        '    printf("\\nMemory Footprint (estimated):\\n");',
        "",
        mem_code,
        "",
        gemma_code,
        "",
        siglip_code,
        "",
        '    printf("=========================================\\n\\n");',
        "#endif",
        "}",
    ]

    code = "\n".join(parts)
    pyperclip.copy(code)
    print("Generated C code copied to clipboard.")


if __name__ == "__main__":
    main()
