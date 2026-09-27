def make_trial_name(i: int, suggestion) -> str:
    """生成 trial 目錄名稱，包含編號、方法與關鍵參數"""
    parts = [f"trial_{i:03d}"]
    mode = suggestion.mode

    # Append the base mode name first
    parts.append(mode)

    if mode == "sparse_unstructured":
        ratio_pct = int((suggestion.sparsity_ratio or 0.5) * 100)
        parts.append(f"{ratio_pct}pct")

    elif mode == "sparse_structured":
        struct = suggestion.sparsity_structure or "2:4"
        parts.append(struct.replace(':', 'x'))

    if mode in ("asvd_only", "hybrid_asvd_bnb") and suggestion.alpha is not None:
        ratio_str = f"{int((suggestion.param_ratio_target or 0.9) * 100):03d}"
        alpha_str = f"{int((suggestion.alpha or 0.5) * 100):02d}"
        parts.append(f"r{ratio_str}_a{alpha_str}")

    # Handle Quantization specific parameters
    if mode == "gptq":
        b = suggestion.quant_bits or 4
        g = suggestion.quant_group_size or 128
        fmt = suggestion.quant_format or "gptq"
        parts.append(f"{b}bit_g{g}_{fmt}")

    elif mode == "awq":
        g = suggestion.quant_group_size or 128
        parts.append(f"4bit_g{g}") # AWQ is fixed to 4-bit in your space

    elif mode == "qqq":
        g = suggestion.quant_group_size or 128
        parts.append(f"4bit_g{g}") # QQQ is fixed to 4-bit

    elif mode in ("bnb", "hybrid_asvd_bnb"):
        b = getattr(suggestion, 'quant_bits', 4)
        parts.append(f"{b}bit")

    return "_".join(parts)
