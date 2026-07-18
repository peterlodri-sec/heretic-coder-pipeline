def export_model(model, tokenizer, merged_dir: str, gguf_dir: str) -> None:
    """Merge LoRA into base -> safetensors, and emit a q4_k_m GGUF (plan.md Export)."""
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")
