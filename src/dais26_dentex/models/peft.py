from __future__ import annotations

import logging
from collections.abc import Iterable

import torch.nn as nn

logger = logging.getLogger(__name__)


def find_target_modules(backbone: nn.Module, patterns: Iterable[str] = ("qkv", "proj")) -> list[str]:
    """Find linear modules in attention blocks matching common QKV/proj naming patterns.

    Returns the named_modules keys (full dotted paths) for use as LoRA target_modules.
    Useful when target_modules isn't explicitly provided.
    """
    targets: list[str] = []
    for name, module in backbone.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1] if "." in name else name
        if any(p in leaf for p in patterns):
            targets.append(name)
    return targets


def unfreeze_last_blocks(backbone: nn.Module, n: int) -> int:
    """Unfreeze the last ``n`` transformer blocks of a (frozen) backbone.

    Heuristic but backbone-agnostic: ViT stacks (timm C-RADIO, HF DINOv3) hold
    their encoder layers in an ``nn.ModuleList``; we pick the LARGEST such list
    (the block stack) and flip ``requires_grad`` on its last ``n`` entries, and
    put those blocks in ``train()`` mode. Returns the number of params unfrozen.

    Raises if no ``ModuleList`` is found so a backbone whose layout we don't
    understand fails loudly instead of silently training nothing.
    """
    if n <= 0:
        return 0
    module_lists = [m for _, m in backbone.named_modules() if isinstance(m, nn.ModuleList)]
    if not module_lists:
        raise RuntimeError(
            "unfreeze_last_blocks: no nn.ModuleList found in backbone; pass an explicit "
            "module path or use backbone_mode='full' instead."
        )
    blocks = max(module_lists, key=len)
    selected = list(blocks)[-n:]
    unfrozen = 0
    for blk in selected:
        blk.train()
        for p in blk.parameters():
            p.requires_grad_(True)
            unfrozen += p.numel()
    logger.info("Unfroze last %d of %d backbone blocks (%d params).", len(selected), len(blocks), unfrozen)
    return unfrozen


def apply_lora(
    backbone: nn.Module,
    rank: int = 8,
    alpha: float = 32.0,
    target_modules: list[str] | None = None,
    target_module_patterns: Iterable[str] = ("qkv", "proj"),
    dropout: float = 0.0,
) -> nn.Module:
    """Inject LoRA adapters into backbone attention linear layers.

    Args:
        backbone: a (frozen) nn.Module.
        rank: LoRA rank (default 8 per plan).
        alpha: LoRA scaling factor (default 32 per plan).
        target_modules: explicit module paths to adapt. If None, auto-discover via
                        target_module_patterns.
        target_module_patterns: substrings to match in module leaf names if
                                target_modules is None.
        dropout: LoRA dropout.

    Returns:
        Backbone with LoRA layers injected. Only LoRA params have requires_grad=True.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise ImportError("peft library required for LoRA. Install: uv pip install 'peft>=0.14,<1.0'") from e

    if target_modules is None:
        target_modules = find_target_modules(backbone, target_module_patterns)
        if not target_modules:
            raise RuntimeError(
                "No LoRA target modules found. Pass target_modules explicitly. "
                f"Patterns tried: {list(target_module_patterns)}"
            )
        logger.info(f"Auto-discovered {len(target_modules)} LoRA target modules.")

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
    )
    # Freeze base first to be sure
    for p in backbone.parameters():
        p.requires_grad_(False)
    peft_model = get_peft_model(backbone, config)
    # Count trainable
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info(f"LoRA injected: trainable={trainable:,} / total={total:,} ({100 * trainable / total:.2f}%)")
    return peft_model


def merge_lora_for_serving(peft_model: nn.Module) -> nn.Module:
    """Merge LoRA adapter weights into base weights for zero-overhead inference.

    After merge, the returned model has no LoRA layers and runs at the same speed
    as the base backbone. Call this before MLflow registration for serving.
    """
    try:
        from peft import PeftModel
    except ImportError as e:
        raise ImportError("peft required") from e
    if not isinstance(peft_model, PeftModel):
        logger.warning(f"Expected PeftModel, got {type(peft_model)}; returning unchanged")
        return peft_model
    return peft_model.merge_and_unload()
