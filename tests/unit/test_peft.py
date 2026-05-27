import pytest
import torch
import torch.nn as nn

# Skip the entire module if peft isn't installed (STRETCH dep)
peft = pytest.importorskip("peft")

from dais26_dentex.models.peft import apply_lora, find_target_modules, merge_lora_for_serving  # noqa: E402


class TinyAttention(nn.Module):
    """Synthetic 'attention block' with qkv + proj linear layers."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(self.qkv(x)[..., :64]))


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = TinyAttention()
        self.block2 = TinyAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(x))


def test_find_target_modules_qkv_proj():
    bb = TinyBackbone()
    targets = find_target_modules(bb, patterns=("qkv", "proj"))
    # Expect 4 hits: block1.qkv, block1.proj, block2.qkv, block2.proj
    assert len(targets) == 4
    assert all("qkv" in t or "proj" in t for t in targets)


def test_find_target_modules_no_hits():
    bb = TinyBackbone()
    targets = find_target_modules(bb, patterns=("nonexistent",))
    assert targets == []


def test_apply_lora_trainable_count():
    bb = TinyBackbone()
    for p in bb.parameters():
        p.requires_grad_(False)
    peft_model = apply_lora(bb, rank=8, alpha=32.0)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    # Should have some trainable LoRA params (4 layers x 2 LoRA matrices x small ranks)
    assert trainable > 0
    # Base params should still be frozen
    base_trainable = sum(
        p.numel() for n, p in peft_model.named_parameters() if p.requires_grad and "lora" not in n.lower()
    )
    assert base_trainable == 0, f"Non-LoRA params should be frozen, found {base_trainable} trainable"


def test_apply_lora_no_targets_raises():
    bb = TinyBackbone()
    with pytest.raises(RuntimeError, match="No LoRA target modules"):
        apply_lora(bb, target_module_patterns=("nonexistent_pattern",))


def test_merge_lora_for_serving():
    bb = TinyBackbone()
    for p in bb.parameters():
        p.requires_grad_(False)
    peft_model = apply_lora(bb, rank=4, alpha=16.0)
    merged = merge_lora_for_serving(peft_model)
    # After merge, no LoRA params should remain trainable
    lora_params = [n for n, _ in merged.named_parameters() if "lora" in n.lower()]
    assert len(lora_params) == 0, f"After merge, LoRA params should be gone: {lora_params}"
    # Forward should still work
    x = torch.randn(1, 4, 64)
    out = merged(x)
    assert out.shape == (1, 4, 64)
