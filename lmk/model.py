from pathlib import Path

from safetensors import safe_open

SNAPSHOT = next((Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots").iterdir())
PROJ = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
N_LAYERS = 28


def linear_names() -> list[str]:
    """Every streamed matvec weight per decode step, in execution order; lm_head is the tied embedding."""
    return [f"model.layers.{i}.{p}.weight" for i in range(N_LAYERS) for p in PROJ] + ["lm_head.weight"]


def load(name: str):
    """float32 CUDA tensor; lm_head.weight resolves to the tied embed_tokens matrix."""
    key = "model.embed_tokens.weight" if name == "lm_head.weight" else name
    with safe_open(SNAPSHOT / "model.safetensors", framework="pt", device="cuda") as f:
        return f.get_tensor(key).float()
