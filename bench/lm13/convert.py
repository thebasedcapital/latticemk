"""Run upstream ExLlamaV2's converter with correct Qwen3 prefill semantics.

ExLlamaV2 0.3.2 defaults Qwen3's RMS Q/K norm to LayerNorm. Its SDPA
attention also passes no causal mask for multi-token prefill on sm_75.
Patch only these config fields before the native converter calibrates.

scripts/gpu.sh bash bench/lm13/run.sh bench/lm13/convert.py -i SNAPSHOT -o WORK -cf OUTPUT -b 4.125 -hb 4
"""
import runpy
from exllamav2 import ExLlamaV2Config

_original_prepare = ExLlamaV2Config.prepare


def _prepare_qwen3(self, *args, **kwargs):
    result = _original_prepare(self, *args, **kwargs)
    if self.arch.arch_string == "Qwen3ForCausalLM":
        self.arch.lm.headnorm = "rmsnorm"
        self.no_sdpa = True
    return result


ExLlamaV2Config.prepare = _prepare_qwen3
runpy.run_module("exllamav2.conversion.convert_exl2", run_name="__main__")
