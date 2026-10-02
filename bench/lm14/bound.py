"""Integrator throughput-only estimate. Not a hardware ceiling or measurement."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
measured=json.loads((HERE/'gemv.json').read_text())
weights=600_000_000  # Integrator's rounded model count, an explicit assumption.
sms=36;clock_ghz=1.8;pairs=weights/sms/2
lds_ms={str(m):pairs*m/32/(clock_ghz*1e6) for m in (1,4)}
hfma_ms=pairs*4/64/(clock_ghz*1e6)
stream=measured['rows'][0]['median_ms'];observed=measured['rows'][1]['median_ms']
result=dict(assumed_weights=weights,sms=sms,assumed_sm_clock_ghz=clock_ghz,weights_per_sm=weights/sms,half2_pairs_per_sm=pairs,lds_cycles_m4=pairs*4/32,lds_ms=lds_ms,hfma2_cycles_m4=pairs*4/64,hfma2_ms_m4=hfma_ms,measured_weight_stream_m1_ms=stream,assumed_compute_band_ms=[0.6,0.9],derived_expected_gemm_m4_band_ms=[1.4,1.6],derived_expected_ratio_band=[1.1,1.2],measured_gemm_m4_ms=observed,derived_gap_ms=observed-1.6,note='Throughput arithmetic supplied by integrator. LDS32 assumes 32 pair loads per SM cycle; HFMA2 assumes 64 lane operations per SM cycle. Ignores dependency latency, instruction scheduling, reduction helpers, control flow, bank conflicts, and other resource interactions. Unexplained gap suggests latency; no stall counters were measured.')
(HERE/'bound.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
