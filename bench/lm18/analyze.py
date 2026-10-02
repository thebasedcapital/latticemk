"""Derive phase deltas and an explicitly assumed serialized-latency allocation."""
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def main():
    phases=json.loads((HERE/'profile.json').read_text());stages=json.loads((HERE/'stages.json').read_text())
    phase={(r['mode'],r['m']):np.array(r['phase_median_ms']) for r in phases}
    sample={(r['shape'],r['m']):np.array(r['corrected_amortized_cycles']) for r in stages}
    names=['qkv','o','gate_up','down','lm_head'];rows=[]
    for mode in ('causal','batch'):
        estimate=np.zeros((2,7))
        for i,shape in enumerate(names):
            costs={}
            for j,m in enumerate((1,4)):
                # This maps diagnostic serialized latency shares onto the real
                # persistent phase duration. It does NOT measure retired stalls.
                c=sample[shape,m];f=c/c.sum();cost=phase[mode,m][i]*f;estimate[j]+=cost;costs[m]=cost
            rows.append(dict(mode=mode,shape=shape,phase_delta_ms=float(phase[mode,4][i]-phase[mode,1][i]),modeled_delta_ms=(costs[4]-costs[1]).tolist()))
        delta=estimate[1]-estimate[0]
        rows.append(dict(mode=mode,shape='all_gemms',phase_delta_ms=float((phase[mode,4][:5]-phase[mode,1][:5]).sum()),modeled_delta_ms=delta.tolist(),attention_delta_ms=float(phase[mode,4][5]-phase[mode,1][5]),other_delta_ms=float(phase[mode,4][6]-phase[mode,1][6]),total_diagnostic_delta_ms=float((phase[mode,4]-phase[mode,1]).sum())))
    gate=json.loads((HERE/'gate.json').read_text())
    correctness=dict(strict_sequential_exact_all=all(r['max_diff_sequential']==0 for r in gate),lm14_equivalent_all=all(r['max_diff_hf']<=.5 and r['max_diff_hf']<=1.25*r['baseline_v2_diff'] and r['bitwise_repeat'] and not r['nonfinite'] and not r['hard_flips'] and r['max_diff_sequential']<=.1 for r in gate),max_diff_sequential=max(r['max_diff_sequential'] for r in gate),max_diff_hf=max(r['max_diff_hf'] for r in gate),bitwise_repeat_all=all(r['bitwise_repeat'] for r in gate))
    decision=json.loads((HERE/'decision-128.json').read_text())
    out=dict(categories=['weight_load_wait','dequant','activation_load_wait','fma_issue','local_reduce','warp_reduce','writeback_issue'],allocation_contract='ASSUMED: corrected/amortized serialized stage-cycle fractions allocate each measured persistent GEMM phase. Negative deltas are modeled reduced contributions. Includes barrier and occupancy effects; not Nsight stall counters and not a unique causal decomposition.',rows=rows,correctness=correctness,decision=decision)
    details=json.loads((HERE/'detail-profile.json').read_text())
    batch4=next(r for r in details if r['mode']=='batch' and r['m']==4)
    combined=batch4['phase_median_ms'][batch4['categories'].index('attention_combine')]
    out['next_fix']=dict(target='parallelize pro_attnc columns in pairs using both active 256-thread groups of the 512-thread CTA',assumed_parallel_efficiency=1.0,derived_saving_ms=combined/2,warning='Perfect two-way overlap estimate, not measured. Does not meet the 1.5x pass limit by itself.')
    out['compiler_contract']=dict(selected_fmad=False,phase_profile_fmad=True,contraction_cost=json.loads((HERE/'contraction-cost.json').read_text()))
    (HERE/'attribution.json').write_text(json.dumps(out,indent=2)+'\n')
    with (HERE/'results.jsonl').open('a') as f:
        f.write(json.dumps(dict(experiment='final_summary',correctness=correctness,decision=decision,selected_fmad=False,next_fix=out['next_fix']))+'\n')
    print(json.dumps(out),flush=True)
if __name__=='__main__':main()
