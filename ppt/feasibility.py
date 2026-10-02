"""Compute feasibility, keeping every measured/derived/assumed quantity explicit."""
import json
from pathlib import Path
from run import ROOT
from sampler import GSM8K

pilot=ROOT/'pilot.jsonl'
rows=[json.loads(l) for l in pilot.read_text().splitlines()] if pilot.exists() else []
complete=next((r for r in rows if r['model']=='1.7B' and r['method']=='ppt'),None)
checkpoint=ROOT/'checkpoint-1.7B-566.json'
cp=json.loads(checkpoint.read_text()) if checkpoint.exists() else None
counts=complete if complete else cp['counts'] if cp else None
if counts is None:
    raise SystemExit('No measured PPT counters')
mean_rate=counts['generated']/counts['wall_seconds']
milestones=cp.get('milestones',[]) if cp else complete.get('milestones',[])
stage_measurements=[]
for previous,current in zip(milestones,milestones[1:]):
    if previous['round']!=-1 or current['round']!=-1:
        continue
    generated=current['generated']-previous['generated']
    seconds=current['wall_seconds']-previous['wall_seconds']
    stage_measurements.append(dict(stage=current['stage'],generated=generated,
        prefill=current['prefill']-previous['prefill'],seconds=seconds,
        generated_tokens_per_second=generated/seconds))
memory_bounded=[r for r in stage_measurements if r['stage']>=7]
rate=memory_bounded[-1]['generated_tokens_per_second'] if memory_bounded else mean_rate
if complete:
    full_tokens=counts['generated']; full_prefill=counts['prefill']; full_seconds=counts['wall_seconds']
    extrapolation=None
else:
    # Freeze each chain's observed terminal/open status. Uniform restarts then
    # give an analytic suffix-length expectation at each future stage.
    lengths=[len(r['tokens']) for r in cp['records']]
    terminal=[r['terminal'] for r in cp['records']]
    stage=cp['stage']; round_=cp['round']; current=(stage+1)*GSM8K['block']
    per_stage=[]
    prompt_length=next(r['prefill'] for r in rows if r['model']=='1.7B' and r['method']=='standard' and r['problem']==566)
    for s in range(stage,16):
        horizon=(s+1)*192
        rounds=10-round_ if s==stage and round_>=0 else 10
        local=sum((l*(l+1)/(2*horizon) if term else (horizon+1)/2) for l,term in zip(lengths,terminal))*rounds
        extension=sum(not term for term in terminal)*192 if s>stage else 0
        prefix=sum((l*prompt_length+l*(l-1)/2)/horizon if term else prompt_length+(horizon-1)/2
                   for l,term in zip(lengths,terminal))*rounds
        extension_prefix=sum(not term for term in terminal)*(prompt_length+horizon-192) if s>stage else 0
        per_stage.append(dict(stage=s+1,horizon=horizon,local_rounds=rounds,
                              expected_generated_tokens=local+extension,expected_prefill_tokens=prefix+extension_prefix))
    full_tokens=counts['generated']+sum(s['expected_generated_tokens'] for s in per_stage)
    full_prefill=counts['prefill']+sum(s['expected_prefill_tokens'] for s in per_stage)
    full_seconds=counts['wall_seconds']+(full_tokens-counts['generated'])/rate
    extrapolation=dict(tag='derived',assumptions='Observed terminal/open pattern stays fixed; regenerated suffix length equals current terminal length or new stage horizon; saved elapsed time plus remaining tokens at last fully measured microbatched stage throughput, or cumulative mean if unavailable',
                       observed_lengths=lengths,observed_terminal=terminal,remaining_stages=per_stage,
                       projected_complete_tokens=full_tokens,projected_complete_seconds=full_seconds,
                       warning='Not an upper/lower bound. Future EOS changes and longer-context cost can change it.')
ratio=250.0/578.6
engines=dict(hf_fp16=dict(tokens_per_second=full_tokens/full_seconds,tag='derived blended rate: saved elapsed time plus remaining work at last measured complete microbatched stage; includes prefill/scoring runtime'),
    batch_06_m4=dict(tokens_per_second=977.0,tag='measured wave-6 at context128; projection ignores prefill and normalizers'),
    batch_06_m5=dict(tokens_per_second=1000.0,tag='measured wave-6 at context128; projection ignores prefill and normalizers'),
    batch_17_m4=dict(tokens_per_second=977.0*ratio,tag='derived, 0.6B batch throughput times 1.7B/0.6B single-stream ratio250/578.6'),
    batch_17_m5=dict(tokens_per_second=1000.0*ratio,tag='derived, same ratio; batch1.7B not ported'))
for name,e in engines.items():
    e['ppt_gpu_hours_for_100']=100*full_tokens/e['tokens_per_second']/3600
    e['ppt_gpu_hours_for_200']=200*full_tokens/e['tokens_per_second']/3600
    if name=='hf_fp16':
        continue
    e['all_matched_compute_groups_gpu_hours_for_100']=5*e['ppt_gpu_hours_for_100']
    e['all_matched_compute_groups_gpu_hours_for_200']=5*e['ppt_gpu_hours_for_200']
    e['all_groups_assumption']='Five equal token budgets: PPT, standard/BoN/vote shared pool, low-temperature, repeated greedy, single-chain. Same throughput assumed for every group; prefill, cache handling and scoring differences excluded.'
controls=[]
control_proof=ROOT/'control-proof.json'
control_rows=json.loads(control_proof.read_text()) if control_proof.exists() else []
for method in ['standard','low_temperature','greedy','single_chain']:
    selected=[r for r in rows if r['model']=='1.7B' and r['method']==method]
    calibration=[r for r in control_rows if r['method']==method] if method!='single_chain' else selected
    if not calibration:
        controls.append(dict(method=method,status='not measured')); continue
    tokens=sum(r['generated'] for r in calibration)
    secs=sum(r.get('seconds',r.get('wall_seconds')) for r in calibration)
    throughput=tokens/secs
    controls.append(dict(method=method,n=len(calibration),calibration_budget_per_problem=tokens/len(calibration),
        measured_prefill_tokens_per_problem=sum(r['prefill'] for r in calibration)/len(calibration),
        measured_seconds_per_problem=secs/len(calibration),measured_tokens_per_second=throughput,
        natural_single_output_pilots=selected if method!='single_chain' else [],
        matched_partial_budget_tokens=counts['generated'],derived_matched_partial_seconds=counts['generated']/throughput,
        derived_matched_complete_seconds=full_tokens/throughput,
        caveat='256-token independent controls and 8192-token single-chain calibrations use batch1. Full independent controls allocate up to6 requests but execute microbatches of2 above horizon768; matched cost extrapolation is not a measured full matched run.'))
baseline_seconds=[r['derived_matched_complete_seconds'] for r in controls if 'derived_matched_complete_seconds' in r]
hf_total_projection=dict(gpu_hours_for_100=100*(full_seconds+sum(baseline_seconds))/3600,
                         gpu_hours_for_200=200*(full_seconds+sum(baseline_seconds))/3600,
                         measured_control_groups=len(baseline_seconds),
                         tag='derived, uses each small control throughput calibration and recorded PPT elapsed plus extrapolated remaining runtime; not an accuracy run')
chunkpath=ROOT/'chunks.jsonl'
chunks=[json.loads(line) for line in chunkpath.read_text().splitlines()] if chunkpath.exists() else []
result=dict(decision='undecided: compute-blocked',completed_ppt_problems=int(bool(complete)),
    measured_counts={k:counts[k] for k in ('generated','prefill','decode_steps','wall_seconds')},
    cumulative_mean_generated_tokens_per_second=mean_rate,
    measured_complete_stages=stage_measurements,
    continuation_charged_seconds=sum(r['wall_seconds'] for r in chunks),
    chunk_accounting_note='Charged wall time conservatively includes lock waits and model reloads; checkpoint counters exclude unsaved in-flight work after OOM/hard timeout.',
    meaningful_full_vocabulary_scoring_positions=counts['generated'],
    meaningful_normalizer_reductions=counts['generated']*6,
    scoring_definition='One full-vocabulary log_softmax and six logsumexp reductions per generated position. Idle batch rows add wasted evaluations not included here. No separate scoring-model pass.',
    rounds_completed=sum(cp['stats']['local_attempts'])//6 if cp else 160,
    projected_complete_tokens=full_tokens,projected_complete_prefill_tokens=full_prefill,
    projected_complete_seconds=full_seconds,extrapolation=extrapolation,
    per_stage_measured_milestones=cp.get('milestones',[]) if cp else complete.get('milestones',[]),
    engines=engines,controls=controls,
    future_decode_tokens_per_second_for_10_second_problem=full_tokens/10,
    future_decode_tokens_per_second_for_60_second_problem=full_tokens/60,
    generated_tokens_per_second_for_all_five_groups_100_problems_in_6_hours=5*full_tokens*100/(6*3600),
    generated_tokens_per_second_for_all_five_groups_200_problems_in_6_hours=5*full_tokens*200/(6*3600),
    hf_total_projection=hf_total_projection,
    single_stream_ratio=ratio)
(ROOT/'feasibility.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
