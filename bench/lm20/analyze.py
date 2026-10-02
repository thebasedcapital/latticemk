"""Derive ratios and slice targets from retained measured samples."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def load_rows(m,ctx=128):
    warps=32 if m==1 else 16
    serial=shared=0;start=ctx-1
    for half in (0,1):
        for vw in range(32):
            ranges=[]
            for col in range(m):
                pos=start+col;mid=(pos+2)//2
                a0,a1=(mid,pos+1) if half else (0,mid)
                per=(a1-a0+31)//32
                lo=a0+vw*per;hi=min(a1,lo+per)
                ranges.append((lo,hi))
                serial+=len([t for t in range(lo,hi) if t<start])
            shared+=len({t for lo,hi in ranges for t in range(lo,hi) if t<start})
    # 16 query heads and two 128-element FP16 rows, across 28 layers.
    return dict(context=ctx,m=m,physical_warps=warps,serial_logical_kv_bytes_per_pass=serial*16*512*28,register_tile_logical_kv_bytes_per_pass=shared*16*512*28,register_over_serial=shared/serial,method='Historical-row address model, including GQA head duplication, not DRAM traffic. Warp-private register tile shares rows across queries; preserved per-query partition boundaries cause bounded cross-warp duplicates.')

def main():
    timing=json.loads((HERE/'timing.json').read_text())
    profile=json.loads((HERE/'detail-profile.json').read_text())
    T={(r['variant'],r['mode'],r['m']):r for r in timing}
    P={(r['variant'],r['mode'],r['m']):r for r in profile}
    result={'full_pass':[],'attention':[],'causal_byte_model':[load_rows(m,ctx) for ctx in (128,2048,8192) for m in (1,2,4)],'context_sweep':[]}
    for mode in ('causal','batch'):
        for m in (1,2,4):
            b=T['mt2',mode,m]['median_ms'];a=T['attention',mode,m]['median_ms']
            result['full_pass'].append(dict(mode=mode,m=m,mt2_ms=b,attention_ms=a,delta_ms=a-b,relative=a/b,attention_M_over_M1=a/T['attention',mode,1]['median_ms']))
        for variant in ('mt2','attention'):
            owned=[];total=[]
            for m in (1,4):
                p=P[variant,mode,m]['phase_median_ms']
                owned.append(sum(p[i] for i in (5,10,13)))
                total.append(sum(p[i] for i in (5,10,13,14)))
            result['attention'].append(dict(variant=variant,mode=mode,owned_M1_ms=owned[0],owned_M4_ms=owned[1],owned_extra_ms=owned[1]-owned[0],whole_attention_M1_ms=total[0],whole_attention_M4_ms=total[1],whole_attention_ratio=total[1]/total[0],batch_owned_target_pass=(owned[1]-owned[0]<=.15) if mode=='batch' else None,causal_whole_target_pass=(total[1]/total[0]<=1.3) if mode=='causal' else None,method='Sum of clock-derived category medians; excludes ending grid barrier and pro_attnc. Owned extra uses scan/prologue/append per assignment baseline; whole attention also includes partial merge.'))
    for ctx in (128,2048,8192):
        if ctx==128:
            full={(v,m):T[v,'causal',m]['median_ms'] for v in ('mt2','attention') for m in (1,4)}
            att={(v,m):sum(P[v,'causal',m]['phase_median_ms'][i] for i in (5,10,13,14)) for v in ('mt2','attention') for m in (1,4)}
        else:
            file=HERE/f'context-{ctx}.json'
            if not file.exists():continue
            c=json.loads(file.read_text())
            full={(r['variant'],r['m']):r['median_ms'] for r in c['full_pass']}
            att={(r['variant'],r['m']):r['attention_ms'] for r in c['phases']}
        result['context_sweep'].append(dict(context=ctx,mt2_m1_ms=full['mt2',1],tile_m1_ms=full['attention',1],mt2_m4_ms=full['mt2',4],tile_m4_ms=full['attention',4],m4_full_relative=full['attention',4]/full['mt2',4],tile_full_m4_over_m1=full['attention',4]/full['attention',1],mt2_attention_m1_ms=att['mt2',1],tile_attention_m1_ms=att['attention',1],mt2_attention_m4_ms=att['mt2',4],tile_attention_m4_ms=att['attention',4],m4_attention_relative=att['attention',4]/att['mt2',4],tile_attention_m4_over_m1=att['attention',4]/att['attention',1]))
    wins=[r['context'] for r in result['context_sweep'] if r['m4_full_relative']<1]
    first=min(wins) if wins else None
    losses=[r['context'] for r in result['context_sweep'] if first is not None and r['context']<first and r['m4_full_relative']>=1]
    result['crossover']=dict(first_tested_winning_context=first,last_tested_losing_context=max(losses) if losses else None,method='Observed sampled contexts, not an interpolated exact crossover. Full-pass M4 median is the selection criterion.')
    (HERE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
