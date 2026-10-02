"""Derive pass and owned-phase differences from retained paired samples."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
latency=json.loads((HERE/'timing.json').read_text())
profile=json.loads((HERE/'detail-profile.json').read_text())
summary={}
for mode in ('batch','causal'):
    full={}
    for m in (1,2,4):
        rows={r['variant']:r for r in latency if r['mode']==mode and r['m']==m}
        old,new=rows['mt2']['median_ms'],rows['pro']['median_ms']
        full[str(m)]={'mt2_ms':old,'pro_ms':new,'delta_ms':new-old,'change_percent':100*(new/old-1)}
    phases={}
    for variant in ('baseline_detail','detail'):
        rows={r['m']:r for r in profile if r['mode']==mode and r['variant']==variant}
        phase={}
        for i in (6,7,8,9,11,12):
            a,b=rows[1]['phase_median_ms'][i],rows[4]['phase_median_ms'][i]
            phase[rows[1]['categories'][i]]={'m1_ms':a,'m4_ms':b,'extra_ms':b-a}
        owned=[v for k,v in phase.items() if k!='grid_barriers']
        phase['owned_sum']={key:sum(v[key] for v in owned) for key in ('m1_ms','m4_ms','extra_ms')}
        phases[variant]=phase
    summary[mode]={'full_pass':full,'pro_m4_m1':full['4']['pro_ms']/full['1']['pro_ms'],'mt2_m4_m1':full['4']['mt2_ms']/full['1']['mt2_ms'],'phases':phases,'slice_pass':phases['detail']['owned_sum']['extra_ms']<=.20}
(HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
