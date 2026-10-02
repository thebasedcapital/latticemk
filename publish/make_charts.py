"""Render publication charts from the recorded benchmark JSONL/JSON evidence.

Run from repository root: .venv/bin/python publish/make_charts.py
Only the original paired 27/30-sample LM-11 and LM-12 Q4_0 runs enter the
headline comparison. Later replications remain in the JSONL audit trail.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "publish/charts"
OUT.mkdir(parents=True, exist_ok=True)
CONTEXTS = (128, 2048, 8192)
CTX_LABEL = {128: "ctx 128", 2048: "ctx 2k", 8192: "ctx 8k"}
INK, INK2, MUTED = "#111827", "#4B5563", "#9CA3AF"
COLORS = {"mega": "#0E7490", "llama": "#C3C8D0", "negative": "#D97a6c", "positive": "#0E7490",
          "q4km": "#7C6FB0", "exl2": "#B08B4F"}
plt.rcParams.update({
    "font.family": ["Noto Sans", "DejaVu Sans"], "font.size": 12,
    "axes.edgecolor": "#D1D5DB", "axes.labelcolor": INK2, "axes.labelsize": 12,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#E5E7EB", "grid.linewidth": 1, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white", "legend.frameon": False,
})


def rows(rel):
    return [json.loads(line) for line in (ROOT / rel).read_text().splitlines() if line]


def pair(source, ctx, mega, llama):
    """Select the first complete, adjacent, same-session 27/30-sample pair."""
    records = rows(source)
    for left, right in zip(records, records[1:]):
        if (left['context'] == ctx and left['kernel'] == mega and left['runs'] == 27
                and right['context'] == ctx and right['kernel'] == llama
                and right['runs'] == 30 and right['flash_attn'] == 1
                and (left.get('paired_with') in (None, 'Q4_0'))):
            return left, right
    raise ValueError(f"Missing same-session pair: {source}, {ctx}")


PAIRS = {
    'Qwen3-0.6B': [pair('bench/lm11/results.jsonl', ctx,
                        'megakernel-v2-gptq', 'llamacpp-q4_0-faon') for ctx in CONTEXTS],
    'Qwen3-1.7B': [pair('bench/lm12/results.jsonl', ctx,
                        'megakernel-scale-gptq', 'llamacpp-Q4_0-faon') for ctx in CONTEXTS],
}


def figure(title, subtitle, source, rect=(.07, .15, .9, .68)):
    """16:9 canvas (2400x1350 px): takeaway title, setup subtitle, source footnote."""
    fig = plt.figure(figsize=(12, 6.75))
    ax = fig.add_axes(rect)
    fig.text(.04, .955, title, fontsize=21, weight="bold", color=INK, va="top")
    fig.text(.04, .885, subtitle, fontsize=12.5, color=INK2, va="top")
    fig.text(.04, .03, source, fontsize=9, color=MUTED)
    fig.text(.96, .03, "Quadro RTX 4000 (Turing, 2018) · batch 1", fontsize=9, color=MUTED, ha="right")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)


def bar_positions(points):
    """x positions with a gap between the two model groups."""
    return np.array([i + (0.6 if model == "Qwen3-1.7B" else 0) for i, (model, _, _) in enumerate(points)])


def model_axis(ax, x, points):
    ax.set_xticks(x, [CTX_LABEL[ctx] for _, ctx, _ in points])
    for model in PAIRS:
        xs = [xi for xi, (m, _, _) in zip(x, points) if m == model]
        ax.text(np.mean(xs), -0.13, model, transform=ax.get_xaxis_transform(), ha="center",
                fontsize=13, weight="bold", color=INK)


points = [(model, ctx, pair_) for model, triples in PAIRS.items()
          for ctx, pair_ in zip(CONTEXTS, triples)]
x = bar_positions(points)
w = .38
mega = [p[0]['tokens_per_s'] for _, _, p in points]
llama = [p[1]['tokens_per_s'] for _, _, p in points]

# Rates are median sample latencies converted to tokens/s by each driver.
fig, ax = figure('One resident kernel beats llama.cpp at every context',
                 'Decode tokens/s. INT4 GPTQ megakernel vs llama.cpp Q4_0 (flash attention on), same card, paired runs.',
                 'Source: bench/lm11/results.jsonl, bench/lm12/results.jsonl. fp16 KV cache. Ratio = paired median tokens/s.')
ax.bar(x - w/2, mega, w, color=COLORS['mega'], zorder=2)
ax.bar(x + w/2, llama, w, color=COLORS['llama'], zorder=2)
top = max(mega) * 1.22
for xi, m, l in zip(x, mega, llama):
    ax.text(xi - w/2, m + top*.012, f'{m:.0f}', ha='center', va='bottom', fontsize=10.5, color=INK, weight='bold')
    ax.text(xi + w/2, l + top*.012, f'{l:.0f}', ha='center', va='bottom', fontsize=10.5, color=INK2)
    ax.text(xi, max(m, l) + top*.085, f'{m/l:.2f}×', ha='center', va='bottom', fontsize=14,
            weight='bold', color=COLORS['mega'])
model_axis(ax, x, points)
ax.set_ylabel('tokens / s')
ax.set_ylim(0, top)
ax.tick_params(axis='x', labelsize=11.5)
ax.legend(handles=[Patch(color=COLORS['mega'], label='This work: megakernel, INT4 GPTQ'),
                   Patch(color=COLORS['llama'], label='llama.cpp Q4_0')],
          loc='upper right', fontsize=11.5, handlelength=1.2)
save(fig, 'speed_vs_llamacpp.png')

# The unmatched Q4_K_M point is deliberately drawn but not given a paired ratio.
quality = json.loads((ROOT/'bench/lm11/ppl.json').read_text())['ppl']
exl2_quality = next(r['ppl'] for r in rows('bench/lm13/ppl_parts.jsonl')
                    if r['model'].endswith('4.65bpw-causal') and r['count_windows'] == 146)
exl2 = next(r for r in rows('bench/lm13/results.jsonl')
            if r['context'] == 128 and r.get('quant_model_dir', '').endswith('4.65bpw-causal'))
q4km = next(r for r in rows('bench/lm11/results.jsonl')
            if r['context'] == 128 and r['kernel'] == 'llamacpp-q4_k_m-faon')
fig, ax = figure('Faster than llama.cpp Q4_0, and lower perplexity',
                 'Qwen3-0.6B. Decode speed at context 128 vs WikiText-2 perplexity (llama-perplexity, -c 2048).',
                 'Sources: bench/lm11, bench/lm13 (results.jsonl, ppl). Q4_K_M timing unpaired; EXL2 timed by its own generator.')
scatter = [
    ('This work (INT4 GPTQ)', PAIRS['Qwen3-0.6B'][0][0]['tokens_per_s'], quality['int4gptq-fake/f16'],
     COLORS['mega'], 260, (0, 16), 'center'),
    ('llama.cpp Q4_0', PAIRS['Qwen3-0.6B'][0][1]['tokens_per_s'], quality['q4_0/f16'],
     COLORS['llama'], 150, (0, 14), 'center'),
    ('llama.cpp Q4_K_M', q4km['tokens_per_s'], quality['q4_k_m/f16'],
     COLORS['q4km'], 150, (0, -24), 'center'),
    ('ExLlamaV2 EXL2 4.65 bpw', exl2['tokens_per_s'], exl2_quality,
     COLORS['exl2'], 150, (0, -24), 'center'),
]
for label, rate, ppl, color, size, offset, ha in scatter:
    ours = label.startswith('This work')
    ax.scatter(rate, ppl, s=size, color=color, zorder=3, edgecolor='white', linewidth=1.5)
    ax.annotate(f'{label}\n{rate:.0f} tok/s · PPL {ppl:.2f}', (rate, ppl), xytext=offset, textcoords='offset points',
                ha=ha, va='bottom' if offset[1] > 0 else 'top', fontsize=11 if ours else 10.5,
                color=INK if ours else INK2, weight='bold' if ours else 'normal', linespacing=1.15)
ax.annotate('better', xy=(.97, .06), xytext=(.84, .25), xycoords='axes fraction', textcoords='axes fraction',
            fontsize=11, color=MUTED, ha='center', va='center',
            arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.4))
ax.grid(axis='x')
ax.spines['left'].set_visible(True)
ax.set_xlabel('decode tokens / s at context 128 (higher is better)')
ax.set_ylabel('perplexity (lower is better)')
ax.set_xlim(0, 720)
ax.set_ylim(min(s[2] for s in scatter) - .3, max(s[2] for s in scatter) + .3)
save(fig, 'ppl_vs_speed.png')

# Each idea keeps its own measured baseline. Missing full-pass evidence is not a zero.
def document(rel):
    return json.loads((ROOT / rel).read_text())


def select(records, **fields):
    matches = [r for r in records if all(r.get(k) == v for k, v in fields.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one evidence row for {fields}, got {len(matches)}")
    return matches[0]


def rate_ratio(records, name, baseline, ctx):
    return (select(records, context=ctx, kernel=name)['tokens_per_s']
            / select(records, context=ctx, kernel=baseline)['tokens_per_s'])


def latency_ratio(records, candidate, baseline, **fields):
    return (select(records, variant=baseline, **fields)['median_ms']
            / select(records, variant=candidate, **fields)['median_ms'])


old = rows('bench/lm03b/results.jsonl')
kv = rows('bench/lm09/results.jsonl')
sync = rows('bench/lm10/results.jsonl')
mt14 = rows('bench/lm14/results.jsonl')
choose18 = document('bench/lm18/choose.json')
mt20 = document('bench/lm20/timing.json')
mt21 = document('bench/lm21/timing.json')
mt23 = rows('bench/lm23/results.jsonl')
compact = document('bench/lm24/timing.json')
session = document('bench/lm24/session.json')
spec = document('spec/analysis.json')
decode8 = select(compact['decode'], name='v2-long', logical=8192, physical=8192)
decode2 = select(compact['decode'], name='v2-short', logical=2048, physical=2048)
kvc2 = select(compact['decode'], name='kvc-compacted', logical=8192, physical=2048)
mt14_m4 = select(mt14, kernel='megakernel-mt-batch', context=128, m_tokens=4, revision='bounded-retry')
mt14_v2 = select(mt14, kernel='megakernel-v2', context=128, revision='bounded-retry')
ideas = [
    ('1', 'Lattice weight codes', None, 'killed', 'No raw timing file; report only'),
    ('2', 'First persistent kernel', rate_ratio(old, 'megakernel-v1', 'separate-graph', 128), 'killed', 'vs separate graph, ctx 128'),
    ('3', 'Occupancy fix', rate_ratio(old, 'megakernel-v2', 'separate-graph', 128), 'kept', 'vs separate graph, ctx 128'),
    ('4', 'Compressed KV', rate_ratio(kv, 'megakernel-kv', 'megakernel-v2', 8192), 'killed', 'vs v2, ctx 8k; correctness failed'),
    ('4', 'Fine-grained sync', rate_ratio(sync, 'megakernel-sync', 'megakernel-v2', 128), 'killed', 'vs v2, ctx 128'),
    ('5', 'Scale to 1.7B', PAIRS['Qwen3-1.7B'][0][0]['tokens_per_s']/PAIRS['Qwen3-1.7B'][0][1]['tokens_per_s'], 'kept', 'vs llama.cpp Q4_0, ctx 128'),
    ('6', 'Multi-token pass', mt14_m4['tokens_per_s']/mt14_v2['tokens_per_s'], 'killed', 'Batch aggregate vs v2; 1.5× cost target missed'),
    ('6', 'Gate mutation audit', None, 'kept', 'Correctness audit, not a speed experiment'),
    ('7', 'GEMM preload', latency_ratio(choose18, 'preload', 'original', kind='full', m=4, context=128), 'killed', 'vs original M4; final 1.5× cost target missed'),
    ('7', 'Three-tier gate', None, 'kept', 'Regression mode catches survivors; no speed ratio'),
    ('7', 'Power tempering', None, 'blocked', 'No completed accuracy campaign or speed ratio'),
    ('8', 'Attention slice', latency_ratio(mt20, 'attention', 'mt2', mode='causal', m=4, context=128), 'kept', 'vs mt2 M4; slice targets missed'),
    ('8', 'Column-local slice', latency_ratio(mt21, 'pro', 'mt2', mode='batch', m=4, context=128), 'kept', 'vs mt2 M4; owned slice target passed'),
    ('8', 'Tensor-core GEMM', None, 'killed', 'Shape probes only; no valid full-pass ratio'),
    ('8', 'Merged multi-token', latency_ratio(mt23, 'mt3', 'mt2', mode='causal', m=4, context=128), 'kept', 'vs mt2 M4; 1.5× cost target still missed'),
    ('8', 'KV compaction', kvc2['tokens_per_s']/decode8['tokens_per_s'], 'kept', 'vs v2 live 8k; retain 2k rows'),
    ('9', 'Lookup: code', spec['categories']['code']['speedup_v2'], 'kept', 'vs v2; evaluated code edits'),
    ('9', 'Lookup: RAG', spec['categories']['rag']['speedup_v2'], 'kept', 'vs v2; evaluated quotation workload'),
    ('9', 'Lookup: summaries', spec['categories']['summarization']['speedup_v2'], 'killed', 'vs v2; disabled'),
    ('9', 'Lookup: continuation', spec['categories']['chat']['speedup_v2'], 'kept', 'Control only; repetitive text, not general chat'),
]
fig, ax = figure('Nine waves kept useful pieces, not every target',
                 'Speed ratio against each idea’s own baseline. Kept can mean a useful slice, even when the full-pass target failed.',
                 'Sources: bench/lm03b,09,10,12,14,18,20,21,23,24; spec/analysis.json. Decisions: reports/wave-1..9.',
                 rect=(.25, .14, .36, .68))
note_ax = fig.add_axes((.64, .14, .33, .68), sharey=ax)
note_ax.axis('off')
for i, (wave, name, value, status, note) in enumerate(ideas):
    color = COLORS['positive'] if status == 'kept' else COLORS['negative'] if status == 'killed' else MUTED
    if value is not None:
        ax.barh(i, value, .65, color=color, zorder=2)
        ax.text(value + .025, i, f'{value:.2f}×', va='center', fontsize=9, color=INK)
    else:
        ax.text(.03, i, 'no speed ratio', va='center', fontsize=9, color=INK2)
    ax.text(-.025, i, f'W{wave}  {name}', transform=ax.get_yaxis_transform(),
            ha='right', va='center', fontsize=9.5, color=INK)
    note_ax.text(0, i, status.upper(), va='center', fontsize=9, color=color, weight='bold')
    note_ax.text(.23, i, note, va='center', fontsize=8.5, color=INK2)
ax.set_ylim(len(ideas)-.5, -.5)
ax.set_yticks([])
ax.grid(axis='y', visible=False)
ax.grid(axis='x')
ax.axvline(1, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
ax.set_xlim(0, 2.25)
ax.set_xticks([0, .5, 1, 1.5, 2], ['0', '0.5×', '1× baseline', '1.5×', '2×'])
save(fig, 'waves.png')

roof = PAIRS['Qwen3-0.6B'][0][0]['gbps'] * 100 / PAIRS['Qwen3-0.6B'][0][0]['pct_roofline']
fig, ax = figure('The megakernel runs closer to the bandwidth ceiling',
                 f'Weight + KV bytes per token ÷ median step time, as % of the measured {roof:.1f} GB/s read ceiling.',
                 'Sources: hw/hw.json, bench/lm11, lm12 results.jsonl. Byte model (packed weights + fp16 KV), not DRAM counters.')
mega_pct = [p[0]['gbps']/roof*100 for _, _, p in points]
llama_pct = [p[1]['gbps']/roof*100 for _, _, p in points]
ax.bar(x - w/2, mega_pct, w, color=COLORS['mega'], zorder=2)
ax.bar(x + w/2, llama_pct, w, color=COLORS['llama'], zorder=2)
for xi, m, l in zip(x, mega_pct, llama_pct):
    ax.text(xi - w/2, m + 1.2, f'{m:.0f}%', ha='center', va='bottom', fontsize=10.5, weight='bold', color=INK)
    ax.text(xi + w/2, l + 1.2, f'{l:.0f}%', ha='center', va='bottom', fontsize=10.5, color=INK2)
ax.axhline(100, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
ax.text(x[-1] + w, 101.5, 'measured read ceiling', ha='right', va='bottom', fontsize=10, color=INK2)
model_axis(ax, x, points)
ax.set_ylim(0, 112)
ax.set_yticks([0, 25, 50, 75, 100], ['0', '25%', '50%', '75%', '100%'])
ax.tick_params(axis='x', labelsize=11.5)
ax.legend(handles=[Patch(color=COLORS['mega'], label='This work'),
                   Patch(color=COLORS['llama'], label='llama.cpp Q4_0')],
          loc='upper left', fontsize=11.5, handlelength=1.2, ncol=2, bbox_to_anchor=(0, 1.0))
save(fig, 'roofline.png')
# Decode scan-length evidence and the complete scripted session use different clocks.
fig, ax = figure('Keeping fewer live KV rows makes long sessions cheaper',
                 f'Qwen3-0.6B. Logical RoPE positions stay unchanged. Scripted session wall time improves {session["speedup"]:.2f}×.',
                 'Sources: bench/lm24/timing.json decode; session.json segments + totals. Retention policy ungraded; not answer quality.',
                 rect=(.07, .2, .37, .6))
rates = [decode8['tokens_per_s'], kvc2['tokens_per_s'], decode2['tokens_per_s']]
ax.bar(np.arange(3), rates, color=[COLORS['llama'], COLORS['mega'], COLORS['q4km']], width=.62)
for i, value in enumerate(rates):
    ax.text(i, value + 8, f'{value:.1f}', ha='center', color=INK, weight='bold')
ax.set_xticks(np.arange(3), ['v2\n8192 live rows', 'kvc\n2048 retained', 'v2 control\n2048 live rows'], fontsize=10)
ax.set_ylim(0, max(rates)*1.22)
ax.set_ylabel('decode tokens / s')
ax.set_title(f'8k → 2k live rows: +{(rates[1]/rates[0]-1)*100:.1f}% speed', loc='left', fontsize=12, weight='bold', pad=14)
chunks = fig.add_axes((.55, .2, .41, .6))
segments = session['segments']
ends = [s['end']/1024 for s in segments]
for key, label, color in [('v2-extended_s', 'v2, extended allocation', COLORS['q4km']),
                          ('kvc_s', 'kvc, with compaction', COLORS['mega'])]:
    chunks.plot(ends, [s[key] for s in segments], marker='o', lw=2, color=color, label=label)
chunks.set_xlabel('generated tokens at chunk end, thousands of 1024')
chunks.set_ylabel('wall seconds per chunk')
chunks.set_xticks(ends, [f'{s["end"]/1024:g}' for s in segments], fontsize=9)
chunks.set_ylim(0, max(s['v2-extended_s'] for s in segments)*1.2)
chunks.legend(loc='upper left', fontsize=10)
chunks.set_title(f'{session["tokens"]:,} tokens: {session["totals_wall_s"]["v2-extended"]:.1f}s → '
                 f'{session["totals_wall_s"]["kvc"]:.1f}s', loc='left', fontsize=12, weight='bold', pad=14)
chunk_sizes = [s['end']-s['begin'] for s in segments]
fig.text(.55, .08, f'Chunk tokens: first {chunk_sizes[0]}, middle {chunk_sizes[1]}, final {chunk_sizes[-1]}.\n'
         f'All {session["compactions"]} compaction events included. One teacher-forced session.', fontsize=9.5, color=INK2)
save(fig, 'compaction.png')

# Prompt-level bootstrap CIs are supplied by the analysis, not recreated from repeats.
categories = spec['categories']
keys = ['code', 'rag', 'summarization', 'chat']
labels = ['Code edits', 'RAG-style answers', 'Summarization', 'Continuation control']
fig, ax = figure('Prompt lookup wins on code and RAG, not summaries',
                 'Decode wall speed vs v2, including CPU drafting, copies and rollback. Lossless to mt3 M1, not to v2.',
                 'Source: spec/analysis.json categories. Paired-prompt bootstrap 95% CIs; 20 prompts/category, 3 repeats; greedy only.',
                 rect=(.19, .23, .45, .57))
accept_ax = fig.add_axes((.72, .23, .24, .57), sharey=ax)
accept_ax.axis('off')
for i, (key, label) in enumerate(zip(keys, labels)):
    d = categories[key]
    value = d['speedup_v2']
    lo, hi = d['speedup_v2_ci95']
    color = COLORS['negative'] if key == 'summarization' else COLORS['q4km'] if key == 'chat' else COLORS['mega']
    ax.errorbar(value, i, xerr=[[value-lo], [hi-value]], fmt='o', markersize=10,
                color=color, capsize=6, linewidth=2)
    ax.text(value, i-.19, f'{value:.2f}× [{lo:.2f}, {hi:.2f}]', ha='center', fontsize=11, weight='bold', color=color)
    accept_ax.text(0, i-.12, f'{d["mean_accepted"]:.2f} accepted drafts / pass', fontsize=11, color=INK, va='center')
    verdict = 'DISABLED' if key == 'summarization' else 'repetitive-text control only' if key == 'chat' else 'enabled for evaluated workload'
    accept_ax.text(0, i+.15, verdict, fontsize=9.5, color=color, va='center')
ax.set_yticks(range(4), labels, fontsize=11)
ax.set_ylim(3.6, -.6)
ax.set_xlim(.75, 1.7)
ax.set_xticks([.8, 1, 1.2, 1.4, 1.6], ['0.8×', '1× baseline', '1.2×', '1.4×', '1.6×'])
ax.grid(axis='y', visible=False)
ax.grid(axis='x')
ax.axvline(1, color=INK2, linestyle=(0, (4, 3)))
repeat_pct = categories['chat']['output_repeated_fourgram_fraction_mean']*100
fig.text(.19, .105, f'Continuation repeats {repeat_pct:.1f}% of generated 4-grams. This is not a general chat speedup.\n'
         'No sampling or 8k-context workload was evaluated.', fontsize=10.5, color=INK2)
save(fig, 'speculative.png')

# A pass verifies positions. It does not generate M accepted dependent tokens.
fig, ax = figure('Multi-token verification still misses the 1.5× cost target',
                 'mt3 causal full-pass medians. One shared KV history. M counts anchor + draft positions, not accepted output tokens.',
                 'Source: bench/lm23/results.jsonl, experiment=full_pass, variant=mt3, mode=causal; 27 paired samples per row.',
                 rect=(.07, .19, .89, .6))
ax.remove()
for i, (ctx, color) in enumerate(zip(CONTEXTS, [COLORS['mega'], COLORS['q4km'], COLORS['exl2']])):
    panel = fig.add_axes((.07+i*.31, .19, .26, .6))
    values = [select(mt23, experiment='full_pass', variant='mt3', mode='causal', context=ctx, m=m)['median_ms']
              for m in range(1, 5)]
    target = values[0]*1.5
    panel.plot(range(1, 5), values, color=color, marker='o', lw=2)
    for m, value in enumerate(values, 1):
        panel.text(m, value + max(values)*.04, f'{value:.2f}', ha='center', fontsize=10, color=INK)
    panel.axhline(target, color=COLORS['negative'], linestyle=(0, (4, 3)))
    panel.text(.03, .94, f'Dashed target: 1.5× M1 = {target:.2f} ms', transform=panel.transAxes,
               fontsize=9, color='#B5523F', va='top')
    panel.set_title(f'{CTX_LABEL[ctx]} · M4/M1 {values[-1]/values[0]:.2f}×', loc='left', fontsize=12, weight='bold', pad=14)
    panel.set_xticks(range(1, 5), ['M1', 'M2', 'M3', 'M4'])
    panel.set_xlim(.8, 4.2)
    panel.set_ylim(0, max(values)*1.23)
    panel.set_xlabel('verified positions / pass')
    if i == 0:
        panel.set_ylabel('milliseconds / pass')
fig.text(.07, .09, 'All three M4 points miss the target. Accepted drafts and host overhead determine real speculative speed.', fontsize=10.5, color=INK2)
save(fig, 'multitoken.png')
print('Generated seven PNGs in publish/charts')
