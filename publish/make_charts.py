"""Render publication charts from the recorded benchmark JSONL/JSON evidence.

Run from repository root: .venv/bin/python publish/make_charts.py
Only the original paired 27/30-sample LM-11 and LM-12 Q4_0 runs enter the
headline comparison. Later replications remain in the JSONL audit trail.
"""
import json
import re
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

# Ratio measurements are drawn from each experiment's own interleaved rows.
old = rows('bench/lm03b/results.jsonl')
graph = next(r for r in old if r['context'] == 128 and r['kernel'] == 'separate-graph')
v1 = next(r for r in old if r['context'] == 128 and r['kernel'] == 'megakernel-v1')
v2 = next(r for r in old if r['context'] == 128 and r['kernel'] == 'megakernel-v2')
kv = rows('bench/lm09/results.jsonl')
sync = rows('bench/lm10/results.jsonl')
wave1 = (ROOT / 'reports/wave-1.md').read_text()
lattice = re.search(r'\| A4-k9 \|[^\n]+\| ([0-9.]+) / ([0-9.]+) / ([0-9.]+) \|', wave1)
if lattice is None:
    raise ValueError('Wave-1 A4-k9 matvec ratio missing')
def rate_ratio(records, name, baseline, ctx):
    a = next(r for r in records if r['context'] == ctx and r['kernel'] == name)
    b = next(r for r in records if r['context'] == ctx and r['kernel'] == baseline)
    return a['tokens_per_s']/b['tokens_per_s']
ideas = [
    ('Wave 1', 'Lattice weight codes', float(lattice.group(3)), False,
     None),
    ('Wave 2', 'First persistent kernel', v1['tokens_per_s']/graph['tokens_per_s'], False,
     'vs separate CUDA-graph kernels'),
    ('Wave 3', 'Occupancy fix', v2['tokens_per_s']/graph['tokens_per_s'], True,
     'vs separate CUDA-graph kernels'),
    ('Wave 4', 'Compressed KV cache', rate_ratio(kv, 'megakernel-kv', 'megakernel-v2', 8192), False,
     'vs fp16 KV, ctx 8k (also failed correctness)'),
    ('Wave 4', 'Fine-grained sync', rate_ratio(sync, 'megakernel-sync', 'megakernel-v2', 128), False,
     'vs grid barriers'),
    ('Wave 5', 'Scale to Qwen3-1.7B', PAIRS['Qwen3-1.7B'][0][0]['tokens_per_s']/PAIRS['Qwen3-1.7B'][0][1]['tokens_per_s'], True,
     'vs llama.cpp Q4_0'),
]
floor = re.search(r'matching INT4 error needs >= ([0-9.]+) b/w', wave1)
if floor is None:
    raise ValueError('Wave-1 lattice bound missing')
ideas[0] = ideas[0][:4] + (f'vs INT4 kernel; any codec needs ≥{floor.group(1)} bits',)
killed = sum(not kept for *_, kept, _ in ideas)
fig, ax = figure(f'{len(ideas)} experiments, {killed} dead ends',
                 'Speed ratio of each idea against its own baseline. Right of the dashed line is faster.',
                 'Sources: reports/wave-1.md; bench/lm03b, lm09, lm10, lm12 results.jsonl. Each ratio from same-process runs.',
                 rect=(.25, .13, .72, .7))
y = np.arange(len(ideas))
ax.barh(y, [v for _, _, v, _, _ in ideas], .62, zorder=2,
        color=[COLORS['positive'] if kept else COLORS['negative'] for *_, kept, _ in ideas])
ax.set_yticks(y, [''] * len(ideas))
for i, (wave, name, v, kept, note) in enumerate(ideas):
    ax.text(-.03, i - .1, name, transform=ax.get_yaxis_transform(), ha='right', va='center',
            fontsize=12.5, weight='bold', color=INK)
    ax.text(-.03, i + .24, wave, transform=ax.get_yaxis_transform(), ha='right', va='center',
            fontsize=10, color=MUTED)
    label_box = dict(facecolor='white', edgecolor='none', pad=1.5)
    ax.text(v + .025, i - .1, f'{v:.2f}×  {"kept" if kept else "killed"}', va='center', fontsize=12.5,
            weight='bold', color=COLORS['positive'] if kept else '#B5523F', bbox=label_box, zorder=4)
    ax.text(v + .025, i + .24, note, va='center', fontsize=10, color=INK2, bbox=label_box, zorder=4)
ax.invert_yaxis()
ax.grid(axis='y', visible=False)
ax.grid(axis='x')
ax.axvline(1, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1.5)
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
print('Generated four PNGs in publish/charts')
