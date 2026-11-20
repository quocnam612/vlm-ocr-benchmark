"""Build LaTeX tables, a static research figure and traceable examples.

Run analyze_results.py first. Requires pandas and matplotlib in addition to
the audit dependencies. Original result/ files are read only.
"""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

csv.field_size_limit(sys.maxsize)
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
GEN = BASE / 'generated'
GEN.mkdir(exist_ok=True)
IMG = BASE.parent / 'img'
MODELS = ['M2', 'M3', 'M4', 'M5']
TYPES = ['Black_Ink', 'Sim', 'Through_Dust', 'Through_Stamp', 'White_Burnt', 'White_Whitener']


def fmt(value, decimals=2):
    return f'{value:.{decimals}f}'.replace('.', ',')


def latex_row(items):
    return ' & '.join(map(str, items)) + r' \\'


text = pd.read_csv(BASE / 'summary_common_text.csv').set_index('model')
cols = ['mean_cer', 'mean_wer', 'mean_token_precision', 'mean_token_recall', 'mean_token_f1']
best = {c: text[c].idxmin() if c in cols[:2] else text[c].idxmax() for c in cols}
rows = []
for model, row in text.iterrows():
    values = [r'\textbf{' + fmt(row[c]*100) + '}' if best[c] == model else fmt(row[c]*100) for c in cols]
    latency = fmt(row.mean_latency_sec, 3)
    if model == 'M4':
        latency = r'\textbf{' + latency + '}'
    rows.append(latex_row([model, *values, latency]))
(GEN / 'common_text_table.tex').write_text(r'''\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrr@{}}
\toprule
\textbf{Mã} & \textbf{CER}$\downarrow$ & \textbf{WER}$\downarrow$ &
\textbf{P}$\uparrow$ & \textbf{R}$\uparrow$ & \textbf{F1}$\uparrow$ & $t\downarrow$ \\
\midrule
''' + '\n'.join(rows) + '\n' + r'''\bottomrule
\end{tabular*}
''', encoding='utf-8')

full = pd.read_csv(BASE / 'summary_common_ocr.csv').set_index('model')
groups = pd.read_csv(BASE / 'common_ocr_by_category.csv').set_index(['model', 'category'])
rows = []
for model in MODELS:
    values = []
    for row in [full.loc[model], groups.loc[(model, 'text')], groups.loc[(model, 'nontext')]]:
        values.extend(fmt(row[c]*100) for c in ['mean_cer', 'mean_wer', 'mean_token_f1'])
    rows.append(latex_row([model, *values]))
(GEN / 'full_ocr_table.tex').write_text(r'''\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrrrrr@{}}
\toprule
& \multicolumn{3}{c}{\textbf{Tổng: 8.999 ảnh}} &
\multicolumn{3}{c}{\textbf{Text: 6.000 ảnh}} &
\multicolumn{3}{c}{\textbf{Nontext: 2.999 ảnh}} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
\textbf{Mã} & CER & WER & F1 & CER & WER & F1 & CER & WER & F1 \\
\midrule
''' + '\n'.join(rows) + '\n' + r'''\bottomrule
\end{tabular*}
''', encoding='utf-8')

types = pd.read_csv(BASE / 'common_ocr_by_occlusion_type.csv')
density = pd.read_csv(BASE / 'common_ocr_by_density.csv').set_index(['model', 'density'])
heatmap = types.pivot(index='model', columns='occlusion_type', values='mean_token_f1').loc[MODELS, TYPES]*100
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(6.85, 1.60), gridspec_kw={'width_ratios': [1.8, 1.0]}, layout='constrained')
ax = axes[0]
im = ax.imshow(heatmap, vmin=30, vmax=85, cmap='YlGnBu', aspect='auto')
ax.set_yticks(range(4), MODELS)
ax.set_xticks(range(6), ['Ink\n1537', 'Sim\n1423', 'Dust\n1537', 'Stamp\n1429', 'Burnt\n1536', 'Whitener\n1537'])
ax.tick_params(length=0, labelsize=7)
for i in range(4):
    for j in range(6):
        v = heatmap.iloc[i, j]
        ax.text(j, i, f'{v:.1f}', ha='center', va='center', color='white' if v>66 else '#152332', fontsize=8)
ax.set_title('F1 theo loại che (%)', loc='left', fontsize=9)
fig.colorbar(im, ax=ax, shrink=.9, pad=.02, ticks=[30, 55, 80])
for spine in ax.spines.values():
    spine.set_visible(False)
ax = axes[1]
for model, color, marker in zip(MODELS, ['#ba6113', '#3164ab', '#00836f', '#8b4b9c'], ['s', '^', 'o', 'D']):
    vals = [density.loc[(model, d), 'mean_token_f1']*100 for d in ['0p5pct', '1p0pct', '2p0pct']]
    ax.plot([.5, 1, 2], vals, marker=marker, color=color, label=model, linewidth=1.3, markersize=3.5)
    ax.text(2.06, vals[-1], model, color=color, va='center', fontsize=7)
ax.set_xlim(.4, 2.4)
ax.set_xticks([.5, 1, 2], ['0,5', '1', '2'])
ax.set_ylim(30, 85)
ax.set_yticks([40, 60, 80])
ax.set_xlabel('Mật độ (%)', labelpad=1)
ax.set_ylabel('F1 (%)', labelpad=1)
ax.set_title('F1 theo mật độ', loc='left', fontsize=9)
ax.grid(axis='y', alpha=.2)
ax.spines[['top', 'right']].set_visible(False)
fig.savefig(IMG / 'robustness.pdf')
fig.savefig(IMG / 'robustness.png', dpi=220)
plt.close(fig)

selected = [
    ('E1', 'M4', 'oprb_paddleocr_v5_en_gpu_results', 'text', 'Black_Ink__1p0pct__10.tar_1601.04736.gz_BiasCorrectedLeastSquares_arxiv2016b_36_ori'),
    ('E2', 'M5', 'oprb_tesseract_eng_fast_cpu_results', 'text', 'Through_Dust__1p0pct__107.tar_1507.03120.gz_arxiv_23_ori'),
    ('E3', 'M4', 'oprb_paddleocr_v5_en_gpu_results', 'nontext', 'White_Burnt__2p0pct__102.tar_1606.00144.gz_math_model_arxiv_28_ori'),
]
examples = []
for eid, model, directory, category, sample_id in selected:
    source = pd.read_csv(ROOT / 'result' / directory / 'per_image_results.csv', engine='python', keep_default_na=False)
    row = source.loc[(source.category == category) & (source.sample_id == sample_id)].iloc[0]
    scores = pd.read_csv(BASE / 'scored' / (model + '.csv'))
    metrics = scores.loc[(scores.category == category) & (scores.sample_id == sample_id)].iloc[0]
    examples.append({'example': eid, 'model': model, 'category': category, 'sample_id': sample_id,
                     'source': str((ROOT / 'result' / directory / 'per_image_results.csv').relative_to(ROOT)),
                     'ground_truth': row.ground_truth, 'prediction': row.prediction,
                     'cer': float(metrics.cer), 'wer': float(metrics.wer), 'f1': float(metrics.token_f1)})
(BASE / 'qualitative_examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding='utf-8')
print('Generated tables, robustness figure and three traceable qualitative examples.')
