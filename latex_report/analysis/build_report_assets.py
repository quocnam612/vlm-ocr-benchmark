"""Create the report's two tables, one figure and three traceable examples.

Run analyze_results.py first. Original result/ files are read only.
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
MODELS = ['M1', 'M2', 'M3', 'M4', 'M5']
NAMES = {'M1': 'Florence', 'M2': 'EasyOCR', 'M3': 'docTR', 'M4': 'PaddleOCR', 'M5': 'Tesseract'}
TYPES = ['Black_Ink', 'Sim', 'Through_Dust', 'Through_Stamp', 'White_Burnt', 'White_Whitener']


def fmt(value, decimals=2):
    return f'{value:.{decimals}f}'.replace('.', ',')


def latex_row(items):
    return ' & '.join(map(str, items)) + r' \\'


full = pd.read_csv(BASE / 'summary_common_test.csv').set_index('model')
cols = ['mean_cer', 'mean_wer', 'mean_token_precision', 'mean_token_recall', 'mean_token_f1', 'mean_latency_sec']
best = {c: full[c].idxmin() if c in ['mean_cer', 'mean_wer', 'mean_latency_sec'] else full[c].idxmax() for c in cols}
rows = []
for model in MODELS:
    values = []
    for col in cols:
        value = fmt(full.loc[model, col], 3) if col == 'mean_latency_sec' else fmt(full.loc[model, col]*100)
        values.append(r'\textbf{' + value + '}' if best[col] == model else value)
    rows.append(latex_row([NAMES[model], *values]))
(GEN / 'overall_table.tex').write_text(r'''\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrr@{}}
\toprule
\textbf{Mô hình} & \textbf{CER}$\downarrow$ & \textbf{WER}$\downarrow$ &
\textbf{P}$\uparrow$ & \textbf{R}$\uparrow$ & \textbf{F1}$\uparrow$ & \textbf{Giây/ảnh}$\downarrow$ \\
\midrule
''' + '\n'.join(rows) + '\n' + r'''\bottomrule
\end{tabular*}
''', encoding='utf-8')

groups = pd.read_csv(BASE / 'common_test_by_category.csv').set_index(['model', 'category'])
rows = []
for model in MODELS:
    values = []
    for category in ['text', 'nontext']:
        values.extend(fmt(groups.loc[(model, category), col]*100) for col in ['mean_cer', 'mean_wer', 'mean_token_f1'])
    rows.append(latex_row([NAMES[model], *values]))
(GEN / 'region_table.tex').write_text(r'''\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrr@{}}
\toprule
& \multicolumn{3}{c}{\textbf{Che vùng chữ: 6.000 ảnh}} &
\multicolumn{3}{c}{\textbf{Che vùng phi văn bản: 2.999 ảnh}} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
\textbf{Mô hình} & CER & WER & F1 & CER & WER & F1 \\
\midrule
''' + '\n'.join(rows) + '\n' + r'''\bottomrule
\end{tabular*}
''', encoding='utf-8')

types = pd.read_csv(BASE / 'common_test_by_occlusion_type.csv')
heatmap = types.pivot(index='model', columns='occlusion_type', values='mean_token_f1').loc[MODELS, TYPES]*100
counts = types.loc[types.model == 'M1'].set_index('occlusion_type').n
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(6.85, 1.58), layout='constrained')
im = ax.imshow(heatmap, vmin=30, vmax=85, cmap='YlGnBu', aspect='auto')
ax.set_yticks(range(5), [NAMES[m] for m in MODELS])
labels = [t + '\n' + f'n = {counts[t]:,}'.replace(',', '.') for t in TYPES]
ax.set_xticks(range(6), labels)
ax.tick_params(length=0, labelsize=8)
for i in range(5):
    for j in range(6):
        value = heatmap.iloc[i, j]
        ax.text(j, i, fmt(value, 1), ha='center', va='center', color='white' if value > 66 else '#152332', fontsize=9)
fig.colorbar(im, ax=ax, shrink=.92, pad=.015, ticks=[30, 55, 80], label='F1 (%)')
for spine in ax.spines.values():
    spine.set_visible(False)
fig.savefig(IMG / 'robustness.pdf')
fig.savefig(IMG / 'robustness.png', dpi=220)
plt.close(fig)

selected = [
    ('E1', 'M4', 'oprb_paddleocr_v5_en_gpu_results', 'text', 'Black_Ink__1p0pct__10.tar_1601.04736.gz_BiasCorrectedLeastSquares_arxiv2016b_36_ori'),
    ('E2', 'M5', 'oprb_tesseract_eng_fast_cpu_results', 'text', 'Through_Dust__1p0pct__107.tar_1507.03120.gz_arxiv_23_ori'),
    ('E3', 'M1', 'oprb_florence_standard_results', 'text', 'White_Burnt__2p0pct__103.tar_1408.3064.gz_WWDR_nil_10_ori'),
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
                     'cer': float(metrics.cer), 'wer': float(metrics.wer), 'f1': float(metrics.token_f1),
                     'possibly_truncated': bool(metrics.possibly_truncated),
                     'generated_tokens': int(float(row.generated_tokens)) if 'generated_tokens' in row and str(row.generated_tokens) else None})
(BASE / 'qualitative_examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding='utf-8')
print('Generated two tables, five-model robustness figure and three exact qualitative examples.')
