"""Audit original result/ CSVs and derive comparable scores without model inference.

Run: python latex_report/analysis/analyze_results.py
Dependencies: pandas, rapidfuzz. Original result/ files are never modified.
"""
from collections import Counter
import csv
from difflib import SequenceMatcher
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import unicodedata

import pandas as pd
from rapidfuzz.distance import Levenshtein

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)
(OUT / 'scored').mkdir(exist_ok=True)
MODELS = {
    'M1': ('Florence', 'oprb_florence_standard_results'),
    'M2': ('EasyOCR', 'oprb_easyocr_en_gpu_results'),
    'M3': ('docTR', 'oprb_doctr_mobilenet_gpu_results'),
    'M4': ('PaddleOCR', 'oprb_paddleocr_v5_en_gpu_results'),
    'M5': ('Tesseract', 'oprb_tesseract_eng_fast_cpu_results'),
}
METRICS = ['cer', 'wer', 'token_precision', 'token_recall', 'token_f1']
LAYOUT_MARKERS = {'##ltline##', '##ltfigure##'}
DENSITIES = {'0p5pct': 0.5, '1p0pct': 1., '2p0pct': 2.}


def normalize(text, filtered=False):
    text = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(text))).strip().casefold()
    return ' '.join(w for w in text.split() if w not in LAYOUT_MARKERS) if filtered else text


def score(reference, prediction):
    assert reference
    refs, hyps = reference.split(), prediction.split()
    rc, hc = Counter(refs), Counter(hyps)
    match = sum((rc & hc).values())
    precision = match / len(hyps) if hyps else 0.
    recall = match / len(refs)
    return {'cer': Levenshtein.distance(reference, prediction) / len(reference),
            'wer': Levenshtein.distance(refs, hyps) / len(refs),
            'token_precision': precision, 'token_recall': recall,
            'token_f1': 2*precision*recall/(precision+recall) if precision+recall else 0.,
            'gt_words': len(refs), 'pred_words': len(hyps)}


def assert_close(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10):
        raise AssertionError((label, actual, expected))


def summarize(frame, groups):
    return frame.groupby(groups, dropna=False).agg(
        n=('sample_id', 'count'),
        **{'mean_' + col: (col, 'mean') for col in METRICS + ['latency_sec']},
    ).reset_index()


def verify_group_csv(frame, saved_file, groups, raw=False):
    source = frame.copy()
    if raw:
        for col in METRICS:
            source[col] = source['raw_' + col]
    calculated = summarize(source, groups)
    saved = pd.read_csv(saved_file)
    pd.testing.assert_frame_equal(
        saved[calculated.columns].sort_values(groups).reset_index(drop=True),
        calculated.sort_values(groups).reset_index(drop=True),
        check_dtype=False, check_exact=False, atol=1e-10, rtol=1e-10,
    )


frames, configs, inputs, failures = {}, {}, [], []
for model, (name, directory) in MODELS.items():
    folder = ROOT / 'result' / directory
    csv_path = folder / 'per_image_results.csv'
    # Some original GT strings contain embedded NUL characters. The C engine
    # truncates those fields; Python's CSV engine preserves the scored text.
    frame = pd.read_csv(csv_path, keep_default_na=False, engine='python')
    assert not frame.duplicated(['category', 'sample_id']).any(), name
    config = json.loads((folder / 'run_config.json').read_text())
    assert config['split'] == 'test' and config['normalize_case'] is True
    assert config['no_finetuning'] is True and config['max_samples_per_category'] is None
    assert frame.model_id.eq(config['model']).all()
    assert all(config[k] == v for k, v in json.loads((folder / 'checkpoint_config.json').read_text()).items())
    for filename in ['per_image_results.csv', 'run_config.json', 'checkpoint_config.json']:
        path = folder / filename
        inputs.append({'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    for row in frame.loc[frame.status != 'ok'].to_dict('records'):
        failures.append({'model': model, 'name': name, 'category': row['category'], 'sample_id': row['sample_id'], 'error': row.get('error', '')})
    frame = frame.set_index(['category', 'sample_id']).sort_index()
    frames[model], configs[model] = frame, config

canonical = frames['M2'][['ground_truth']].copy()
assert len(canonical) == 9000 and canonical.index.get_level_values('category').value_counts().to_dict() == {'text': 6000, 'nontext': 3000}
canonical['raw_ref'] = canonical.ground_truth.map(normalize)
canonical['ref'] = canonical.ground_truth.map(lambda s: normalize(s, True))
canonical['valid_reference'] = canonical.ref.ne('')
canonical['occlusion_type'] = [key[1].split('__', 1)[0] for key in canonical.index]
canonical['density'] = [key[1].split('__', 2)[1] if len(key[1].split('__', 2)) > 1 and key[1].split('__', 2)[1] in DENSITIES else 'unspecified' for key in canonical.index]
canonical['markers_removed'] = canonical.raw_ref.map(lambda s: sum(w in LAYOUT_MARKERS for w in s.split()))
invalid_keys = set(canonical.index[~canonical.valid_reference])
valid_keys = set(canonical.index[canonical.valid_reference])
canonical.reset_index().groupby(['category', 'occlusion_type']).size().rename('n').reset_index().to_csv(OUT / 'dataset_by_type.csv', index=False)
canonical.reset_index().groupby(['category', 'density']).size().rename('n').reset_index().to_csv(OUT / 'dataset_by_density.csv', index=False)
canonical.loc[~canonical.valid_reference, ['ground_truth']].reset_index().to_csv(OUT / 'excluded_references.csv', index=False)

coverage, details, checked_rows = [], [], 0
scored, cohorts = {}, {}
for model, (name, directory) in MODELS.items():
    frame, config = frames[model], configs[model]
    folder = ROOT / 'result' / directory
    assert set(frame.index) <= set(canonical.index)
    succeeded = frame.loc[frame.status == 'ok'].copy()
    for col in METRICS + ['latency_sec', 'gt_words', 'pred_words']:
        succeeded[col] = pd.to_numeric(succeeded[col], errors='raise')
    assert succeeded.latency_sec.gt(0).all()
    assert succeeded.ground_truth.equals(canonical.loc[succeeded.index, 'ground_truth'])
    cleaned = []
    uses_filtered = config.get('metric_policy_id') == 'oprb_macro_nfkc_layout_markers_removed_v1'
    for key, row in succeeded.iterrows():
        raw_ref = canonical.at[key, 'raw_ref']
        raw_hyp = normalize(row.prediction)
        raw = score(raw_ref, raw_hyp)
        ref, hyp = canonical.at[key, 'ref'], normalize(row.prediction, True)
        filtered = score(ref, hyp) if ref else None
        expected = filtered if uses_filtered else raw
        assert expected is not None
        for col, value in expected.items():
            assert_close(row[col], value, (model, key, col))
        if uses_filtered:
            for col, value in raw.items():
                assert_close(row['raw_' + col], value, (model, key, 'raw_' + col))
            assert float(row.n_gt_markers_removed) == canonical.at[key, 'markers_removed']
            assert float(row.n_pred_markers_removed) == sum(w in LAYOUT_MARKERS for w in raw_hyp.split())
        checked_rows += 1
        if filtered is not None:
            cleaned.append({'category': key[0], 'sample_id': key[1], 'model': model, 'name': name,
                            'occlusion_type': canonical.at[key, 'occlusion_type'], 'density': canonical.at[key, 'density'],
                            'latency_sec': float(row.latency_sec), 'empty_prediction': not hyp,
                            'possibly_truncated': str(row.get('possibly_truncated', False)).lower() == 'true',
                            **filtered})
    for col in METRICS + ['latency_sec']:
        assert_close(config['metrics']['mean_' + col], succeeded[col].mean(), (model, 'overall', col))
    assert config['metrics']['n_success'] == len(succeeded)
    assert config['metrics']['n_errors'] == len(frame) - len(succeeded)
    for filename, groups in [('summary_by_category.csv', ['category']),
                             ('summary_by_occlusion_type.csv', ['category', 'occlusion_type']),
                             ('summary_by_density.csv', ['category', 'occlusion_density'])]:
        verify_group_csv(succeeded.reset_index(), folder / filename, groups)
        if uses_filtered:
            rawframe = succeeded.reset_index()
            for col in METRICS:
                rawframe['raw_' + col] = pd.to_numeric(rawframe['raw_' + col])
            verify_group_csv(rawframe, folder / ('raw_' + filename), groups, raw=True)
    cleaned = pd.DataFrame(cleaned).set_index(['category', 'sample_id']).sort_index()
    cleaned.reset_index().to_csv(OUT / 'scored' / (model + '.csv'), index=False)
    scored[model] = cleaned
    coverage.append({'model': model, 'name': name, 'selected': 9000, 'attempted': len(frame),
                     'status_ok': len(succeeded), 'scorable': len(cleaned), 'errors': len(frame)-len(succeeded),
                     'unattempted': 9000-len(frame), 'empty_predictions': int(cleaned.empty_prediction.sum()),
                     'truncated': int(cleaned.possibly_truncated.sum()),
                     'text_scorable': int((cleaned.index.get_level_values('category') == 'text').sum()),
                     'nontext_scorable': int((cleaned.index.get_level_values('category') == 'nontext').sum())})
    details.append({'model': model, 'max_cer': float(cleaned.cer.max()), 'n_cer_gt_one': int(cleaned.cer.gt(1).sum()),
                    'max_wer': float(cleaned.wer.max()), 'n_wer_gt_one': int(cleaned.wer.gt(1).sum())})
    print(f'Verified {name}: {len(frame)} attempted, {len(succeeded)} successful, {len(cleaned)} valid references.', flush=True)

pd.DataFrame(coverage).to_csv(OUT / 'coverage.csv', index=False)
pd.DataFrame(failures).to_csv(OUT / 'execution_errors.csv', index=False)
all_available = pd.concat([frame.reset_index() for frame in scored.values()], ignore_index=True)
summarize(all_available, ['model', 'name']).to_csv(OUT / 'summary_all_available.csv', index=False)
common_all = set.intersection(*(set(frame.index) for frame in scored.values()))
common_ocr = set.intersection(*(set(scored[model].index) for model in ['M2', 'M3', 'M4', 'M5']))
assert len(common_all) == 5790 and {key[0] for key in common_all} == {'text'}
assert common_ocr == valid_keys and len(common_ocr) == 8999
for cohort_name, keys, models in [('common_text', common_all, list(MODELS)), ('common_ocr', common_ocr, ['M2', 'M3', 'M4', 'M5'])]:
    rows = pd.concat([scored[model].loc[sorted(keys)].reset_index() for model in models], ignore_index=True)
    summarize(rows, ['model', 'name']).to_csv(OUT / ('summary_' + cohort_name + '.csv'), index=False)
    for group in ['category', 'occlusion_type', 'density']:
        summarize(rows, ['model', 'name', group]).to_csv(OUT / (cohort_name + '_by_' + group + '.csv'), index=False)
    summarize(rows, ['model', 'name', 'category', 'occlusion_type']).to_csv(OUT / (cohort_name + '_by_category_type.csv'), index=False)
    pd.DataFrame(sorted(keys), columns=['category', 'sample_id']).to_csv(OUT / (cohort_name + '_ids.csv'), index=False)

# Candidate examples are restricted to common images and retain exact original text.
examples = []
for model in MODELS:
    subset = scored[model].loc[sorted(common_all)]
    candidates = set(subset.nsmallest(8, 'cer').index) | set(subset.nlargest(8, 'wer').index)
    candidates |= set(subset.loc[subset.possibly_truncated].nsmallest(3, 'wer').index)
    candidates |= set(subset.loc[(subset.token_f1 > .75) & (subset.wer > .7)].head(3).index)
    for key in sorted(candidates):
        row = frames[model].loc[key]
        metrics = scored[model].loc[key]
        ref, hyp = canonical.at[key, 'ref'], normalize(row.prediction, True)
        matcher = SequenceMatcher(None, ref.split(), hyp.split(), autojunk=False)
        edits = []
        for tag, i, j, k, l in matcher.get_opcodes():
            if tag != 'equal':
                edits.append({'operation': tag, 'gt': ' '.join(ref.split()[i:j])[:240], 'prediction': ' '.join(hyp.split()[k:l])[:240]})
        examples.append({'model': model, 'category': key[0], 'sample_id': key[1],
                         'cer': float(metrics.cer), 'wer': float(metrics.wer), 'f1': float(metrics.token_f1),
                         'possibly_truncated': bool(metrics.possibly_truncated),
                         'gt': str(row.ground_truth), 'prediction': str(row.prediction), 'edits': edits[:12]})
(OUT / 'qualitative_candidates.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding='utf-8')
audit = {'date': '2026-11-20', 'source_directory': 'result/', 'csv_reader': 'pandas engine=python, preserve embedded NUL characters', 'metric_policy': 'NFKC, casefold, collapsed whitespace; remove exact layout tokens in GT and prediction',
         'excluded_layout_tokens': sorted(LAYOUT_MARKERS), 'metric_aggregation': 'Unweighted arithmetic mean of per-image rates',
         'verified_success_records': checked_rows, 'invalid_reference_count': len(invalid_keys),
         'common_all_models_count': len(common_all), 'common_ocr_count': len(common_ocr),
         'gt_layout_marker_count': int(canonical.markers_removed.sum()),
         'unspecified_density_count': int(canonical.density.eq('unspecified').sum()),
         'coverage': coverage, 'rate_outliers': details, 'inputs': inputs}
(OUT / 'audit_summary.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
print('\nCOMMON TEXT (all five):\n', pd.read_csv(OUT / 'summary_common_text.csv').to_string(index=False), flush=True)
print('\nFULL VALID OCR TEST:\n', pd.read_csv(OUT / 'summary_common_ocr.csv').to_string(index=False), flush=True)
print('Audit passed. Original result/ files untouched. Derived data:', OUT, flush=True)
