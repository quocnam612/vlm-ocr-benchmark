from __future__ import annotations
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description='OPRB PaddleOCR benchmark in an isolated environment')
parser.add_argument('--config', required=True)
args = parser.parse_args()
SETTINGS = json.loads(Path(args.config).read_text(encoding='utf-8'))
if sys.prefix == sys.base_prefix:
    raise RuntimeError('Run this script with the isolated virtual-environment Python.')
if importlib.util.find_spec('torch') is not None:
    raise RuntimeError('The isolated Paddle environment must not contain PyTorch or inherit system site-packages.')
os.environ['PADDLE_PDX_EAGER_INIT'] = 'False'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['PADDLE_PDX_DISABLE_DEVICE_FALLBACK'] = 'True'
os.environ['PADDLE_PDX_CACHE_HOME'] = SETTINGS.get('sdk_cache_dir', '/tmp/oprb_paddle_cache')
os.environ['MPLBACKEND'] = 'Agg'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/oprb_paddle_matplotlib')
print('ISOLATED RUNTIME:', sys.executable, '; Python', sys.version.split()[0], '; PyTorch absent', flush=True)


print('Import Paddle and validate SDK', flush=True)

import gc, hashlib, json, os, platform, random, re, shutil, tarfile, time, unicodedata
from collections import Counter
from importlib.metadata import version
from pathlib import Path

# Chỉ tắt kiểm tra kết nối tới các kho khác; trọng số tải rõ ràng từ Hugging Face ở Mục 7.
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['PADDLE_PDX_DISABLE_DEVICE_FALLBACK'] = 'True'
# Chỉ dùng inference, không khởi tạo sớm các repository training của PaddleX.
# Phải đặt trước lần import paddlex/paddleocr đầu tiên trong kernel.
os.environ['PADDLE_PDX_EAGER_INIT'] = 'False'

import numpy as np
import pandas as pd
import cv2
import paddle
import yaml
from PIL import Image
def display(value):
    if isinstance(value, pd.DataFrame):
        print(value.to_string(index=False), flush=True)
from huggingface_hub import hf_hub_download
from jiwer import cer, wer
from tqdm import tqdm
import matplotlib.pyplot as plt

CUDA_AVAILABLE = bool(paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0)
print('Python:', platform.python_version(), flush=True)
print('Paddle:', paddle.__version__, '; CUDA GPU dùng được:', CUDA_AVAILABLE, flush=True)
if CUDA_AVAILABLE:
    print('GPU đầu tiên:', paddle.device.cuda.get_device_name(0), flush=True)
else:
    print('Sẽ chạy trên CPU. Muốn dùng GPU: bật GPU và chạy lại Mục 1 trong kernel mới.', flush=True)


def load_paddleocr_class():
    """Kiểm tra SDK sớm; không sửa/xóa trạng thái nội bộ PaddleX trong kernel lỗi."""
    for package, expected in [('paddleocr', '3.4.1'), ('paddlex', '3.4.0')]:
        if version(package) != expected:
            raise RuntimeError(f'Cần {package}=={expected}; chạy Mục 1 và restart kernel nếu đã import bản khác.')
    os.environ['PADDLE_PDX_EAGER_INIT'] = 'False'
    try:
        from paddleocr import PaddleOCR
    except RuntimeError as exc:
        if 'PDX has already been initialized' in str(exc):
            raise RuntimeError(
                'PaddleX còn trạng thái khởi tạo trong kernel cũ. Restart kernel, '
                'rồi chạy từ Mục 2; không chỉ chạy lại cell lỗi. Không cần xóa dữ liệu/checkpoint.'
            ) from exc
        raise
    return PaddleOCR


PaddleOCR = load_paddleocr_class()
print('SDK import OK:', 'PaddleOCR', version('paddleocr'), '; PaddleX', version('paddlex'), flush=True)



print('Configuration', flush=True)

REPO_ID = SETTINGS.get('repo_id', 'kpurkayastha/OPRB')
MODEL_ID = 'PaddleOCR:PP-OCRv5_mobile_det+en_PP-OCRv5_mobile_rec'
DATA_ROOT = Path(SETTINGS['data_root'])
MODEL_DIR = Path(SETTINGS['model_dir'])
DEVICE_PREFERENCE = SETTINGS['device_preference']
assert DEVICE_PREFERENCE in {'auto', 'gpu', 'cpu'}
if DEVICE_PREFERENCE == 'gpu' and (not CUDA_AVAILABLE):
    raise RuntimeError('Đã chọn gpu nhưng Paddle chưa dùng được GPU. Kiểm tra Mục 1–2.')
USE_CUDA = CUDA_AVAILABLE and DEVICE_PREFERENCE != 'cpu'
DEVICE = 'gpu:0' if USE_CUDA else 'cpu'
DEVICE_NAME = paddle.device.cuda.get_device_name(0) if USE_CUDA else 'CPU'
paddle.set_device(DEVICE)
OUTPUT_DIR = Path(SETTINGS['output_dir'])
RESUME_FROM = Path(SETTINGS['resume_from']) if SETTINGS.get('resume_from') else None
CATEGORIES = SETTINGS['categories']
MAX_SAMPLES_PER_CATEGORY = SETTINGS['max_samples_per_category']
RANDOM_SEED = SETTINGS.get('random_seed', 42)
NORMALIZE_CASE = SETTINGS.get('normalize_case', True)
MAX_EVAL_HOURS = SETTINGS.get('max_eval_hours', 10.0)
CPU_THREADS = SETTINGS.get('cpu_threads', 2)
GT_LAYOUT_MARKERS = ['##LTLine##', '##LTFigure##']
METRIC_POLICY_ID = 'oprb_macro_nfkc_layout_markers_removed_v1'
DETECT_NETWORK = 'PP-OCRv5_mobile_det'
RECOG_NETWORK = 'en_PP-OCRv5_mobile_rec'
MODEL_SPECS = {'detector': {'name': DETECT_NETWORK, 'repo_id': f'PaddlePaddle/{DETECT_NETWORK}', 'revision': '0d63e78e2b680928f6b1747d76a08db6e645efb7'}, 'recognizer': {'name': RECOG_NETWORK, 'repo_id': f'PaddlePaddle/{RECOG_NETWORK}', 'revision': '267c36e24c331595590fe7bd72bde2436fd286f2'}}
MODEL_FILES = ['inference.json', 'inference.pdiparams', 'inference.yml']
RECOGNITION_BATCH_SIZE = 16 if USE_CUDA else 4
OCR_KWARGS = {'use_doc_orientation_classify': False, 'use_doc_unwarping': False, 'use_textline_orientation': False, 'text_det_limit_side_len': 960, 'text_det_limit_type': 'max', 'text_det_thresh': 0.3, 'text_det_box_thresh': 0.6, 'text_det_unclip_ratio': 1.5, 'text_recognition_batch_size': RECOGNITION_BATCH_SIZE, 'text_rec_score_thresh': 0.0, 'return_word_box': False}
RUNTIME_KWARGS = {'device': DEVICE, 'enable_hpi': False, 'use_tensorrt': False, 'precision': 'fp32', 'enable_mkldnn': False, 'enable_cinn': False, 'cpu_threads': CPU_THREADS}
assert MAX_EVAL_HOURS is None or MAX_EVAL_HOURS > 0
assert MAX_SAMPLES_PER_CATEGORY is None or MAX_SAMPLES_PER_CATEGORY > 0
assert CPU_THREADS > 0 and RECOGNITION_BATCH_SIZE > 0
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
paddle.seed(RANDOM_SEED)
print('Pipeline:', MODEL_ID, flush=True)
print('Thiết bị:', DEVICE, DEVICE_NAME, flush=True)
print('Mỗi loại:', MAX_SAMPLES_PER_CATEGORY, '; seed:', RANDOM_SEED, flush=True)
print('Điểm chính:', METRIC_POLICY_ID, flush=True)
print('Output:', OUTPUT_DIR, flush=True)

runtime_packages = {
    distribution.metadata['Name']: distribution.version
    for distribution in importlib.metadata.distributions()
    if distribution.metadata.get('Name')
}
(OUTPUT_DIR / 'runtime_packages.json').write_text(
    json.dumps(dict(sorted(runtime_packages.items())), indent=2), encoding='utf-8'
)




print('Download test images and scoring GT', flush=True)

def extract_flat(archive_path: str, destination: Path, allowed_suffix: str) -> int:
    """Safely copy only files of the specified type; never use archive-supplied paths."""
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(archive_path, mode='r:gz') as archive:
        for member in tqdm(archive, desc=f'Extract {destination.parent.name}/{destination.name}', leave=False):
            if not member.isfile() or not member.name.lower().endswith(allowed_suffix):
                continue
            filename = Path(member.name).name
            target = destination / filename
            if not target.exists():
                stream = archive.extractfile(member)
                if stream is None:
                    continue
                with stream, target.open('wb') as out:
                    shutil.copyfileobj(stream, out)
            count += 1
    return count

def download_test_category(category: str):
    if category not in {'text', 'nontext'}:
        raise ValueError(f'Unknown category: {category}')
    for kind, suffix in (('img', '.jpg'), ('gt', '.txt')):
        repo_path = f'test/{category}/{kind}.tar.gz'
        target_dir = DATA_ROOT / category / kind
        existing = list(target_dir.glob(f'*{suffix}')) if target_dir.exists() else []
        expected = 6000 if category == 'text' else 3000
        if len(existing) >= expected:
            print(f'Reuse {category}/{kind}: {len(existing)} files')
            continue
        print(f'Downloading {repo_path} ...')
        tar_path = hf_hub_download(repo_id=REPO_ID, repo_type='dataset', filename=repo_path)
        n = extract_flat(tar_path, target_dir, suffix)
        print(f'  {category}/{kind}: {n:,} archive members of type {suffix} -> {target_dir}')
if SETTINGS.get('download_data', True):
    for category in CATEGORIES:
        download_test_category(category)
else:
    print('Using local test images/GT from:', DATA_ROOT, flush=True)




print('Pair test images and GT', flush=True)

def parse_occlusion_metadata(stem: str):
    # OPRB filename convention: OcclusionType__density__batchID... 
    parts = stem.split("__", 2)
    return {
        "occlusion_type": parts[0] if parts else "unknown",
        "occlusion_density": parts[1] if len(parts) > 1 else "unknown",
    }


def collect_test_pairs(categories):
    pairs = []
    for category in categories:
        image_dir = DATA_ROOT / category / "img"
        gt_dir = DATA_ROOT / category / "gt"
        images = {p.stem: p for p in image_dir.glob("*.jpg")}
        gts = {p.stem: p for p in gt_dir.glob("*.txt")}
        common = sorted(images.keys() & gts.keys())
        print(f"{category}: images={len(images):,}, gt={len(gts):,}, paired={len(common):,}")
        if not common:
            raise RuntimeError(f"No paired files for {category}. Check extraction and naming.")
        if MAX_SAMPLES_PER_CATEGORY is not None:
            rng = random.Random(RANDOM_SEED)
            common = sorted(rng.sample(common, min(MAX_SAMPLES_PER_CATEGORY, len(common))))
        for stem in common:
            pairs.append({
                "category": category,
                "sample_id": stem,
                "image_path": str(images[stem]),
                "gt_path": str(gts[stem]),
                **parse_occlusion_metadata(stem),
            })
    return pairs


pairs = collect_test_pairs(CATEGORIES)
print(f"Selected {len(pairs):,} samples. Set MAX_SAMPLES_PER_CATEGORY=None to use the entire test set.")
display(pd.DataFrame(pairs)[["category", "occlusion_type", "occlusion_density", "sample_id"]].head(8))

with Image.open(pairs[0]["image_path"]) as image:
    preview = image.copy()
preview.thumbnail((650, 850))
display(preview)
if MAX_SAMPLES_PER_CATEGORY is None:
    expected_counts = {'text': 6000, 'nontext': 3000}
    actual_counts = Counter(sample['category'] for sample in pairs)
    for category in CATEGORIES:
        if actual_counts[category] != expected_counts[category]:
            raise RuntimeError(f'Full test requires {expected_counts[category]} paired {category} pages; got {actual_counts[category]}.')
preview.save(OUTPUT_DIR / 'preview.png')




print('Metrics: layout tokens removed + legacy raw scores', flush=True)

def read_gt_tokens(gt_path: str):
    tokens = []
    with open(gt_path, 'r', encoding='utf-8-sig', errors='replace') as file:
        for line in file:
            fields = line.rstrip('\r\n').split('\t')
            if len(fields) >= 10 and fields[0].strip():
                tokens.append(fields[0].strip())
    if not tokens:
        raise ValueError(f'No GT tokens read from {gt_path}. Check annotation format.')
    return tokens


def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.casefold() if NORMALIZE_CASE else text


def clean_text(text: str) -> str:
    normalized = normalize_text(text)
    markers = {normalize_text(marker) for marker in GT_LAYOUT_MARKERS}
    return ' '.join(token for token in normalized.split() if token not in markers)


def bag_of_words_prf(reference: str, prediction: str):
    ref, hyp = Counter(reference.split()), Counter(prediction.split())
    matched = sum((ref & hyp).values())
    ref_n, hyp_n = sum(ref.values()), sum(hyp.values())
    precision = matched / hyp_n if hyp_n else (1.0 if ref_n == 0 else 0.0)
    recall = matched / ref_n if ref_n else (1.0 if hyp_n == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score_normalized_pair(ref: str, hyp: str):
    if not ref:
        raise ValueError('Empty GT transcript after normalization; cannot compute CER/WER.')
    precision, recall, f1 = bag_of_words_prf(ref, hyp)
    return {
        'cer': float(cer(ref, hyp)), 'wer': float(wer(ref, hyp)),
        'token_precision': precision, 'token_recall': recall, 'token_f1': f1,
        'gt_words': len(ref.split()), 'pred_words': len(hyp.split()),
    }


def compute_metrics(ground_truth: str, prediction: str):
    ref_raw, hyp_raw = normalize_text(ground_truth), normalize_text(prediction)
    ref, hyp = clean_text(ground_truth), clean_text(prediction)
    return {
        **score_normalized_pair(ref, hyp),
        **{f'raw_{key}': value for key, value in score_normalized_pair(ref_raw, hyp_raw).items()},
        'n_gt_markers_removed': len(ref_raw.split()) - len(ref.split()),
        'n_pred_markers_removed': len(hyp_raw.split()) - len(hyp.split()),
    }


assert compute_metrics('The cat sat', 'The cat')['wer'] == 1/3
assert compute_metrics('Hello ##LTLine## world', 'hello world')['wer'] == 0
assert compute_metrics('Hello ##LTLine## world', 'hello world')['raw_wer'] == 1/3
print('Toy check:', compute_metrics('Hello ##LTLine## world', 'hello world'), flush=True)
print('First GT tokens:', read_gt_tokens(pairs[0]['gt_path'])[:35], flush=True)



print('Load official pretrained detector and recognizer', flush=True)

PaddleOCR = load_paddleocr_class()

if paddle.__version__ != '3.2.0':
    raise RuntimeError('Cần PaddlePaddle 3.2.0 như Mục 1.')


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


MODEL_COMPONENTS = []
MODEL_PATHS = {}
for component_name, specification in MODEL_SPECS.items():
    local_dir = MODEL_DIR / specification['name']
    local_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for filename in MODEL_FILES:
        cached = Path(hf_hub_download(
            repo_id=specification['repo_id'], repo_type='model',
            revision=specification['revision'], filename=filename,
        ))
        target = local_dir / filename
        if not target.is_file() or sha256_file(target) != sha256_file(cached):
            shutil.copy2(cached, target)
        artifacts.append({'filename': filename, 'sha256': sha256_file(target), 'bytes': target.stat().st_size})
    inference_config = yaml.safe_load((local_dir / 'inference.yml').read_text(encoding='utf-8'))
    if inference_config.get('Global', {}).get('model_name') != specification['name']:
        raise ValueError(f'Model config không khớp: {local_dir}')
    MODEL_PATHS[component_name] = local_dir
    MODEL_COMPONENTS.append({
        'component': component_name,
        'architecture': specification['name'],
        'checkpoint_repo': specification['repo_id'],
        'checkpoint_revision': specification['revision'],
        'artifacts': artifacts,
        'weight_file_bytes': (local_dir / 'inference.pdiparams').stat().st_size,
        'parameter_count': None,
        'parameter_count_note': 'Not inferred from serialized inference-file size; report weight MiB.',
        'inference_config': inference_config,
    })

ocr = PaddleOCR(
    text_detection_model_name=DETECT_NETWORK,
    text_detection_model_dir=str(MODEL_PATHS['detector']),
    text_recognition_model_name=RECOG_NETWORK,
    text_recognition_model_dir=str(MODEL_PATHS['recognizer']),
    **OCR_KWARGS,
    **RUNTIME_KWARGS,
)
pipeline_config_path = OUTPUT_DIR / 'paddlex_pipeline.yaml'
ocr.export_paddlex_config_to_yaml(str(pipeline_config_path))
PIPELINE_CONFIG = yaml.safe_load(pipeline_config_path.read_text(encoding='utf-8'))
if PIPELINE_CONFIG.get('use_doc_preprocessor') or PIPELINE_CONFIG.get('use_textline_orientation'):
    raise RuntimeError('Không được bật mạng phụ trong thí nghiệm này.')
EFFECTIVE_OCR_SETTINGS = json.loads(json.dumps({
    'pipeline': PIPELINE_CONFIG,
    'explicit_ocr_options': OCR_KWARGS,
    'runtime': RUNTIME_KWARGS,
}))
print('Đã tải:', [(c['component'], c['architecture']) for c in MODEL_COMPONENTS], flush=True)
print('Tổng trọng số (MiB):', sum(c['weight_file_bytes'] for c in MODEL_COMPONENTS) / 1024**2, flush=True)
print('Thiết bị:', DEVICE_NAME, '; batch crop:', RECOGNITION_BATCH_SIZE, flush=True)
print('Config:', pipeline_config_path, flush=True)



print('Single-image warm-up', flush=True)

def transcribe_page(image_path: str):
    """Chỉ nhận đường dẫn ảnh che khuất, không nhận GT."""
    if USE_CUDA:
        paddle.device.cuda.synchronize(0)
    started = time.perf_counter()
    pages = list(ocr.predict(input=str(image_path)))
    if len(pages) != 1:
        raise RuntimeError(f'Expected one page; got {len(pages)}')
    page = pages[0]
    if 'error' in page:
        raise RuntimeError(f'PaddleOCR error: {page["error"]}')
    texts = page['rec_texts']
    if not isinstance(texts, (list, tuple)) or not all(isinstance(text, str) for text in texts):
        raise TypeError('Unexpected PaddleOCR rec_texts schema; expected a list of strings.')
    prediction = ' '.join(text.strip() for text in texts if text.strip())
    if USE_CUDA:
        paddle.device.cuda.synchronize(0)
    elapsed_sec = time.perf_counter() - started
    return prediction, elapsed_sec, int(len(texts))


sample = pairs[0]
example_gt = ' '.join(read_gt_tokens(sample['gt_path']))
example_prediction, example_time, example_regions = transcribe_page(sample['image_path'])
print('CATEGORY:', sample['category'], flush=True)
print('IMAGE:', Path(sample['image_path']).name, flush=True)
print('LATENCY (s):', round(example_time, 2), flush=True)
print('OCR REGIONS:', example_regions, flush=True)
print('GT RAW (first 700 chars):', example_gt[:700], flush=True)
print('PREDICTION (first 700 chars):', example_prediction[:700], flush=True)
print('METRICS:', compute_metrics(example_gt, example_prediction), flush=True)



print('Evaluation and checkpoint', flush=True)

RESULTS_CSV = OUTPUT_DIR / 'per_image_results.csv'
CHECKPOINT_CONFIG = OUTPUT_DIR / 'checkpoint_config.json'
checkpoint_settings = {
    'dataset': REPO_ID,
    'split': 'test',
    'runtime_isolation': 'venv_without_system_site_packages',
    'python_version': platform.python_version(),
    'model': MODEL_ID,
    'detector': DETECT_NETWORK,
    'recognizer': RECOG_NETWORK,
    'components': MODEL_COMPONENTS,
    'quantization': None,
    'precision_option': 'fp32',
    'metric_policy_id': METRIC_POLICY_ID,
    'excluded_layout_tokens': GT_LAYOUT_MARKERS,
    'ocr_settings': EFFECTIVE_OCR_SETTINGS,
    'reading_order': 'PaddleOCR rec_texts order; join nonempty region texts with spaces',
    'normalize_case': NORMALIZE_CASE,
    'metric_definition': 'Macro per-image metrics after layout-token removal; raw_* retains legacy unfiltered scores',
    'device': DEVICE,
    'device_name': DEVICE_NAME,
    'cpu_threads': CPU_THREADS,
    'paddleocr_version': version('paddleocr'),
    'paddlex_version': version('paddlex'),
    'paddle_version': str(paddle.__version__),
    'numpy_version': np.__version__,
    'opencv_version': cv2.__version__,
    'jiwer_version': version('jiwer'),
}

if RESUME_FROM is not None and not RESULTS_CSV.exists():
    resume_root = Path(RESUME_FROM)
    for filename in ('checkpoint_config.json', 'per_image_results.csv'):
        if not (resume_root / filename).is_file():
            raise FileNotFoundError(f'Missing resume file: {resume_root / filename}')
    for filename in ('checkpoint_config.json', 'per_image_results.csv'):
        source = resume_root / filename
        target = OUTPUT_DIR / filename
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise FileExistsError(f'Không ghi đè checkpoint khác: {target}')
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    print('Restored checkpoint from:', resume_root, flush=True)

if CHECKPOINT_CONFIG.exists():
    saved_settings = json.loads(CHECKPOINT_CONFIG.read_text(encoding='utf-8'))
    if saved_settings != checkpoint_settings:
        raise ValueError('Checkpoint khác mô hình/cấu hình/thiết bị. Chọn OUTPUT_DIR mới cho thí nghiệm khác.')
elif RESULTS_CSV.exists():
    raise ValueError('CSV thiếu checkpoint_config.json. Phục hồi cả hai tệp hoặc chọn OUTPUT_DIR mới.')
else:
    CHECKPOINT_CONFIG.write_text(json.dumps(checkpoint_settings, indent=2), encoding='utf-8')


def sample_key(record):
    return record['category'], record['sample_id']


selected_keys = {sample_key(sample) for sample in pairs}
if RESULTS_CSV.exists():
    previous = pd.read_csv(RESULTS_CSV, keep_default_na=False).to_dict(orient='records')
    if any(record.get('model_id') != MODEL_ID for record in previous):
        raise ValueError('CSV chứa mô hình khác. Chọn OUTPUT_DIR riêng.')
    saved_rows = {sample_key(record): record for record in previous}
else:
    saved_rows = {}
finished = {key for key, record in saved_rows.items() if record.get('status') == 'ok'}
pending = [sample for sample in pairs if sample_key(sample) not in finished]
print(f'Reusing {len(finished & selected_keys):,} successful selected pages; {len(pending):,} remaining.', flush=True)
evaluation_started = time.perf_counter()
last_progress_log = evaluation_started
successful_count = len(finished & selected_keys)
errors_this_run = 0


def format_duration(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


for attempted_this_run, sample in enumerate(tqdm(pending, desc='PaddleOCR v5 Mobile evaluation', unit='page', mininterval=60), start=1):
    if MAX_EVAL_HOURS is not None and time.perf_counter() - evaluation_started >= MAX_EVAL_HOURS * 3600:
        print('Time budget reached. Export this checkpoint, then resume later.', flush=True)
        break
    result = dict(sample)
    result['model_id'] = MODEL_ID
    try:
        ground_truth = ' '.join(read_gt_tokens(sample['gt_path']))
        if not clean_text(ground_truth):
            raise ValueError('GT chỉ chứa nhãn cấu trúc; không có văn bản để chấm điểm.')
        prediction, elapsed, n_regions = transcribe_page(sample['image_path'])
        result.update({
            'status': 'ok',
            'prediction': prediction,
            'ground_truth': ground_truth,
            'latency_sec': elapsed,
            'n_text_regions': n_regions,
            **compute_metrics(ground_truth, prediction),
        })
    except Exception as exc:
        result.update({'status': 'error', 'error': repr(exc)})
        print('Failure:', sample['sample_id'], repr(exc), flush=True)
        gc.collect()
        if USE_CUDA:
            paddle.device.cuda.empty_cache()
    saved_rows[sample_key(sample)] = result
    temporary_csv = RESULTS_CSV.with_suffix('.csv.tmp')
    pd.DataFrame(saved_rows.values()).to_csv(temporary_csv, index=False)
    temporary_csv.replace(RESULTS_CSV)
    successful_count += result['status'] == 'ok'
    errors_this_run += result['status'] == 'error'
    now = time.perf_counter()
    if attempted_this_run == 1 or now - last_progress_log >= 60 or attempted_this_run == len(pending):
        elapsed = now - evaluation_started
        average_sec = elapsed / attempted_this_run
        unattempted = len(pending) - attempted_this_run
        print(
            f'PROGRESS: {successful_count:,}/{len(pairs):,} successful ({100 * successful_count / len(pairs):.1f}%) | '
            f'attempted this run={attempted_this_run:,}/{len(pending):,} | errors this run={errors_this_run:,} | '
            f'elapsed={format_duration(elapsed)} | avg={average_sec:.2f}s/page | '
            f'ETA for unattempted pages={format_duration(unattempted * average_sec)}',
            flush=True,
        )
        last_progress_log = now

rows = [record for key, record in saved_rows.items() if key in selected_keys]
results = pd.DataFrame(rows)
if results.empty:
    raise RuntimeError('Chưa có ảnh được thử. Kiểm tra giới hạn thời gian và checkpoint.')
print('Saved:', RESULTS_CSV, flush=True)
print('Successful:', (results.status == 'ok').sum(), '/', len(pairs), 'selected pages', flush=True)
display(results.reindex(columns=[
    'category', 'occlusion_type', 'cer', 'wer', 'token_f1',
    'latency_sec', 'n_text_regions', 'status', 'error'
]).head(10))



print('Summaries', flush=True)

ok = results.loc[results['status'] == 'ok'].copy()
assert len(ok), 'Không có trang chấm điểm được. Kiểm tra cột error.'
metric_cols = ['cer', 'wer', 'token_precision', 'token_recall', 'token_f1', 'latency_sec']
# CSV có hàng lỗi có thể khiến pandas đọc các số thành chuỗi khi resume.
for column in metric_cols + ['n_text_regions']:
    ok[column] = pd.to_numeric(ok[column], errors='raise')


def aggregate_metrics(df, group_by):
    return df.groupby(group_by, dropna=False).agg(
        n=('sample_id', 'count'),
        **{f'mean_{name}': (name, 'mean') for name in metric_cols},
        mean_text_regions=('n_text_regions', 'mean'),
    ).reset_index()


overall = {f'mean_{name}': float(ok[name].mean()) for name in metric_cols}
overall.update({
    'n_selected': int(len(pairs)),
    'n_success': int(len(ok)),
    'n_errors': int(len(results) - len(ok)),
    'n_remaining': int(len(pairs) - len(ok)),
    'evaluation_complete': bool(len(ok) == len(pairs)),
    'n_empty_predictions': int(ok['prediction'].fillna('').astype(str).str.strip().eq('').sum()),
    'mean_text_regions': float(ok['n_text_regions'].mean()),
})
by_category = aggregate_metrics(ok, ['category'])
by_type = aggregate_metrics(ok, ['category', 'occlusion_type'])
by_density = aggregate_metrics(ok, ['category', 'occlusion_density'])
print('OVERALL macro-averaged metrics:', flush=True)
print(json.dumps(overall, indent=2), flush=True)
print('BY OCCLUSION REGION:', flush=True)
display(by_category)
print('BY OCCLUSION TYPE:', flush=True)
display(by_type)
print('BY OCCLUSION DENSITY:', flush=True)
display(by_density)
by_category.to_csv(OUTPUT_DIR / 'summary_by_category.csv', index=False)
by_type.to_csv(OUTPUT_DIR / 'summary_by_occlusion_type.csv', index=False)
by_density.to_csv(OUTPUT_DIR / 'summary_by_density.csv', index=False)

region_times = by_category.set_index('category')['mean_latency_sec']
if {'text', 'nontext'}.issubset(region_times.index):
    estimated_full_hours = (6000 * region_times['text'] + 3000 * region_times['nontext']) / 3600
    print(f'Ước lượng suy luận 9.000 trang trên {DEVICE_NAME}: {estimated_full_hours:.1f} giờ.', flush=True)
print('Số ảnh còn cần chấm thành công:', overall['n_remaining'], flush=True)


# Điểm theo chính sách cũ để đối chiếu các CSV trước khi lọc marker.
raw_metric_cols = ['cer', 'wer', 'token_precision', 'token_recall', 'token_f1']
for name in raw_metric_cols + ['gt_words', 'pred_words']:
    ok[f'raw_{name}'] = pd.to_numeric(ok[f'raw_{name}'], errors='raise')
for column in ['n_gt_markers_removed', 'n_pred_markers_removed']:
    ok[column] = pd.to_numeric(ok[column], errors='raise')
raw_overall = {f'mean_{name}': float(ok[f'raw_{name}'].mean()) for name in raw_metric_cols}
raw_overall['mean_latency_sec'] = overall['mean_latency_sec']
overall['n_gt_markers_removed'] = int(ok['n_gt_markers_removed'].sum())
overall['n_pred_markers_removed'] = int(ok['n_pred_markers_removed'].sum())
raw_ok = ok.copy()
for name in raw_metric_cols:
    raw_ok[name] = raw_ok[f'raw_{name}']
for filename, groups in [
    ('raw_summary_by_category.csv', ['category']),
    ('raw_summary_by_occlusion_type.csv', ['category', 'occlusion_type']),
    ('raw_summary_by_density.csv', ['category', 'occlusion_density']),
]:
    aggregate_metrics(raw_ok, groups).to_csv(OUTPUT_DIR / filename, index=False)
print('RAW/LEGACY macro metrics (còn nhãn, để đối chiếu CSV cũ):', flush=True)
print(json.dumps(raw_overall, indent=2), flush=True)
print('GT marker tokens removed:', overall['n_gt_markers_removed'], flush=True)
print('Không so trực tiếp điểm chính đã lọc với điểm gốc chưa lọc của model khác.', flush=True)



print('Save plots', flush=True)

ax = by_category.set_index('category')[['mean_cer', 'mean_wer']].plot(
    kind='bar', figsize=(7, 4), rot=0, title='OPRB test: mean CER and WER by region'
)
ax.set_xlabel('Occlusion region')
ax.set_ylabel('Error rate (lower is better)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cer_wer_by_category.png', dpi=150)
plt.close()

ax = by_type.set_index('occlusion_type').groupby(level=0)['mean_wer'].mean().sort_values().plot(
    kind='barh', figsize=(9, 5), title='Mean WER by occlusion type (unweighted category average)'
)
ax.set_xlabel('Mean word error rate (lower is better)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'wer_by_occlusion_type.png', dpi=150)
plt.close()



print('Export results', flush=True)

run_config = {
    **checkpoint_settings,
    'categories': CATEGORIES,
    'no_finetuning': True,
    'max_samples_per_category': MAX_SAMPLES_PER_CATEGORY,
    'random_seed': RANDOM_SEED,
    'max_eval_hours_per_run': MAX_EVAL_HOURS,
    'metrics': overall,
    'raw_metrics': raw_overall,
    'ground_truth_storage': 'Original GT tokens retained in ground_truth; layout tokens removed only while scoring',
    'gpu_name': DEVICE_NAME if USE_CUDA else None,
    'cuda_version': str(paddle.version.cuda()) if USE_CUDA else None,
    'python_version': platform.python_version(),
    'platform': platform.platform(),
    'runtime_python': sys.executable,
    'torch_available_in_runtime': False,
    'runtime_packages_file': 'runtime_packages.json',
    'jiwer_version': version('jiwer'),
    'pandas_version': version('pandas'),
    'numpy_version': np.__version__,
    'output_token_limit': None,
    'output_token_limit_note': 'No page-wide VLM token generation limit; crop resolution and CTC decoding remain limitations.',
}
(OUTPUT_DIR / 'run_config.json').write_text(json.dumps(run_config, indent=2), encoding='utf-8')
zip_name = shutil.make_archive(str(OUTPUT_DIR), 'zip', root_dir=OUTPUT_DIR)
print('Results ZIP:', zip_name, flush=True)
print('Tải ZIP trong Output của Kaggle.', flush=True)
print('Lần sau: gắn output phiên này làm Input rồi đặt RESUME_FROM đến thư mục CSV/JSON.', flush=True)
if not overall['evaluation_complete']:
    print('Đây là kết quả một phần. Tiếp tục checkpoint để hoàn tất tập đã chọn.', flush=True)

