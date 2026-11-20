"""Losslessly package or restore original per-image benchmark CSVs.

    python scripts/results_archive.py pack
    python scripts/results_archive.py restore

Only the standard library is required. Existing different CSVs are never
overwritten by restore. Archives omit timestamps for reproducible Git blobs.
"""
import argparse
import gzip
import hashlib
from pathlib import Path
import shutil

RESULTS = Path(__file__).resolve().parents[1] / 'result'


def digest(path, compressed=False):
    opener = gzip.open if compressed else open
    result = hashlib.sha256()
    with opener(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def pack():
    sources = sorted(RESULTS.glob('*/per_image_results.csv'))
    if not sources:
        raise FileNotFoundError('No original CSVs found in result/.')
    for source in sources:
        archive = source.with_suffix('.csv.gz')
        temporary = archive.with_suffix('.gz.tmp')
        try:
            with source.open('rb') as src, temporary.open('wb') as raw:
                with gzip.GzipFile(filename='', mode='wb', fileobj=raw,
                                   compresslevel=6, mtime=0) as dst:
                    shutil.copyfileobj(src, dst)
            if digest(temporary, compressed=True) != digest(source):
                raise ValueError(f'Archive verification failed: {source}')
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'Packed {source.parent.name}: {source.stat().st_size:,} -> '
              f'{archive.stat().st_size:,} bytes; SHA-256 verified.')


def restore(destination):
    archives = sorted(RESULTS.glob('*/per_image_results.csv.gz'))
    if not archives:
        raise FileNotFoundError('No result archives found in result/.')
    for archive in archives:
        target = destination / archive.parent.name / 'per_image_results.csv'
        expected = digest(archive, compressed=True)
        if target.exists():
            if digest(target) != expected:
                raise FileExistsError(f'Existing CSV differs; refusing to overwrite: {target}')
            print(f'Already restored: {target.parent.name}')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.csv.tmp')
        try:
            with gzip.open(archive, 'rb') as src, temporary.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            if digest(temporary) != expected:
                raise ValueError(f'Restored CSV verification failed: {target}')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'Restored {target.parent.name}; SHA-256 verified.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['pack', 'restore'])
    parser.add_argument('--destination', type=Path, default=RESULTS,
                        help='Destination result/ directory for restore.')
    args = parser.parse_args()
    if args.mode == 'pack':
        pack()
    else:
        restore(args.destination.resolve())


if __name__ == '__main__':
    main()
