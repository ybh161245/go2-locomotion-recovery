"""Archive public staged source separately from locally retained experiment data."""
from datetime import datetime, timezone, timedelta
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def write_archive(path, entries):
    hashes = {}
    with ZipFile(path, 'w', compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for file, name in entries:
            content = file.read_bytes()
            archive.writestr(name, content)
            hashes[name] = hashlib.sha256(content).hexdigest()
        archive.writestr('ARCHIVE-MANIFEST.json', json.dumps(hashes, indent=2))
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('ZIP integrity check failed')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix('.zip.sha256').write_text(f'{digest}  {path.name}\n', encoding='utf-8')
    return dict(name=path.name, files=len(hashes), bytes=path.stat().st_size, sha256=digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, default=2)
    args = parser.parse_args()
    if args.stage < 1:
        parser.error('--stage must be positive')
    prefix = f'stage-{args.stage:02d}'
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S')
    out = ROOT / 'artifacts/source-snapshots' / stamp
    out.mkdir(parents=True)
    # The Git index controls the public source allowlist; ignored/private files
    # and the virtual environment are excluded. Archive current saved contents.
    names = subprocess.check_output(['git', 'ls-files', '--cached', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    source = [(ROOT / name, name) for name in names if name and (ROOT / name).is_file()]
    reports = [write_archive(out / f'{prefix}-source.zip', source)]
    experiments = []
    for run in sorted((ROOT / 'artifacts/runs').iterdir()):
        if not ((run / 'manifest.json').exists() or (run / 'summary.json').exists() or (run / 'failure.txt').exists()):
            continue
        for file in sorted(run.rglob('*')):
            if file.is_file():
                experiments.append((file, file.relative_to(ROOT / 'artifacts/runs').as_posix()))
    reports.append(write_archive(out / f'{prefix}-experiments.zip', experiments))
    (out / 'archives.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps(dict(directory=str(out), archives=reports), indent=2))


if __name__ == '__main__':
    main()
