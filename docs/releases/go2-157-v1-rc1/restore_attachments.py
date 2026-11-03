"""Restore declared runtime attachments from an existing exported157 directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[3]
CATALOG=Path(__file__).with_name('attachments.json')


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from-directory',required=True,help='Existing validated157 bundle or equivalent attachment root.')
    p.add_argument('--verify-only',action='store_true',help='Check the source bytes without copying.')
    args=p.parse_args();source=Path(args.from_directory).resolve();entries=json.loads(CATALOG.read_text(encoding='utf-8'))['attachments']
    required=[a for a in entries if a['role']=='required_runtime']
    for a in required:
        src=(source/a['path']).resolve();dest=(ROOT/a['path']).resolve()
        assert src.is_relative_to(source) and dest.is_relative_to(ROOT),a['path']
        assert src.is_file() and src.stat().st_size==a['bytes'] and digest(src)==a['sha256'],a['path']
        if dest.exists():assert digest(dest)==a['sha256'],f'Existing different file retained: {dest}'
    if not args.verify_only:
        for a in required:
            dest=ROOT/a['path']
            if not dest.exists():
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/a['path'],dest)
    print(f'{len(required)} required attachments verified; copied={not args.verify_only}; no physics or training.')


if __name__=='__main__':main()
