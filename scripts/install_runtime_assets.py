"""Verify the versioned runtime archive, then restore only declared files."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
def digest(data): return hashlib.sha256(data).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    catalog=json.loads((ROOT/'release/assets.json').read_text(encoding='utf-8'))
    expected=catalog['archives']['go2-v0.1.0-runtime.zip']
    assert args.archive.stat().st_size==expected['bytes']
    assert digest(args.archive.read_bytes())==expected['sha256'], 'Archive checksum mismatch'
    entries=[a for a in catalog['files'] if a['role']=='required_runtime']
    with zipfile.ZipFile(args.archive) as z:
        assert len(z.namelist())==len(entries) and set(z.namelist())=={a['path'] for a in entries}
        for a in entries:
            dest=(ROOT/a['path']).resolve()
            assert dest.is_relative_to(ROOT) and not dest.is_symlink(), a['path']
            data=z.read(a['path'])
            assert len(data)==a['bytes'] and digest(data)==a['sha256'], a['path']
            if dest.exists(): assert digest(dest.read_bytes())==a['sha256'], 'Existing different file retained: '+a['path']
        if not args.verify_only:
            for a in entries:
                dest=ROOT/a['path']
                if not dest.exists():
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with dest.open('xb') as f: f.write(z.read(a['path']))
    print(str(len(entries))+' runtime files verified; copied='+str(not args.verify_only))
if __name__=='__main__': main()
