"""Vendor the upstream BSD-licensed Go2 model and lock every downloaded byte."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'third_party/unitree_go2'
BASE = 'https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/unitree_go2/'


def download(name):
    path = DEST / name
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(BASE + name, headers={'User-Agent': 'quadruped-simulation-project'})
    with urllib.request.urlopen(request, timeout=45) as response:
        content = response.read()
    path.write_bytes(content)
    print(f'DOWNLOADED {name}: {len(content)} bytes', flush=True)
    return name, dict(url=BASE+name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


def verify_assets():
    source = json.loads((DEST / 'SOURCE.json').read_text(encoding='utf-8'))
    for name, entry in source['files'].items():
        content = (DEST / name).read_bytes()
        if len(content) != entry['bytes'] or hashlib.sha256(content).hexdigest() != entry['sha256']:
            raise RuntimeError(f'Upstream asset differs from lock: {name}')
    return source


def main():
    if (DEST / 'SOURCE.json').exists():
        print(f'Using verified vendored Go2 files: {len(verify_assets()["files"])}')
        return
    entries = dict(download(name) for name in ('go2.xml', 'scene.xml', 'LICENSE', 'README.md'))
    xml = (DEST / 'go2.xml').read_text(encoding='utf-8')
    meshes = [f'assets/{element.attrib["file"]}' for element in ET.fromstring(xml).findall('./asset/mesh')]
    with ThreadPoolExecutor(max_workers=4) as pool:
        entries.update(pool.map(download, meshes))
    # Check that the XML source did not change while collecting its asset set.
    _, second = download('go2.xml')
    if second['sha256'] != entries['go2.xml']['sha256']:
        raise RuntimeError('Upstream XML changed during snapshot; retry before publishing')
    source = dict(upstream='https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2',
                  reference='main', commit=None,
                  version_lock='Vendored file SHA-256 snapshot; commit lookup unavailable',
                  fetched_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  license='BSD-3-Clause; see LICENSE', files=dict(sorted(entries.items())))
    (DEST / 'SOURCE.json').write_text(json.dumps(source, indent=2), encoding='utf-8', newline='\n')
    verify_assets()
    print(f'VERIFIED {len(entries)} files, {sum(e["bytes"] for e in entries.values())} bytes', flush=True)


if __name__ == '__main__':
    main()
