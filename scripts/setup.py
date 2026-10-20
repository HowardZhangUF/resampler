"""Recreate pinned repositories and CPU environments on Linux x86_64."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
uv=shutil.which('uv')
if not uv:
    raise SystemExit('Install uv first: python -m pip install uv. Then run this script again.')


def run(*args):subprocess.run(list(map(str,args)),cwd=ROOT,check=True)


for source in json.loads((ROOT/'sources.lock.json').read_text()):
    path=ROOT/source['path']
    if not path.exists():
        run('git','clone','--no-checkout','https://github.com/'+source['repository']+'.git',path)
        run('git','-C',path,'checkout',source['commit'])
    actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    if actual!=source['commit']:raise SystemExit(f'{path} has revision {actual}; expected {source["commit"]}. Existing checkout left unchanged.')
for folder,version in [('.venv','3.12'),('.venv-tf','3.11')]:
    if not (ROOT/folder/'bin/python').exists():run(uv,'venv','--python',version,'--managed-python',folder)
tool=json.loads((ROOT/'tools.lock.json').read_text())
run(uv,'pip','install','--python','.venv/bin/python',tool['torch_cpu'],'--index-url','https://download.pytorch.org/whl/cpu')
run(uv,'pip','install','--python','.venv/bin/python','-r','requirements-main.lock')
run(uv,'pip','install','--python','.venv-tf/bin/python','-r','requirements-tf.lock')
run(uv,'pip','install','--python','.venv/bin/python','--no-deps','-e','vendor/cuthbert/pkg/cuthbertlib','-e','vendor/cuthbert/pkg/cuthbert','-e','vendor/pfjax')
tools=ROOT/'.tools';tools.mkdir(exist_ok=True)
if not (tools/('julia-'+tool['julia_version'])/'bin/julia').exists():
    archive=tools/'julia.tar.gz'
    if not archive.exists():urllib.request.urlretrieve(tool['julia_url'],archive)
    with archive.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual!=tool['julia_sha256']:raise SystemExit('Julia archive checksum does not match tools.lock.json')
    with tarfile.open(archive) as f:f.extractall(tools,filter='data')
print('Setup complete. See README.md for run commands.')
