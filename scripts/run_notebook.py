"""Execute the existing notebook in a clean kernel, persisting outputs after each cell."""
from pathlib import Path
from datetime import datetime
import os
import sys
import nbformat
from nbclient import NotebookClient
ROOT = Path(__file__).resolve().parents[1]
path = ROOT / '01_freshretail_project.ipynb'
nb = nbformat.read(path, as_version=4)
os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH','')

def start(cell, cell_index, **kwargs):
    print(f'{datetime.now().isoformat(timespec="seconds")} START cell {cell_index}', flush=True)

def done(cell, cell_index, **kwargs):
    # Persist executed state to the same user-visible notebook.
    nbformat.write(nb, path)
    for output in cell.get('outputs',[]):
        if output.output_type == 'stream':
            print(output.text, end='', flush=True)
    print(f'{datetime.now().isoformat(timespec="seconds")} DONE cell {cell_index}', flush=True)

client = NotebookClient(nb, timeout=1800, kernel_name='python3',
    resources={'metadata':{'path':str(ROOT)}},
    on_cell_start=start, on_cell_executed=done, allow_errors=False)
try:
    client.execute()
finally:
    nbformat.write(nb,path)
nbformat.validate(nb)
assert all(c.get('execution_count') is not None for c in nb.cells if c.cell_type == 'code')
assert not any(o.output_type == 'error' for c in nb.cells for o in c.get('outputs',[]))
print('SUCCESS: all code cells executed and saved.',flush=True)
