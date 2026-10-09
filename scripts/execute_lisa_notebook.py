"""Execute Lisa's notebook from a fresh kernel and export the complete HTML report."""
from pathlib import Path
import json
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "02_lisa_pipeline.ipynb"
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
client.execute()
nbformat.write(nb, path)
html, _ = HTMLExporter(template_name="lab").from_notebook_node(nb)
out = ROOT / "results" / "v2"
(out / "02_lisa_pipeline.html").write_text(html)
errors = [o for c in nb.cells for o in c.get("outputs", []) if o.output_type == "error"]
receipt = {"code_cells": sum(c.cell_type == "code" for c in nb.cells),
    "executed_code_cells": sum(c.cell_type == "code" and c.execution_count is not None for c in nb.cells),
    "errors": len(errors), "inline_images": sum("image/png" in o.get("data", {}) for c in nb.cells for o in c.get("outputs", []))}
(out / "notebook_execution.json").write_text(json.dumps(receipt, indent=2))
print(receipt)
if errors:
    raise RuntimeError("Notebook contains execution errors")
