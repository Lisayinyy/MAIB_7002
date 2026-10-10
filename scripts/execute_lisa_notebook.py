"""Execute Lisa's notebook from a fresh kernel and export the complete HTML report."""
from pathlib import Path
import json
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "02_lisa_pipeline.ipynb"
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
client.execute()
nbformat.write(nb, path)
html, _ = HTMLExporter(template_name="lab").from_notebook_node(nb)
# Keep runnable source available without putting implementation before the story.
# Native HTML details controls work offline and need no JavaScript dependency.
soup = BeautifulSoup(html, "html.parser")
code_toggles = 0
for cell in soup.select(".jp-CodeCell"):
    source = cell.select_one(".jp-Cell-inputWrapper")
    if source is not None:
        details = soup.new_tag("details", attrs={"class": "story-source"})
        summary = soup.new_tag("summary")
        summary.string = "Show Python for this step"
        source.wrap(details)
        details.insert(0, summary)
        code_toggles += 1
style = soup.new_tag("style")
style.string = """
body { background: #fbfcfa; }
main.jp-Notebook { max-width: 1100px; margin: 0 auto; padding-top: 36px; }
.jp-RenderedHTMLCommon { line-height: 1.7; }
.jp-RenderedHTMLCommon h1 { color: #145c50; }
.jp-RenderedHTMLCommon h2 { color: #145c50; margin-top: 32px; }
.story-source { border: 1px solid #dce8e2; border-radius: 6px; margin: 6px 20px 12px 64px; background: #f4f8f5; }
.story-source summary { cursor: pointer; color: #376457; padding: 9px 14px; font: 13px system-ui, sans-serif; }
.story-source[open] summary { border-bottom: 1px solid #dce8e2; }
.jp-OutputArea-output { overflow-x: auto; }
@media (max-width: 600px) { .story-source { margin-left: 12px; margin-right: 12px; } }
"""
soup.head.append(style)
if soup.title:
    soup.title.string = "Should we discount tomorrow? — FreshRetail project story"
html = str(soup)
out = ROOT / "results" / "v2"
(out / "02_lisa_pipeline.html").write_text(html)
errors = [o for c in nb.cells for o in c.get("outputs", []) if o.output_type == "error"]
receipt = {"code_cells": sum(c.cell_type == "code" for c in nb.cells),
    "executed_code_cells": sum(c.cell_type == "code" and c.execution_count is not None for c in nb.cells),
    "errors": len(errors), "inline_images": sum("image/png" in o.get("data", {}) for c in nb.cells for o in c.get("outputs", [])),
    "collapsible_code_blocks": code_toggles}
(out / "notebook_execution.json").write_text(json.dumps(receipt, indent=2))
print(receipt)
if errors:
    raise RuntimeError("Notebook contains execution errors")
