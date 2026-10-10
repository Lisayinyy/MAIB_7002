"""Execute Lisa's notebook from a fresh kernel and export the complete HTML report."""
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone
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
# A short navigation panel makes a detailed notebook readable without removing
# its evidence. Native links and details controls also work without JavaScript.
headings = soup.select(".jp-MarkdownCell h2")
navigation = soup.new_tag("nav", attrs={"class": "story-navigation", "aria-label": "Report chapters"})
nav_details = soup.new_tag("details", attrs={"open": ""})
nav_summary = soup.new_tag("summary")
nav_summary.string = "Follow the investigation · 15 chapters and a technical appendix"
nav_details.append(nav_summary)
nav_list = soup.new_tag("ol")
for i, heading in enumerate(headings, 1):
    anchor = heading.get("id") or f"story-chapter-{i}"
    heading["id"] = anchor
    item = soup.new_tag("li")
    link = soup.new_tag("a", href=f"#{anchor}")
    link.string = heading.get_text(" ", strip=True).replace("¶", "").strip()
    item.append(link); nav_list.append(item)
nav_details.append(nav_list); navigation.append(nav_details)
first = soup.select_one(".jp-MarkdownCell")
if first is not None:
    first.insert_after(navigation)
style = soup.new_tag("style")
style.string = """
body { background: #fbfcfa; }
main.jp-Notebook { max-width: 1100px; margin: 0 auto; padding-top: 36px; }
.jp-RenderedHTMLCommon { line-height: 1.7; }
.jp-RenderedHTMLCommon h1 { color: #145c50; }
.jp-RenderedHTMLCommon h2 { color: #145c50; margin-top: 32px; }
.jp-RenderedHTMLCommon h3 { color: #234e42; margin-top: 24px; }
.jp-RenderedHTMLCommon p, .jp-RenderedHTMLCommon li { font-size: 15px; }
.jp-RenderedHTMLCommon table { font-size: 13px; }
.jp-RenderedHTMLCommon th { background: #e9f2ed; color: #234e42; }
.jp-RenderedHTMLCommon blockquote { border-left-color: #cf8538; background: #fcf8ef; padding: 8px 18px; }
.story-navigation { margin: 12px 24px 32px 64px; padding: 18px 22px; border: 1px solid #cfdfd5; border-radius: 10px; background: #edf5f0; color: #234e42; font: 14px/1.65 system-ui, sans-serif; }
.story-navigation summary { cursor: pointer; font-weight: 650; }
.story-navigation ol { columns: 2; padding-left: 0; list-style: none; margin-bottom: 0; }
.story-navigation li { break-inside: avoid; padding-bottom: 6px; }
.story-navigation a { color: #235e4e; text-decoration: none; }
.story-navigation a:hover { text-decoration: underline; }
html { scroll-behavior: smooth; }
.story-source { border: 1px solid #dce8e2; border-radius: 6px; margin: 6px 20px 12px 64px; background: #f4f8f5; }
.story-source summary { cursor: pointer; color: #376457; padding: 9px 14px; font: 13px system-ui, sans-serif; }
.story-source[open] summary { border-bottom: 1px solid #dce8e2; }
.jp-OutputArea-output { overflow-x: auto; }
@media (max-width: 600px) { .story-source, .story-navigation { margin-left: 12px; margin-right: 12px; } .story-navigation ol { columns: 1; } }
@media print { .story-navigation { break-inside: avoid; } .story-source { display: none; } }
"""
soup.head.append(style)
if soup.title:
    soup.title.string = "Should we discount tomorrow? — FreshRetail project story"
html = str(soup)
out = ROOT / "results" / "v2"
(out / "02_lisa_pipeline.html").write_text(html)
errors = [o for c in nb.cells for o in c.get("outputs", []) if o.output_type == "error"]
receipt = {"completed_utc": datetime.now(timezone.utc).isoformat(),
    "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "builder_sha256": hashlib.sha256((ROOT / "scripts/build_lisa_notebook.py").read_bytes()).hexdigest(),
    "helper_sha256": hashlib.sha256((ROOT / "src/finalproject_pricingml/storytelling.py").read_bytes()).hexdigest(),
    "cells": len(nb.cells), "code_cells": sum(c.cell_type == "code" for c in nb.cells),
    "executed_code_cells": sum(c.cell_type == "code" and c.execution_count is not None for c in nb.cells),
    "errors": len(errors), "inline_images": sum("image/png" in o.get("data", {}) for c in nb.cells for o in c.get("outputs", [])),
    "collapsible_code_blocks": code_toggles, "navigation_chapters": len(headings),
    "live_demonstration_fits": 3, "full_grid_retrained": False}
(out / "notebook_execution.json").write_text(json.dumps(receipt, indent=2))
print(receipt)
if errors:
    raise RuntimeError("Notebook contains execution errors")
