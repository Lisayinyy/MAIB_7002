"""Print the current local authenticated notebook URL, with the mapped host port."""
import os
from jupyter_server.serverapp import list_running_servers
from urllib.parse import urlencode

servers = list(list_running_servers())
if not servers:
    raise SystemExit("No running Jupyter server in this container. Start docker compose up -d first.")
server = servers[0]
port = os.environ.get("HOST_JUPYTER_PORT", "19888")
query = urlencode({"token": server["token"]}) if server.get("token") else ""
print(f"http://localhost:{port}/lab/tree/02_lisa_pipeline.ipynb" + (f"?{query}" if query else ""))
