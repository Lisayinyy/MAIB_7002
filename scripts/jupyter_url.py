#!/usr/bin/env python3
"""Print this local Compose Jupyter login URL, with the correct host port."""
import os
from jupyter_server.serverapp import list_running_servers

servers = list(list_running_servers())
if not servers:
    raise SystemExit('Jupyter is not running. Start it with: docker compose up -d lab')
port = os.environ.get('HOST_JUPYTER_PORT', '18888')
for server in servers:
    print(f"http://localhost:{port}/lab/tree/01_freshretail_project.ipynb?token={server['token']}")
