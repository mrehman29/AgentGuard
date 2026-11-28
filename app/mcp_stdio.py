"""AgentGuard MCP server entry point for stdio transport (PLAN.md section 9, C.1).

Logs to stderr only so that stdout remains clean for MCP JSON-RPC protocol messages.
"""

import logging
from pathlib import Path
import sys

# Ensure project root is on sys.path so app can be imported when run from any directory (C-15)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.mcp_server import mcp
from core import budget_store
from core.config import get_settings

if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    get_settings()  # a bad setting stops the server here, with its name
    budget_store.get_tracker()
    mcp.run(transport="stdio")
