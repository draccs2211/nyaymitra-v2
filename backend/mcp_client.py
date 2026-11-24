"""
MCP client for NyayMitra's MCP servers — timeout, retry and fallback.

Servers run in-process by default (FastMCP in-memory transport: no subprocess,
same protocol). Set MCP_<NAME>_URL (e.g. MCP_INDIANKANOON_URL=http://host/mcp)
to call a remote server instead.

Callers get either the tool's dict result or an MCPCallError, and decide their
own fallback; nothing here raises into the pipeline otherwise.
"""
import asyncio
import importlib
import os
import threading
from typing import Any, Dict, Optional

from fastmcp import Client
from fastmcp.exceptions import ToolError

from backend.config import MCP_RETRIES, MCP_TIMEOUT_SECONDS

SERVERS = {
    "indiankanoon": "backend.mcp_servers.indiankanoon_mcp",
    "legal_reference": "backend.mcp_servers.legal_reference_mcp",
    "dlsa": "backend.mcp_servers.dlsa_mcp",
    "templates": "backend.mcp_servers.templates_mcp",
}


class MCPCallError(Exception):
    """Tool call failed after all retries (or failed non-retryably)."""


def _transport(server: str):
    if server not in SERVERS:
        raise MCPCallError(f"Unknown MCP server: {server}")
    url = os.getenv(f"MCP_{server.upper()}_URL")
    if url:
        return url
    return importlib.import_module(SERVERS[server]).mcp


async def _call_once(server: str, tool: str, args: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    async with Client(_transport(server), timeout=timeout) as client:
        result = await client.call_tool(tool, args)
    data = result.data if isinstance(result.data, dict) else result.structured_content
    if not isinstance(data, dict):
        raise MCPCallError(f"{server}.{tool} returned non-dict result")
    return data


async def call_tool(
    server: str,
    tool: str,
    args: Optional[Dict[str, Any]] = None,
    *,
    timeout: float = MCP_TIMEOUT_SECONDS,
    retries: int = MCP_RETRIES,
    backoff: float = 0.5,
) -> Dict[str, Any]:
    """
    Call `tool` on `server`. Each attempt is bounded by `timeout` seconds.
    Retries transient failures (timeouts, transport errors) with exponential
    backoff. A ToolError (the tool itself raised) is not retried.
    """
    args = args or {}
    last_error: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            return await asyncio.wait_for(_call_once(server, tool, args, timeout), timeout)
        except (ToolError, MCPCallError) as e:
            raise MCPCallError(f"{server}.{tool} failed: {e}") from e
        except Exception as e:      # timeout / transport / connection
            last_error = e
            if attempt < retries:
                await asyncio.sleep(backoff * (2 ** attempt))
    raise MCPCallError(
        f"{server}.{tool} failed after {retries + 1} attempt(s): {type(last_error).__name__}: {last_error}"
    ) from last_error


def run_sync(coro):
    """
    Run a coroutine from sync code (agents are sync). Uses asyncio.run when no
    loop is running in this thread; otherwise runs it on a helper thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box: Dict[str, Any] = {}

    def runner():
        try:
            box["result"] = asyncio.run(coro)
        except BaseException as e:   # re-raised in caller thread
            box["error"] = e

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["result"]
