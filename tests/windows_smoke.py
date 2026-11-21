"""Exercise the installed, relocated runtime and the actual stdio transport on Windows."""
import asyncio
from datetime import date
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from qmt import desktop_profile
from qmt.data.tushare_client import TushareClient


async def check_mcp():
    config = desktop_profile.mcp_config()["mcpServers"]["qmt"]
    assert Path(config["command"]).resolve() == Path(sys.executable).resolve()
    params = StdioServerParameters(**config)
    async with stdio_client(params) as streams, ClientSession(*streams) as client:
        await client.initialize()
        names = {tool.name for tool in (await client.list_tools()).tools}
        assert names == {"qmt_status", "fund_nav", "analyze_cached_fund"}
        status = await client.call_tool("qmt_status", {})
        assert not status.isError and json.loads(status.content[0].text)["configured"]
        desktop_profile.remove_key()
        status = await client.call_tool("qmt_status", {})
        assert not json.loads(status.content[0].text)["configured"]
        denied = await client.call_tool("fund_nav", {"fund_code": "000001", "start": "2024-01-01", "end": "2024-01-31"})
        assert denied.isError


def check_dashboard():
    launcher = Path(sys.executable).parent.parent / "QMT Desktop.exe"
    subprocess.run([str(launcher)], check=True, timeout=15)
    url = "http://127.0.0.1:8770"
    deadline = time.monotonic() + 90
    while True:
        try:
            with urlopen(url + "/api/health", timeout=2) as r:
                assert json.load(r)["service"] == "qmt-controlplane"
            break
        except (URLError, TimeoutError):
            if time.monotonic() > deadline:
                raise
            time.sleep(.5)
    assert "QMT" in urlopen(url).read().decode()
    assert urlopen(url + "/api/decisions").status == 200
    assert json.load(urlopen(url + "/api/funds"))["count"] == 0
    try:
        urlopen(Request(url, headers={"Origin": "https://example.org"}))
    except HTTPError as error:
        assert error.code == 403
    else:
        raise AssertionError("cross-origin request accepted")


if __name__ == "__main__":
    cache = desktop_profile.configure()
    assert not list(cache.iterdir())
    assert not desktop_profile.read_key()
    from tkinter import Tk
    from qmt.desktop import Desktop
    # Exercise the real Settings callbacks using a disposable CI profile.
    root = Tk()
    root.withdraw()
    from unittest.mock import patch
    with patch.object(Desktop, "serve"):
        window = Desktop(root)
        window.key_entry.insert(0, "ci-synthetic-key")
        window.store_key()
        assert window.key_entry.get() == "" and desktop_profile.read_key() == "ci-synthetic-key"
        assert TushareClient().has_token
        window.copy_config()
        assert json.loads(root.clipboard_get()) == desktop_profile.mcp_config()
        window.clear_key()
        assert not desktop_profile.read_key()
        window.close()
    try:
        desktop_profile.save_key("ci-synthetic-key")
        asyncio.run(check_mcp())
    finally:
        desktop_profile.remove_key()
    check_dashboard()
    print("Installed Windows desktop, credential storage, local HTTP and MCP stdio verified.")
