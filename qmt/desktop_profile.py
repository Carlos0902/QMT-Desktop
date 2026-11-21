"""Per-user Windows runtime shared by the desktop app and the stdio server."""
from __future__ import annotations

import os
from pathlib import Path
import sys

CREDENTIAL = "QMT Desktop/Tushare Pro"


def configure():
    root = Path(os.environ["LOCALAPPDATA"]) / "QMT Desktop"
    os.environ["QMT_DATA_CACHE_DIR"] = str(root / "data")
    os.environ["QMT_DESKTOP"] = "1"
    from qmt.config import get_settings
    get_settings.cache_clear()
    return get_settings().ensure_data_cache()


def read_key():
    import pywintypes
    import win32cred
    try:
        record = win32cred.CredRead(CREDENTIAL, win32cred.CRED_TYPE_GENERIC)
    except pywintypes.error as exc:
        if exc.winerror == 1168:
            return ""
        raise
    return record["CredentialBlob"].decode("utf-16-le")


def save_key(value):
    import win32cred
    value = value.strip()
    if not value or len(value) > 512 or any(c.isspace() for c in value):
        raise ValueError("请输入有效的 Tushare Pro Key。")
    win32cred.CredWrite({
        "Type": win32cred.CRED_TYPE_GENERIC, "TargetName": CREDENTIAL,
        "CredentialBlob": value.encode("utf-16-le"),
        "Persist": win32cred.CRED_PERSIST_LOCAL_MACHINE,
    }, 0)


def remove_key():
    import pywintypes
    import win32cred
    try:
        win32cred.CredDelete(CREDENTIAL, win32cred.CRED_TYPE_GENERIC)
    except pywintypes.error as exc:
        if exc.winerror != 1168:
            raise


def mcp_config():
    python = Path(sys.executable).with_name("python.exe")
    return {"mcpServers": {"qmt": {"command": str(python), "args": [
        "-I", "-X", "utf8", "-m", "qmt.desktop_mcp"]}}}
