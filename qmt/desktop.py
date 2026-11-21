"""Native settings window and loopback dashboard for Windows."""
from __future__ import annotations

import json
import os
import queue
import socket
import threading
import tkinter as tk
from tkinter import messagebox, ttk
import webbrowser

from qmt.desktop_profile import configure, mcp_config, read_key, remove_key, save_key

PORT = 8770
URL = f"http://127.0.0.1:{PORT}"


def dashboard_app():
    from fastapi.responses import PlainTextResponse
    from qmt.controlplane.api import create_app
    app = create_app()

    @app.middleware("http")
    async def local_only(request, call_next):
        if (request.headers.get("host") != f"127.0.0.1:{PORT}"
                or request.headers.get("origin", URL) != URL
                or request.headers.get("sec-fetch-site") == "cross-site"):
            return PlainTextResponse("Local requests only", status_code=403)
        response = await call_next(request)
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    return app


class Desktop:
    def __init__(self, root):
        self.root = root
        self.server = None
        self.closing = threading.Event()
        self.events = queue.Queue()
        root.title("QMT Desktop")
        root.minsize(530, 290)
        tabs = ttk.Notebook(root)
        tabs.pack(fill="both", expand=True, padx=14, pady=14)
        main = ttk.Frame(tabs, padding=18)
        settings = ttk.Frame(tabs, padding=18)
        tabs.add(main, text="基金分析")
        tabs.add(settings, text="设置")
        ttk.Label(main, text="场外基金分析控制台", font=("Microsoft YaHei UI", 15)).pack(anchor="w")
        self.status = tk.StringVar(value="正在启动本地控制台…")
        ttk.Label(main, textvariable=self.status, wraplength=470).pack(anchor="w", pady=12)
        self.open_button = ttk.Button(main, text="打开控制台", state="disabled",
                                      command=lambda: webbrowser.open(URL))
        self.open_button.pack(anchor="w")
        ttk.Label(settings, text="Tushare Pro Key").grid(row=0, column=0, sticky="w")
        self.key_entry = ttk.Entry(settings, show="●", width=44)
        self.key_entry.grid(row=1, column=0, sticky="ew", pady=6)
        ttk.Button(settings, text="保存", command=self.store_key).grid(row=1, column=1, padx=6)
        ttk.Button(settings, text="清除", command=self.clear_key).grid(row=1, column=2)
        self.key_status = tk.StringVar()
        ttk.Label(settings, textvariable=self.key_status).grid(row=2, column=0, columnspan=3, sticky="w")
        configured = bool(read_key())
        self.key_status.set("已保存到 Windows 凭据管理器。" if configured else "尚未配置数据来源。")
        ttk.Separator(settings).grid(row=3, column=0, columnspan=3, sticky="ew", pady=14)
        ttk.Label(settings, text="本地 MCP Server（stdio）").grid(row=4, column=0, sticky="w")
        ttk.Button(settings, text="复制 AI 客户端配置", command=self.copy_config).grid(row=5, column=0, sticky="w", pady=6)
        settings.columnconfigure(0, weight=1)
        if not configured:
            tabs.select(settings)
        root.protocol("WM_DELETE_WINDOW", self.close)
        threading.Thread(target=self.serve, daemon=True).start()
        root.after(125, self.poll)

    def store_key(self):
        try:
            save_key(self.key_entry.get())
        except (ValueError, OSError) as exc:
            messagebox.showerror("保存失败", str(exc), parent=self.root)
            return
        self.key_entry.delete(0, "end")
        self.key_status.set("已保存到 Windows 凭据管理器，下次查询生效。")

    def clear_key(self):
        remove_key()
        self.key_entry.delete(0, "end")
        self.key_status.set("已清除数据来源 Key。")

    def copy_config(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(json.dumps(mcp_config(), indent=2, ensure_ascii=False))
        self.key_status.set("MCP 配置已复制，可粘贴到 AI 客户端。")

    def serve(self):
        import uvicorn
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                sock.bind(("127.0.0.1", PORT))
                sock.listen()
                server = uvicorn.Server(uvicorn.Config(dashboard_app(), host="127.0.0.1", port=PORT,
                                                       log_config=None, access_log=False))
                self.server = server
                if not self.closing.is_set():
                    server.run(sockets=[sock])
        except Exception as exc:
            self.events.put(f"控制台启动失败：{exc}")

    def poll(self):
        try:
            error = self.events.get_nowait()
        except queue.Empty:
            if self.server is not None and self.server.started:
                self.open_button.configure(state="normal")
                self.status.set(f"控制台已就绪 · {URL}")
                return
            self.root.after(125, self.poll)
        else:
            self.status.set(error)

    def close(self):
        self.closing.set()
        if self.server:
            self.server.should_exit = True
        self.root.destroy()


def main():
    import multiprocessing
    multiprocessing.freeze_support()
    root = tk.Tk()
    try:
        configure()
        Desktop(root)
        root.mainloop()
    except Exception as exc:
        messagebox.showerror("QMT Desktop", str(exc), parent=root)
        root.destroy()


if __name__ == "__main__":
    main()
