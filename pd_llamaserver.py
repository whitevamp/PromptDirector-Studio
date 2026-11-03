"""
pd_llamaserver.py — let the app start and stop llama.cpp's llama-server itself.

Because the app owns the server process, it can stop it to free all VRAM (for example before ComfyUI renders),
restart it with a different GGUF, and shut it down when the app closes. GGUF files are used in place.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

log = logging.getLogger("promptdirector")

DEFAULT_SERVER = {
    "managed": False,              # ON = the app starts/stops llama-server (backend type 'llamacpp')
    "llama_cpp_dir": "",           # your llama.cpp folder (custom build) or the folder holding llama-server
    "gguf_dir": "",                # folder with your .gguf files (fills the model drop-down)
    "gpu": "0",                    # CUDA_VISIBLE_DEVICES ('' = leave unchanged)
    "n_gpu_layers": 99,
    "flash_attn": "on",            # on | off | auto ('' = don't pass -fa)
    "parallel": 1,                 # -np: 1 gives one request the whole context
    "cache_type": "",              # '' | q8_0 | q4_0 ... (KV cache; needs flash attention)
    "extra_args": "",              # anything else, e.g. --mlock --threads 8
    "startup_timeout": 300,        # seconds to wait for the model to load
    "stop_when_app_closes": True,
}
BINARY_NAMES = ("llama-server", "llama-server.exe", "server")
BINARY_SUBDIRS = ("", "build/bin", "bin", "build/bin/Release", "build", "build-cuda/bin", "build/bin/Debug")


class ServerError(Exception):
    pass


def find_binary(llama_cpp_dir: str) -> Path:
    """llama-server inside a llama.cpp folder (custom builds put it in build/bin), else on PATH."""
    if llama_cpp_dir:
        base = Path(llama_cpp_dir).expanduser()
        if base.is_file() and os.access(base, os.X_OK):
            return base
        if not base.is_dir():
            raise ServerError(f"llama.cpp folder not found: {base}")
        for sub in BINARY_SUBDIRS:
            for name in BINARY_NAMES:
                p = base / sub / name
                if p.is_file() and os.access(p, os.X_OK):
                    return p
        raise ServerError(f"No llama-server binary found in {base} (looked in "
                          f"{', '.join(s or '.' for s in BINARY_SUBDIRS)}). Build it with: "
                          "cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -t llama-server")
    found = shutil.which("llama-server")
    if found:
        return Path(found)
    raise ServerError("Set the llama.cpp folder (llama-server was not found on PATH)")


def list_ggufs(gguf_dir: str) -> list[str]:
    """All .gguf files under a folder (multimodal projector files are skipped). Full paths, sorted."""
    if not gguf_dir:
        return []
    root = Path(gguf_dir).expanduser()
    if not root.is_dir():
        return []
    out = []
    for p in root.rglob("*.gguf"):
        name = p.name.lower()
        if "mmproj" in name or not p.is_file():
            continue
        if "-0000" in name and "-00001-of-" not in name:   # split models: list only the first part
            continue
        out.append(str(p))
    return sorted(out, key=lambda s: Path(s).name.lower())


def build_command(binary: Path, sc: dict, model_path: str, n_ctx: int, host: str, port: int) -> list[str]:
    cmd = [str(binary), "-m", str(Path(model_path).expanduser()), "-c", str(int(n_ctx)),
           "-ngl", str(int(sc.get("n_gpu_layers", 99))), "-np", str(max(1, int(sc.get("parallel") or 1))),
           "--host", host, "--port", str(port)]
    if sc.get("flash_attn"):
        cmd += ["-fa", str(sc["flash_attn"])]
    if sc.get("cache_type"):
        cmd += ["-ctk", sc["cache_type"], "-ctv", sc["cache_type"]]
    if sc.get("extra_args"):
        cmd += shlex.split(sc["extra_args"])
    return cmd


def _get(url: str, timeout: float = 3.0) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read().decode("utf-8", errors="replace")
            return r.status, (json.loads(body) if body.strip().startswith("{") else {})
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8", errors="replace") or "{}")
        except ValueError:
            return e.code, {}
    except (urllib.error.URLError, OSError, ValueError):
        return 0, {}


class LlamaServerManager:
    """One managed llama-server per app. Thread-safe start/stop."""

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.cmd: list[str] | None = None
        self.log_path: Path | None = None
        self._log_file = None
        self._lock = threading.RLock()
        atexit.register(self.stop)

    # ------------------------------------------------------------------ state
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def status(self) -> str:
        if not self.running():
            return "stopped"
        model = Path(self.cmd[self.cmd.index("-m") + 1]).name if self.cmd and "-m" in self.cmd else "?"
        return f"running (pid {self.proc.pid}, {model})"

    def _tail(self, n: int = 15) -> str:
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-n:])
        except (OSError, AttributeError):
            return ""

    # ------------------------------------------------------------------ start / stop
    def ensure_running(self, cfg: dict, log_dir: Path, stop_event=None) -> str:
        """Start llama-server for cfg (or keep the running one if nothing changed). Returns a status line."""
        sc, m, b = cfg["llama_server"], cfg["model"], cfg["backend"]
        if not m.get("path"):
            raise ServerError("Pick a GGUF model file (Backend & Model > Model file)")
        if not Path(m["path"]).expanduser().is_file():
            raise ServerError(f"Model file not found: {m['path']}")
        u = urlparse(b["host"] or "http://127.0.0.1:8080")
        host, port = u.hostname or "127.0.0.1", u.port or 8080
        if host not in ("127.0.0.1", "localhost", "0.0.0.0", "::1"):
            raise ServerError(f"A managed llama-server must run on this machine, but the host is {b['host']}")
        binary = find_binary(sc.get("llama_cpp_dir", ""))
        cmd = build_command(binary, sc, m["path"], m["n_ctx"], "127.0.0.1" if host == "localhost" else host, port)
        with self._lock:
            if self.running() and self.cmd == cmd:
                return f"llama-server already running with {Path(m['path']).name}"
            if self.running():
                log.info("llama-server settings changed - restarting")
                self.stop()
            code, _ = _get(f"http://127.0.0.1:{port}/health", 2)
            if code:
                raise ServerError(f"Port {port} is already in use (another llama-server you started yourself?). "
                                  "Stop it, or change the port in the Host URL.")
            log_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = log_dir / "llama-server.log"
            self._log_file = open(self.log_path, "a", encoding="utf-8")
            self._log_file.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n$ {shlex.join(cmd)}\n")
            self._log_file.flush()
            env = dict(os.environ)
            if str(sc.get("gpu", "")).strip():
                env["CUDA_VISIBLE_DEVICES"] = str(sc["gpu"]).strip()
            log.info("Starting llama-server: %s", shlex.join(cmd))
            try:
                self.proc = subprocess.Popen(cmd, stdout=self._log_file, stderr=subprocess.STDOUT, env=env,
                                             cwd=str(binary.parent), start_new_session=True)
            except OSError as e:
                raise ServerError(f"Could not start {binary}: {e}") from e
            self.cmd = cmd
        # wait for the model to load: /health is 503 while loading, 200 when ready
        t0 = time.monotonic()
        timeout = float(sc.get("startup_timeout") or 300)
        while time.monotonic() - t0 < timeout:
            if stop_event is not None and stop_event.is_set():
                self.stop()
                raise ServerError("Stopped while llama-server was loading")
            if self.proc.poll() is not None:
                tail = self._tail()
                self.proc = None
                hint = ""
                if "out of memory" in tail.lower() or "failed to allocate" in tail.lower():
                    hint = "\nHint: not enough VRAM - lower the context size or GPU layers, or use a smaller model."
                elif "invalid argument" in tail.lower() or "unknown argument" in tail.lower():
                    hint = "\nHint: this llama.cpp build doesn't know one of the flags (older builds use '-fa' without a value)."
                raise ServerError(f"llama-server exited during startup. Last output:\n{tail}{hint}")
            code, _ = _get(f"http://127.0.0.1:{port}/health", 3)
            if code == 200:
                secs = time.monotonic() - t0
                log.info("llama-server ready in %.1fs (log: %s)", secs, self.log_path)
                return f"llama-server started with {Path(m['path']).name} in {secs:.0f}s"
            time.sleep(0.5)
        self.stop()
        raise ServerError(f"llama-server did not become ready within {int(timeout)}s. See {self.log_path}")

    def stop(self) -> bool:
        """Stop the managed server (frees its VRAM). Returns True if one was running."""
        with self._lock:
            proc, self.proc = self.proc, None
            if proc is None or proc.poll() is not None:
                self._close_log()
                return False
            log.info("Stopping llama-server (pid %d)", proc.pid)
            try:
                os.killpg(proc.pid, signal.SIGTERM) if hasattr(os, "killpg") else proc.terminate()
                proc.wait(timeout=15)
            except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError, OSError):
                try:
                    os.killpg(proc.pid, signal.SIGKILL) if hasattr(os, "killpg") else proc.kill()
                    proc.wait(timeout=5)
                except Exception:
                    pass
            self._close_log()
            return True

    def _close_log(self):
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None


MANAGER = LlamaServerManager()
