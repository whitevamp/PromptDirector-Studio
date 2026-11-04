#!/usr/bin/env python3
# PromptDirector Studio - Copyright (c) 2026 whitevamp
# SPDX-License-Identifier: CC-BY-NC-SA-4.0 (see LICENSE)
"""
promptdirector_gui.py — CustomTkinter front end for pd_core.

    pip install customtkinter
    python promptdirector_gui.py [--config path/to/config.toml]

Tabs: Scene | Results | Log | Backend & Model | Generation | Output & Logging | Presets | System Prompt
The bottom bar (count, Generate, Stop, Preview, progress) is visible on every tab.
"""
from __future__ import annotations

import argparse
import copy
import logging
import os
import platform
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

try:
    import customtkinter as ctk
except ImportError:
    print("This GUI needs CustomTkinter:  pip install customtkinter", file=sys.stderr)
    sys.exit(1)

import pd_core as core


def env(name: str, default: str = "") -> str:
    """PD_<name> environment variable (the old KPG_<name> spelling still works)."""
    return os.environ.get(f"PD_{name}", os.environ.get(f"KPG_{name}", default))
import pd_llamaserver as lserver
from pd_picker import SearchPicker
import pd_workflow as wfmod


# ---------------------------------------------------------------------------------------------
# Private font set. Opening a font makes Tk ask fontconfig to sort EVERY installed font into a
# fallback list; with tens of thousands of fonts each new font/size costs ~50 ms, and a GUI uses
# dozens. So before Tk starts, point this process at a tiny fontconfig setup with only the fonts
# the GUI uses. System fonts are untouched. Disable with PD_GUI_FONTCONFIG=0.
# ---------------------------------------------------------------------------------------------
PRIVATE_FC_STATUS = "not set up"
UI_FONT_FAMILIES = ("Roboto", "DejaVu Sans", "DejaVu Sans Mono", "Noto Sans", "CustomTkinter_shapes_font")


def setup_private_fontconfig() -> str:
    global PRIVATE_FC_STATUS
    if not sys.platform.startswith("linux") or env("GUI_FONTCONFIG") == "0":
        PRIVATE_FC_STATUS = "off"
        return PRIVATE_FC_STATUS
    if os.environ.get("FONTCONFIG_FILE"):
        PRIVATE_FC_STATUS = "skipped (FONTCONFIG_FILE already set)"
        return PRIVATE_FC_STATUS
    if not shutil.which("fc-list"):
        PRIVATE_FC_STATUS = "skipped (fc-list not found)"
        return PRIVATE_FC_STATUS
    files: list[Path] = []
    for fam in UI_FONT_FAMILIES:
        try:
            out = subprocess.run(["fc-list", "-f", "%{file}\n", f":family={fam}"], capture_output=True,
                                 text=True, timeout=30).stdout
        except Exception:
            continue
        found = [Path(l.strip()) for l in out.splitlines() if l.strip()]
        files += [f for f in found if f.is_file()][:40]
    # CustomTkinter bundles Roboto and its corner-shape font; add them in case fontconfig doesn't list them
    assets = Path(ctk.__file__).resolve().parent / "assets" / "fonts"
    if assets.is_dir():
        files += [f for f in assets.rglob("*") if f.suffix.lower() in (".ttf", ".otf")]
    files = list(dict.fromkeys(f.resolve() for f in files))
    if not files:
        PRIVATE_FC_STATUS = "skipped (no UI fonts found)"
        return PRIVATE_FC_STATUS
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "promptdirector" / "fontconfig"
    fdir, cdir = base / "fonts", base / "cache"
    try:
        fdir.mkdir(parents=True, exist_ok=True)
        cdir.mkdir(parents=True, exist_ok=True)
        for old in fdir.iterdir():
            if old.is_symlink() or old.is_file():
                old.unlink()
        for i, f in enumerate(files):
            (fdir / f"{i:03d}_{f.name}").symlink_to(f)
        conf = base / "fonts.conf"
        conf.write_text(
            '<?xml version="1.0"?>\n<!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">\n<fontconfig>\n'
            f"  <dir>{fdir}</dir>\n  <cachedir>{cdir}</cachedir>\n"
            "  <alias><family>sans-serif</family><prefer><family>Roboto</family><family>DejaVu Sans</family>"
            "<family>Noto Sans</family></prefer></alias>\n"
            "  <alias><family>monospace</family><prefer><family>DejaVu Sans Mono</family></prefer></alias>\n"
            '  <match target="font">\n'
            '    <edit name="antialias" mode="assign"><bool>true</bool></edit>\n'
            '    <edit name="hinting" mode="assign"><bool>true</bool></edit>\n'
            '    <edit name="hintstyle" mode="assign"><const>hintslight</const></edit>\n'
            '    <edit name="rgba" mode="assign"><const>none</const></edit>\n'
            "  </match>\n</fontconfig>\n", encoding="utf-8")
    except OSError as e:
        PRIVATE_FC_STATUS = f"skipped ({e})"
        return PRIVATE_FC_STATUS
    os.environ["FONTCONFIG_FILE"] = str(conf)
    PRIVATE_FC_STATUS = f"private ({len(files)} font files, {conf})"
    return PRIVATE_FC_STATUS


def _find_draw_engine():
    """CustomTkinter's DrawEngine class (location differs between versions)."""
    de = getattr(ctk, "DrawEngine", None)
    if de is None:
        try:
            from customtkinter.windows.widgets.core_rendering import DrawEngine as de  # type: ignore
        except Exception:
            de = None
    return de


def set_draw_method() -> str:
    """On Linux CTk draws rounded corners with glyphs from its own font ('font_shapes'). On systems with
    very large font collections that makes every button/entry/switch slow to create; 'circle_shapes'
    draws the same corners with plain canvas shapes. Override with PD_GUI_DRAW=font_shapes|circle_shapes|
    polygon_shapes."""
    de = _find_draw_engine()
    if de is None or not hasattr(de, "preferred_drawing_method"):
        return "unknown (CustomTkinter internals not found)"
    wanted = env("GUI_DRAW").strip()
    if wanted in ("font_shapes", "circle_shapes", "polygon_shapes"):
        de.preferred_drawing_method = wanted
    elif (sys.platform.startswith("linux") and de.preferred_drawing_method == "font_shapes"
          and not PRIVATE_FC_STATUS.startswith("private")):
        de.preferred_drawing_method = "circle_shapes"
    return str(de.preferred_drawing_method)


def _patch_ctk_scroll_bug() -> None:
    """CustomTkinter bug: CTkScrollableFrame binds the mouse wheel app-wide, and when the wheel is
    used over a dropdown list Tk passes the widget as a *string* path, so CTk raises
    "AttributeError: 'str' object has no attribute 'master'" on every wheel click."""
    cls = getattr(ctk, "CTkScrollableFrame", None)
    orig = getattr(cls, "_check_if_valid_scroll", None)
    if orig is None or getattr(orig, "_pd_patched", False):
        return

    def safe_check(self, widget):
        if isinstance(widget, str):
            try:
                widget = self.nametowidget(widget)
            except Exception:
                return False  # Tcl-only widget (e.g. a dropdown menu): not ours, don't scroll
        try:
            return orig(self, widget)
        except AttributeError:
            return False

    safe_check._pd_patched = True  # type: ignore[attr-defined]
    cls._check_if_valid_scroll = safe_check


_patch_ctk_scroll_bug()

APP_TITLE = f"PromptDirector Studio {core.__version__}"
TIMING = env("GUI_TIMING") == "1"     # print startup / tab timings to the terminal
LAZY_TABS = ("Comic", "Backend & Model", "Generation", "Output & Logging")   # built on first visit
MUTED = ("gray40", "gray65")
PAD = {"padx": 8, "pady": 4}


class QueueLogHandler(logging.Handler):
    """Sends log records to the GUI thread."""

    def __init__(self, q: queue.Queue):
        super().__init__(logging.INFO)
        self.q = q

    def emit(self, record):
        try:
            self.q.put(("log", record.levelname, self.format(record)))
        except Exception:
            self.handleError(record)


def open_path(path: Path) -> None:
    path = Path(path)
    if not path.exists():
        messagebox.showwarning(APP_TITLE, f"Not found yet:\n{path}")
        return
    try:
        if platform.system() == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        messagebox.showerror(APP_TITLE, f"Could not open {path}:\n{e}")


class App(ctk.CTk):
    def __init__(self, config_path: str | None = None):
        super().__init__()
        self._t0 = time.perf_counter()
        # Build everything while the window is hidden: on some XWayland setups every widget added to a
        # visible window waits on a slow round trip to the display server.
        self.withdraw()
        self._setup_fonts()            # before any widget: pick an installed font, share font objects
        self.title(APP_TITLE)
        self.geometry("1180x820")
        self.minsize(980, 680)

        self.q: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.last_result: core.JobResult | None = None
        self.rng = random.Random()
        self.vars: dict[tuple[str, str], tuple[tk.Variable, type]] = {}

        try:
            self.cfg, path = core.load_config(config_path)
        except core.PDError as e:
            messagebox.showerror(APP_TITLE, f"{e}\n\nStarting with default settings.")
            self.cfg, path = copy.deepcopy(core.DEFAULT_CONFIG), None
        self.config_path = Path(path) if path else core.APP_DIR / "config.toml"
        self.presets = core.load_presets(self.cfg)
        self.saved_snapshot = copy.deepcopy(self.cfg)

        _prof = self._profile_start()
        self._make_all_vars()          # every setting has a variable even before its tab is built
        self._ollama_models: list[str] = []
        self._comfy_lists: dict[str, list[str]] = {}
        self.wf_doc = None
        self.wf_original = ""
        self.wf_rows: list = []
        self._bar_widgets: list = []
        self._smooth_bar(self._build_toolbar)
        self.tabs = ctk.CTkTabview(self, command=self._on_tab_change)
        self.tabs.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        for name in ("Scene", "Comic", "ComfyUI", "Workflow", "Results", "Log", "Backend & Model", "Generation",
                     "Output & Logging", "Presets", "System Prompt"):
            self.tabs.add(name)
        self._lazy_builders = {
            "Comic": self._build_comic_tab, "ComfyUI": self._build_comfy_tab, "Workflow": self._build_workflow_tab,
            "Backend & Model": self._build_backend_tab,
            "Generation": self._build_generation_tab, "Output & Logging": self._build_output_tab,
        }
        self._built: set[str] = set()
        self._build_scene_tab(self.tabs.tab("Scene"))
        self._build_results_tab(self.tabs.tab("Results"))
        self._build_log_tab(self.tabs.tab("Log"))
        self._build_presets_tab(self.tabs.tab("Presets"))
        self._build_sysprompt_tab(self.tabs.tab("System Prompt"))
        self._smooth_bar(self._build_runbar)
        self.bind("<Configure>", self._on_window_configure, add="+")

        self.load_into_widgets(self.cfg)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(100, self._poll_queue)
        self._profile_report(_prof, "startup")
        self.deiconify()
        if TIMING:
            self._timing("widgets built")
            self.after_idle(lambda: self._timing("window drawn (first idle)"))

    # ------------------------------------------------------------------ helpers
    def _setup_fonts(self):
        """Every CTk widget creates a font. If the theme's family (Roboto on Linux) isn't installed, each
        creation goes through a slow fontconfig fallback, so use an installed family and share font objects."""
        t = time.perf_counter()
        fams: set[str] = set()
        try:
            import tkinter.font as tkfont
            fams = set(tkfont.families(self))
        except Exception:
            pass
        wanted = [env("GUI_FONT"), "Roboto", "Noto Sans", "DejaVu Sans", "Cantarell",
                  "Liberation Sans", "Ubuntu", "Inter", "Arial", "Helvetica"]
        family = next((f for f in wanted if f and f in fams), None)
        mono = next((f for f in ("DejaVu Sans Mono", "Noto Sans Mono", "Liberation Mono", "JetBrains Mono",
                                 "Ubuntu Mono", "Hack", "Courier New") if f in fams), "TkFixedFont")
        try:
            if family:
                ctk.ThemeManager.theme["CTkFont"]["family"] = family
        except Exception:
            pass
        self.font_small = ctk.CTkFont(size=11)
        self.font_bold = ctk.CTkFont(weight="bold")
        self.font_head = ctk.CTkFont(size=14, weight="bold")
        self.font_text = ctk.CTkFont(size=13)
        self.font_mono = ctk.CTkFont(family=mono, size=12)
        self._fast_draw = (env("GUI_DRAW") == "" and sys.platform.startswith("linux")
                           and not PRIVATE_FC_STATUS.startswith("private"))
        draw = set_draw_method()
        ime = "on"
        if env("GUI_IME") == "0":
            try:   # each text-entry widget otherwise opens a slow X input-method context (fcitx/ibus via XWayland)
                self.tk.call("tk", "useinputmethods", "0")
                ime = "off"
            except Exception:
                ime = "unchanged (not supported)"
        if TIMING:
            print(f"[timing] fontconfig: {PRIVATE_FC_STATUS}", file=sys.stderr)
            print(f"[timing] X input methods: {ime}", file=sys.stderr)
            print(f"[timing] corner drawing method: {draw}", file=sys.stderr)
            print(f"[timing] fonts: family={family or 'theme default'}, mono={mono}, {len(fams)} families "
                  f"installed, setup {time.perf_counter() - t:.2f}s", file=sys.stderr)

    def _smooth_bar(self, build):
        """Build the toolbar / bottom bar with CTk's smooth (font-based) corners. It is slow per widget,
        but these bars only hold ~10 widgets and are always visible; everything else uses fast shapes."""
        de = _find_draw_engine()
        smooth = self._fast_draw and de is not None and hasattr(de, "preferred_drawing_method")
        prev = de.preferred_drawing_method if smooth else None
        if smooth:
            de.preferred_drawing_method = "font_shapes"
        try:
            bar = build()
        finally:
            if smooth:
                de.preferred_drawing_method = prev
        stack = list(bar.winfo_children()) if bar is not None else []
        while stack:
            w = stack.pop()
            stack.extend(w.winfo_children() if hasattr(w, "winfo_children") else [])
            eng = getattr(w, "_draw_engine", None)
            if smooth and eng is not None:
                try:
                    eng.preferred_drawing_method = "font_shapes"    # instance setting survives redraws
                except Exception:
                    pass
            if hasattr(w, "_draw"):
                self._bar_widgets.append(w)

    def _repaint_bars(self):
        """Redraw toolbar/bottom-bar widgets (they can stay blank after resizes under XWayland)."""
        self._repaint_pending = False
        for w in self._bar_widgets:
            try:
                w._draw()
            except Exception:
                pass

    def _on_window_configure(self, event):
        if event.widget is self and not getattr(self, "_repaint_pending", False):
            self._repaint_pending = True
            self.after(200, self._repaint_bars)

    @staticmethod
    def _profile_start():
        if not TIMING:
            return None
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
        return pr

    @staticmethod
    def _profile_report(pr, title: str, top: int = 25):
        if pr is None:
            return
        pr.disable()
        import io
        import pstats
        for key in ("tottime", "cumulative"):
            buf = io.StringIO()
            st = pstats.Stats(pr, stream=buf).strip_dirs().sort_stats(key)
            st.print_stats(top)
            lines = [l for l in buf.getvalue().splitlines() if l.strip()]
            start = next((i for i, l in enumerate(lines) if "ncalls" in l), 0)
            print(f"\n[profile] {title} - top {top} by {key}:", file=sys.stderr)
            print("\n".join(lines[start:start + top + 1]), file=sys.stderr)

    def _timing(self, what: str):
        print(f"[timing] {what}: {time.perf_counter() - self._t0:.2f}s since start", file=sys.stderr)

    def _make_all_vars(self):
        """Create a Tk variable for every setting up front (lists are handled separately)."""
        def walk(section: str, d: dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    walk(f"{section}.{k}" if section else k, v)
                elif not isinstance(v, (list, tuple)) and section:
                    self.bind_var(section, k, type(v) if isinstance(v, (bool, int, float)) else str)
        walk("", core.DEFAULT_CONFIG)
        self.format_vars = {f: tk.BooleanVar(self) for f in core.VALID_FORMATS}
        self.people_vars = [{k: tk.StringVar(self) for k in core.PERSON_KEYS} for _ in range(core.MAX_CHARACTERS)]
        self.rot_models_var = tk.StringVar(self)          # rotation model list, one per line
        self.target_vars = {t: tk.BooleanVar(self) for t in core.VALID_TARGETS}

    def _ensure_built(self, name: str):
        if name in self._lazy_builders and name not in self._built:
            t = time.perf_counter()
            self._built.add(name)
            prof = self._profile_start() if name == "Generation" else None
            self._lazy_builders[name](self.tabs.tab(name))
            self._profile_report(prof, f"building tab '{name}'", 20)
            if name == "Comic":
                self._update_comic_info()
            elif name == "Backend & Model":
                self._backend_changed()
            if TIMING:
                print(f"[timing] built tab '{name}' in {time.perf_counter() - t:.2f}s", file=sys.stderr)

    def bind_var(self, section: str, key: str, kind: type, var: tk.Variable | None = None) -> tk.Variable:
        if var is None and (section, key) in self.vars:
            existing = self.vars[(section, key)][0]   # reuse: created up front by _make_all_vars
            self.vars[(section, key)] = (existing, kind)
            return existing
        if var is None:
            var = tk.BooleanVar(self) if kind is bool else tk.StringVar(self)
        self.vars[(section, key)] = (var, kind)
        return var

    def form(self, parent, title: str | None = None) -> ctk.CTkScrollableFrame:
        f = ctk.CTkScrollableFrame(parent, label_text=title)
        f.pack(fill="both", expand=True)
        f.grid_columnconfigure(1, weight=1)
        f._row = 0  # type: ignore[attr-defined]
        return f

    def row(self, f, label: str, section: str, key: str, kind: type = str, *, choices=None,
            combo=None, hint: str = "", browse: str | None = None, secret=False, pick=None, pick_kw=None):
        r = f._row
        ctk.CTkLabel(f, text=label, anchor="w").grid(row=r, column=0, sticky="w", **PAD)
        var = self.bind_var(section, key, kind)
        if kind is bool:
            w = ctk.CTkSwitch(f, text="", variable=var, onvalue=True, offvalue=False)
            w.grid(row=r, column=1, sticky="w", **PAD)
        elif choices:
            w = ctk.CTkOptionMenu(f, values=list(choices), variable=var)
            w.grid(row=r, column=1, sticky="w", **PAD)
        elif pick is not None:      # long lists: entry + searchable picker window (stays open until chosen)
            w = ctk.CTkEntry(f, textvariable=var)
            w.grid(row=r, column=1, sticky="ew", **PAD)
            kw = dict(pick_kw or {})
            ctk.CTkButton(f, text="Choose...", width=80,
                          command=lambda: self.open_picker(var, label, pick, **kw)).grid(row=r, column=2, **PAD)
            browse = None
        elif combo is not None:
            w = ctk.CTkComboBox(f, values=list(combo), variable=var)
            w.grid(row=r, column=1, sticky="ew", **PAD)
        else:
            w = ctk.CTkEntry(f, textvariable=var, show="•" if secret else "")
            w.grid(row=r, column=1, sticky="ew", **PAD)
        if browse:
            ctk.CTkButton(f, text="Browse…", width=80,
                          command=lambda: self._browse(var, browse)).grid(row=r, column=2, **PAD)
        if hint:
            ctk.CTkLabel(f, text=hint, text_color=MUTED, anchor="w", font=self.font_small
                         ).grid(row=r, column=3, sticky="w", **PAD)
        f._row += 1
        return w

    def open_picker(self, var, title, items_fn, refresh=None, browse=None, clear_label=None):
        """Searchable picker for long lists. items_fn/refresh return [(display, value)]."""
        try:
            items = items_fn()
        except Exception as e:  # noqa: BLE001
            items = []
            log_msg = str(e)
        else:
            log_msg = ""
        browse_fn = None
        if browse:
            def browse_fn():
                kinds = {"gguf": [("GGUF model", "*.gguf"), ("All files", "*.*")]}
                return filedialog.askopenfilename(filetypes=kinds.get(browse, [("All files", "*.*")])) or None
        picker = SearchPicker(self, f"Choose: {title}", items, var.get(), var.set, refresh=refresh,
                              browse=browse_fn, clear_label=clear_label, font=self.font_text)
        if log_msg:
            picker.msg.configure(text=f"ERROR: {log_msg.splitlines()[0]}")
        self._last_picker = picker
        return picker

    def _picker_entry(self, parent, var, title: str, items_fn, width: int | None = None):
        """Text box + '...' button that opens the searchable picker (long lists, stays open until chosen)."""
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.grid_columnconfigure(0, weight=1)
        e = ctk.CTkEntry(box, textvariable=var, **({"width": width} if width else {}))
        e.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(box, text="...", width=28, command=lambda: self.open_picker(
            var, title, lambda: [(x, x) for x in items_fn()], clear_label="Clear")).grid(row=0, column=1, padx=(3, 0))
        box.entry = e
        return box

    def _field_choices(self, key: str) -> list[str]:
        vals = set(self.presets.get(key, []))
        if key == "action":
            vals |= set(self.presets.get("action_group", []))
        return sorted(v for v in vals if v)

    def _person_choices(self, key: str) -> list[str]:
        if key == "role":
            vals = self.presets.get("role") or self.presets.get("subject") or []
        else:
            vals = list(self.presets.get(key) or [])
            if key == "age":
                vals += ["25", "30", "35", "40", "50"]
        return sorted(set(v for v in vals if v), key=lambda x: (key == "age", x)) if key != "age" else list(dict.fromkeys(vals))

    def section_label(self, f, text: str):
        ctk.CTkLabel(f, text=text, font=self.font_head, anchor="w"
                     ).grid(row=f._row, column=0, columnspan=4, sticky="w", padx=8, pady=(14, 2))
        f._row += 1

    def _browse(self, var: tk.Variable, kind: str):
        if kind == "dir":
            p = filedialog.askdirectory(initialdir=var.get() or os.getcwd())
        elif kind == "gguf":
            p = filedialog.askopenfilename(filetypes=[("GGUF model", "*.gguf"), ("All files", "*.*")])
        elif kind == "json":
            p = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        else:
            p = filedialog.askopenfilename(filetypes=[("Text", "*.txt"), ("All files", "*.*")])
        if p:
            var.set(p)

    def run_bg(self, func, on_done, *args):
        """Run func(*args) in a thread; on_done(result, error) on the GUI thread."""
        def target():
            try:
                self.q.put(("call", on_done, func(*args), None))
            except Exception as e:  # noqa: BLE001
                self.q.put(("call", on_done, None, e))
        threading.Thread(target=target, daemon=True, name="bg").start()

    def _on_tab_change(self):
        name = self.tabs.get()
        self._ensure_built(name)
        if name == "Workflow":
            self._wf_update_suggestion()
        self.after(50, self._repaint_bars)
        if TIMING:
            t = time.perf_counter()
            self.after_idle(lambda: print(f"[timing] switch to '{name}' drawn in {time.perf_counter() - t:.2f}s",
                                          file=sys.stderr))

    # ------------------------------------------------------------------ toolbar
    def _build_toolbar(self):
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", padx=8, pady=(8, 4))
        ctk.CTkLabel(bar, text="Config:").pack(side="left", padx=(4, 2))
        self.cfg_label = ctk.CTkLabel(bar, text="", text_color=MUTED)
        self.cfg_label.pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Save", width=70, command=self.save_config).pack(side="left", padx=3)
        ctk.CTkButton(bar, text="Save As…", width=80, command=self.save_config_as).pack(side="left", padx=3)
        ctk.CTkButton(bar, text="Load…", width=70, command=self.load_config_dialog).pack(side="left", padx=3)
        ctk.CTkButton(bar, text="Defaults", width=80, fg_color="gray40",
                      command=self.reset_defaults).pack(side="left", padx=3)
        self.appearance = ctk.CTkOptionMenu(bar, values=["System", "Dark", "Light"], width=100,
                                            command=ctk.set_appearance_mode)
        self.appearance.pack(side="right", padx=4)
        ctk.CTkLabel(bar, text="Theme:").pack(side="right")
        return bar

    # ------------------------------------------------------------------ scene tab
    def _build_scene_tab(self, tab):
        top = ctk.CTkFrame(tab, fg_color="transparent")
        top.pack(fill="x", pady=(0, 2))
        self.random_master = self.bind_var("scene", "random_enabled", bool)
        self.random_switch = ctk.CTkSwitch(top, text="", variable=self.random_master, onvalue=True,
                                           offvalue=False, font=self.font_bold)
        self.random_switch.pack(side="left", padx=6)
        self.random_master.trace_add("write", lambda *_: self._update_random_label())
        self._update_random_label()

        btns = ctk.CTkFrame(tab, fg_color="transparent")
        btns.pack(fill="x", pady=(0, 4))
        for text, cmd in (("Random: all on", lambda: self.set_all_random(True)),
                          ("Random: all off", lambda: self.set_all_random(False)),
                          ("Roll all empty", self.roll_all),
                          ("Clear all fields", self.clear_fields)):
            ctk.CTkButton(btns, text=text, width=130, command=cmd).pack(side="left", padx=6)

        ctk.CTkLabel(tab, text=("'Random' ticked (and Random ON): a new random value every prompt; any text in the box is "
                                "ignored (shown greyed out). 'Random' unticked: the box's value is used for every prompt; "
                                "an empty box is left for the LLM. 'Roll' picks a value now and unticks 'Random'."),
                     text_color=MUTED, wraplength=1100, justify="left", anchor="w").pack(fill="x", padx=6)

        f = ctk.CTkScrollableFrame(tab)
        f.pack(fill="both", expand=True, pady=4)
        f.grid_columnconfigure(1, weight=1)
        for c, h in enumerate(("Field", "Value (empty = random / LLM decides)", "Random", "", "")):
            ctk.CTkLabel(f, text=h, text_color=MUTED, font=self.font_small).grid(row=0, column=c, sticky="w", padx=8)
        self.field_widgets: dict = {}
        for r, (key, label, hint, can_random) in enumerate(core.SCENE_FIELDS, start=1):
            ctk.CTkLabel(f, text=f"{label}\n{hint}", anchor="w", justify="left"
                         ).grid(row=r, column=0, sticky="w", padx=8, pady=3)
            var = self.bind_var("scene.fields", key, str)
            if key == "characters":
                cb = ctk.CTkComboBox(f, values=[""] + [str(i) for i in range(1, core.MAX_CHARACTERS + 1)],
                                     variable=var)
            elif key == "extra":
                cb = ctk.CTkEntry(f, textvariable=var)
            else:
                cb = self._picker_entry(f, var, label, lambda k=key: self._field_choices(k))
            cb.grid(row=r, column=1, sticky="ew", padx=8, pady=3)
            self.field_widgets[key] = cb
            if can_random:
                rv = self.bind_var("scene.random", key, bool)
                ctk.CTkCheckBox(f, text="", variable=rv, onvalue=True, offvalue=False, width=30
                                ).grid(row=r, column=2, padx=8)
                ctk.CTkButton(f, text="Roll", width=50, command=lambda k=key: self.roll_one(k)
                              ).grid(row=r, column=3, padx=2)
            ctk.CTkButton(f, text="Clear", width=50, fg_color="gray40",
                          command=lambda v=var: v.set("")).grid(row=r, column=4, padx=(2, 8))
        self._build_people_section(f, len(core.SCENE_FIELDS) + 1)
        for (sect, _key), (var, _kind) in list(self.vars.items()):
            if sect in ("scene.random", "scene.random_people"):
                var.trace_add("write", self._update_ignored_text)
        self.random_master.trace_add("write", self._update_ignored_text)
        self._update_ignored_text()

        box = ctk.CTkFrame(tab)
        box.pack(fill="x", pady=(4, 0))
        ctk.CTkLabel(box, text="Free-text scene ideas (optional, one per line — used in turn; "
                               "your filled-in fields are still applied, random fields are not):",
                     anchor="w").pack(fill="x", padx=8, pady=(6, 0))
        self.scene_ideas = ctk.CTkTextbox(box, height=70)
        self.scene_ideas.pack(fill="x", padx=8, pady=6)

    def _update_random_label(self):
        on = bool(self.random_master.get())
        self.random_switch.configure(text=(
            "Random ON - empty fields with 'Random' ticked get a random value per prompt" if on else
            "Random OFF - empty fields are left for the LLM to decide"))

    def _build_people_section(self, f, row: int):
        """One row per character; rows follow the 'Number of characters' field."""
        ctk.CTkLabel(f, text="Characters", font=self.font_head, anchor="w").grid(
            row=row, column=0, columnspan=5, sticky="w", padx=8, pady=(14, 0))
        self.people_note = ctk.CTkLabel(f, text="", text_color=MUTED, anchor="w", justify="left", wraplength=1000)
        self.people_note.grid(row=row + 1, column=0, columnspan=5, sticky="w", padx=8)
        pf = ctk.CTkFrame(f, fg_color="transparent")
        pf.grid(row=row + 2, column=0, columnspan=5, sticky="ew", padx=4)
        for c in range(1, len(core.PERSON_FIELDS) + 1):
            pf.grid_columnconfigure(c, weight=1)
        ctk.CTkLabel(pf, text="#", text_color=MUTED).grid(row=0, column=0, padx=4)
        for c, (k, label) in enumerate(core.PERSON_FIELDS, start=1):
            head = ctk.CTkFrame(pf, fg_color="transparent")
            head.grid(row=0, column=c, sticky="w", padx=4)
            ctk.CTkLabel(head, text=label, font=self.font_small).pack(side="left")
            ctk.CTkCheckBox(head, text="random", variable=self.bind_var("scene.random_people", k, bool),
                            onvalue=True, offvalue=False, width=24, font=self.font_small).pack(side="left", padx=6)
        self.people_rows = []
        for i in range(core.MAX_CHARACTERS):
            widgets = [ctk.CTkLabel(pf, text=str(i + 1), width=20)]
            widgets[0].grid(row=i + 1, column=0, padx=4, pady=2)
            for c, (k, label) in enumerate(core.PERSON_FIELDS, start=1):
                cb = self._picker_entry(pf, self.people_vars[i][k], f"Character {i + 1}: {label}",
                                        lambda k=k: self._person_choices(k), width=110)
                cb.grid(row=i + 1, column=c, sticky="ew", padx=4, pady=2)
                widgets.append(cb)
            self.people_rows.append(widgets)
        self.vars[("scene.fields", "characters")][0].trace_add("write", lambda *_: self._update_people_rows())
        self._update_people_rows()

    def _people_count(self) -> int:
        v = self.vars[("scene.fields", "characters")][0].get().strip()
        return min(int(v), core.MAX_CHARACTERS) if v.isdigit() else 0

    def _update_people_rows(self):
        if not hasattr(self, "people_rows"):
            return
        n = self._people_count()
        for i, widgets in enumerate(self.people_rows):
            for w in widgets:
                (w.grid if i < n else w.grid_remove)()
        self.people_note.configure(text=(
            f"Describe each of the {n} character(s). A ticked 'random' column is random for everyone (typed text is "
            f"ignored); unticked columns use what you typed, and empty boxes are left for the LLM. "
            f"Ages must be 18+ (e.g. 30 or early 30s). Typed roles build the Subject line."
            if n else "Set 'Number of characters' above (1-6) to describe each person here. With a random count, "
                      "ticked 'random' columns are still filled in for every character."))

    TEXT_ON = ("gray10", "#DCE4EE")
    TEXT_IGNORED = ("gray60", "gray40")

    def _update_ignored_text(self, *_):
        """Grey out typed values that a ticked 'Random' box overrides."""
        if not hasattr(self, "random_master"):
            return
        master = bool(self.random_master.get())
        for key, w in getattr(self, "field_widgets", {}).items():
            rv = self.vars.get(("scene.random", key))
            ignored = master and rv is not None and bool(rv[0].get())
            target = getattr(w, "entry", w)
            try:
                target.configure(text_color=self.TEXT_IGNORED if ignored else self.TEXT_ON)
            except Exception:
                pass
        for widgets in getattr(self, "people_rows", []):
            for (k, _), w in zip(core.PERSON_FIELDS, widgets[1:]):
                ignored = master and bool(self.vars[("scene.random_people", k)][0].get())
                try:
                    getattr(w, "entry", w).configure(text_color=self.TEXT_IGNORED if ignored else self.TEXT_ON)
                except Exception:
                    pass

    def refresh_field_choices(self):
        pass                                   # pickers read self.presets each time they open

    def _n_chars(self) -> int | None:
        v = self.vars[("scene.fields", "characters")][0].get().strip()
        return int(v) if v.isdigit() else None

    def roll_one(self, key: str):
        var = self.vars[("scene.fields", key)][0]
        var.set(core.roll_field(key, self.presets, self.rng, self._n_chars()))
        if ("scene.random", key) in self.vars:
            self.vars[("scene.random", key)][0].set(False)     # rolled = use this value

    def roll_all(self):
        for key, _, _, can_random in core.SCENE_FIELDS:
            if can_random and not self.vars[("scene.fields", key)][0].get().strip():
                self.roll_one(key)
        for i in range(self._people_count()):
            for k in core.PERSON_KEYS:
                var = self.people_vars[i][k]
                pool = self.presets.get(k) or (self.presets.get("subject") if k == "role" else [])
                if not var.get().strip() and pool:
                    var.set(self.rng.choice(pool))
                    self.vars[("scene.random_people", k)][0].set(False)

    def clear_fields(self):
        for key in core.FIELD_KEYS:
            self.vars[("scene.fields", key)][0].set("")
        for row in self.people_vars:
            for v in row.values():
                v.set("")
        self.scene_ideas.delete("1.0", "end")

    def set_all_random(self, on: bool):
        for (sect, _), (var, _) in self.vars.items():
            if sect == "scene.random":
                var.set(on)

    # ------------------------------------------------------------------ comic tab
    def _build_comic_tab(self, tab):
        f = self.form(tab)
        self.section_label(f, "Comic panel layout (for Krea 2)")
        self.row(f, "Comic panel layout", "comic", "enabled", bool,
                 hint="ON = one image split into panels; uses the Scene tab for cast and style")
        self.row(f, "Grid (columns x rows)", "comic", "layout", choices=core.COMIC_LAYOUTS,
                 hint="2x3 = 6 panels, 2 columns by 3 rows")
        self.row(f, "Panel mode", "comic", "mode", choices=core.COMIC_MODES,
                 hint="vignettes = separate moments | story = consecutive beats")
        self.row(f, "Prompt format", "comic", "format", choices=core.COMIC_FORMATS,
                 hint="prose = 'Panel 1 (Top Left): ...' | bbox = layout spec for krea2-turbo-bbox")
        self.row(f, "Location per panel", "comic", "location_mode", choices=core.COMIC_LOCATION_MODES,
                 hint="auto: vignettes = different places, story = one place")
        self.row(f, "Speech bubbles", "comic", "speech_bubbles", bool, hint="short lines of dialogue (max 8 words)")
        self.comic_info = ctk.CTkLabel(f, text="", justify="left", anchor="w", text_color=MUTED)
        self.comic_info.grid(row=f._row, column=0, columnspan=4, sticky="w", padx=8, pady=(12, 4))
        f._row += 1
        ctk.CTkLabel(f, justify="left", anchor="w", wraplength=1000, text=(
            "Tips: the cast (style, characters, lighting, weather) comes from the Scene tab and is shared by every "
            "panel; camera angle and shot (and location/action, depending on the mode) are rolled per panel. "
            "Generate the image at the suggested size so panels stay roughly square. The prose format suits stock "
            "Krea 2 and mixes (FLUX handles 2-4 panels); the bbox format is for the krea2-turbo-bbox finetune.")
                     ).grid(row=f._row, column=0, columnspan=4, sticky="w", padx=8, pady=4)
        f._row += 1
        for key in ("enabled", "layout", "format"):
            self.vars[("comic", key)][0].trace_add("write", lambda *_: self._update_comic_info())

    def _update_comic_info(self):
        if not hasattr(self, "comic_info"):
            return
        try:
            lay = self.vars[("comic", "layout")][0].get()
            on = bool(self.vars[("comic", "enabled")][0].get())
            fmt = self.vars[("comic", "format")][0].get()
            panels = core.comic_grid(lay)
            txt = (f"{'ON' if on else 'OFF'} - {len(panels)} panels: " + ", ".join(p["label"] for p in panels) +
                   f"\nSuggested image size: {core.suggested_size(lay)}  |  format: {fmt}")
            self.comic_info.configure(text=txt)
        except Exception:
            pass

    # ------------------------------------------------------------------ ComfyUI tab
    def _build_comfy_tab(self, tab):
        f = self.form(tab)
        self.section_label(f, "ComfyUI")
        self.row(f, "Send prompts to ComfyUI", "comfy", "enabled", bool,
                 hint="ON = every finished prompt is rendered by ComfyUI")
        self.row(f, "Also save prompt files", "comfy", "save_prompts", bool,
                 hint="OFF = send to ComfyUI only, don't write prompt files (logs are always kept)")
        self.row(f, "ComfyUI URL", "comfy", "url", hint="where ComfyUI runs, e.g. http://127.0.0.1:8188")
        ctk.CTkButton(f, text="Test", width=80, command=self.comfy_test).grid(row=f._row - 1, column=2, **PAD)
        self.comfy_status = ctk.CTkLabel(f, text="", anchor="w", justify="left", wraplength=900)
        self.comfy_status.grid(row=f._row, column=0, columnspan=4, sticky="w", padx=8)
        f._row += 1
        self.row(f, "ComfyUI models folder", "comfy", "models_dir", browse="dir",
                 hint="optional: ComfyUI's 'models' folder (or one of its subfolders); used while ComfyUI is off")
        r = f._row
        ctk.CTkButton(f, text="Refresh model lists", width=150, command=self.comfy_refresh_models
                      ).grid(row=r, column=0, sticky="w", **PAD)
        self.comfy_cache_label = ctk.CTkLabel(f, text="", text_color=MUTED, anchor="w")
        self.comfy_cache_label.grid(row=r, column=1, columnspan=3, sticky="w", **PAD)
        f._row += 1

        self.section_label(f, "Workflow and models")
        self.row(f, "Workflow", "comfy", "workflow", choices=core.comfy.WORKFLOWS,
                 hint="custom = your exported workflow (Krea 2, Moody mix, LoRAs) | builtin_sdxl (Illustrious) | builtin_flux")
        self.row(f, "Custom workflow file", "comfy", "workflow_file", browse="json",
                 hint="ComfyUI: Workflow > Export (API). Put %PROMPT% in its prompt box first")
        self.comfy_fields = {}
        for key, label, lst, hint in (
                ("checkpoint", "Checkpoint", "checkpoints", "SDXL / Illustrious (builtin_sdxl)"),
                ("diffusion_model", "Diffusion model", "diffusion_models", "FLUX / Krea 2 model (blank = keep workflow's)"),
                ("text_encoder", "Text encoder", "text_encoders", "FLUX: t5xxl | Krea 2: qwen3vl (blank = keep workflow's)"),
                ("text_encoder_2", "Text encoder 2", "text_encoders", "FLUX only: clip_l"),
                ("vae", "VAE", "vae", "FLUX: ae | Krea 2: qwen_image_vae (blank = default)")):
            self.comfy_fields[key] = (self.row(
                f, label, "comfy", key, hint=hint,
                pick=lambda lst=lst: [(x, x) for x in self._comfy_lists.get(lst, [])],
                pick_kw={"refresh": lambda lst=lst: self._comfy_refresh_sync(lst),
                         "clear_label": "Use workflow's own"}), lst)
        self.row(f, "Prompt to send", "comfy", "prompt_source", choices=core.comfy.PROMPT_SOURCES,
                 hint="auto = Illustrious tags for builtin_sdxl (tick the Illustrious target), prose otherwise")
        self.row(f, "Negative prompt", "comfy", "negative_prompt", hint="SDXL/Illustrious, or %NEGATIVE% in custom")

        self.section_label(f, "Image settings (0 / blank = automatic or keep the workflow's)")
        self.row(f, "Width", "comfy", "width", int, hint="0 = comic suggested size, else 1024 / workflow's")
        self.row(f, "Height", "comfy", "height", int)
        self.row(f, "Steps", "comfy", "steps", int, hint="0 = sdxl 28, flux 24, custom: keep")
        self.row(f, "CFG", "comfy", "cfg", float, hint="0 = sdxl 5.5, flux 1.0, custom: keep")
        self.row(f, "FLUX guidance", "comfy", "guidance", float)
        self.row(f, "Sampler", "comfy", "sampler", hint="blank = default / keep")
        self.row(f, "Scheduler", "comfy", "scheduler", hint="blank = default / keep")
        self.row(f, "CLIP skip", "comfy", "clip_skip", int, hint="builtin_sdxl; Illustrious usually 2")
        self.row(f, "Seed", "comfy", "seed", int, hint="0 = random per image")
        self.row(f, "Images per prompt", "comfy", "images_per_prompt", int)
        self.row(f, "File name prefix", "comfy", "filename_prefix", hint="ComfyUI SaveImage prefix")

        self._build_rotation_section(f)

        self.section_label(f, "Run behaviour")
        self.row(f, "When to send", "comfy", "send_mode", choices=core.comfy.SEND_MODES,
                 hint="end = after all prompts (LLM unloaded first, best on one GPU) | each = right away")
        self.row(f, "Unload LLM before images", "comfy", "unload_llm", bool,
                 hint="Ollama/llama-cpp-python; llama-server must be stopped by hand")
        self.row(f, "Wait for images", "comfy", "wait_for_images", bool, hint="OFF = just queue them in ComfyUI")
        self.row(f, "Download images", "comfy", "download_images", bool, hint="copy them into the run folder/images")
        self.row(f, "Free ComfyUI VRAM when done", "comfy", "free_vram_after", bool)
        self.row(f, "Wait for LLM VRAM release (s)", "comfy", "vram_wait", float,
                 hint="after unloading the LLM, wait until the memory is really free (0 = don't wait)")
        self.row(f, "Timeout per image (s)", "comfy", "timeout", float)
        self._apply_comfy_cache(core.comfy.load_model_cache(self.cfg["comfy"], core.APP_DIR))

    def _build_rotation_section(self, f):
        self.section_label(f, "Model rotation (several models in one run)")
        self.row(f, "Rotate models", "comfy", "rotation", choices=core.comfy.ROTATION_MODES,
                 hint="random = N different cached models | list = the models below, in order")
        self.row(f, "Model type to rotate", "comfy", "rotation_target", choices=core.comfy.ROTATION_TARGETS,
                 hint="auto = from the workflow (diffusion model for Krea 2/FLUX, checkpoint for SDXL)")
        self.row(f, "Random: number of models", "comfy", "rotation_count", int,
                 hint="must be <= number of prompts (no limit when comparing); consecutive prompts share a model")
        self.row(f, "Compare models (same prompt on each)", "comfy", "compare_models", bool,
                 hint="ON = every prompt is rendered on EVERY model above, same seed; set Prompts to 1 for a "
                      "test bed. Makes compare.html in the run folder")
        r = f._row
        ctk.CTkLabel(f, text="List: models in order", anchor="w").grid(row=r, column=0, sticky="nw", **PAD)
        self.rot_text = ctk.CTkTextbox(f, height=90)
        self.rot_text.grid(row=r, column=1, sticky="ew", **PAD)
        self.rot_text.insert("end", self.rot_models_var.get())
        self.rot_text.bind("<KeyRelease>", lambda e: self.rot_models_var.set(self.rot_text.get("1.0", "end-1c")))
        bb = ctk.CTkFrame(f, fg_color="transparent")
        bb.grid(row=r, column=2, sticky="n", **PAD)
        ctk.CTkButton(bb, text="Add model...", width=110, command=self._rot_add).pack(pady=(0, 4))
        ctk.CTkButton(bb, text="Clear list", width=110, fg_color="gray40", command=self._rot_clear).pack()
        ctk.CTkLabel(f, text="one model per line; 8 prompts with 4 models = 2 prompts each", text_color=MUTED,
                     font=self.font_small, anchor="w").grid(row=r, column=3, sticky="nw", **PAD)
        f._row += 1
        self.row(f, "Only compatible models", "comfy", "rotation_compatible_only", bool,
                 hint="same architecture as the workflow's model (checked from the model files)")
        self.row(f, "Model search folders", "comfy", "model_search_dirs", browse="dir",
                 hint="where your model files live, separated by ; (for the check)")
        r = f._row
        ctk.CTkButton(f, text="Check models", width=120, command=self._rot_check).grid(
            row=r, column=0, sticky="w", **PAD)
        self.rot_report = ctk.CTkLabel(f, text="", anchor="w", justify="left", wraplength=900)
        self.rot_report.grid(row=r, column=1, columnspan=3, sticky="w", **PAD)
        f._row += 1
        self.rot_models_var.trace_add("write", self._rot_sync_text)

    def _rot_sync_text(self, *_):
        if hasattr(self, "rot_text") and self.rot_text.get("1.0", "end-1c") != self.rot_models_var.get():
            self.rot_text.delete("1.0", "end")
            self.rot_text.insert("end", self.rot_models_var.get())

    def _rot_folder(self) -> str:
        cc = self._comfy_cfg()
        tmpl = None
        if cc.get("workflow") == "custom" and cc.get("workflow_file"):
            try:
                tmpl = core.comfy.load_custom_workflow(cc["workflow_file"])
            except Exception:
                tmpl = None
        return core.comfy.TARGET_FOLDER[core.comfy.rotation_target(cc, tmpl)]

    def _rot_add(self):
        holder = tk.StringVar(self)
        folder = self._rot_folder()

        def picked(*_):
            name = holder.get().strip()
            if name:
                cur = [x for x in self.rot_models_var.get().splitlines() if x.strip()]
                self.rot_models_var.set("\n".join(cur + [name]))
                self.vars[("comfy", "rotation")][0].set("list")
        holder.trace_add("write", picked)
        self.open_picker(holder, f"Add model ({folder})", lambda: [(x, x) for x in self._comfy_lists.get(folder, [])],
                         refresh=lambda: self._comfy_refresh_sync(folder))

    def _rot_clear(self):
        self.rot_models_var.set("")

    def _rot_check(self):
        cfg = self.collect_cfg(quiet=True, validate=False) or self.cfg
        self.rot_report.configure(text="Checking models (reads the model file headers)...", text_color=MUTED)

        def work():
            cc = cfg["comfy"]
            cc["rotation_models"] = [x.strip() for x in self.rot_models_var.get().splitlines() if x.strip()]
            tmpl = core.comfy.load_custom_workflow(cc["workflow_file"]) if (
                cc.get("workflow") == "custom" and cc.get("workflow_file")) else None
            lists = core.comfy.load_model_cache(cc, core.APP_DIR).get("lists", {})
            target = core.comfy.rotation_target(cc, tmpl)
            folder = core.comfy.TARGET_FOLDER[target]
            ref = core.comfy.workflow_model(cc, tmpl, target)
            cands = cc["rotation_models"] if cc.get("rotation") == "list" else list(lists.get(folder, []))
            verdicts = core.comfy.check_compat(cc, ref, cands, folder)
            ok = [m for m, v in verdicts.items() if v[0] == "ok"]
            no = [m for m, v in verdicts.items() if v[0] == "no"]
            unk = [m for m, v in verdicts.items() if v[0] == "unknown"]
            lines = [f"Workflow model ({target.replace('_', ' ')}): {ref or '(not found)'}",
                     f"{len(ok)} compatible, {len(no)} not compatible, {len(unk)} unknown (of {len(cands)})"]
            if cc.get("rotation") == "list":
                lines += [f"  {m}: {v[0]} - {v[1]}" for m, v in verdicts.items()]
            elif no or unk:
                lines.append("  e.g. not compatible: " + ", ".join(no[:3]) if no else "")
            n = cfg["generation"]["count"]
            if cc.get("rotation") != "off":
                try:
                    _, rep = core.comfy.plan_rotation(cc, n, lists, tmpl, random.Random())
                    lines.append(rep)
                except core.comfy.ComfyError as e:
                    lines.append(f"ERROR: {e}")
            return "\n".join(l for l in lines if l)

        def done(text, err):
            self.rot_report.configure(text=f"ERROR: {err}" if err else text,
                                      text_color="#e05555" if err or "ERROR" in (text or "") else MUTED)
        self.run_bg(work, done)

    def _comfy_cfg(self) -> dict:
        cfg = self.collect_cfg(quiet=True, validate=False) or self.cfg
        return cfg["comfy"]

    def _comfy_refresh_sync(self, lst: str) -> list[tuple[str, str]]:
        cache = core.comfy.refresh_model_cache(self._comfy_cfg(), core.APP_DIR)
        self._apply_comfy_cache(cache)
        return [(x, x) for x in self._comfy_lists.get(lst, [])]

    def _apply_comfy_cache(self, cache: dict):
        lists = cache.get("lists", {})
        self._comfy_lists = {k: list(v) for k, v in lists.items()}
        n = sum(len(v) for v in lists.values())
        if hasattr(self, "comfy_cache_label"):
            self.comfy_cache_label.configure(
                text=(f"{n} cached model files from {cache.get('source')} ({cache.get('updated')})" if n
                      else "no cached model list yet - press Refresh"))

    def comfy_test(self):
        self.comfy_status.configure(text="Testing...", text_color=MUTED)

        def done(msg, err):
            self.comfy_status.configure(text=f"ERROR: {err}" if err else msg,
                                        text_color="#e05555" if err else "#3aa655")
        self.run_bg(core.comfy.test_connection, done, self._comfy_cfg())

    def comfy_refresh_models(self):
        self.comfy_cache_label.configure(text="Refreshing...")

        def done(cache, err):
            if err:
                self.comfy_cache_label.configure(text=f"ERROR: {str(err).splitlines()[0]}", text_color="#e05555")
                self.comfy_status.configure(text=str(err), text_color="#e05555")
                return
            self.comfy_cache_label.configure(text_color=MUTED)
            self._apply_comfy_cache(cache)
        self.run_bg(core.comfy.refresh_model_cache, done, self._comfy_cfg(), core.APP_DIR)

    # ------------------------------------------------------------------ Workflow tab (edit the ComfyUI workflow)
    def _build_workflow_tab(self, tab):
        head = ctk.CTkFrame(tab, fg_color="transparent")
        head.pack(fill="x")
        ctk.CTkButton(head, text="Load", width=70, command=self.wf_load).pack(side="left", padx=4)
        ctk.CTkButton(head, text="Save", width=70, command=self.wf_save).pack(side="left", padx=4)
        ctk.CTkButton(head, text="Save As...", width=90, command=self.wf_save_as).pack(side="left", padx=4)
        ctk.CTkButton(head, text="Revert", width=70, fg_color="gray40", command=self.wf_revert).pack(side="left", padx=4)
        self.wf_status = ctk.CTkLabel(head, text="", anchor="w", justify="left", text_color=MUTED, wraplength=900)
        self.wf_status.pack(side="left", padx=10, fill="x", expand=True)
        ctk.CTkLabel(tab, anchor="w", justify="left", text_color=MUTED, wraplength=1100, text=(
            "Edits the workflow file set on the ComfyUI tab. Your original export is never overwritten: Save writes "
            "a copy named ..._pd.json next to it and the app switches to that copy. Unsaved edits are saved "
            "automatically when you press Generate.")).pack(fill="x", padx=6, pady=(2, 4))
        self.wf_body = ctk.CTkScrollableFrame(tab)
        self.wf_body.pack(fill="both", expand=True)
        self.wf_body.grid_columnconfigure(1, weight=1)
        for key in (("comic", "enabled"), ("comic", "layout"), ("comfy", "comic_scale")):
            self.vars[key][0].trace_add("write", self._wf_update_suggestion)
        if self.vars[("comfy", "workflow_file")][0].get().strip():
            self.wf_load(quiet=True)
        else:
            self._wf_note("No workflow file yet: pick one on the ComfyUI tab (Custom workflow file), then press Load.")

    def _wf_note(self, text, error=False):
        self.wf_status.configure(text=text, text_color="#e05555" if error else MUTED)

    def wf_load(self, quiet=False):
        path = self.vars[("comfy", "workflow_file")][0].get().strip()
        if not path:
            self._wf_note("Set the Custom workflow file on the ComfyUI tab first", error=not quiet)
            return
        try:
            self.wf_doc = wfmod.WorkflowDoc.load(path)
        except Exception as e:  # noqa: BLE001
            self.wf_doc = None
            self._wf_note(f"ERROR: {e}", error=True)
            self._wf_render()
            return
        p = Path(path)
        orig = p
        for suffix in ("_pd.json", "_kpg.json"):          # _kpg = copies made by the old app name
            if p.name.endswith(suffix):
                orig = p.with_name(p.name[: -len(suffix)] + ".json")
        self.wf_original = str(orig) if orig.exists() else path
        try:
            self.wf_original_size = wfmod.WorkflowDoc.load(self.wf_original).size()
        except Exception:
            self.wf_original_size = None
        if not self._comfy_lists:
            self._apply_comfy_cache(core.comfy.load_model_cache(self._comfy_cfg(), core.APP_DIR))
        self._wf_render()
        self._wf_note(f"Loaded {path}")

    # ---- building the editor rows
    def _wf_render(self):
        body = self.wf_body
        for w in body.winfo_children():
            w.destroy()
        self.wf_rows = []
        body._row = 0
        doc = self.wf_doc
        if doc is None:
            return
        lists = self._comfy_lists

        def heading(text):
            self.section_label(body, text)

        def entry_row(label, value, pick_list=None, hint=""):
            r = body._row
            ctk.CTkLabel(body, text=label, anchor="w").grid(row=r, column=0, sticky="w", **PAD)
            var = tk.StringVar(self, value="" if value is None else str(value))
            ctk.CTkEntry(body, textvariable=var).grid(row=r, column=1, sticky="ew", **PAD)
            if pick_list is not None:
                ctk.CTkButton(body, text="Choose...", width=80, command=lambda: self.open_picker(
                    var, label, lambda: [(x, x) for x in lists.get(pick_list, [])],
                    refresh=lambda: self._comfy_refresh_sync(pick_list))).grid(row=r, column=2, **PAD)
            if hint:
                ctk.CTkLabel(body, text=hint, text_color=MUTED, anchor="w", font=self.font_small
                             ).grid(row=r, column=3, sticky="w", **PAD)
            body._row += 1
            return var

        heading("Prompt")
        ctk.CTkLabel(body, text=doc.prompt_location(), anchor="w").grid(row=body._row, column=0, columnspan=4,
                                                                       sticky="w", **PAD)
        body._row += 1

        loaders = doc.loaders()
        if loaders:
            heading("Models loaded by the workflow")
            kind = {"ckpt_name": "checkpoints", "unet_name": "diffusion_models", "vae_name": "vae"}
            for nid, field, val in loaders:
                lst = kind.get(field, "text_encoders")
                var = entry_row(f"{doc.title(nid)} {field}", val, lst)
                self.wf_rows.append(("input", (nid, field, str), var))

        heading("LoRAs")
        loras = doc.loras()
        if not loras:
            ctk.CTkLabel(body, text="No LoRAs in this workflow.", text_color=MUTED, anchor="w").grid(
                row=body._row, column=0, columnspan=4, sticky="w", **PAD)
            body._row += 1
        for e in loras:
            r = body._row
            on = tk.BooleanVar(self, value=e["on"])
            ctk.CTkSwitch(body, text=f"{doc.title(e['node'])}" + (f" {e['key']}" if e["key"] else ""), variable=on,
                          onvalue=True, offvalue=False).grid(row=r, column=0, sticky="w", **PAD)
            name = tk.StringVar(self, value=e["name"])
            ctk.CTkEntry(body, textvariable=name).grid(row=r, column=1, sticky="ew", **PAD)
            ctk.CTkButton(body, text="Choose...", width=80, command=lambda v=name: self.open_picker(
                v, "LoRA", lambda: [(x, x) for x in lists.get("loras", [])],
                refresh=lambda: self._comfy_refresh_sync("loras"))).grid(row=r, column=2, **PAD)
            sub = ctk.CTkFrame(body, fg_color="transparent")
            sub.grid(row=r, column=3, sticky="w", **PAD)
            ctk.CTkLabel(sub, text="strength").pack(side="left")
            strength = tk.StringVar(self, value=str(e["strength"]))
            ctk.CTkEntry(sub, textvariable=strength, width=60).pack(side="left", padx=4)
            ctk.CTkButton(sub, text="Remove", width=70, fg_color="#b03a3a", hover_color="#8a2d2d",
                          command=lambda e=e: self._wf_remove_lora(e)).pack(side="left", padx=4)
            body._row += 1
            self.wf_rows.append(("lora", e, (on, name, strength)))
        ctk.CTkButton(body, text="Add LoRA...", width=110, command=self._wf_add_lora).grid(
            row=body._row, column=0, sticky="w", **PAD)
        body._row += 1

        heading("Resolution")
        size = doc.size()
        self.wf_size_vars = None
        if size:
            wv = entry_row("Width (workflow)", size[0])
            hv = entry_row("Height (workflow)", size[1])
            self.wf_rows.append(("size", None, (wv, hv)))
            self.wf_size_vars = (wv, hv)
            r = body._row
            self.wf_sugg_label = ctk.CTkLabel(body, text="", font=self.font_bold, anchor="w")
            self.wf_sugg_label.grid(row=r, column=0, sticky="w", **PAD)
            self.wf_sugg_why = ctk.CTkLabel(body, text="", anchor="w", justify="left", wraplength=700,
                                            text_color=MUTED)
            self.wf_sugg_why.grid(row=r, column=1, columnspan=2, sticky="w", **PAD)
            ctk.CTkButton(body, text="Use suggestion", width=120, command=self._wf_use_suggestion).grid(
                row=r, column=3, sticky="w", **PAD)
            body._row += 1
            self._wf_update_suggestion()
        else:
            ctk.CTkLabel(body, text="No latent size node found in this workflow.", text_color=MUTED).grid(
                row=body._row, column=0, columnspan=4, sticky="w", **PAD)
            body._row += 1
        r = body._row
        ctk.CTkLabel(body, text="Size used when sending", anchor="w").grid(row=r, column=0, sticky="w", **PAD)
        ctk.CTkOptionMenu(body, values=list(core.comfy.SIZE_MODES), variable=self.vars[("comfy", "size_mode")][0]
                          ).grid(row=r, column=1, sticky="w", **PAD)
        ctk.CTkLabel(body, text="auto = comic pages use the app's size x comic scale | workflow = keep the size "
                                "above | custom = ComfyUI tab Width/Height", text_color=MUTED, font=self.font_small,
                     anchor="w").grid(row=r, column=2, columnspan=2, sticky="w", **PAD)
        body._row += 1
        r = body._row
        ctk.CTkLabel(body, text="Comic scale", anchor="w").grid(row=r, column=0, sticky="w", **PAD)
        ctk.CTkEntry(body, textvariable=self.vars[("comfy", "comic_scale")][0], width=80).grid(
            row=r, column=1, sticky="w", **PAD)
        up = doc.upscale_factor()
        if up:
            ctk.CTkButton(body, text=f"Match upscale (x{1 / up[0]:.2f})", width=150, command=lambda: (
                self.vars[("comfy", "comic_scale")][0].set(f"{1 / up[0]:.3f}"),
                self.vars[("comfy", "size_mode")][0].set("auto"))).grid(row=r, column=2, sticky="w", **PAD)
        body._row += 1

        samplers = doc.samplers()
        if samplers:
            heading("Sampling passes")
            for nid in samplers:
                inp = doc.nodes[nid]["inputs"]
                ctk.CTkLabel(body, text=doc.title(nid), font=self.font_bold, anchor="w").grid(
                    row=body._row, column=0, columnspan=4, sticky="w", padx=8, pady=(8, 0))
                body._row += 1
                for f in wfmod.SAMPLER_FIELDS:
                    if f in inp and not isinstance(inp[f], list):
                        typ = int if f == "steps" else (float if f in ("cfg", "denoise") else str)
                        var = entry_row(f"  {f}", inp[f])
                        self.wf_rows.append(("input", (nid, f, typ), var))

    # ---- size suggestion (recomputed whenever comic settings change or the tab is opened)
    def _wf_update_suggestion(self, *_):
        if self.wf_doc is None or not getattr(self, "wf_size_vars", None) or not hasattr(self, "wf_sugg_label"):
            return
        cfg = self.collect_cfg(quiet=True, validate=False) or self.cfg
        sug, why = wfmod.suggest_size(self.wf_doc, cfg, getattr(self, "wf_original_size", None))
        self.wf_sugg = sug
        self.wf_sugg_label.configure(text=f"App suggests: {sug[0]}x{sug[1]}" if sug else "App suggests: -")
        self.wf_sugg_why.configure(text=why)

    def _wf_use_suggestion(self):
        self._wf_update_suggestion()
        sug = getattr(self, "wf_sugg", None)
        if not sug or not self.wf_size_vars:
            return
        wv, hv = self.wf_size_vars
        wv.set(str(sug[0]))
        hv.set(str(sug[1]))
        self.vars[("comfy", "size_mode")][0].set("workflow")
        self._wf_note(f"Workflow size set to {sug[0]}x{sug[1]} (not saved yet; saved automatically on Generate)")

    # ---- applying / saving
    def _wf_apply(self) -> list[str]:
        """Copy the editor fields into the workflow. Returns error messages (bad numbers)."""
        doc, errs = self.wf_doc, []
        if doc is None:
            return errs
        for kind, ref, v in self.wf_rows:
            if kind == "input":
                nid, field, typ = ref
                if nid not in doc.nodes:
                    continue
                raw = v.get().strip()
                try:
                    val = typ(float(raw)) if typ in (int, float) else raw
                except ValueError:
                    errs.append(f"{doc.title(nid)} {field}: '{raw}' is not a number")
                    continue
                if doc.nodes[nid]["inputs"].get(field) != val:
                    doc.set_input(nid, field, val)
            elif kind == "lora":
                on, name, strength = v
                try:
                    st = float(strength.get())
                except ValueError:
                    errs.append(f"LoRA {name.get()}: strength '{strength.get()}' is not a number")
                    continue
                if (on.get(), name.get(), st) != (ref["on"], ref["name"], float(ref["strength"])):
                    doc.set_lora(ref, name=name.get(), strength=st, on=on.get())
                    ref.update(on=on.get(), name=name.get(), strength=st)
            elif kind == "size":
                wv, hv = v
                try:
                    w, h = int(float(wv.get())), int(float(hv.get()))
                except ValueError:
                    errs.append("Width/Height must be numbers")
                    continue
                if doc.size() != (w, h):
                    doc.set_size(w, h)
        return errs

    def _wf_remove_lora(self, entry):
        self._wf_apply()
        self.wf_doc.remove_lora(entry)
        self._wf_render()
        self._wf_note(f"Removed LoRA {entry['name']} (not saved yet)")

    def _wf_add_lora(self):
        holder = tk.StringVar(self)

        def picked(*_):
            name = holder.get()
            if not name:
                return
            self._wf_apply()
            try:
                msg = self.wf_doc.add_lora(name, 1.0)
            except ValueError as e:
                self._wf_note(f"ERROR: {e}", error=True)
                return
            self._wf_render()
            self._wf_note(f"LoRA {name} {msg} (not saved yet)")
        holder.trace_add("write", picked)
        self.open_picker(holder, "Add LoRA", lambda: [(x, x) for x in self._comfy_lists.get("loras", [])],
                         refresh=lambda: self._comfy_refresh_sync("loras"))

    def wf_save(self, path: str | None = None) -> bool:
        if self.wf_doc is None:
            self._wf_note("Nothing loaded", error=True)
            return False
        errs = self._wf_apply()
        if errs:
            self._wf_note("ERROR: " + "; ".join(errs), error=True)
            return False
        target = path or wfmod.WorkflowDoc.edited_copy_path(self.wf_original or self.wf_doc.path)
        try:
            saved = self.wf_doc.save(target)
        except OSError as e:
            self._wf_note(f"ERROR: {e}", error=True)
            return False
        self.vars[("comfy", "workflow_file")][0].set(saved)
        self._wf_note(f"Saved {saved} - the ComfyUI tab now uses this copy")
        return True

    def wf_save_as(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("ComfyUI API workflow", "*.json")])
        if p:
            self.wf_save(p)

    def wf_revert(self):
        if not self.wf_original:
            return
        self.vars[("comfy", "workflow_file")][0].set(self.wf_original)
        self.wf_load()
        self._wf_note(f"Reverted to the original {self.wf_original}")

    def _wf_autosave(self):
        """Before a run: save unsaved editor changes so ComfyUI gets them."""
        if self.wf_doc is None:
            return
        self._wf_apply()
        if self.wf_doc.dirty:
            if self.wf_save():
                self.q.put(("log", "INFO", "Workflow edits saved before the run: " +
                            self.vars[("comfy", "workflow_file")][0].get()))

    # ------------------------------------------------------------------ results / log
    def _build_results_tab(self, tab):
        bar = ctk.CTkFrame(tab, fg_color="transparent")
        bar.pack(fill="x")
        ctk.CTkButton(bar, text="Copy all", width=90, command=self.copy_results).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Clear", width=70, fg_color="gray40",
                      command=lambda: self._set_text(self.results, "")).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Open output folder", width=140, command=self.open_output).pack(side="left", padx=4)
        self.show_scene = tk.BooleanVar(self, value=True)
        ctk.CTkCheckBox(bar, text="Show scene details above each prompt", variable=self.show_scene
                        ).pack(side="left", padx=12)
        self.results = ctk.CTkTextbox(tab, wrap="word", font=self.font_text)
        self.results.pack(fill="both", expand=True, pady=(6, 0))

    def _build_log_tab(self, tab):
        bar = ctk.CTkFrame(tab, fg_color="transparent")
        bar.pack(fill="x")
        ctk.CTkButton(bar, text="Clear view", width=90, fg_color="gray40",
                      command=lambda: self._set_text(self.logbox, "")).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Open log folder", width=120, command=self.open_logs).pack(side="left", padx=4)
        ctk.CTkLabel(bar, text="Full details are in the log files; enable the debug log on 'Output & Logging'.",
                     text_color=MUTED).pack(side="left", padx=10)
        self.logbox = ctk.CTkTextbox(tab, wrap="word", font=self.font_mono)
        self.logbox.pack(fill="both", expand=True, pady=(6, 0))
        tb = self.logbox._textbox  # underlying tk.Text for colored tags
        tb.tag_config("WARNING", foreground="#d4a017")
        tb.tag_config("ERROR", foreground="#e05555")
        tb.tag_config("CRITICAL", foreground="#ff3333")

    def _set_text(self, box: ctk.CTkTextbox, text: str):
        box.delete("1.0", "end")
        if text:
            box.insert("end", text)

    def copy_results(self):
        self.clipboard_clear()
        self.clipboard_append(self.results.get("1.0", "end-1c"))
        self.status.configure(text="Results copied to clipboard")

    # ------------------------------------------------------------------ backend tab
    def _build_backend_tab(self, tab):
        f = self.form(tab)
        self.section_label(f, "Backend")
        self.row(f, "Backend type", "backend", "type", choices=core.VALID_BACKENDS,
                 hint="ollama | llamacpp = llama-server | openai = LM Studio/vLLM | llamacpp_python = load .gguf")
        self.vars[("backend", "type")][0].trace_add("write", lambda *_: self._backend_changed())
        self.row(f, "Host URL", "backend", "host", hint="empty = default for the backend type")
        self.host_hint = ctk.CTkLabel(f, text="", text_color=MUTED, anchor="w")
        self.host_hint.grid(row=f._row, column=1, sticky="w", padx=8)
        f._row += 1
        self.row(f, "API key", "backend", "api_key", secret=True, hint="only if your server requires one")
        self.row(f, "Timeout (s)", "backend", "timeout", float)
        self.row(f, "Max retries", "backend", "max_retries", int)
        self.row(f, "Retry backoff (s)", "backend", "retry_backoff", float, hint="doubles after each retry")
        self.row(f, "Verify before run", "backend", "verify_connection", bool)

        self.section_label(f, "Model")
        self.model_entry = self.row(f, "Model name", "model", "name",
                                    pick=lambda: ([(m, m) for m in self._ollama_models] if self._ollama_models
                                                  else self._list_models_sync()),
                                    pick_kw={"refresh": self._list_models_sync},
                                    hint="Ollama tag or server model id - Choose... lists them")
        self.gguf_entry = self.row(f, "Model file (.gguf)", "model", "path", pick=self._gguf_items,
                                   pick_kw={"refresh": self._gguf_items, "browse": "gguf"},
                                   hint="managed llama-server / llamacpp_python")
        self.row(f, "Context size (n_ctx)", "model", "n_ctx", int)
        self.row(f, "GPU layers", "model", "n_gpu_layers", int, hint="llamacpp_python; -1 = all")
        self.row(f, "Temperature", "model", "temperature", float, hint="0-2; higher = more varied")
        self.row(f, "Top-p", "model", "top_p", float)
        self.row(f, "Max tokens", "model", "max_tokens", int, hint="per prompt")
        self.row(f, "Keep alive", "model", "keep_alive", hint="ollama only, e.g. 10m")
        self.row(f, "Thinking (reasoning models)", "model", "thinking", choices=("off", "on", "auto"),
                 hint="Qwen3 / DeepSeek-R1 etc.: off = answer directly (recommended); on needs a much larger Max tokens")

        bar = ctk.CTkFrame(f, fg_color="transparent")
        bar.grid(row=f._row, column=0, columnspan=4, sticky="w", padx=8, pady=12)
        ctk.CTkButton(bar, text="Test connection", command=self.test_connection).pack(side="left")
        ctk.CTkButton(bar, text="Refresh model list", command=self.refresh_models).pack(side="left", padx=8)
        self._build_server_section(f)
        self.test_label = ctk.CTkLabel(bar, text="", wraplength=700, justify="left")
        self.test_label.pack(side="left", padx=12)

    def _build_server_section(self, f):
        f._row += 1
        self.section_label(f, "Managed llama-server (the app starts and stops llama.cpp)")
        self.row(f, "Let the app run llama-server", "llama_server", "managed", bool,
                 hint="backend type 'llamacpp': starts it with the GGUF above, stops it before ComfyUI / on exit")
        self.row(f, "llama.cpp folder", "llama_server", "llama_cpp_dir", browse="dir",
                 hint="your build; llama-server is found in build/bin, bin or the folder itself")
        self.row(f, "GGUF folder", "llama_server", "gguf_dir", browse="dir", hint="fills the Model file picker")
        self.row(f, "GPU (CUDA_VISIBLE_DEVICES)", "llama_server", "gpu", hint="0 = your 5060 Ti; blank = all")
        self.row(f, "GPU layers (-ngl)", "llama_server", "n_gpu_layers", int)
        self.row(f, "Flash attention (-fa)", "llama_server", "flash_attn", choices=("on", "off", "auto", ""),
                 hint="blank = don't pass -fa (older builds)")
        self.row(f, "Parallel slots (-np)", "llama_server", "parallel", int, hint="1 = one request gets the whole -c")
        self.row(f, "KV cache type", "llama_server", "cache_type", combo=["", "q8_0", "q4_0", "f16"],
                 hint="q8_0 saves VRAM (needs flash attention)")
        self.row(f, "Extra arguments", "llama_server", "extra_args", hint="e.g. --threads 8 --mlock")
        self.row(f, "Startup timeout (s)", "llama_server", "startup_timeout", int)
        self.row(f, "Stop it when the app closes", "llama_server", "stop_when_app_closes", bool)
        bar = ctk.CTkFrame(f, fg_color="transparent")
        bar.grid(row=f._row, column=0, columnspan=4, sticky="w", padx=8, pady=8)
        ctk.CTkButton(bar, text="Start server", width=110, command=self.server_start).pack(side="left")
        ctk.CTkButton(bar, text="Stop server", width=110, fg_color="#b03a3a", hover_color="#8a2d2d",
                      command=self.server_stop).pack(side="left", padx=8)
        self.server_label = ctk.CTkLabel(bar, text=f"llama-server: {lserver.MANAGER.status()}", anchor="w",
                                         justify="left", wraplength=750)
        self.server_label.pack(side="left", padx=8)
        f._row += 1
        self.vars[("llama_server", "gguf_dir")][0].trace_add("write", lambda *_: self._refresh_ggufs())
        self._refresh_ggufs()

    def _refresh_ggufs(self):
        pass                                   # the picker scans the GGUF folder when opened

    def _gguf_items(self) -> list[tuple[str, str]]:
        root = self.vars[("llama_server", "gguf_dir")][0].get()
        out = []
        for full in lserver.list_ggufs(root):
            try:
                disp = str(Path(full).relative_to(Path(root).expanduser()))
            except ValueError:
                disp = full
            out.append((disp, full))
        return out

    def _list_models_sync(self) -> list[tuple[str, str]]:
        cfg = self.collect_cfg(quiet=True, validate=False) or self.cfg
        models = core.list_models(cfg)
        self._ollama_models = list(models)
        return [(m, m) for m in models]

    def server_start(self):
        cfg = self.collect_cfg()
        if cfg is None:
            return
        if cfg["backend"]["type"] != "llamacpp":
            self.server_label.configure(text="Set Backend type to 'llamacpp' first", text_color="#e05555")
            return
        self.server_label.configure(text="Starting llama-server (loading the model)...", text_color=MUTED)
        log_dir = core.output_root(cfg) / (cfg["logging"]["directory"] or "logs")

        def start():
            core.validate_config(cfg)
            return lserver.MANAGER.ensure_running(cfg, log_dir)

        def done(msg, err):
            self.server_label.configure(text=f"ERROR: {err}" if err else msg,
                                        text_color="#e05555" if err else "#3aa655")
        self.run_bg(start, done)

    def server_stop(self):
        stopped = lserver.MANAGER.stop()
        self.server_label.configure(text="llama-server stopped (VRAM freed)" if stopped else "llama-server was not running",
                                    text_color=MUTED)

    def _backend_changed(self):
        if not hasattr(self, "host_hint"):
            return
        t = self.vars[("backend", "type")][0].get()
        d = core.DEFAULT_HOSTS.get(t, "")
        self.host_hint.configure(text=f"default: {d}" if d else "no server: the model loads inside this app")

    def refresh_models(self):
        cfg = self.collect_cfg(quiet=True)
        if cfg is None:
            return
        self.test_label.configure(text="Listing models…")

        def done(models, err):
            if err:
                self.test_label.configure(text=f"ERROR: {err}", text_color="#e05555")
                return
            self._ollama_models = list(models)
            self.test_label.configure(text=f"{len(models)} model(s) found", text_color=MUTED)
        self.run_bg(core.list_models, done, cfg)

    def test_connection(self):
        cfg = self.collect_cfg()
        if cfg is None:
            return
        self.test_label.configure(text="Testing… (first load of a model can take a while)", text_color=MUTED)

        def done(msg, err):
            if err:
                self.test_label.configure(text=f"ERROR: {err}", text_color="#e05555")
            else:
                self.test_label.configure(text=f"{msg}", text_color="#3aa655")
        self.run_bg(core.test_backend, done, cfg)

    # ------------------------------------------------------------------ generation tab
    def _build_generation_tab(self, tab):
        f = self.form(tab)
        self.section_label(f, "Target image models")
        r = f._row
        ctk.CTkLabel(f, text="Prompts are for", anchor="w").grid(row=r, column=0, sticky="w", **PAD)
        fr = ctk.CTkFrame(f, fg_color="transparent")
        fr.grid(row=r, column=1, columnspan=3, sticky="w", **PAD)
        names = {"krea2": "Krea 2 (150-300 words)", "flux": "FLUX (100-220 words)",
                 "illustrious": "Illustrious (shorter + tag line)"}
        for t in core.VALID_TARGETS:
            v = self.target_vars[t]
            ctk.CTkCheckBox(fr, text=names[t], variable=v).pack(side="left", padx=(0, 14))
        f._row += 1
        self.row(f, "Min words", "generation", "min_words", int, hint="0 = use the targets' range")
        self.row(f, "Max words", "generation", "max_words", int, hint="0 = use the targets' range")
        self.row(f, "Illustrious tag prefix", "generation", "tag_prefix", hint="put in front of every tag line")
        self.section_label(f, "Prompts")
        self.row(f, "Default count", "generation", "count", int, hint="also set in the bottom bar")
        self.row(f, "Seed mode", "generation", "seed_mode", choices=core.VALID_SEED_MODES,
                 hint="builtin = fields | llm = LLM writes a scene idea first | file = scenes file")
        self.row(f, "Scenes file", "generation", "scenes_file", browse="file", hint="one idea per line (file mode)")
        self.row(f, "Content rating", "generation", "content_rating", choices=("as_file", "sfw"),
                 hint="sfw strips the NSFW rule from the system prompt")
        self.row(f, "Random seed", "generation", "random_seed", int, hint="0 = different each run")
        self.row(f, "User message template", "generation", "user_template", hint="must contain {scene}")
        self.section_label(f, "Files")
        self.row(f, "System prompt file", "generation", "system_prompt_file", browse="file")
        self.row(f, "Presets file", "generation", "presets_file", browse="json")
        self.section_label(f, "Quality checks")
        self.row(f, "Min paragraphs", "generation", "min_paragraphs", int)
        self.row(f, "Attempts per prompt", "generation", "max_attempts_per_prompt", int,
                 hint="re-asks the LLM when a reply fails the checks")
        self.row(f, "Reject duplicates", "generation", "dedupe", bool)
        self.row(f, "Strip <think> blocks", "generation", "strip_reasoning", bool)
        self.row(f, "Clean output", "generation", "clean_output", bool,
                 hint="removes preambles, labels, bullets, code fences")
        self.row(f, "Filler check", "generation", "filler_check", bool,
                 hint="re-ask when the prompt describes mood/feelings instead of visible things")
        self.row(f, "Max filler phrases", "generation", "max_filler_hits", int)
        self.row(f, "Compatibility rules", "generation", "compat_rules", bool,
                 hint="random picks avoid contradictions (studio light outdoors, close-up of a group, ...)")

    # ------------------------------------------------------------------ output tab
    def _build_output_tab(self, tab):
        f = self.form(tab)
        self.section_label(f, "Output")
        self.row(f, "Output folder", "output", "directory", browse="dir", hint="empty = current folder")
        r = f._row
        ctk.CTkLabel(f, text="Formats", anchor="w").grid(row=r, column=0, sticky="w", **PAD)
        fr = ctk.CTkFrame(f, fg_color="transparent")
        fr.grid(row=r, column=1, columnspan=3, sticky="w", **PAD)
        tips = {"txt": "txt (all prompts)", "lines": "lines (one per line)", "jsonl": "jsonl (+details)",
                "separate": "separate files", "tags": "Illustrious tags (auto with that target)"}
        for fmt in core.VALID_FORMATS:
            v = self.format_vars[fmt]
            ctk.CTkCheckBox(fr, text=tips[fmt], variable=v).pack(side="left", padx=(0, 14))
        f._row += 1
        self.row(f, "Filename prefix", "output", "filename_prefix")
        self.row(f, "Timestamped run folder", "output", "timestamp_subfolder", bool)
        self.section_label(f, "Logging")
        self.row(f, "Log folder", "logging", "directory", browse="dir", hint="relative = inside output folder")
        self.row(f, "Log file", "logging", "log_file", hint="always written")
        self.row(f, "Debug log", "logging", "debug", bool, hint="separate file with full requests/responses")
        self.row(f, "Debug log file", "logging", "debug_file")
        self.row(f, "Console level", "logging", "console_level", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
        self.row(f, "Rotate at (bytes)", "logging", "max_bytes", int)
        self.row(f, "Backups kept", "logging", "backup_count", int)

    # ------------------------------------------------------------------ presets tab
    def _build_presets_tab(self, tab):
        bar = ctk.CTkFrame(tab, fg_color="transparent")
        bar.pack(fill="x")
        ctk.CTkLabel(bar, text="Random list:").pack(side="left", padx=4)
        self.preset_key = tk.StringVar(self, value="style")
        self.preset_menu = ctk.CTkOptionMenu(bar, values=list(self.presets), variable=self.preset_key,
                                             command=self._preset_switch)
        self.preset_menu.pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Save presets", width=110, command=self.save_presets).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Reset this list", width=110, fg_color="gray40",
                      command=self.reset_preset).pack(side="left", padx=4)
        ctk.CTkLabel(tab, text=("One value per line. Repeat a line to make it more likely. 'action_group' is only "
                                "used with 2+ characters. Edits apply to the next run right away; "
                                "'Save presets' writes them to the presets file."),
                     text_color=MUTED, anchor="w", justify="left", wraplength=1100).pack(fill="x", padx=6, pady=4)
        self.preset_box = ctk.CTkTextbox(tab, wrap="none")
        self.preset_box.pack(fill="both", expand=True)
        self._preset_current = "style"
        self._set_text(self.preset_box, "\n".join(self.presets.get("style", [])))

    def _store_preset_box(self):
        lines = [ln.strip() for ln in self.preset_box.get("1.0", "end-1c").splitlines() if ln.strip()]
        self.presets[self._preset_current] = lines

    def _preset_switch(self, key):
        self._store_preset_box()
        self._preset_current = key
        self._set_text(self.preset_box, "\n".join(self.presets.get(key, [])))

    def reset_preset(self):
        key = self._preset_current
        self.presets[key] = list(core.DEFAULT_PRESETS.get(key, []))
        self._set_text(self.preset_box, "\n".join(self.presets[key]))

    def save_presets(self):
        self._store_preset_box()
        cfg = self.collect_cfg(quiet=True, validate=False) or self.cfg
        try:
            p = core.save_presets(self.presets, cfg)
            self.refresh_field_choices()
            self.status.configure(text=f"Presets saved to {p}")
        except OSError as e:
            messagebox.showerror(APP_TITLE, f"Could not save presets:\n{e}")

    # ------------------------------------------------------------------ system prompt tab
    def _build_sysprompt_tab(self, tab):
        bar = ctk.CTkFrame(tab, fg_color="transparent")
        bar.pack(fill="x")
        ctk.CTkButton(bar, text="Reload from file", width=130, command=self.reload_sysprompt).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Save to file", width=110, command=self.save_sysprompt).pack(side="left", padx=4)
        self.sp_label = ctk.CTkLabel(bar, text="", text_color=MUTED)
        self.sp_label.pack(side="left", padx=10)
        ctk.CTkLabel(tab, text="The text below is what gets sent to the LLM on the next run (saved or not).",
                     text_color=MUTED, anchor="w").pack(fill="x", padx=6, pady=4)
        self.sp_box = ctk.CTkTextbox(tab, wrap="word", font=self.font_mono)
        self.sp_box.pack(fill="both", expand=True)
        self._sp_loaded = ""

    def reload_sysprompt(self):
        cfg = self.collect_cfg(quiet=True, validate=False) or self.cfg
        try:
            text = core.read_system_prompt_file(cfg)
            self._sp_loaded = text
            self._set_text(self.sp_box, text)
            self.sp_label.configure(text=str(core.resolve_path(cfg["generation"]["system_prompt_file"])))
        except core.PDError as e:
            self.sp_label.configure(text=f"ERROR: {e}")

    def save_sysprompt(self):
        cfg = self.collect_cfg(quiet=True, validate=False) or self.cfg
        p = core.resolve_path(cfg["generation"]["system_prompt_file"])
        try:
            text = self.sp_box.get("1.0", "end-1c")
            p.write_text(text, encoding="utf-8")
            self._sp_loaded = text
            self.sp_label.configure(text=f"Saved {p}")
        except OSError as e:
            messagebox.showerror(APP_TITLE, f"Could not save:\n{e}")

    # ------------------------------------------------------------------ run bar
    def _build_runbar(self):
        bar = ctk.CTkFrame(self)
        bar.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkLabel(bar, text="Prompts:").pack(side="left", padx=(10, 4), pady=8)
        self.count_var = tk.StringVar(self)
        ctk.CTkEntry(bar, textvariable=self.count_var, width=60).pack(side="left")
        self.gen_btn = ctk.CTkButton(bar, text="Generate", width=120, command=self.start_generation,
                                     fg_color="#2e8b57", hover_color="#246b44")
        self.gen_btn.pack(side="left", padx=(10, 4))
        self.stop_btn = ctk.CTkButton(bar, text="Stop", width=80, command=self.stop_generation,
                                      fg_color="#b03a3a", hover_color="#8a2d2d", state="disabled")
        self.stop_btn.pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Preview scenes", width=120, fg_color="gray40",
                      command=self.preview).pack(side="left", padx=4)
        self.progress = ctk.CTkProgressBar(bar, width=220)
        self.progress.set(0)
        self.progress.pack(side="left", padx=12)
        self.status = ctk.CTkLabel(bar, text="Ready", anchor="w")
        self.status.pack(side="left", fill="x", expand=True, padx=6)
        return bar

    # ------------------------------------------------------------------ config <-> widgets
    def load_into_widgets(self, cfg: dict):
        for (sect, key), (var, kind) in self.vars.items():
            d = cfg
            for part in sect.split("."):
                d = d.get(part, {})
            val = d.get(key, "" if kind is not bool else False)
            var.set(bool(val) if kind is bool else ("" if val is None else str(val)))
        for fmt, v in self.format_vars.items():
            v.set(fmt in cfg["output"]["formats"])
        for t, v in self.target_vars.items():
            v.set(t in cfg["generation"]["targets"])
        self.rot_models_var.set("\n".join(cfg["comfy"].get("rotation_models") or []))
        people = cfg["scene"].get("people") or []
        for i, row in enumerate(self.people_vars):
            p = people[i] if i < len(people) and isinstance(people[i], dict) else {}
            for k, v in row.items():
                v.set(str(p.get(k, "") or ""))
        self.count_var.set(str(cfg["generation"]["count"]))
        self._show_cfg_path()
        self._backend_changed()
        self._update_comic_info()
        self.reload_sysprompt()

    def _show_cfg_path(self):
        p = str(self.config_path)
        if len(p) > 60:
            p = "..." + p[-57:]
        self.cfg_label.configure(text=p + ("" if self.config_path.exists() else "  (not saved yet)"))

    def collect_cfg(self, quiet: bool = False, validate: bool = True) -> dict | None:
        """Settings as shown on screen. validate=False skips the cross-setting checks, so helper buttons
        (Test, Refresh, Open folder...) always use what is on screen even while other settings are incomplete."""
        cfg = copy.deepcopy(core.DEFAULT_CONFIG)
        cfg = core.deep_merge(cfg, {k: v for k, v in self.cfg.items() if k in cfg})  # keep non-GUI keys
        errors = []
        for (sect, key), (var, kind) in self.vars.items():
            d = cfg
            for part in sect.split("."):
                d = d.setdefault(part, {})
            raw = var.get()
            if kind is bool:
                d[key] = bool(raw)
            elif kind in (int, float):
                txt = str(raw).strip()
                try:
                    d[key] = kind(txt)
                except ValueError:
                    try:          # e.g. a decimal in a field whose default is a whole number
                        num = float(txt)
                        d[key] = int(num) if num.is_integer() else num
                    except ValueError:
                        errors.append(f"{sect}.{key}: '{raw}' is not a number")
            else:
                d[key] = str(raw)
        cfg["output"]["formats"] = [f for f, v in self.format_vars.items() if v.get()]
        cfg["generation"]["targets"] = [t for t, v in self.target_vars.items() if v.get()]
        cfg["scene"]["people"] = [{k: v.get().strip() for k, v in row.items()} for row in self.people_vars]
        cfg["comfy"]["rotation_models"] = [x.strip() for x in self.rot_models_var.get().splitlines() if x.strip()]
        try:
            cfg["generation"]["count"] = int(self.count_var.get().strip())
        except ValueError:
            errors.append(f"Prompts count '{self.count_var.get()}' is not a number")
        if errors:
            if not quiet:
                messagebox.showerror(APP_TITLE, "Please fix:\n\n" + "\n".join(errors))
            return None
        if not validate:
            return cfg
        try:
            core.validate_config(copy.deepcopy(cfg))
        except core.ConfigError as e:
            if not quiet:
                messagebox.showerror(APP_TITLE, str(e))
            return None
        return cfg

    def save_config(self, path: Path | None = None) -> bool:
        cfg = self.collect_cfg()
        if cfg is None:
            return False
        path = Path(path or self.config_path)
        try:
            core.save_config(cfg, path)
        except OSError as e:
            messagebox.showerror(APP_TITLE, f"Could not save config:\n{e}")
            return False
        self.cfg, self.config_path, self.saved_snapshot = cfg, path, copy.deepcopy(cfg)
        self._show_cfg_path()
        self.status.configure(text=f"Config saved to {path}")
        return True

    def save_config_as(self):
        p = filedialog.asksaveasfilename(defaultextension=".toml", initialfile=self.config_path.name,
                                         filetypes=[("TOML", "*.toml"), ("JSON", "*.json")])
        if p:
            self.save_config(Path(p))

    def load_config_dialog(self):
        p = filedialog.askopenfilename(filetypes=[("Config", "*.toml *.json"), ("All files", "*.*")])
        if not p:
            return
        try:
            cfg, path = core.load_config(p)
        except core.PDError as e:
            messagebox.showerror(APP_TITLE, str(e))
            return
        self.cfg, self.config_path, self.saved_snapshot = cfg, Path(path), copy.deepcopy(cfg)
        self.presets = core.load_presets(cfg)
        self.refresh_field_choices()
        self.preset_menu.configure(values=list(self.presets))
        self._preset_current = self.preset_key.get()
        self._set_text(self.preset_box, "\n".join(self.presets.get(self._preset_current, [])))
        self.load_into_widgets(cfg)
        self.status.configure(text=f"Loaded {path}")

    def reset_defaults(self):
        if messagebox.askyesno(APP_TITLE, "Reset every setting and field to the defaults?\n"
                                          "(Your config file is not changed until you Save.)"):
            self.cfg = copy.deepcopy(core.DEFAULT_CONFIG)
            self.load_into_widgets(self.cfg)

    # ------------------------------------------------------------------ generation
    def _scene_ideas(self) -> list[str]:
        return [ln.strip() for ln in self.scene_ideas.get("1.0", "end-1c").splitlines() if ln.strip()]

    def preview(self):
        cfg = self.collect_cfg()
        if cfg is None:
            return
        self._store_preset_box()
        try:
            seeds = core.preview_scenes(cfg, min(cfg["generation"]["count"], 20), self._scene_ideas(), self.presets)
        except core.PDError as e:
            messagebox.showerror(APP_TITLE, str(e))
            return
        text = "\n\n".join(f"----- Scene {i} -----\n{s.text}" for i, s in enumerate(seeds, 1))
        self._set_text(self.results, "SCENE PREVIEW (no LLM calls; random values change every time)\n\n" + text + "\n")
        self.tabs.set("Results")

    def start_generation(self):
        if self.worker and self.worker.is_alive():
            return
        self._wf_autosave()
        cfg = self.collect_cfg()
        if cfg is None:
            return
        self._store_preset_box()
        sp_text = self.sp_box.get("1.0", "end-1c")
        ideas = self._scene_ideas()
        self.stop_event = threading.Event()
        self._set_text(self.results, "")
        self.progress.set(0)
        self.gen_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status.configure(text="Starting…")
        self.tabs.set("Results")
        handler = QueueLogHandler(self.q)
        presets = copy.deepcopy(self.presets)

        def work():
            try:
                res = core.run_job(
                    cfg, fixed_scenes=ideas, system_prompt_text=sp_text, stop_event=self.stop_event,
                    presets=presets, log_handlers=[handler], console_log=False,
                    on_prompt=lambda i, n, p, seed, rec: self.q.put(("prompt", i, n, p, seed, rec)),
                    on_progress=lambda i, n, msg: self.q.put(("progress", i, n, msg)),
                    on_comfy=lambda i, info: self.q.put(("comfy", i, info)),
                )
                self.q.put(("done", res, None))
            except Exception as e:  # noqa: BLE001
                core.log.exception("Run failed")
                self.q.put(("done", None, e))

        self.worker = threading.Thread(target=work, daemon=True, name="generator")
        self.worker.start()

    def stop_generation(self):
        self.stop_event.set()
        self.status.configure(text="Stopping after the current request…")
        self.stop_btn.configure(state="disabled")

    def _poll_queue(self):
        try:
            while True:
                item = self.q.get_nowait()
                kind = item[0]
                if kind == "log":
                    _, level, msg = item
                    self.logbox.insert("end", msg + "\n", level if level in ("WARNING", "ERROR", "CRITICAL") else None)
                    self.logbox.see("end")
                elif kind == "prompt":
                    _, i, n, text, seed, rec = item
                    head = f"========== Prompt {i}/{n} ({rec.get('words', '?')} words) ==========\n"
                    if self.show_scene.get():
                        head += "[" + seed.text.replace("\n", " | ") + "]\n\n"
                    tail = f"\n\n[Illustrious tags]\n{rec['tags']}" if rec.get("tags") else ""
                    if rec.get("comic"):
                        head = head.replace(f"({rec.get('words', '?')} words)",
                                            f"(comic {rec['comic']['layout']}, generate at {rec['comic']['suggested_size']})")
                    if rec.get("problems"):
                        tail += f"\n[saved with issues: {'; '.join(rec['problems'])}]"
                    self.results.insert("end", head + text + tail + "\n\n")
                    self.results.see("end")
                elif kind == "comfy":
                    _, i, info = item
                    if info.get("error"):
                        line = f"[ComfyUI] prompt {i}: ERROR {info['error'].splitlines()[0]}\n"
                    else:
                        imgs = ", ".join(info.get("images") or []) or "queued"
                        line = f"[ComfyUI] prompt {i}: {imgs} (seed {info.get('seed')}, {info.get('sent')})\n"
                    self.results.insert("end", line)
                    self.results.see("end")
                elif kind == "progress":
                    _, i, n, msg = item
                    self.progress.set(i / max(n, 1))
                    self.status.configure(text=msg)
                elif kind == "done":
                    self._finish(item[1], item[2])
                elif kind == "call":
                    _, cb, result, err = item
                    cb(result, err)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _finish(self, res: core.JobResult | None, err: Exception | None):
        self.gen_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        if err:
            self.status.configure(text="FAILED - see the Log tab")
            messagebox.showerror(APP_TITLE, f"Generation failed:\n\n{err}")
            return
        self.last_result = res
        self.progress.set(1 if not res.stopped else self.progress.get())
        comfy_txt = (f" | ComfyUI {res.comfy_sent} sent, {res.comfy_failed} failed" if res.comfy_sent or res.comfy_failed
                     else "")
        self.status.configure(text=(f"{'Stopped' if res.stopped else 'Done'}: {res.ok} ok, {res.flagged} with issues, "
                                    f"{res.failed} failed — {res.total_seconds or res.wall_seconds:.0f}s — saved to {res.run_dir}{comfy_txt}"))

    def open_output(self):
        if self.last_result and self.last_result.run_dir:
            open_path(self.last_result.run_dir)
        else:
            cfg = self.collect_cfg(quiet=True, validate=False) or self.cfg
            open_path(core.output_root(cfg))

    def open_logs(self):
        if self.last_result and self.last_result.main_log:
            open_path(self.last_result.main_log.parent)
        else:
            cfg = self.collect_cfg(quiet=True, validate=False) or self.cfg
            d = Path(cfg["logging"]["directory"]).expanduser()
            open_path(d if d.is_absolute() else core.output_root(cfg) / d)

    def on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno(APP_TITLE, "A run is in progress. Stop it and quit?"):
                return
        if lserver.MANAGER.running():
            if self.vars[("llama_server", "stop_when_app_closes")][0].get():
                lserver.MANAGER.stop()
            else:
                lserver.MANAGER.proc = None      # leave it running after the app exits
            self.stop_event.set()
        cfg = self.collect_cfg(quiet=True)
        if cfg is not None and cfg != self.saved_snapshot:
            ans = messagebox.askyesnocancel(APP_TITLE, f"Save your settings to {self.config_path}?")
            if ans is None:
                return
            if ans and not self.save_config():
                return
        self.destroy()


def main():
    ap = argparse.ArgumentParser(description="PromptDirector Studio")
    ap.add_argument("-c", "--config", help="config file to open")
    a = ap.parse_args()
    setup_private_fontconfig()          # must run before Tk opens any font
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    App(a.config).mainloop()


if __name__ == "__main__":
    main()
