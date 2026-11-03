"""
pd_picker.py — searchable list picker for long lists (thousands of models).

Replaces drop-down menus, which close on mouse release and can't scroll comfortably through big lists.
The window stays open until something is chosen (Enter / double-click / Choose) or it is cancelled (Esc).
Typing filters the list immediately; every space-separated word must match, in any order, case-insensitive.
"""
from __future__ import annotations

import tkinter as tk
from typing import Callable

try:
    import customtkinter as ctk
except ImportError:  # pragma: no cover
    ctk = None


def filter_items(items: list[tuple[str, str]], query: str) -> list[tuple[str, str]]:
    words = query.lower().split()
    if not words:
        return list(items)
    return [it for it in items if all(w in it[0].lower() for w in words)]


class SearchPicker(tk.Toplevel):
    def __init__(self, master, title: str, items: list[tuple[str, str]], current: str,
                 on_pick: Callable[[str], None], *, refresh: Callable[[], list[tuple[str, str]]] | None = None,
                 browse: Callable[[], str | None] | None = None, clear_label: str | None = None,
                 font=None):
        super().__init__(master)
        self.title(title)
        self.transient(master)
        self.items = list(items)
        self.shown: list[tuple[str, str]] = []
        self.on_pick = on_pick
        self.refresh_fn = refresh
        self.browse_fn = browse
        self._after_id = None
        dark = (ctk.get_appearance_mode() == "Dark") if ctk else True
        bg, fg, sel = ("#242424", "#dce4ee", "#1f6aa5") if dark else ("#f2f2f2", "#1a1a1a", "#3b8ed0")
        self.configure(bg=bg)

        top = tk.Frame(self, bg=bg)
        top.pack(fill="x", padx=10, pady=(10, 4))
        self.query = tk.StringVar(self)
        Entry = ctk.CTkEntry if ctk else tk.Entry
        kw = {"placeholder_text": "type to search, e.g.  krea fp8"} if ctk else {}
        self.entry = Entry(top, **kw)
        self.entry.pack(side="left", fill="x", expand=True)
        self.count = tk.Label(top, text="", bg=bg, fg=fg)
        self.count.pack(side="left", padx=10)

        mid = tk.Frame(self, bg=bg)
        mid.pack(fill="both", expand=True, padx=10, pady=4)
        self.listbox = tk.Listbox(mid, activestyle="none", exportselection=False, bg=bg, fg=fg,
                                  selectbackground=sel, selectforeground="#ffffff", highlightthickness=0,
                                  borderwidth=0, font=font)
        sb = tk.Scrollbar(mid, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        bottom = tk.Frame(self, bg=bg)
        bottom.pack(fill="x", padx=10, pady=(4, 10))
        self.msg = tk.Label(bottom, text="", bg=bg, fg="#e05555", anchor="w")
        self.msg.pack(side="left", fill="x", expand=True)
        Button = ctk.CTkButton if ctk else tk.Button

        def btn(text, cmd, **extra):
            b = Button(bottom, text=text, command=cmd, **({"width": 90, **extra} if ctk else {}))
            b.pack(side="right", padx=4)
            return b
        btn("Cancel", self.cancel, fg_color="gray40")
        btn("Choose", self.choose)
        if clear_label:
            btn(clear_label, lambda: self._finish(""), width=150, fg_color="gray40")
        if browse:
            btn("Browse...", self._browse)
        if refresh:
            btn("Refresh", self._refresh)

        # typing filters; keys navigate without leaving the search box
        self.entry.bind("<KeyRelease>", self._on_type)
        for w in (self.entry, self.listbox):
            w.bind("<Return>", lambda e: self.choose())
            w.bind("<KP_Enter>", lambda e: self.choose())
            w.bind("<Escape>", lambda e: self.cancel())
        self.entry.bind("<Down>", lambda e: self._move(1))
        self.entry.bind("<Up>", lambda e: self._move(-1))
        self.entry.bind("<Next>", lambda e: self._move(15))
        self.entry.bind("<Prior>", lambda e: self._move(-15))
        self.listbox.bind("<Double-Button-1>", lambda e: self.choose())
        self.protocol("WM_DELETE_WINDOW", self.cancel)

        self._apply_filter(select=current)
        self._place(master)
        self.after(60, self._grab)

    # ------------------------------------------------------------------ window
    def _place(self, master):
        try:
            master.update_idletasks()
            w, h = 820, 560
            x = master.winfo_rootx() + max(0, (master.winfo_width() - w) // 2)
            y = master.winfo_rooty() + max(0, (master.winfo_height() - h) // 3)
            self.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            self.geometry("820x560")

    def _grab(self, tries: int = 10):
        try:
            self.grab_set()                     # modal: stays until a choice is made or cancelled
            self.entry.focus_set()
        except tk.TclError:
            if tries:
                self.after(80, lambda: self._grab(tries - 1))

    # ------------------------------------------------------------------ list
    def _on_type(self, event=None):
        if event is not None and event.keysym in ("Up", "Down", "Next", "Prior", "Return", "KP_Enter", "Escape"):
            return
        if self._after_id:
            self.after_cancel(self._after_id)
        self._after_id = self.after(90, self._apply_filter)

    def _apply_filter(self, select: str | None = None):
        self._after_id = None
        self.shown = filter_items(self.items, self.entry.get())
        self.listbox.delete(0, "end")
        for disp, _ in self.shown:
            self.listbox.insert("end", disp)
        self.count.configure(text=f"{len(self.shown):,} of {len(self.items):,}")
        idx = 0
        if select:
            idx = next((i for i, (_, v) in enumerate(self.shown) if v == select), 0)
        if self.shown:
            self._select(idx)

    def _select(self, idx: int):
        idx = max(0, min(idx, len(self.shown) - 1))
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(idx)
        self.listbox.activate(idx)
        self.listbox.see(idx)

    def _move(self, step: int):
        if not self.shown:
            return "break"
        cur = self.listbox.curselection()
        self._select((cur[0] if cur else -1) + step)
        return "break"

    # ------------------------------------------------------------------ actions
    def choose(self):
        cur = self.listbox.curselection()
        if not cur or not self.shown:
            self.msg.configure(text="Nothing selected" if self.items else "The list is empty - try Refresh")
            return
        self._finish(self.shown[cur[0]][1])

    def _finish(self, value: str):
        try:
            self.grab_release()
        except tk.TclError:
            pass
        self.destroy()
        self.on_pick(value)

    def cancel(self):
        try:
            self.grab_release()
        except tk.TclError:
            pass
        self.destroy()

    def _refresh(self):
        self.msg.configure(text="Refreshing...", fg="#888888")
        self.update_idletasks()
        try:
            self.items = list(self.refresh_fn())
            self.msg.configure(text=f"{len(self.items):,} items", fg="#3aa655")
        except Exception as e:  # noqa: BLE001
            self.msg.configure(text=f"ERROR: {str(e).splitlines()[0]}", fg="#e05555")
        self._apply_filter()

    def _browse(self):
        val = self.browse_fn()
        if val:
            self._finish(val)
