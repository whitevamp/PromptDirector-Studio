#!/usr/bin/env python3
"""
pd_gui_bench.py — measures why CustomTkinter widgets are slow to create on this machine.

    python pd_gui_bench.py

Opens a small window for a few seconds, prints timings, and closes it.
Send the printed output back for diagnosis.
"""
import os
import platform
import shutil
import subprocess
import sys
import time
import tkinter as tk
import tkinter.font as tkfont

N = 30


def timed(label, fn):
    t = time.perf_counter()
    fn()
    dt = time.perf_counter() - t
    print(f"{label:<52} {dt:6.2f}s  ({dt / N * 1000:6.1f} ms each)")
    return dt


def main():
    import customtkinter as ctk
    print(f"Python {sys.version.split()[0]} | Tk {tk.TkVersion} (patchlevel {tk.Tcl().call('info', 'patchlevel')}) "
          f"| CustomTkinter {getattr(ctk, '__version__', '?')} | {platform.platform()}")
    print(f"Session: {os.environ.get('XDG_SESSION_TYPE', '?')} | DISPLAY={os.environ.get('DISPLAY', '')} "
          f"| WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY', '')} | desktop={os.environ.get('XDG_CURRENT_DESKTOP', '?')}")
    if shutil.which("fc-match"):
        for fam in ("Roboto", "Courier"):
            out = subprocess.run(["fc-match", fam], capture_output=True, text=True).stdout.strip()
            print(f"fc-match {fam:<8} -> {out}")
    if shutil.which("fc-match"):
        out = subprocess.run(["fc-match", "CustomTkinter_shapes_font"], capture_output=True, text=True).stdout.strip()
        print(f"fc-match CustomTkinter_shapes_font -> {out}")
    if shutil.which("fc-list"):
        n = len(subprocess.run(["fc-list"], capture_output=True, text=True).stdout.splitlines())
        print(f"fontconfig: {n} font files installed")

    print("Input method env: " + " | ".join(f"{k}={os.environ.get(k, '')}" for k in
                                         ("XMODIFIERS", "GTK_IM_MODULE", "QT_IM_MODULE")))
    root = ctk.CTk()
    root.geometry("600x400")
    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True)

    t = time.perf_counter()
    fams = set(tkfont.families(root))
    print(f"tkfont.families(): {len(fams)} families in {time.perf_counter() - t:.2f}s")
    theme_family = ctk.ThemeManager.theme["CTkFont"].get("family", "?")
    print(f"CustomTkinter theme font family: {theme_family!r} -> installed: {theme_family in fams}")
    alt = next((f for f in ("Noto Sans", "DejaVu Sans", "Cantarell", "Liberation Sans", "Ubuntu", "Inter")
                if f in fams), None)
    print(f"Installed alternative used below: {alt!r}\n")

    timed(f"{N} plain tk.Label", lambda: [tk.Label(frame, text="x") for _ in range(N)])
    timed(f"{N} tkinter Font(family={theme_family!r})",
          lambda: [tkfont.Font(root, family=theme_family, size=-13) for _ in range(N)])
    if alt:
        timed(f"{N} tkinter Font(family={alt!r})", lambda: [tkfont.Font(root, family=alt, size=-13) for _ in range(N)])
    timed(f"{N} CTkLabel (theme font {theme_family!r})", lambda: [ctk.CTkLabel(frame, text="x") for _ in range(N)])
    shared = ctk.CTkFont(size=11)
    timed(f"{N} CTkLabel sharing one CTkFont", lambda: [ctk.CTkLabel(frame, text="x", font=shared) for _ in range(N)])
    timed(f"{N} CTkEntry", lambda: [ctk.CTkEntry(frame) for _ in range(N)])
    timed(f"{N} CTkSwitch", lambda: [ctk.CTkSwitch(frame, text="") for _ in range(N)])
    timed(f"{N} CTkButton", lambda: [ctk.CTkButton(frame, text="x") for _ in range(N)])
    if alt:
        ctk.ThemeManager.theme["CTkFont"]["family"] = alt
        timed(f"{N} CTkLabel (theme font switched to {alt!r})", lambda: [ctk.CTkLabel(frame, text="x") for _ in range(N)])
        timed(f"{N} CTkButton (theme font switched to {alt!r})", lambda: [ctk.CTkButton(frame, text="x") for _ in range(N)])

    de = getattr(ctk, "DrawEngine", None)
    if de is None:
        try:
            from customtkinter.windows.widgets.core_rendering import DrawEngine as de
        except Exception:
            de = None
    if de is None:
        print("\nDrawEngine not found in this CustomTkinter version; cannot compare drawing methods.")
    else:
        print(f"\nDefault corner drawing method: {de.preferred_drawing_method!r}")
        for method in ("font_shapes", "circle_shapes", "polygon_shapes"):
            de.preferred_drawing_method = method
            timed(f"{N} CTkButton  [{method}]", lambda: [ctk.CTkButton(frame, text="x") for _ in range(N)])
            timed(f"{N} CTkEntry   [{method}]", lambda: [ctk.CTkEntry(frame) for _ in range(N)])
            timed(f"{N} CTkSwitch  [{method}]", lambda: [ctk.CTkSwitch(frame, text="") for _ in range(N)])
        timed(f"{N} CTkButton  corner_radius=0", lambda: [ctk.CTkButton(frame, text="x", corner_radius=0) for _ in range(N)])
        de.preferred_drawing_method = "circle_shapes"
    vals = [f"value {i}" for i in range(50)]
    print("\nText widgets, input methods ON (default):")
    timed(f"{N} plain tk.Entry", lambda: [tk.Entry(frame) for _ in range(N)])
    timed(f"{N} plain tk.Text", lambda: [tk.Text(frame, height=2) for _ in range(N)])
    timed(f"{N} CTkEntry", lambda: [ctk.CTkEntry(frame) for _ in range(N)])
    timed(f"{N} CTkComboBox (50 values)", lambda: [ctk.CTkComboBox(frame, values=vals) for _ in range(N)])
    try:
        root.tk.call("tk", "useinputmethods", "0")
        print("Text widgets, input methods OFF (tk useinputmethods 0):")
        timed(f"{N} plain tk.Entry", lambda: [tk.Entry(frame) for _ in range(N)])
        timed(f"{N} plain tk.Text", lambda: [tk.Text(frame, height=2) for _ in range(N)])
        timed(f"{N} CTkEntry", lambda: [ctk.CTkEntry(frame) for _ in range(N)])
        timed(f"{N} CTkComboBox (50 values)", lambda: [ctk.CTkComboBox(frame, values=vals) for _ in range(N)])
    except tk.TclError as e:
        print(f"tk useinputmethods not available: {e}")

    def round_trip(label):
        t = time.perf_counter()
        for _ in range(N):
            root.winfo_pointerxy()          # one synchronous request/reply with the X server each
        dt = time.perf_counter() - t
        print(f"{label:<52} {dt:6.2f}s  ({dt / N * 1000:6.1f} ms each)")

    print("\nDisplay-server round trips and window state:")
    round_trip(f"{N} X round trips (window hidden)")
    timed(f"{N} tk.Menu (window hidden)", lambda: [tk.Menu(root, tearoff=0) for _ in range(N)])
    timed(f"{N} tk.Frame (window hidden)", lambda: [tk.Frame(frame) for _ in range(N)])
    timed(f"{N} CTkLabel (window hidden)", lambda: [ctk.CTkLabel(frame, text="x") for _ in range(N)])
    root.deiconify()
    root.update()
    time.sleep(0.3)
    root.update()
    round_trip(f"{N} X round trips (window shown)")
    timed(f"{N} tk.Frame (window shown)", lambda: [tk.Frame(frame) for _ in range(N)])
    timed(f"{N} tk.Label (window shown)", lambda: [tk.Label(frame, text="x") for _ in range(N)])
    timed(f"{N} CTkLabel (window shown)", lambda: [ctk.CTkLabel(frame, text="x") for _ in range(N)])
    timed(f"{N} CTkButton (window shown)", lambda: [ctk.CTkButton(frame, text="x") for _ in range(N)])

    print("\nFont realization (is every NEW font slow to open?):")
    M = 10
    def per(label, fn, n=M):
        t = time.perf_counter(); fn(); dt = time.perf_counter() - t
        print(f"{label:<52} {dt:6.2f}s  ({dt / n * 1000:6.1f} ms each)")
    per(f"{M} tk.Label, each a NEW Roboto size", lambda: [tk.Label(frame, text="x", font=("Roboto", -(20 + i))) for i in range(M)])
    per(f"{M} tk.Label, each a NEW DejaVu Sans size", lambda: [tk.Label(frame, text="x", font=("DejaVu Sans", -(20 + i))) for i in range(M)])
    per(f"{M} tk.Label, all the SAME new size", lambda: [tk.Label(frame, text="x", font=("Roboto", -41)) for i in range(M)])
    per(f"{M} tk.Label, Roboto bold new sizes", lambda: [tk.Label(frame, text="x", font=("Roboto", -(50 + i), "bold")) for i in range(M)])
    per(f"{M} tk.Label text 'Save As\u2026' (ellipsis)", lambda: [tk.Label(frame, text="Save As\u2026") for _ in range(M)])
    per(f"{M} tk.Label text with '\u2014' (em dash)", lambda: [tk.Label(frame, text="a \u2014 b") for _ in range(M)])
    muted = ("gray40", "gray65")
    small = ctk.CTkFont(size=11)
    cf = ctk.CTkFrame(root)
    cf.pack()
    sf = ctk.CTkScrollableFrame(root)
    sf.pack()
    per(f"{M} CTkLabel like the GUI hints (CTkFrame)", lambda: [ctk.CTkLabel(cf, text=f"hint text e.g. value {i}",
        text_color=muted, anchor="w", font=small) for i in range(M)])
    per(f"{M} CTkLabel like the GUI hints (ScrollableFrame)", lambda: [ctk.CTkLabel(sf, text=f"hint text e.g. value {i}",
        text_color=muted, anchor="w", font=small) for i in range(M)])
    per(f"{M} CTkLabel text_color only", lambda: [ctk.CTkLabel(cf, text="x", text_color=muted) for _ in range(M)])
    per(f"{M} CTkLabel font=CTkFont(size=11) only", lambda: [ctk.CTkLabel(cf, text="x", font=small) for _ in range(M)])
    per(f"{M} CTkButton in CTkFrame", lambda: [ctk.CTkButton(cf, text="Save") for _ in range(M)])
    if shutil.which("fc-cache"):
        out = subprocess.run(["fc-cache", "-v"], capture_output=True, text=True, timeout=120).stdout
        stale = [l for l in out.splitlines() if "invalid cache" in l.lower() or "caching" in l.lower()]
        print(f"\nfc-cache -v: {len(out.splitlines())} lines, {len(stale)} mention invalid/re-caching")

    root.after(500, root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
