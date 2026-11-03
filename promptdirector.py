#!/usr/bin/env python3
"""
promptdirector.py — command-line front end. The GUI is promptdirector_gui.py.
Both use pd_core.py, and both read/write the same config.toml.

Precedence: DEFAULT_CONFIG in pd_core.py -> config file -> CLI flags.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path

import pd_core as core


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="promptdirector",
        description="Generate character-centric image prompts with a local LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "scene fields: anything you set is used for every prompt; empty fields are random\n"
            "(if random is on for that field) or left for the LLM to choose.\n\n"
            "examples:\n"
            "  %(prog)s -n 20\n"
            "  %(prog)s -n 5 --style \"anime illustration\" --characters 2 --camera-angle \"low angle\"\n"
            "  %(prog)s -n 5 --subject \"a knight and a sorceress\" --location \"a ruined chapel\" --no-random\n"
            "  %(prog)s --backend llamacpp --host http://localhost:8080 -n 5\n"
            "  %(prog)s --backend llamacpp_python --model-path ~/models/model.gguf -n 5\n"
            "  %(prog)s -n 10 --target krea2 illustrious   # length for Krea 2 + Illustrious tag lines\n"
            "  %(prog)s -n 5 --comic 2x3 --comic-mode story       # 6-panel comic pages for Krea 2\n"
            "  %(prog)s -n 5 --comic 2x2 --comic-format bbox      # layout spec for krea2-turbo-bbox\n"
            "  %(prog)s -n 5 --comfy --comfy-workflow krea2_api.json  # also render in ComfyUI\n"
            "  %(prog)s -n 10 --manage-server --llama-cpp-dir ~/llama.cpp --gguf /models/model.gguf --gpu 0\n"
            "  %(prog)s --dry-run -n 10\n"
        ),
    )
    p.add_argument("-n", "--count", type=int, help="number of prompts")
    p.add_argument("-c", "--config", help="config file (.toml/.json); default ./config.toml if present")
    p.add_argument("--init-config", nargs="?", const="config.toml", metavar="PATH",
                   help="write a config file with all defaults and exit")

    g = p.add_argument_group("backend / model")
    g.add_argument("--backend", choices=core.VALID_BACKENDS)
    g.add_argument("--host", help="server URL")
    g.add_argument("--api-key")
    g.add_argument("--timeout", type=float)
    g.add_argument("-m", "--model", help="model name/tag")
    g.add_argument("--model-path", help=".gguf file (llamacpp_python)")
    g.add_argument("-t", "--temperature", type=float)
    g.add_argument("--max-tokens", type=int)

    g = p.add_argument_group("scene fields (empty = random / LLM's choice)")
    for key, label, hint, _ in core.SCENE_FIELDS:
        g.add_argument(f"--{key.replace('_', '-')}", dest=f"field_{key}", metavar="TEXT", help=f"{label}: {hint}")
    r = g.add_mutually_exclusive_group()
    r.add_argument("--random", dest="random", action="store_true", default=None, help="random on (master switch)")
    r.add_argument("--no-random", dest="random", action="store_false",
                   help="random off: empty fields are left for the LLM to decide")
    g.add_argument("--random-fields", nargs="+", metavar="FIELD",
                   help=f"only these fields may be random: {', '.join(k for k in core.FIELD_KEYS if k != 'extra')}")
    g.add_argument("-s", "--scene", action="append", help="free-text scene idea (repeatable; cycles)")
    g.add_argument("--person", action="append", metavar="KEY=VALUE;...",
                   help="describe one character (repeat per person, in order), e.g. "
                        "\"role=a knight;age=30;ethnicity=East Asian;hair_color=black;hair_length=long;clothing=plate armor\"")

    g = p.add_argument_group("comic panel layout (Krea 2)")
    g.add_argument("--comic", nargs="?", const="2x3", choices=core.COMIC_LAYOUTS, metavar="GRID",
                   help=f"turn comic mode on; grid columns x rows ({', '.join(core.COMIC_LAYOUTS)}), default 2x3")
    g.add_argument("--no-comic", action="store_true", help="turn comic mode off (overrides the config)")
    g.add_argument("--comic-mode", choices=core.COMIC_MODES)
    g.add_argument("--comic-format", choices=core.COMIC_FORMATS, help="prose or bbox (krea2-turbo-bbox)")
    g.add_argument("--comic-location", choices=core.COMIC_LOCATION_MODES)
    g.add_argument("--speech-bubbles", action="store_true")

    g = p.add_argument_group("managed llama-server (the app starts/stops it)")
    g.add_argument("--manage-server", action="store_true", help="start llama-server with --gguf, stop it when done")
    g.add_argument("--llama-cpp-dir", help="your llama.cpp folder (custom build)")
    g.add_argument("--gguf", help="GGUF model file for the managed server")
    g.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES for the managed server, e.g. 0")
    g.add_argument("--list-ggufs", metavar="FOLDER", help="list the GGUF files in a folder and exit")

    g = p.add_argument_group("ComfyUI")
    g.add_argument("--comfy", action="store_true", help="send each prompt to ComfyUI (and save prompt files)")
    g.add_argument("--comfy-only", action="store_true", help="send to ComfyUI but do NOT save prompt files")
    g.add_argument("--comfy-url", help="ComfyUI address, default http://127.0.0.1:8188")
    g.add_argument("--comfy-workflow", metavar="FILE", help="custom workflow (ComfyUI: Workflow > Export (API))")
    g.add_argument("--comfy-builtin", choices=("sdxl", "flux"), help="use a built-in workflow instead")
    g.add_argument("--comfy-send", choices=("end", "each"), help="send after all prompts (default) or after each")
    g.add_argument("--comfy-models", action="store_true", help="refresh + print ComfyUI model lists and exit")
    g.add_argument("--rotate-random", type=int, metavar="N", help="use N different random compatible models")
    g.add_argument("--rotate-models", nargs="+", metavar="MODEL", help="use these models in turn (grouped)")
    g.add_argument("--any-model", action="store_true", help="rotation: don't limit to compatible models")
    g.add_argument("--model-dirs", help="folders (separated by ;) where model files live, for the check")
    g.add_argument("--comfy-test", action="store_true", help="test the ComfyUI connection and exit")

    g = p.add_argument_group("generation")
    g.add_argument("--target", nargs="+", choices=core.VALID_TARGETS,
                   help="image model(s) the prompts are for; sets length, illustrious adds a tag line")
    g.add_argument("--min-words", type=int, help="override the target's minimum words")
    g.add_argument("--max-words", type=int, help="override the target's maximum words")
    g.add_argument("--no-filler-check", action="store_true", help="allow mood/interpretation phrases")
    g.add_argument("--no-compat", action="store_true", help="random picks may contradict each other")
    g.add_argument("--system-prompt", help="system prompt file")
    g.add_argument("--presets", help="presets JSON file (random value lists)")
    g.add_argument("--seed-mode", choices=core.VALID_SEED_MODES)
    g.add_argument("--scenes-file", help="one scene idea per line (implies --seed-mode file)")
    g.add_argument("--seed", type=int, help="random seed (0 = random)")
    g.add_argument("--sfw", action="store_true", help="strip NSFW rules, request safe-for-work output")

    g = p.add_argument_group("output / logging")
    g.add_argument("-o", "--output-dir")
    g.add_argument("-f", "--format", nargs="+", choices=core.VALID_FORMATS)
    g.add_argument("--no-subfolder", action="store_true", help="no timestamped run folder")
    d = g.add_mutually_exclusive_group()
    d.add_argument("--debug", dest="debug", action="store_true", default=None, help="enable separate debug log")
    d.add_argument("--no-debug", dest="debug", action="store_false")
    g.add_argument("--log-dir")
    g.add_argument("-q", "--quiet", action="store_true", help="console: warnings and errors only")

    g = p.add_argument_group("utilities")
    g.add_argument("--list-models", action="store_true")
    g.add_argument("--check", action="store_true", help="test backend + model and exit")
    g.add_argument("--dry-run", action="store_true", help="show scene seeds only, no LLM")
    g.add_argument("--show-config", action="store_true")
    g.add_argument("--export-presets", metavar="PATH", help="write the built-in presets to a JSON file and exit")
    p.add_argument("--version", action="version", version=f"%(prog)s {core.__version__}")
    return p


def apply_cli(cfg: dict, a: argparse.Namespace) -> dict:
    b, m, g, o, lg, sc = (cfg[k] for k in ("backend", "model", "generation", "output", "logging", "scene"))
    pairs = [(a.count, g, "count"), (a.backend, b, "type"), (a.host, b, "host"), (a.api_key, b, "api_key"),
             (a.timeout, b, "timeout"), (a.model, m, "name"), (a.model_path, m, "path"),
             (a.temperature, m, "temperature"), (a.max_tokens, m, "max_tokens"),
             (a.system_prompt, g, "system_prompt_file"), (a.presets, g, "presets_file"),
             (a.seed_mode, g, "seed_mode"), (a.seed, g, "random_seed"), (a.output_dir, o, "directory"),
             (a.format, o, "formats"), (a.debug, lg, "debug"), (a.log_dir, lg, "directory"),
             (a.random, sc, "random_enabled")]
    for val, sect, key in pairs:
        if val is not None and val != "":
            sect[key] = val
    if a.scenes_file:
        g["scenes_file"] = a.scenes_file
        if not a.seed_mode:
            g["seed_mode"] = "file"
    if a.sfw:
        g["content_rating"] = "sfw"
    if a.target:
        g["targets"] = a.target
    if a.min_words is not None:
        g["min_words"] = a.min_words
    if a.max_words is not None:
        g["max_words"] = a.max_words
    if a.no_filler_check:
        g["filler_check"] = False
    if a.no_compat:
        g["compat_rules"] = False
    ls = cfg["llama_server"]
    if a.manage_server:
        ls["managed"] = True
        b["type"] = "llamacpp"
    if a.llama_cpp_dir:
        ls["llama_cpp_dir"] = a.llama_cpp_dir
    if a.gguf:
        m["path"] = a.gguf
    if a.gpu is not None:
        ls["gpu"] = a.gpu
    cc = cfg["comfy"]
    if a.comfy or a.comfy_only:
        cc["enabled"] = True
        cc["save_prompts"] = not a.comfy_only
    if a.comfy_url:
        cc["url"] = a.comfy_url
    if a.comfy_workflow:
        cc["workflow"], cc["workflow_file"] = "custom", a.comfy_workflow
    if a.comfy_builtin:
        cc["workflow"] = f"builtin_{a.comfy_builtin}"
    if a.comfy_send:
        cc["send_mode"] = a.comfy_send
    if a.rotate_random:
        cc["rotation"], cc["rotation_count"] = "random", a.rotate_random
    if a.rotate_models:
        cc["rotation"], cc["rotation_models"] = "list", a.rotate_models
    if a.any_model:
        cc["rotation_compatible_only"] = False
    if a.model_dirs:
        cc["model_search_dirs"] = a.model_dirs
    cm = cfg.setdefault("comic", dict(core.DEFAULT_CONFIG["comic"]))
    if a.comic:
        cm["enabled"], cm["layout"] = True, a.comic
    if a.no_comic:
        cm["enabled"] = False
    for val, key in ((a.comic_mode, "mode"), (a.comic_format, "format"), (a.comic_location, "location_mode")):
        if val:
            cm[key] = val
    if a.speech_bubbles:
        cm["speech_bubbles"] = True
    if a.no_subfolder:
        o["timestamp_subfolder"] = False
    if a.quiet:
        lg["console_level"] = "WARNING"
    for key in core.FIELD_KEYS:
        v = getattr(a, f"field_{key}")
        if v is not None:
            sc["fields"][key] = v
    if a.person:
        people = []
        for spec in a.person[:core.MAX_CHARACTERS]:
            p = {k: "" for k in core.PERSON_KEYS}
            for part in spec.split(";"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    k = k.strip().lower().replace(" ", "_")
                    if k not in p:
                        raise core.ConfigError(f"--person: unknown key '{k}' (use {', '.join(core.PERSON_KEYS)})")
                    p[k] = v.strip()
            people.append(p)
        sc["people"] = people + [{k: "" for k in core.PERSON_KEYS}] * (core.MAX_CHARACTERS - len(people))
        if not sc["fields"].get("characters"):
            sc["fields"]["characters"] = str(len(people))
    if a.random_fields:
        bad = [f for f in a.random_fields if f not in sc["random"]]
        if bad:
            raise core.ConfigError(f"Unknown --random-fields: {bad}")
        for k in sc["random"]:
            sc["random"][k] = k in a.random_fields
    return cfg


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.init_config:
        t = Path(a.init_config)
        if t.exists():
            print(f"Refusing to overwrite {t}", file=sys.stderr)
            return 1
        core.save_config(core.DEFAULT_CONFIG, t)
        print(f"Wrote {t}")
        return 0
    if a.list_ggufs:
        from pd_llamaserver import list_ggufs
        print("\n".join(list_ggufs(a.list_ggufs)) or "(no .gguf files found)")
        return 0
    if a.export_presets:
        Path(a.export_presets).write_text(json.dumps(core.DEFAULT_PRESETS, indent=2), encoding="utf-8")
        print(f"Wrote {a.export_presets}")
        return 0

    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        cfg, cfg_path = core.load_config(a.config)
        cfg = core.validate_config(apply_cli(cfg, a))
    except core.PDError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if a.show_config:
        print(json.dumps(core.redacted(cfg), indent=2))
        return 0
    if a.dry_run:
        try:
            for i, s in enumerate(core.preview_scenes(cfg, cfg["generation"]["count"], a.scene), 1):
                print(f"--- {i} ({s.source})\n{s.text}")
        except core.PDError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        return 0
    if a.comfy_test or a.comfy_models:
        try:
            if a.comfy_test:
                print(core.comfy.test_connection(cfg["comfy"]))
            if a.comfy_models:
                cache = core.comfy.refresh_model_cache(cfg["comfy"], core.APP_DIR)
                print(f"source: {cache['source']}  (cached in {core.comfy.cache_path(cfg['comfy'], core.APP_DIR)})")
                for k, v in cache["lists"].items():
                    print(f"\n[{k}] {len(v)}")
                    print("\n".join(f"  {x}" for x in v))
            return 0
        except core.comfy.ComfyError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
    if a.list_models or a.check:
        try:
            if a.list_models:
                print("\n".join(core.list_models(cfg)) or "(no models reported)")
            else:
                print(core.test_backend(cfg))
            return 0
        except core.PDError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    stop = threading.Event()

    def on_sigint(sig, frame):
        if stop.is_set():
            raise KeyboardInterrupt
        stop.set()
        core.log.warning("Ctrl+C — stopping after the current request (Ctrl+C again to abort now)")

    signal.signal(signal.SIGINT, on_sigint)
    try:
        if cfg_path:
            print(f"Config: {cfg_path}", file=sys.stderr)
        res = core.run_job(cfg, fixed_scenes=a.scene, stop_event=stop)
        if res.stopped:
            return 130
        return 0 if (res.ok + res.flagged) else 1
    except core.PDError as e:
        core.log.error("%s", e)
        return 2
    except KeyboardInterrupt:
        core.log.warning("Aborted")
        return 130
    except OSError as e:
        core.log.error("File/folder error: %s", e)
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except Exception:
        core.log.exception("Unexpected error")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
