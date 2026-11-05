# PromptDirector Studio - Copyright (c) 2026 whitevamp
# SPDX-License-Identifier: CC-BY-NC-SA-4.0 (see LICENSE)
"""
pd_core.py — shared engine for promptdirector (CLI) and promptdirector_gui (GUI).

Contains: config (defaults / load / save / validate), logging, LLM backends,
scene builder (per-field user values or random presets), response cleaning,
output writers and the generation job.

Only the Python standard library is required.
Optional: tomli (TOML config on Python < 3.11), llama-cpp-python (in-process GGUF).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import logging.handlers
import os
import random
import math
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pd_comfy as comfy
import pd_llamaserver as lserver

__version__ = "2.9.2"

APP_DIR = Path(__file__).resolve().parent
log = logging.getLogger("promptdirector")
log.addHandler(logging.NullHandler())

# =============================================================================
# SCENE FIELDS — the points from the system prompt the user can set.
# key, label, hint, randomizable
# =============================================================================
SCENE_FIELDS: list[tuple[str, str, str, bool]] = [
    ("style", "Image style", "e.g. photorealistic photography, anime illustration", True),
    ("characters", "Number of characters", "1-6; empty = random or decided by the subject", True),
    ("subject", "Subject / characters", "who is in the image, e.g. a knight and a sorceress", True),
    ("action", "Action / interaction", "what they are doing", True),
    ("camera_angle", "Camera angle", "e.g. low angle, eye level, overhead", True),
    ("shot", "Shot / camera distance", "e.g. close-up, medium shot, wide shot", True),
    ("lighting", "Lighting", "e.g. soft window light, neon light", True),
    ("location", "Location / environment", "where the scene takes place", True),
    ("time_weather", "Time of day / weather", "e.g. sunset, heavy rain", True),
    ("composition", "Focus / composition", "e.g. deep focus, subject centered", True),
    ("extra", "Extra notes", "anything else the prompt must include (never random)", False),
]
FIELD_KEYS = [f[0] for f in SCENE_FIELDS]
FIELD_LABELS = {f[0]: f[1] for f in SCENE_FIELDS}
SEED_LABELS = {
    "style": "Style", "characters": "Number of characters", "subject": "Characters",
    "action": "Action", "camera_angle": "Camera angle", "shot": "Shot / distance",
    "lighting": "Lighting", "location": "Location", "time_weather": "Time / weather",
    "composition": "Focus / composition", "extra": "Additional notes",
}
MAX_CHARACTERS = 6
# Per-character details (Scene tab > Characters). key, label
PERSON_FIELDS: list[tuple[str, str]] = [
    ("role", "Role"), ("age", "Age"), ("ethnicity", "Ethnicity"), ("hair_color", "Hair color"),
    ("hair_length", "Hair length / style"), ("clothing", "Clothing"),
]
PERSON_KEYS = [k for k, _ in PERSON_FIELDS]
PERSON_LABELS = {"role": "", "age": "age", "ethnicity": "", "hair_color": "hair", "hair_length": "hair",
                 "clothing": "wearing"}
CHAR_COUNT_WEIGHTS = {1: 35, 2: 40, 3: 15, 4: 10}

# =============================================================================
# CONFIG SECTION — defaults. A config file, the GUI or CLI flags override these.
# =============================================================================
DEFAULT_CONFIG: dict[str, Any] = {
    "backend": {
        # ollama | llamacpp (llama-server) | openai (LM Studio, vLLM, ...) | llamacpp_python (in-process GGUF)
        "type": "ollama",
        "host": "",              # empty = default host for the backend type
        "api_key": "",
        "timeout": 300,
        "max_retries": 3,
        "retry_backoff": 2.0,
        "verify_connection": True,
    },
    "model": {
        "name": "llama3.1:8b",
        "path": "",              # .gguf file, llamacpp_python only
        "n_ctx": 8192,
        "n_gpu_layers": -1,
        "temperature": 0.9,
        "top_p": 0.95,
        "max_tokens": 1500,
        "keep_alive": "10m",     # ollama only
        # Reasoning ("thinking") models such as Qwen3 / DeepSeek-R1:
        #   off  = ask the model to answer directly (recommended: thinking can use up the token budget)
        #   on   = let it think first (raise Max tokens a lot)
        #   auto = don't tell the server anything
        "thinking": "off",
    },
    "generation": {
        "count": 5,
        # Image models the prompts are for: any of krea2, flux, illustrious.
        # Sets the word range; 'illustrious' also produces a Danbooru-style tag line per prompt.
        "targets": ["krea2"],
        "min_words": 0,                   # 0 = use the targets' default range
        "max_words": 0,
        "filler_check": True,             # re-ask when the prompt is padded with mood/interpretation
        "detail_check": True,             # re-ask (with a correction) when ages/hair/camera/lighting... don't match
        "max_filler_hits": 2,
        "compat_rules": True,             # random picks avoid contradicting each other / your fields
        "tag_prefix": "masterpiece, best quality, amazing quality",   # put in front of Illustrious tags
        "system_prompt_file": "system_prompt.txt",
        "presets_file": "presets.json",   # random-value lists; created on first save from the GUI
        # builtin -> scene built from the fields / random presets
        # llm     -> the LLM first writes a short scene idea from those details (2 calls per prompt)
        # file    -> random line from scenes_file, plus any fields you filled in
        "seed_mode": "builtin",
        "scenes_file": "",
        "user_template": "Scene description:\n{scene}",
        "random_seed": 0,                 # 0 = new random seed each run
        "content_rating": "as_file",      # as_file | sfw
        "min_paragraphs": 3,
        "max_attempts_per_prompt": 3,
        "dedupe": True,
        "strip_reasoning": True,
        "clean_output": True,
    },
    "llama_server": copy.deepcopy(lserver.DEFAULT_SERVER),   # app-managed llama-server (backend 'llamacpp')
    "comfy": copy.deepcopy(comfy.DEFAULT_COMFY),   # send prompts to ComfyUI (see pd_comfy.py)
    "comic": {
        # One image split into panels. Made for Krea 2 (the prose format also suits FLUX for 2-4 panels).
        "enabled": False,
        "layout": "2x3",          # columns x rows: 2x1, 1x2, 3x1, 1x3, 2x2, 2x3, 3x2, 3x3
        "mode": "vignettes",      # vignettes = separate moments, same cast | story = consecutive beats
        # prose = "Panel 1 (Top Left): ..." paragraphs (stock Krea 2 and mixes)
        # bbox  = grounded layout spec for the krea2-turbo-bbox finetune (p[...], ac:id[...], ...)
        "format": "prose",
        "location_mode": "auto",  # auto (vignettes = varied, story = same) | same | varied
        "speech_bubbles": False,
    },
    "scene": {
        # Master switch: off = empty fields are left for the LLM to decide (nothing random).
        "random_enabled": True,
        # What you type here is used for EVERY prompt. Empty = random (if allowed) or LLM's choice.
        "fields": {k: "" for k in FIELD_KEYS},
        # Per-field random switch (used only when the field is empty and random_enabled is on).
        # lighting/composition default off so the LLM can match them to the scene ("NO FIXED STYLE").
        "random": {
            "style": True, "characters": True, "subject": True, "action": True,
            "camera_angle": True, "shot": True, "lighting": False, "location": True,
            "time_weather": True, "composition": False,
        },
        # One entry per character slot (only the first 'Number of characters' are used).
        "people": [{k: "" for k in ["role", "age", "ethnicity", "hair_color", "hair_length", "clothing"]}
                   for _ in range(6)],
        # Per-column random switch for empty per-character boxes (needs random_enabled).
        "random_people": {"role": True, "age": True, "ethnicity": False, "hair_color": True,
                          "hair_length": True, "clothing": False},
    },
    "output": {
        "directory": "",                  # empty = current working directory
        "formats": ["txt", "jsonl"],      # txt | lines | jsonl | separate
        "filename_prefix": "prompts",
        "timestamp_subfolder": True,
        "separator": "\n\n----------------------------------------\n\n",
    },
    "logging": {
        "directory": "logs",              # relative = inside output.directory
        "log_file": "promptdirector.log",          # always on
        "debug": False,
        "debug_file": "promptdirector.debug.log",
        "console_level": "INFO",
        "max_bytes": 5_000_000,
        "backup_count": 5,
    },
}

DEFAULT_HOSTS = {
    "ollama": "http://localhost:11434",
    "llamacpp": "http://localhost:8080",
    "openai": "http://localhost:1234",
    "llamacpp_python": "",
}
VALID_BACKENDS = tuple(DEFAULT_HOSTS)
VALID_FORMATS = ("txt", "lines", "jsonl", "separate", "tags")
VALID_TARGETS = ("krea2", "flux", "illustrious")
COMIC_LAYOUTS = ("2x1", "1x2", "3x1", "1x3", "2x2", "2x3", "3x2", "3x3")
COMIC_MODES = ("vignettes", "story")
COMIC_FORMATS = ("prose", "bbox")
COMIC_LOCATION_MODES = ("auto", "same", "varied")
# Word ranges per target. Krea 2 and FLUX read up to 512 tokens (~380 words); SDXL-based
# Illustrious reads 75-token chunks, so its prose is kept shorter and a tag line is added.
TARGET_WORDS = {"krea2": (150, 300), "flux": (100, 220), "illustrious": (90, 180)}
VALID_SEED_MODES = ("builtin", "llm", "file")
CONFIG_SEARCH_NAMES = ("config.toml", "config.json")

# =============================================================================
# RANDOM PRESETS — editable in the GUI (Presets tab) or in presets.json.
# =============================================================================
DEFAULT_PRESETS: dict[str, list[str]] = {
    "style": [
        "photorealistic photography", "photorealistic photography", "photorealistic photography",
        "cinematic still", "cinematic still", "documentary photography", "fashion photography",
        "studio photography", "street photography", "editorial portrait photography",
        "film photography", "anime illustration", "digital painting", "concept art", "3D render",
        "historical painting", "oil painting", "watercolor illustration", "comic book illustration",
    ],
    "subject": [
        "a street musician", "an elderly fisherman", "a young chef", "a nurse on a night shift",
        "a mechanic", "a ballet dancer", "a mountain climber", "a bookstore owner", "a boxer",
        "a farmer", "a university student", "a tattoo artist", "a soldier", "a barista",
        "a detective", "a gardener", "a surfer", "a painter", "a librarian", "a skateboarder",
        "a bride", "a groom", "a businesswoman", "a businessman", "a grandmother", "a grandfather",
        "a photographer", "a blacksmith", "a pilot", "a market vendor", "a knight", "a sorceress",
        "a samurai", "an astronaut", "a cyberpunk hacker", "a cowboy", "a lifeguard",
        "a jazz singer", "a model", "a factory worker", "a monk", "a florist", "a firefighter",
        "a scientist", "a fitness trainer", "a bartender", "a dancer", "a tourist",
    ],
    "action": [  # works for one person or more
        "waiting", "dancing", "laughing", "taking a break", "preparing for an event",
        "repairing something", "posing for a photo", "reading", "training", "celebrating",
        "resting", "looking at something off-camera", "carrying heavy objects", "getting dressed",
        "cooking", "cleaning up after a party", "walking toward the camera", "writing a letter",
    ],
    "action_group": [  # only used when there are 2+ characters
        "working together", "having a quiet conversation", "arguing", "sharing a meal",
        "saying goodbye", "meeting for the first time", "walking side by side", "embracing",
        "playing a game", "competing", "dancing together", "helping one of them up",
    ],
    "camera_angle": [
        "eye level", "eye level", "low angle", "high angle", "overhead top-down view",
        "worm's-eye view", "over-the-shoulder view", "dutch angle", "three-quarter side view",
        "profile side view", "slightly elevated angle",
    ],
    "shot": [
        "close-up", "medium close-up", "medium shot", "medium shot", "cowboy shot",
        "full-body shot", "full-body shot", "wide shot", "wide establishing shot", "extreme close-up",
    ],
    "lighting": [
        "soft natural daylight", "harsh midday sun", "golden hour sunlight", "overcast diffused light",
        "soft window light", "single hard key light", "neon light", "candlelight",
        "fluorescent overhead light", "moonlight", "studio softbox lighting", "backlight with rim light",
        "practical lamps in the scene", "flash photography",
    ],
    "location": [
        "a crowded night market", "a rainy city street", "a small kitchen", "a hospital corridor",
        "an auto repair garage", "a rehearsal studio", "a snowy mountain ridge", "a cluttered bookstore",
        "a boxing gym", "a wheat field", "a university library", "a neon-lit alley", "a train platform",
        "a desert highway", "a rooftop", "a beach at low tide", "a forest clearing", "a 1920s jazz club",
        "a medieval castle courtyard", "a space station module", "a subway car", "a greenhouse",
        "a laundromat", "a hotel room", "a lakeside dock", "a busy office", "a wedding reception hall",
        "a diner booth", "an abandoned warehouse", "a botanical garden", "a ferry deck", "a hair salon",
        "a vineyard", "a recording studio", "a living room", "a bedroom", "a swimming pool",
        "an art gallery", "a parking garage", "a fishing harbor", "a temple courtyard", "a ski lodge",
    ],
    "time_weather": [
        "early morning", "midday", "late afternoon", "sunset", "blue hour", "night",
        "overcast day", "heavy rain", "light fog", "snowfall", "clear sunny day", "after rain, wet ground",
        "thunderstorm", "windy day", "hazy summer afternoon",
    ],
    "role": [],   # empty = use the 'subject' list for per-character roles
    "age": ["early 20s", "mid 20s", "late 20s", "early 30s", "mid 30s", "late 30s", "early 40s", "late 40s",
            "50s", "60s", "70s"],
    "ethnicity": ["East Asian", "South Asian", "Southeast Asian", "Black", "White", "Hispanic/Latino",
                  "Middle Eastern", "North African", "Pacific Islander", "Indigenous American", "mixed-race"],
    "hair_color": ["black", "dark brown", "light brown", "auburn", "red", "copper", "blonde", "platinum blonde",
                   "grey", "silver", "white", "dyed blue", "dyed pink", "dyed green", "black with highlights"],
    "hair_length": ["shaved head", "buzz cut", "short", "short curly", "pixie cut", "chin-length bob",
                    "shoulder-length", "shoulder-length wavy", "long straight", "long wavy", "very long",
                    "ponytail", "high bun", "braided", "dreadlocks", "afro"],
    "clothing": ["casual jeans and t-shirt", "business suit", "summer dress", "leather jacket and dark jeans",
                 "hoodie and joggers", "evening gown", "work overalls", "sportswear", "traditional kimono",
                 "medieval armor", "lab coat", "uniform matching the role", "knitted sweater and skirt",
                 "trench coat", "swimwear", "festival outfit", "flannel shirt and boots"],
    "composition": [
        "deep focus, everything sharp", "shallow depth of field, background softened",
        "subject centered, symmetrical framing", "rule-of-thirds framing", "foreground objects framing the subjects",
        "subjects small in a large environment", "tight framing", "leading lines toward the subjects",
    ],
}

# =============================================================================
# Errors
# =============================================================================
class PDError(Exception):
    pass


class ConfigError(PDError):
    pass


class BackendError(PDError):
    pass


class FatalBackendError(BackendError):
    """Retrying can't help (e.g. the prompt doesn't fit the server's context): stop the run."""


class ContextSizeError(FatalBackendError):
    pass


CONTEXT_ERR_RE = re.compile(r"exceed[s_ ]?\w*[_ ]context|context[_ ]?(length|size|window)[^\n]{0,40}(exceed|too|limit)|"
                            r"maximum context|context_length_exceeded", re.I)


# =============================================================================
# Paths & config
# =============================================================================
def resolve_path(p: str) -> Path:
    """Relative paths: CWD first, then the app folder."""
    path = Path(p).expanduser()
    if path.is_absolute():
        return path
    for base in (Path.cwd(), APP_DIR):
        if (base / path).exists():
            return base / path
    return APP_DIR / path


def deep_merge(base: dict, override: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        full = f"{path}.{key}" if path else key
        if key not in out:
            log.warning("Unknown config key '%s' (ignored)", full)
            out[key] = val
        elif isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = deep_merge(out[key], val, full)
        else:
            out[key] = val
    return out


def _load_toml(text: str) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError as e:
            raise ConfigError("TOML config needs Python 3.11+ or `pip install tomli` (or use config.json)") from e
    return tomllib.loads(text)


def load_config_file(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text) if path.suffix.lower() == ".json" else _load_toml(text)
    except ConfigError:
        raise
    except Exception as e:
        raise ConfigError(f"Invalid config file {path}: {e}") from e


def find_config_file(explicit: str | None = None) -> Path | None:
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("PD_CONFIG") or os.environ.get("KPG_CONFIG")
    if env:
        return Path(env).expanduser()
    for folder in (Path.cwd(), APP_DIR):
        for name in CONFIG_SEARCH_NAMES:
            if (folder / name).is_file():
                return folder / name
    return None


def load_config(explicit: str | None = None) -> tuple[dict, Path | None]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    path = find_config_file(explicit)
    if path:
        cfg = deep_merge(cfg, load_config_file(path))
        cc = cfg.get("comfy") or {}
        if str(cc.get("filename_prefix", "")).strip().lower() == "kpg":    # the old app name's default
            cc["filename_prefix"] = "pds"
            log.info("ComfyUI image prefix: the old default 'kpg' is now 'pds' (saved with your next Save)")
    return cfg, path


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):                     # inline table, e.g. one entry of scene.people
        return "{" + ", ".join(f"{json.dumps(str(k))} = {_toml_value(x)}" for k, x in v.items()) + "}"
    return json.dumps(str(v), ensure_ascii=False)  # JSON strings are valid TOML basic strings


def dump_toml(data: dict, prefix: str = "") -> str:
    lines, tables = [], []
    for k, v in data.items():
        if isinstance(v, dict):
            tables.append((k, v))
        else:
            lines.append(f"{k} = {_toml_value(v)}")
    out = ""
    if prefix and (lines or not tables):
        out += f"\n[{prefix}]\n"
    if lines:
        out += "\n".join(lines) + "\n"
    for k, v in tables:
        out += dump_toml(v, f"{prefix}.{k}" if prefix else k)
    return out


def save_config(cfg: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        header = f"# promptdirector {__version__} configuration (saved {datetime.now():%Y-%m-%d %H:%M})\n"
        path.write_text(header + dump_toml(cfg).lstrip("\n"), encoding="utf-8")


def validate_config(cfg: dict) -> dict:
    """Validates and normalizes in place; returns cfg. Raises ConfigError."""
    b, m, g, o, s = (cfg[k] for k in ("backend", "model", "generation", "output", "scene"))
    errs = []
    if b["type"] not in VALID_BACKENDS:
        errs.append(f"backend.type must be one of {VALID_BACKENDS}, got '{b['type']}'")
    for sect, key, typ in (("generation", "count", int), ("backend", "timeout", float),
                           ("backend", "max_retries", int), ("model", "max_tokens", int),
                           ("model", "n_ctx", int), ("model", "n_gpu_layers", int),
                           ("model", "temperature", float), ("model", "top_p", float),
                           ("generation", "random_seed", int), ("generation", "min_paragraphs", int),
                           ("generation", "max_attempts_per_prompt", int), ("backend", "retry_backoff", float)):
        try:
            cfg[sect][key] = typ(cfg[sect][key])
        except (TypeError, ValueError):
            errs.append(f"{sect}.{key} must be a number, got '{cfg[sect][key]}'")
    if not errs:
        if g["count"] < 1:
            errs.append("generation.count must be >= 1")
        if not 0 <= m["temperature"] <= 2:
            errs.append("model.temperature should be between 0 and 2")
        if g["max_attempts_per_prompt"] < 1:
            errs.append("generation.max_attempts_per_prompt must be >= 1")
    if g["seed_mode"] not in VALID_SEED_MODES:
        errs.append(f"generation.seed_mode must be one of {VALID_SEED_MODES}")
    if g["seed_mode"] == "file" and not g["scenes_file"]:
        errs.append("generation.seed_mode = 'file' needs generation.scenes_file")
    if g["content_rating"] not in ("as_file", "sfw"):
        errs.append("generation.content_rating must be 'as_file' or 'sfw'")
    if "{scene}" not in g["user_template"]:
        errs.append("generation.user_template must contain '{scene}'")
    if isinstance(g["targets"], str):
        g["targets"] = [g["targets"]]
    if not g["targets"] or any(t not in VALID_TARGETS for t in g["targets"]):
        errs.append(f"generation.targets must be a non-empty list from {VALID_TARGETS}")
    for k in ("min_words", "max_words", "max_filler_hits"):
        try:
            g[k] = int(g[k])
        except (TypeError, ValueError):
            errs.append(f"generation.{k} must be a number")
    if not errs and g["max_words"] and g["min_words"] > g["max_words"]:
        errs.append("generation.min_words is larger than generation.max_words")
    ls = cfg.setdefault("llama_server", copy.deepcopy(lserver.DEFAULT_SERVER))
    for k, v in lserver.DEFAULT_SERVER.items():
        ls.setdefault(k, v)
    for k in ("n_gpu_layers", "parallel", "startup_timeout"):
        try:
            ls[k] = int(float(ls[k]))
        except (TypeError, ValueError):
            errs.append(f"llama_server.{k} must be a number")
    ls["gpu"] = str(ls.get("gpu", "")).strip()
    if b["type"] == "llamacpp" and ls["managed"] and not m.get("path"):
        errs.append("Managed llama-server: pick a GGUF model file (Backend & Model > Model file)")
    cc = cfg.setdefault("comfy", copy.deepcopy(comfy.DEFAULT_COMFY))
    for k, v in comfy.DEFAULT_COMFY.items():
        cc.setdefault(k, v)
    if cc["workflow"] not in comfy.WORKFLOWS:
        errs.append(f"comfy.workflow must be one of {comfy.WORKFLOWS}")
    if cc["send_mode"] not in comfy.SEND_MODES:
        errs.append(f"comfy.send_mode must be one of {comfy.SEND_MODES}")
    if cc["prompt_source"] not in comfy.PROMPT_SOURCES:
        errs.append(f"comfy.prompt_source must be one of {comfy.PROMPT_SOURCES}")
    if cc.get("rotation", "off") not in comfy.ROTATION_MODES:
        errs.append(f"comfy.rotation must be one of {comfy.ROTATION_MODES}")
    if cc.get("rotation_target", "auto") not in comfy.ROTATION_TARGETS:
        errs.append(f"comfy.rotation_target must be one of {comfy.ROTATION_TARGETS}")
    if isinstance(cc.get("rotation_models"), str):
        cc["rotation_models"] = [x.strip() for x in cc["rotation_models"].splitlines() if x.strip()]
    cc["rotation_models"] = [str(x).strip() for x in (cc.get("rotation_models") or []) if str(x).strip()]
    try:
        cc["rotation_count"] = int(float(cc.get("rotation_count") or 0))
    except (TypeError, ValueError):
        errs.append("comfy.rotation_count must be a number")
    if cc.get("size_mode", "auto") not in comfy.SIZE_MODES:
        errs.append(f"comfy.size_mode must be one of {comfy.SIZE_MODES}")
    for k, typ in (("width", int), ("height", int), ("steps", int), ("cfg", float), ("guidance", float), ("comic_scale", float),
                   ("clip_skip", int), ("seed", int), ("images_per_prompt", int), ("timeout", float)):
        try:
            cc[k] = typ(cc[k] or 0)
        except (TypeError, ValueError):
            errs.append(f"comfy.{k} must be a number")
    cc["compare_models"] = bool(cc.get("compare_models", False))
    try:
        cc["vram_wait"] = max(0.0, float(cc.get("vram_wait", 30) or 0))
    except (TypeError, ValueError):
        errs.append("comfy.vram_wait must be a number")
    if cc["enabled"] and cc["compare_models"] and cc.get("rotation", "off") == "off":
        errs.append("ComfyUI: 'Compare models' needs 'Rotate models' set to random or list "
                    "(those are the models every prompt is rendered on)")
    if cc["enabled"] and cc["workflow"] == "custom" and not cc["workflow_file"]:
        errs.append("ComfyUI: the custom workflow needs a workflow file (ComfyUI: Workflow > Export (API))")
    if not cc["enabled"]:
        cc["save_prompts"] = True          # nothing would happen otherwise
    cm = cfg.setdefault("comic", copy.deepcopy(DEFAULT_CONFIG["comic"]))
    if cm["layout"] not in COMIC_LAYOUTS:
        errs.append(f"comic.layout must be one of {COMIC_LAYOUTS}")
    if cm["mode"] not in COMIC_MODES:
        errs.append(f"comic.mode must be one of {COMIC_MODES}")
    if cm["format"] not in COMIC_FORMATS:
        errs.append(f"comic.format must be one of {COMIC_FORMATS}")
    if cm["location_mode"] not in COMIC_LOCATION_MODES:
        errs.append(f"comic.location_mode must be one of {COMIC_LOCATION_MODES}")
    if isinstance(o["formats"], str):
        o["formats"] = [o["formats"]]
    if not o["formats"] or any(f not in VALID_FORMATS for f in o["formats"]):
        errs.append(f"output.formats must be a non-empty list from {VALID_FORMATS}")
    if m.get("thinking", "off") not in ("off", "on", "auto"):
        errs.append("model.thinking must be off, on or auto")
    if b["type"] == "llamacpp_python" and not m["path"]:
        errs.append("model.path (.gguf file) is required for backend 'llamacpp_python'")
    if b["type"] == "ollama" and not m["name"]:
        errs.append("model.name is required for the ollama backend")
    # scene
    for k in FIELD_KEYS:
        s["fields"].setdefault(k, "")
        s["fields"][k] = str(s["fields"][k] or "").strip()
    for k in FIELD_KEYS:
        if k != "extra":
            s["random"].setdefault(k, False)
    s.setdefault("random_people", copy.deepcopy(DEFAULT_CONFIG["scene"]["random_people"]))
    for k in PERSON_KEYS:
        s["random_people"].setdefault(k, False)
    people = list(s.get("people") or [])
    people = [p if isinstance(p, dict) else {} for p in people][:MAX_CHARACTERS]
    people += [{} for _ in range(MAX_CHARACTERS - len(people))]
    s["people"] = [{k: str(p.get(k, "") or "").strip() for k in PERSON_KEYS} for p in people]
    for i, p in enumerate(s["people"], 1):
        age = p["age"]
        nums = [int(x) for x in re.findall(r"\d+", age)]
        if nums and min(nums) < 18:
            errs.append(f"Character {i}: age must be 18 or older (got '{age}')")
    nc = s["fields"]["characters"]
    if nc and (not nc.isdigit() or not 1 <= int(nc) <= MAX_CHARACTERS):
        errs.append(f"Number of characters must be 1-{MAX_CHARACTERS} (or empty), got '{nc}'")
    if errs:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errs))
    if not b["host"]:
        b["host"] = DEFAULT_HOSTS[b["type"]]
    b["host"] = str(b["host"]).rstrip("/")
    return cfg


def redacted(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    if c["backend"].get("api_key"):
        c["backend"]["api_key"] = "***redacted***"
    return c


# =============================================================================
# Presets
# =============================================================================
def load_presets(cfg: dict) -> dict[str, list[str]]:
    presets = copy.deepcopy(DEFAULT_PRESETS)
    pf = cfg["generation"].get("presets_file")
    if pf:
        p = resolve_path(pf)
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                for k, v in data.items():
                    if isinstance(v, list):
                        presets[k] = [str(x).strip() for x in v if str(x).strip()]
                log.debug("Presets loaded from %s", p)
            except (OSError, json.JSONDecodeError) as e:
                log.warning("Could not read presets file %s (%s); using built-in presets", p, e)
    return presets


def save_presets(presets: dict, cfg: dict) -> Path:
    p = resolve_path(cfg["generation"]["presets_file"] or "presets.json")
    p.write_text(json.dumps(presets, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


# =============================================================================
# Logging
# =============================================================================
def setup_logging(cfg: dict, output_root: Path, extra_handlers: list[logging.Handler] | None = None,
                  console: bool = True) -> tuple[Path, Path | None]:
    lc = cfg["logging"]
    log_dir = Path(lc["directory"]).expanduser()
    if not log_dir.is_absolute():
        log_dir = output_root / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt_file = logging.Formatter("%(asctime)s.%(msecs)03d | %(levelname)-8s | %(threadName)s | %(message)s",
                                 datefmt="%Y-%m-%d %H:%M:%S")
    fmt_debug = logging.Formatter("%(asctime)s.%(msecs)03d | %(levelname)-8s | %(threadName)s | "
                                  "%(funcName)s:%(lineno)d | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fmt_console = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    for h in list(log.handlers):
        log.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    log.propagate = False
    log.setLevel(logging.DEBUG if lc["debug"] else logging.INFO)
    level = getattr(logging, str(lc["console_level"]).upper(), logging.INFO)

    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(level)
        ch.setFormatter(fmt_console)
        log.addHandler(ch)
    for h in extra_handlers or []:
        if h.formatter is None:
            h.setFormatter(fmt_console)
        log.addHandler(h)

    main_path = log_dir / lc["log_file"]
    mh = logging.handlers.RotatingFileHandler(main_path, maxBytes=int(lc["max_bytes"]),
                                              backupCount=int(lc["backup_count"]), encoding="utf-8")
    mh.setLevel(logging.INFO)
    mh.setFormatter(fmt_file)
    log.addHandler(mh)

    debug_path = None
    if lc["debug"]:
        debug_path = log_dir / lc["debug_file"]
        dh = logging.handlers.RotatingFileHandler(debug_path, maxBytes=int(lc["max_bytes"]),
                                                  backupCount=int(lc["backup_count"]), encoding="utf-8")
        dh.setLevel(logging.DEBUG)
        dh.setFormatter(fmt_debug)
        log.addHandler(dh)
    return main_path, debug_path


def trunc(text: str, n: int = 4000) -> str:
    return text if len(text) <= n else text[:n] + f"... [truncated, {len(text)} chars total]"


# =============================================================================
# LLM backends
# =============================================================================
@dataclass
class LLMResult:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    seconds: float = 0.0


class BaseBackend:
    name = "base"

    def __init__(self, cfg: dict, stop_event: threading.Event | None = None):
        self.b = cfg["backend"]
        self.m = cfg["model"]
        self.stop_event = stop_event

    def _request(self, method: str, url: str, payload: dict | None = None, retries: int | None = None) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.b.get("api_key"):
            headers["Authorization"] = f"Bearer {self.b['api_key']}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        retries = int(self.b["max_retries"]) if retries is None else retries
        delay = float(self.b["retry_backoff"])
        for attempt in range(retries + 1):
            log.debug("HTTP %s %s (attempt %d/%d)", method, url, attempt + 1, retries + 1)
            if payload is not None:
                log.debug("Request body: %s", trunc(json.dumps(payload, ensure_ascii=False)))
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=float(self.b["timeout"])) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                    log.debug("HTTP %d, body: %s", resp.status, trunc(body))
                    return json.loads(body) if body else {}
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                log.debug("HTTP error %d body: %s", e.code, trunc(body))
                if 400 <= e.code < 500 and e.code not in (408, 429):
                    if CONTEXT_ERR_RE.search(body):
                        raise ContextSizeError(self._explain_context_error(body)) from e
                    raise BackendError(self._explain_http_error(e.code, body, url)) from e
                err = f"HTTP {e.code}: {trunc(body, 300)}"
            except urllib.error.URLError as e:
                err = f"cannot reach {url}: {e.reason}"
            except (TimeoutError, ConnectionError, OSError) as e:
                err = f"{type(e).__name__}: {e}"
            except json.JSONDecodeError as e:
                err = f"server returned invalid JSON: {e}"
            if attempt < retries and not (self.stop_event and self.stop_event.is_set()):
                log.warning("Request failed (%s); retrying in %.1fs", err, delay)
                if self.stop_event:
                    self.stop_event.wait(delay)
                else:
                    time.sleep(delay)
                delay *= 2
            else:
                raise BackendError(f"Request failed after {attempt + 1} attempt(s): {err}")
        raise BackendError("unreachable")

    def _explain_context_error(self, body: str) -> str:
        n_prompt = n_ctx = None
        try:
            err = json.loads(body).get("error", {})
            n_prompt, n_ctx = err.get("n_prompt_tokens"), err.get("n_ctx")
        except (ValueError, AttributeError):
            pass
        need = (n_prompt or 2500) + int(self.m["max_tokens"])
        suggest = next(x for x in (4096, 8192, 16384, 32768, 65536) if x >= need * 1.1) if need < 60000 else need
        msg = "The prompt does not fit in the model's context window"
        if n_prompt and n_ctx:
            msg += f" ({n_prompt} prompt tokens, server context is {n_ctx})"
        msg += f".\nThe prompt plus up to {self.m['max_tokens']} output tokens needs about {need} tokens.\n"
        if self.name == "llamacpp":
            slots = getattr(self, "server_slots", None) or 0
            if slots > 1:
                msg += (f"llama-server is running {slots} parallel slots, which split the -c context between them "
                        f"({n_ctx or '?'} each).\nFix: add  -np 1  to the llama-server command "
                        f"(and keep -c at {suggest} or more).")
            else:
                msg += (f"Fix: restart llama-server with  -c {suggest}  (and -np 1; newer llama-server versions "
                        "split -c between several parallel slots by default).")
            if getattr(self, "n_ctx_train", None) and self.n_ctx_train < need:
                msg += (f"\nNote: this model was trained for only {self.n_ctx_train} tokens of context; output past "
                        "that is usually poor. A newer long-context model is recommended.")
        elif self.name == "llamacpp_python":
            msg += f"Fix: set Context size (n_ctx) to {suggest} or more."
        else:
            msg += f"Fix: load the model with a context length of {suggest} or more in your server's settings."
        return msg

    def _explain_http_error(self, code: int, body: str, url: str) -> str:
        msg = f"HTTP {code} from {url}: {trunc(body, 500)}"
        if code == 404 and self.name == "ollama":
            msg += f"\nHint: is the model pulled?  ollama pull {self.m['name']}"
        if code in (401, 403):
            msg += "\nHint: check the API key"
        return msg

    def check(self) -> None: ...
    def list_models(self) -> list[str]: return []
    def chat(self, messages, temperature=None, max_tokens=None, seed=None) -> LLMResult: raise NotImplementedError
    def close(self) -> None: ...


class OllamaBackend(BaseBackend):
    name = "ollama"

    def list_models(self) -> list[str]:
        data = self._request("GET", f"{self.b['host']}/api/tags", retries=0)
        return sorted(m.get("name", "?") for m in data.get("models", []))

    def check(self) -> None:
        models = self.list_models()
        want = self.m["name"]
        names = {want, f"{want}:latest"} if ":" not in want else {want}
        if not names & set(models):
            raise BackendError(f"Model '{want}' is not available in Ollama at {self.b['host']}.\n"
                               f"Installed: {', '.join(models) or '(none)'}\nPull it with:  ollama pull {want}")

    def chat(self, messages, temperature=None, max_tokens=None, seed=None) -> LLMResult:
        options = {
            "temperature": self.m["temperature"] if temperature is None else temperature,
            "top_p": self.m["top_p"],
            "num_predict": max_tokens or self.m["max_tokens"],
            "num_ctx": self.m["n_ctx"],
        }
        if seed is not None:
            options["seed"] = seed
        payload = {"model": self.m["name"], "messages": messages, "stream": False,
                   "options": options, "keep_alive": self.m["keep_alive"]}
        think = self.m.get("thinking", "off")
        if think in ("off", "on"):
            payload["think"] = think == "on"
        t0 = time.monotonic()
        try:
            data = self._request("POST", f"{self.b['host']}/api/chat", payload)
        except BackendError as e:
            if "think" in payload and "think" in str(e).lower() and "not support" in str(e).lower():
                payload.pop("think")              # older Ollama or a model without thinking support
                data = self._request("POST", f"{self.b['host']}/api/chat", payload)
            else:
                raise
        msg = data.get("message") or {}
        text = msg.get("content", "") or ""
        if not text.strip() and (msg.get("thinking") or "").strip():
            log.warning("The model spent its whole token budget thinking and gave no answer. Set Thinking to 'off' "
                        "(Backend & Model tab) or raise Max tokens.")
        return LLMResult(text, data.get("prompt_eval_count"), data.get("eval_count"), time.monotonic() - t0)


class OpenAICompatBackend(BaseBackend):
    def __init__(self, cfg, stop_event=None):
        super().__init__(cfg, stop_event)
        self.name = cfg["backend"]["type"]
        host = self.b["host"]
        self.base = host if host.endswith("/v1") else f"{host}/v1"

    def list_models(self) -> list[str]:
        data = self._request("GET", f"{self.base}/models", retries=0)
        return sorted(m.get("id", "?") for m in data.get("data", []))

    server_ctx: int | None = None      # usable context per request (per slot)
    server_slots: int | None = None
    n_ctx_train: int | None = None     # context length the model was trained for

    def _llamacpp_info(self) -> None:
        """llama-server: GET /props (slot context, slot count) and /v1/models (model metadata)."""
        root = self.base[:-3] if self.base.endswith("/v1") else self.base
        try:
            data = self._request("GET", f"{root}/props", retries=0)
            dgs = data.get("default_generation_settings") or {}
            for v in (dgs.get("n_ctx"), (dgs.get("params") or {}).get("n_ctx"), data.get("n_ctx")):
                if isinstance(v, int) and v > 0:
                    self.server_ctx = v
                    break
            if isinstance(data.get("total_slots"), int):
                self.server_slots = data["total_slots"]
        except BackendError as e:
            log.debug("Could not read llama-server /props: %s", e)
        try:
            models = self._request("GET", f"{self.base}/models", retries=0).get("data") or []
            meta = (models[0].get("meta") or {}) if models else {}
            if isinstance(meta.get("n_ctx_train"), int):
                self.n_ctx_train = meta["n_ctx_train"]
        except (BackendError, AttributeError, IndexError) as e:
            log.debug("Could not read llama-server model metadata: %s", e)

    def check(self) -> None:
        models = self.list_models()
        if self.name == "llamacpp":
            self._llamacpp_info()
            if self.server_ctx:
                slots = self.server_slots or 1
                log.info("llama-server context: %d tokens per request%s%s", self.server_ctx,
                         f" ({slots} parallel slots share {self.server_ctx * slots}; start with -np 1 to give "
                         f"one request all of it)" if slots > 1 else "",
                         f" | model trained for {self.n_ctx_train}" if self.n_ctx_train else "")
        if self.name != "llamacpp" and self.m["name"] and models and self.m["name"] not in models:
            log.warning("Model '%s' not in server list %s — the server may ignore or reject it",
                        self.m["name"], models)

    def chat(self, messages, temperature=None, max_tokens=None, seed=None) -> LLMResult:
        payload = {"model": self.m["name"] or "local-model", "messages": messages,
                   "temperature": self.m["temperature"] if temperature is None else temperature,
                   "top_p": self.m["top_p"], "max_tokens": max_tokens or self.m["max_tokens"], "stream": False}
        if seed is not None:
            payload["seed"] = seed
        think = self.m.get("thinking", "off")
        if think in ("off", "on"):                # Qwen3-style templates (llama-server, vLLM) read this
            payload["chat_template_kwargs"] = {"enable_thinking": think == "on"}
        t0 = time.monotonic()
        data = self._request("POST", f"{self.base}/chat/completions", payload)
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise BackendError(f"Unexpected response: {trunc(json.dumps(data), 500)}") from e
        usage = data.get("usage") or {}
        content = msg.get("content") or ""
        if not re.sub(r"(?s)<think>.*?(</think>|$)", "", content).strip() and \
                ((msg.get("reasoning_content") or "").strip() or "<think>" in content):
            log.warning("The model spent its whole token budget thinking and gave no answer "
                        "(%s tokens).%s", usage.get("completion_tokens", "?"),
                        " Retrying with /no_think." if self.m.get("thinking", "off") == "off"
                        else " Set Thinking to 'off' (Backend & Model tab) or raise Max tokens.")
        return LLMResult(content, usage.get("prompt_tokens"),
                         usage.get("completion_tokens"), time.monotonic() - t0)


class LlamaCppPythonBackend(BaseBackend):
    name = "llamacpp_python"
    _cache: dict = {}   # keep a loaded model between GUI runs

    def _load(self):
        key = (str(Path(self.m["path"]).expanduser()), int(self.m["n_ctx"]), int(self.m["n_gpu_layers"]))
        if key in self._cache:
            self.llm = self._cache[key]
            return
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as e:
            raise BackendError("backend 'llamacpp_python' needs:  pip install llama-cpp-python") from e
        if not Path(key[0]).is_file():
            raise BackendError(f"Model file not found: {key[0]}")
        log.info("Loading GGUF model %s (n_ctx=%s, n_gpu_layers=%s) ...", *key)
        t0 = time.monotonic()
        self._cache.clear()
        self.llm = Llama(model_path=key[0], n_ctx=key[1], n_gpu_layers=key[2], verbose=False)
        self._cache[key] = self.llm
        log.info("Model loaded in %.1fs", time.monotonic() - t0)

    def check(self) -> None:
        self._load()

    def list_models(self) -> list[str]:
        p = Path(self.m["path"]).expanduser()
        folder = p if p.is_dir() else p.parent
        return sorted(str(x) for x in folder.glob("*.gguf")) if folder.is_dir() else []

    def chat(self, messages, temperature=None, max_tokens=None, seed=None) -> LLMResult:
        self._load()
        log.debug("llama-cpp-python request: %s", trunc(json.dumps(messages, ensure_ascii=False)))
        kw = dict(messages=messages, temperature=self.m["temperature"] if temperature is None else temperature,
                  top_p=self.m["top_p"], max_tokens=max_tokens or self.m["max_tokens"])
        if seed is not None:
            kw["seed"] = seed
        t0 = time.monotonic()
        try:
            data = self.llm.create_chat_completion(**kw)
        except Exception as e:
            raise BackendError(f"llama-cpp-python generation failed: {e}") from e
        log.debug("llama-cpp-python response: %s", trunc(json.dumps(data, default=str, ensure_ascii=False)))
        usage = data.get("usage") or {}
        return LLMResult(data["choices"][0]["message"].get("content") or "", usage.get("prompt_tokens"),
                         usage.get("completion_tokens"), time.monotonic() - t0)


def make_backend(cfg: dict, stop_event: threading.Event | None = None) -> BaseBackend:
    t = cfg["backend"]["type"]
    if t == "ollama":
        return OllamaBackend(cfg, stop_event)
    if t in ("llamacpp", "openai"):
        return OpenAICompatBackend(cfg, stop_event)
    if t == "llamacpp_python":
        return LlamaCppPythonBackend(cfg, stop_event)
    raise ConfigError(f"Unknown backend type: {t}")


def list_models(cfg: dict) -> list[str]:
    validate_config(cfg)
    return make_backend(cfg).list_models()


def test_backend(cfg: dict) -> str:
    """Check server + model and do a tiny generation. Returns a status message."""
    validate_config(cfg)
    be = make_backend(cfg)
    be.check()
    res = be.chat([{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=20)
    return f"OK — {cfg['backend']['type']} replied in {res.seconds:.1f}s: {clean_response(res.text)[:60]!r}"


# =============================================================================
# System prompt
# =============================================================================
SFW_ADDENDUM = ("\n\nCONTENT RULE\nAll output must be safe for work: no nudity, no sexual content, "
                "no graphic violence. Characters are clothed appropriately for the setting.")


def word_range(cfg: dict) -> tuple[int, int]:
    g = cfg["generation"]
    prose_targets = [t for t in g["targets"] if t != "illustrious"] or ["illustrious"]  # tags cover Illustrious
    ranges = [TARGET_WORDS[t] for t in prose_targets if t in TARGET_WORDS] or [TARGET_WORDS["krea2"]]
    lo, hi = max(r[0] for r in ranges), min(r[1] for r in ranges)
    if lo > hi:
        lo = int(hi * 0.6)
    return (g["min_words"] or lo), (g["max_words"] or hi)


# =============================================================================
# Comic panel mode
# =============================================================================
COMIC_GUTTER = 12            # on the 0-1000 layout grid
COMIC_MAX_WORDS = 360        # ~480 tokens: Krea 2 reads 512 (507 usable)
BBOX_MAX_CHARS = 1800


def comic_on(cfg: dict) -> bool:
    return bool(cfg.get("comic", {}).get("enabled"))


def comic_grid(layout: str) -> list[dict]:
    """Panels in reading order with label and [x0,y0,x1,y1] box on the 0-1000 grid."""
    cols, rows = (int(x) for x in layout.split("x"))
    rn = {1: [""], 2: ["Top", "Bottom"], 3: ["Top", "Middle", "Bottom"]}[rows]
    cn = {1: [""], 2: ["Left", "Right"], 3: ["Left", "Center", "Right"]}[cols]
    g = COMIC_GUTTER
    w, h = (1000 - g * (cols - 1)) / cols, (1000 - g * (rows - 1)) / rows
    panels = []
    for r in range(rows):
        for c in range(cols):
            x0, y0 = round(c * (w + g)), round(r * (h + g))
            box = [x0, y0, min(1000, round(x0 + w)), min(1000, round(y0 + h))]
            pos = " ".join(x for x in (rn[r], cn[c]) if x)
            panels.append({"n": len(panels) + 1, "pos": pos, "box": box,
                           "label": f"Panel {len(panels) + 1}" + (f" ({pos})" if pos else "")})
    return panels


def char_slots(box: list[int], k: int) -> list[list[int]]:
    """k side-by-side character boxes inside a panel box."""
    x0, y0, x1, y1 = box
    pw, ph = x1 - x0, y1 - y0
    mx, top, bot = pw * 0.06, y0 + ph * 0.08, y1 - ph * 0.03
    sw = (pw - 2 * mx) / max(k, 1)
    return [[round(x0 + mx + i * sw), round(top), round(x0 + mx + (i + 1) * sw), round(bot)] for i in range(k)]


def suggested_size(layout: str, area: int = 1536 * 1536) -> str:
    """Image size (multiples of 16, ~Krea 2's 1536x1536 area) giving roughly square panels."""
    cols, rows = (int(x) for x in layout.split("x"))
    ratio = cols / rows
    w = (area * ratio) ** 0.5
    h = area / w
    return f"{int(round(w / 16) * 16)}x{int(round(h / 16) * 16)}"


def _role_id(role: str) -> str:
    words = re.sub(r"[^a-z0-9 ]", "", role.lower()).split()
    words = [w for w in words if w not in ("a", "an", "the", "on", "of", "in", "at")]
    return "_".join(words[-2:]) or "char"


def build_comic_seed(cfg: dict, presets: dict, rng: random.Random, base: str | None = None,
                     allow_random: bool = True) -> "SceneSeed":
    cm, sc = cfg["comic"], cfg["scene"]
    f, rnd = sc["fields"], sc["random"]
    use_random = allow_random and sc["random_enabled"]
    compat = cfg["generation"].get("compat_rules", True)
    panels = comic_grid(cm["layout"])
    shared, origin = build_details(cfg, presets, rng, allow_random)
    shared.pop("composition", None)
    origin.pop("composition", None)
    n = int(shared["characters"]) if shared.get("characters", "").isdigit() else None
    loc_mode = cm["location_mode"]
    vary_loc = loc_mode == "varied" or (loc_mode == "auto" and cm["mode"] == "vignettes")
    per_keys = ["camera_angle", "shot"]
    if vary_loc:
        per_keys.append("location")
    if cm["mode"] == "vignettes":
        per_keys.append("action")
    for k in per_keys:                 # these are rolled per panel instead
        if origin.get(k) == "random":
            shared.pop(k, None)
            origin.pop(k, None)

    used_locs: set[str] = set()
    for p in panels:
        pd = {}
        for k in per_keys:
            if f.get(k) and not (use_random and rnd.get(k, False)):
                pd[k] = f[k]
                continue
            if not (use_random and rnd.get(k, False)):
                continue
            base_c = compat_conflicts({**shared, **pd}, n) if compat else 0
            for _ in range(40 if compat else 1):
                cand = roll_field(k, presets, rng, n)
                if not cand:
                    break
                if k == "location" and cand in used_locs and len(presets.get("location", [])) > len(panels):
                    continue
                if not compat or compat_conflicts({**shared, **pd, k: cand}, n) <= base_c:
                    pd[k] = cand
                    break
        if pd.get("location"):
            used_locs.add(pd["location"])
        p["details"] = pd

    cols, rows = (int(x) for x in cm["layout"].split("x"))
    head = (f"Comic page: {len(panels)} panels in a grid of {cols} column{'s' * (cols > 1)} by "
            f"{rows} row{'s' * (rows > 1)}, read left to right, top to bottom.\n"
            f"Mode: {'story - consecutive moments of one short sequence' if cm['mode'] == 'story' else 'vignettes - separate moments with the same characters'}.")
    lines = [head]
    if base:
        lines.append(f"Scene idea: {base}")
    body = details_to_text(shared)
    if body:
        lines.append(body)
    ids = []
    if cm["format"] == "bbox":
        roles = [r.strip() for r in re.split(r",| and ", shared.get("subject", "")) if r.strip()]
        ids = [_role_id(r) for r in roles]
        if ids:
            lines.append("Character ids: " + ", ".join(f"{i} = {r}" for i, r in zip(ids, roles)))
    k_slots = n or len(ids) or 2
    for p in panels:
        parts = [f"{SEED_LABELS[k]}: {v}" for k, v in p["details"].items()]
        if cm["format"] == "bbox":
            b = p["box"]
            slots = "; ".join(f"[{s[0]},{s[1]},{s[2]},{s[3]}]" for s in char_slots(b, min(k_slots, 3)))
            lines.append(f"{p['label']}: p[{b[0]},{b[1]},{b[2]},{b[3]}]" + (" | " + " | ".join(parts) if parts else "")
                         + f" | character slots: {slots}")
        else:
            lines.append(f"{p['label']}:" + (" " + " | ".join(parts) if parts else " (your choice)"))
    meta = {"comic": {"layout": cm["layout"], "mode": cm["mode"], "format": cm["format"],
                      "panels": [{"label": p["label"], "box": p["box"], **p["details"]} for p in panels],
                      "suggested_size": suggested_size(cm["layout"])}}
    return SceneSeed("\n".join(lines), "comic", {**shared, **meta}, origin)


def comic_addendum(cfg: dict) -> str:
    cm = cfg["comic"]
    panels = comic_grid(cm["layout"])
    n = len(panels)
    cols, rows = (int(x) for x in cm["layout"].split("x"))
    mode = ("The panels show consecutive moments of one short sequence; each panel moves the action forward."
            if cm["mode"] == "story" else
            "Each panel is a separate moment with the same characters; the panels do not need to connect.")
    head = ("\n\n________________________________________\n"
            f"13. COMIC PANEL MODE\n\nThis image is ONE comic page divided into {n} panels arranged as "
            f"{cols} column{'s' * (cols > 1)} by {rows} row{'s' * (rows > 1)}, read left to right, top to bottom. "
            f"{mode} Characters must look identical in every panel.\n")
    no_mood = ("Spend words only on visible things; never describe mood, atmosphere, feelings or meaning.\n")
    if cm["format"] == "bbox":
        bubbles = ('Speech bubbles: in some panels add one line t[x0,y0,x1,y1]"<at most 8 words>" '
                   "speech bubble, white with black text, with the box inside that panel near the speaker.\n"
                   if cm["speech_bubbles"] else "Do not add any text elements.\n")
        return head + (
            "This section replaces the paragraph structure in sections 2, 4, 5 and 9. Output ONLY a grounded "
            "layout spec, one element per line, no paragraphs, no explanations, no markdown:\n"
            "Line 1: one plain sentence describing the whole page.\n"
            "Line 2: @<art style>; <medium>\n"
            "Line 3 (only when every panel shares one setting): ~<background or setting>\n"
            "Then one line per panel, copying its box exactly from the scene description: "
            "p[x0,y0,x1,y1] <the shot, camera angle, location and moment of that panel>\n"
            "Then the characters of each panel: ac:<id>[x0,y0,x1,y1] <description> for illustrated or anime styles, "
            "pe:<id>[x0,y0,x1,y1] <description> for photographic or realistic styles. Use the given character ids, "
            "reuse the same id in every panel, and repeat each character's key look (adult age, hair, clothing) "
            "briefly in every line together with that panel's pose and limbs. Every character box must lie "
            "inside its panel box; use the suggested character slots.\n"
            "Optional objects: o[x0,y0,x1,y1] <short description>, inside their panel.\n" + bubbles +
            "Coordinates are integers from 0 to 1000, x first: [x0,y0,x1,y1] with x0<x1 and y0<y1, origin top-left. "
            "No space between the tag and '['.\n"
            f"Keep every description short: the whole spec must stay under {BBOX_MAX_CHARS} characters.\n" + no_mood)
    per = max(25, (COMIC_MAX_WORDS - 90) // n)
    bubbles = ('Speech bubbles: in some panels add one short line written as: a speech bubble near the '
               '<character> reads "<at most 8 words>".\n' if cm["speech_bubbles"]
               else "Do not add speech bubbles, captions or any other text.\n")
    return head + (
        "This section replaces the paragraph structure in sections 2, 4, 5 and 9. Write the prompt in this order:\n"
        "1. One opening sentence: a comic page, the style, the number of panels and the grid, thin black panel "
        "borders and white gutters.\n"
        "2. A paragraph that starts with \"Characters:\" and describes each character once in one sentence: "
        "adult age, build, skin tone, hair, face and clothing. Later panels refer to them only by role "
        "(for example \"the knight\").\n"
        "3. One paragraph per panel, in order, each starting exactly with its label as given, for example "
        f"\"{panels[0]['label']}:\". In 1-3 sentences give the shot and camera angle, which characters are "
        "visible and where they are inside the panel (left, center, right, foreground, background), their poses "
        "with specific limbs, and the panel's location.\n"
        "4. One final sentence on the shared lighting, color palette and line style.\n"
        f"Keep each panel to about {per} words and the whole prompt under {COMIC_MAX_WORDS} words.\n"
        + bubbles + no_mood)


_BBOX_LINE = re.compile(r"^(p|o|t|pe|ac|fc)(:[a-z0-9_]+)?\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\](.*)$")


def normalize_bbox(text: str) -> str:
    """Tidy a grounded spec: fix 'p [' spacing, drop stray lines after the header."""
    out = []
    for i, ln in enumerate(l.strip() for l in text.splitlines()):
        if not ln:
            continue
        ln = re.sub(r"^(p|o|t|pe|ac|fc)(\s*:\s*([A-Za-z0-9_]+))?\s+\[", lambda m: m.group(1) + (
            ":" + m.group(3).lower() if m.group(3) else "") + "[", ln)
        ln = re.sub(r"^(pe|ac|fc):([A-Za-z0-9_]+)\[", lambda m: f"{m.group(1)}:{m.group(2).lower()}[", ln)
        if _BBOX_LINE.match(ln) or ln[0] in "@~" or not out:
            out.append(ln)
    return "\n".join(out)


def validate_comic(text: str, cfg: dict) -> list[str]:
    cm = cfg["comic"]
    panels = comic_grid(cm["layout"])
    n = len(panels)
    problems = []
    if cm["format"] == "bbox":
        lines = [l for l in text.splitlines() if l.strip()]
        boxes, chars = [], 0
        for ln in lines:
            m = _BBOX_LINE.match(ln)
            if not m:
                continue
            tag, x0, y0, x1, y1 = m.group(1), *(int(m.group(i)) for i in range(3, 7))
            if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
                problems.append(f"bad box {ln[:30]}")
                continue
            if tag == "p":
                boxes.append((x0, y0, x1, y1))
            else:
                if tag in ("pe", "ac", "fc"):
                    chars += 1
                    if not m.group(2):
                        problems.append("character line without an id")
                if boxes or True:
                    inside = any(x0 >= b[0] - 20 and y0 >= b[1] - 20 and x1 <= b[2] + 20 and y1 <= b[3] + 20
                                 for b in boxes)
                    if boxes and not inside:
                        problems.append(f"element outside its panel: {ln[:30]}")
        if len(boxes) != n:
            problems.append(f"{len(boxes)} panel lines, expected {n}")
        if not chars:
            problems.append("no character lines")
        if len(text) > BBOX_MAX_CHARS * 1.1:
            problems.append(f"too long ({len(text)} chars, max {BBOX_MAX_CHARS})")
        return problems[:4]
    labels = re.findall(r"(?im)^\s*panel\s+(\d+)\b", text)
    if len(set(labels)) != n:
        problems.append(f"{len(set(labels))} panel paragraphs, expected {n}")
    wc = word_count(text)
    if wc > COMIC_MAX_WORDS * 1.12:
        problems.append(f"too long ({wc} words, max {COMIC_MAX_WORDS})")
    if not re.search(r"(?im)^\s*characters?\s*:", text):
        problems.append("missing 'Characters:' paragraph")
    return problems


def generator_addendum(cfg: dict) -> str:
    lo, hi = word_range(cfg)
    return (
        "\n\n________________________________________\n"
        "12. LENGTH AND FOCUS\n\n"
        f"The complete prompt must be between {lo} and {hi} words.\n"
        "Keep the opening to one sentence, each character paragraph to 2-4 sentences, and the environment "
        "and composition paragraphs to 2-3 sentences each.\n"
        "Spend words only on what a camera can see. Never describe mood, atmosphere, feelings, meaning, "
        "symbolism or the viewer's experience (no phrases such as \"a sense of\", \"evoking\", "
        "\"poignant\", \"dreamlike\", \"ethereal\", \"serene atmosphere\", \"captures the essence\").\n"
        "If the requested details are physically hard to combine, choose the most plausible visible "
        "arrangement and describe only that, without commenting on it."
    )


# Content guard. Always on when NSFW is allowed; it cannot be switched off.
SCENE_INPUT_ADDENDUM = (
    "\n\n________________________________________\n"
    "11. SCENE INPUT FORMAT\n\n"
    "The scene description you receive is a list of labeled details (for example Style, Number of characters, "
    "Characters, Action, Camera angle, Shot / distance, Lighting, Location, Time / weather, Focus / composition, "
    "Additional notes), sometimes preceded by a short free-text scene idea.\n"
    "Some details were chosen by the user and some were assembled at random; treat all of them as the user's request.\n"
    "Every detail that is given must appear in the final prompt and must not be contradicted: keep the exact style, "
    "number of characters, character roles, camera angle, shot distance, lighting, location and time/weather that "
    "are listed.\n"
    "Place each detail where the instructions above require it (style, camera angle, distance and lighting in the "
    "opening sentence; characters in their own paragraphs; location and weather in the environment paragraph; "
    "focus and composition in the final paragraph).\n"
    "Lines starting with \"Character 1:\", \"Character 2:\" and so on describe the characters in that order: use "
    "each line's role, age, ethnicity, hair and clothing for that character's paragraph.\n"
    "Any detail that is not listed is your choice: pick what best fits the scene.\n"
    "Invent all remaining concrete visual details yourself (appearance, clothing, posture, limbs, objects, surfaces).\n"
    "Each response is an independent image; do not refer to previous prompts.\n"
    "The output must consist solely of the prompt itself, with no explanations, preambles or postscripts."
)

ADULT_RULE = (
    "\n\nCONTENT SAFETY (always applies)\n"
    "Every character is an adult (at least 18) with a clearly adult face and body; give each character an adult age. "
    "Never depict minors or childlike characters, and never use school settings or school uniforms, in any nude, "
    "sexual or romantic context. Intimate content is only between clearly consenting, fully aware adults who are not "
    "related to each other, with no teacher-student or other coercive power dynamic."
)
_MINOR_RE = re.compile(
    r"\b(child|children|kid|kids|minor|minors|underage|under-age|teen|teens|teenage|teenaged|teenager|teenagers|"
    r"adolescent|adolescents|preteen|pre-teen|tween|tweens|schoolgirl|schoolgirls|schoolboy|schoolboys|"
    r"school uniform|serafuku|randoseru|kindergarten|elementary school|middle school|junior high|high school|"
    r"loli|lolis|lolicon|shota|shotacon|little girl|little boy|young girl|young boy|toddler|infant|pubescent|"
    r"prepubescent|aged down|child-like|childlike|petite child)\b"
    r"|\b(1[0-7]|[1-9])[- ]?(year|yr)s?[- ]old\b|\baged? (1[0-7]|[1-9])\b"
    r"|\b(early|mid|late)[- ]teens\b|\bin (her|his|their) teens\b", re.I)
_FAMILY_RE = re.compile(
    r"\b(incest|incestuous|step-?(mother|father|mom|dad|sister|brother|daughter|son|sibling)s?|"
    r"(his|her|their|my|your) (own )?(mother|father|mom|dad|sister|brother|daughter|son|aunt|uncle|niece|nephew|"
    r"cousin|grandmother|grandfather|grandson|granddaughter|sibling)s?|siblings|twin (sisters|brothers))\b", re.I)
_NONCON_RE = re.compile(
    r"\b(rape|raped|raping|non-?consensual|against (her|his|their) will|forced|forcing|coerced|coercion|"
    r"molest\w*|drugged|unconscious|passed out|asleep|sleeping|intoxicated|drunk|blackmail\w*|sexual assault|"
    r"assaulted|teacher and (his |her |their )?student|student and (his |her |their )?teacher|"
    r"student-teacher|teacher-student|(professor|tutor) and (his |her |their )?student)\b", re.I)
_SEXUAL_RE = re.compile(
    r"\b(nude|nudity|naked|topless|bottomless|undressed|sex|sexual|sexually|erotic|aroused|arousal|orgasm|"
    r"intercourse|penetrat\w*|genital\w*|penis|vagina|vulva|labia|clitoris|erection|nipples?|breasts? exposed|"
    r"bare (breasts|chest|buttocks)|lingerie|nsfw|explicit|lewd)\b", re.I)


def content_problem(text: str, cfg: dict) -> str | None:
    """Returns a reason string if the text breaks the content-safety rule, else None."""
    if cfg["generation"]["content_rating"] == "sfw":
        sexual = bool(_SEXUAL_RE.search(text))
        return "content rule: sexual content in SFW mode" if sexual and _MINOR_RE.search(text) else None
    if _MINOR_RE.search(text):
        return "content rule: describes a minor"
    sexual = bool(_SEXUAL_RE.search(text))
    if sexual and _FAMILY_RE.search(text):
        return "content rule: intimate content between relatives"
    if sexual and _NONCON_RE.search(text):
        return "content rule: non-consensual content"
    return None


def check_scene_inputs(cfg: dict, fixed_scenes: list[str] | None, presets: dict | None) -> None:
    """Refuse to start when user-entered scene details break the content rule; drop bad preset entries."""
    texts = [v for v in cfg["scene"]["fields"].values() if v] + list(fixed_scenes or [])
    texts += [v for p in cfg["scene"].get("people", []) for v in p.values() if v]
    for t in texts:
        if content_problem(t, cfg) or (cfg["generation"]["content_rating"] != "sfw" and _MINOR_RE.search(t)):
            raise ConfigError("A scene field or scene idea describes a minor or other content this tool won't "
                              "generate. With NSFW allowed, every character must be an adult. Please change it.")
    if presets and cfg["generation"]["content_rating"] != "sfw":
        for k, vals in presets.items():
            bad = [v for v in vals if _MINOR_RE.search(v) or _FAMILY_RE.search(v) or _NONCON_RE.search(v)]
            if bad:
                presets[k] = [v for v in vals if v not in bad]
                log.warning("Presets '%s': skipped %d entr%s that conflict with the content rule",
                            k, len(bad), "y" if len(bad) == 1 else "ies")


SYSTEM_PROMPT_URL = "https://civitai.red/models/2976901/moody-krea2-prompt-writing-skill-llm-system-prompt"


def read_system_prompt_file(cfg: dict) -> str:
    path = resolve_path(cfg["generation"]["system_prompt_file"])
    if not path.is_file():
        raise ConfigError(
            f"System prompt file not found: {path}\n"
            "It isn't included with the app. Download it from " + SYSTEM_PROMPT_URL +
            " and save it as system_prompt.txt in the app folder, or write your own "
            "(the System Prompt tab can save it). See 'Notes: the system prompt' in the README.")
    return path.read_text(encoding="utf-8", errors="replace")


def prepare_system_prompt(cfg: dict, text: str | None = None) -> str:
    text = (text if text is not None and text.strip() else read_system_prompt_file(cfg)).strip()
    if not text:
        raise ConfigError("System prompt is empty")
    if cfg["generation"]["content_rating"] == "sfw":
        kept = [ln for ln in text.splitlines() if "nsfw" not in ln.lower()]
        log.info("SFW mode: removed %d NSFW line(s) from the system prompt", len(text.splitlines()) - len(kept))
        text = "\n".join(kept) + SFW_ADDENDUM
    else:
        text += ADULT_RULE
    if not comic_on(cfg) and "SCENE INPUT FORMAT" not in text.upper():
        text += SCENE_INPUT_ADDENDUM      # how to read the app's labeled scene details (any system prompt)
    text += comic_addendum(cfg) if comic_on(cfg) else generator_addendum(cfg)
    log.info("System prompt ready (%d chars, target %d-%d words)", len(text), *word_range(cfg))
    log.debug("System prompt:\n%s", text)
    return text


# =============================================================================
# Scene builder
# =============================================================================
@dataclass
class SceneSeed:
    text: str
    source: str
    details: dict = field(default_factory=dict)   # key -> value
    origin: dict = field(default_factory=dict)    # key -> "user" | "random"


def _join_roles(roles: list[str]) -> str:
    return roles[0] if len(roles) == 1 else ", ".join(roles[:-1]) + " and " + roles[-1]


def roll_field(key: str, presets: dict, rng: random.Random, n_chars: int | None = None) -> str:
    """One random value for a field (used by the GUI's dice buttons too)."""
    if key == "characters":
        return str(rng.choices(list(CHAR_COUNT_WEIGHTS), weights=list(CHAR_COUNT_WEIGHTS.values()))[0])
    if key == "subject":
        n = n_chars or int(roll_field("characters", presets, rng))
        pool = presets.get("subject") or ["a person"]
        roles = rng.sample(pool, min(n, len(pool))) if len(pool) >= n else [rng.choice(pool) for _ in range(n)]
        return _join_roles(roles)
    if key == "action":
        pool = list(presets.get("action", []))
        if n_chars is None or n_chars >= 2:
            pool += presets.get("action_group", [])
        if n_chars and n_chars >= 2 and presets.get("action_group") and rng.random() < 0.5:
            pool = list(presets["action_group"])
        return rng.choice(pool) if pool else ""
    pool = presets.get(key) or []
    if key == "shot" and n_chars and n_chars >= 3:   # close shots can't show 3+ people
        pool = [x for x in pool if "close" not in x.lower()] or pool
    return rng.choice(pool) if pool else ""


_OUTDOOR_LOC = re.compile(r"street|field|ridge|highway|beach|clearing|courtyard|rooftop|dock|harbor|harbour|"
                          r"vineyard|garden|alley|market|deck|platform|forest|park|mountain|desert|lake|river|"
                          r"bridge|parking lot|meadow|cliff|shore|plaza|square|outdoor|outside", re.I)
_INDOOR_LIGHT = re.compile(r"softbox|studio|fluorescent|practical lamp|candle|window light|lamp", re.I)
_SUN_LIGHT = re.compile(r"midday sun|harsh sun|golden hour|sunlight", re.I)
_WEATHER = re.compile(r"rain|snow|fog|storm|thunder|wind|wet ground|haz", re.I)
_DARK_TIME = re.compile(r"night|blue hour", re.I)


def compat_conflicts(d: dict, n: int | None) -> int:
    """Number of physically/visually contradictory combinations in a scene's details."""
    style, loc, light = d.get("style", ""), d.get("location", ""), d.get("lighting", "")
    tw, shot, cam, comp = d.get("time_weather", ""), d.get("shot", ""), d.get("camera_angle", ""), d.get("composition", "")
    outdoor = bool(loc) and bool(_OUTDOOR_LOC.search(loc))
    indoor = bool(loc) and not outdoor
    c = 0
    if "studio" in style.lower() and outdoor:
        c += 1
    if "street photography" in style.lower() and indoor:
        c += 1
    if light and _INDOOR_LIGHT.search(light) and outdoor:
        c += 1
    if light and _SUN_LIGHT.search(light) and (indoor and "window" not in light.lower()):
        c += 1
    if tw and _WEATHER.search(tw) and indoor:
        c += 1
    if light and tw:
        l, t = light.lower(), tw.lower()
        if _SUN_LIGHT.search(l) and (_DARK_TIME.search(t) or re.search(r"overcast|rain|snow|fog|storm", t)):
            c += 1
        if "golden hour" in l and "midday" in t:
            c += 1
        if "midday" in l and re.search(r"sunset|morning|late afternoon", t):
            c += 1
        if "moonlight" in l and not re.search(r"night|blue hour|snow|fog", t):
            c += 1
        if "daylight" in l and _DARK_TIME.search(t):
            c += 1
    s = shot.lower()
    if n:
        if "extreme close" in s and n > 1:
            c += 1
        elif "close-up" in s and "medium" not in s and n > 1:
            c += 1
        elif "medium close" in s and n > 2:
            c += 1
        if "over-the-shoulder" in cam.lower() and n < 2:
            c += 1
    if "cowboy" in s and re.search(r"overhead|top-down|worm", cam, re.I):
        c += 1
    if "small in a large" in comp.lower() and "wide" not in s:
        c += 1
    if "tight framing" in comp.lower() and "wide" in s:
        c += 1
    if "close" in s and re.search(r"environment|establishing", comp, re.I):
        c += 1
    return c


def fixed_fields(cfg: dict, allow_random: bool = True) -> dict:
    """Typed scene values that are actually used: a ticked Random box (with Random ON) wins over typed text."""
    sc = cfg["scene"]
    use_random = allow_random and sc["random_enabled"]
    return {k: v for k, v in sc["fields"].items()
            if v and not (use_random and sc["random"].get(k, False))}


def build_details(cfg: dict, presets: dict, rng: random.Random, allow_random: bool = True) -> tuple[dict, dict]:
    sc = cfg["scene"]
    rnd = sc["random"]
    use_random = allow_random and sc["random_enabled"]
    compat = cfg["generation"].get("compat_rules", True)
    f = {k: "" for k in FIELD_KEYS}
    f.update(fixed_fields(cfg, allow_random))            # random-ticked fields count as empty here
    details = {k: f[k] for k in FIELD_KEYS if f[k]}
    origin = {k: "user" for k in details}

    def can_roll(k):
        return use_random and rnd.get(k, False) and k not in details

    n = int(f["characters"]) if f["characters"] else None
    if n is None and can_roll("characters") and not f["subject"]:
        n = int(roll_field("characters", presets, rng))
        details["characters"], origin["characters"] = str(n), "random"

    for k in ("style", "subject", "action", "location", "time_weather", "lighting", "shot",
              "camera_angle", "composition"):
        if not can_roll(k):
            continue
        base = compat_conflicts(details, n) if compat else 0
        val = ""
        for _ in range(40 if compat else 1):
            cand = roll_field(k, presets, rng, n)
            if not cand:
                break
            if not compat or compat_conflicts({**details, k: cand}, n) <= base:
                val = cand
                break
        if val:
            details[k], origin[k] = val, "random"
    if compat and compat_conflicts(details, n) and any(origin.get(k) == "user" for k in details):
        log.debug("Your fixed fields contain a combination that may be hard to show in one image: %s", details)
    ordered = {k: details[k] for k in FIELD_KEYS if k in details}
    out_origin = {k: origin[k] for k in ordered}
    people = build_people(cfg, presets, rng, n, use_random, ordered, out_origin)
    if people:
        ordered["people"] = people
    return ordered, out_origin


def build_people(cfg: dict, presets: dict, rng: random.Random, n: int | None, use_random: bool,
                 details: dict, origin: dict) -> list[dict]:
    """Per-character details for the first n characters (user value, random, or left to the LLM)."""
    sc = cfg["scene"]
    slots, rnd = sc.get("people") or [], sc.get("random_people") or {}
    if not n:
        n = len([p for p in slots if any(p.values())]) or 0
    if not n:
        return []
    people = []
    for i in range(min(n, MAX_CHARACTERS)):
        slot = slots[i] if i < len(slots) else {}
        p = {}
        for k in PERSON_KEYS:
            v = (slot.get(k) or "").strip()
            if use_random and rnd.get(k):
                v = ""                                   # ticked Random wins over typed text
            if not v and use_random and rnd.get(k) and k != "role":
                pool = presets.get(k) or []
                v = rng.choice(pool) if pool else ""
            if v:
                p[k] = v
        people.append(p)
    # roles: per-person roles build the subject when the subject field is empty
    user_roles = [p.get("role") for p in people]
    if any(user_roles) and origin.get("subject") != "user":
        pool = presets.get("role") or presets.get("subject") or ["a person"]
        roles = [r or (rng.choice(pool) if use_random and rnd.get("role") else "") for r in user_roles]
        if all(roles):
            details["subject"] = _join_roles(roles)
            origin["subject"] = "user" if all(user_roles) else "mixed"
        for p, r in zip(people, roles):
            if r:
                p["role"] = r
    elif details.get("subject") and not any(user_roles):
        roles = [r.strip() for r in re.split(r",\s*|\s+and\s+", details["subject"]) if r.strip()]
        if len(roles) == len(people):          # random or typed subject: label each character line
            for p, r in zip(people, roles):
                p.setdefault("role", r)
    return people if any(people) else []


def person_line(i: int, p: dict) -> str:
    parts = []
    if p.get("role"):
        parts.append(p["role"])
    if p.get("age"):
        parts.append(f"age {p['age']}")
    if p.get("ethnicity"):
        parts.append(p["ethnicity"])
    color, length = p.get("hair_color"), p.get("hair_length")
    if color and length:
        parts.append(f"{color} hair ({length})")
    elif color:
        parts.append(f"{color} hair")
    elif length:
        parts.append(f"hair: {length}")
    if p.get("clothing"):
        parts.append(f"wearing {p['clothing']}")
    return f"Character {i}: " + "; ".join(parts) if parts else ""


def details_to_text(details: dict) -> str:
    lines = []
    for k in FIELD_KEYS:
        if details.get(k):
            lines.append(f"{SEED_LABELS[k]}: {details[k]}")
        if k == "subject":
            for i, p in enumerate(details.get("people") or [], 1):
                line = person_line(i, p)
                if line:
                    lines.append(line)
    return "\n".join(lines)


class SceneGenerator:
    def __init__(self, cfg: dict, rng: random.Random, backend: BaseBackend | None = None,
                 fixed_scenes: list[str] | None = None, presets: dict | None = None):
        self.cfg = cfg
        self.g = cfg["generation"]
        self.rng = rng
        self.backend = backend
        self.fixed = [s for s in (fixed_scenes or []) if s.strip()]
        self.presets = presets or load_presets(cfg)
        self.file_scenes: list[str] = []
        if self.g["seed_mode"] == "file" and not self.fixed:
            p = resolve_path(self.g["scenes_file"])
            if not p.is_file():
                raise ConfigError(f"Scenes file not found: {p}")
            self.file_scenes = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
                                if ln.strip() and not ln.lstrip().startswith("#")]
            if not self.file_scenes:
                raise ConfigError(f"Scenes file has no usable lines: {p}")
            log.info("Loaded %d scene(s) from %s", len(self.file_scenes), p)

    def _compose(self, base: str | None, details: dict, origin: dict, source: str) -> SceneSeed:
        body = details_to_text(details)
        if base and body:
            text = f"Scene idea: {base}\n{body}"
        elif base:
            text = f"Scene idea: {base}"
        elif body:
            text = body
        else:
            text = "No details given: invent an original scene of your choice with at least one character."
        return SceneSeed(text, source, details, origin)

    def next(self, index: int, dry_run: bool = False) -> SceneSeed:
        if comic_on(self.cfg):
            if self.fixed:
                return build_comic_seed(self.cfg, self.presets, self.rng, self.fixed[index % len(self.fixed)], False)
            if self.g["seed_mode"] == "file":
                return build_comic_seed(self.cfg, self.presets, self.rng, self.rng.choice(self.file_scenes), False)
            return build_comic_seed(self.cfg, self.presets, self.rng)
        if self.fixed:   # explicit scene text: only the user's own field values are added
            d, o = build_details(self.cfg, self.presets, self.rng, allow_random=False)
            return self._compose(self.fixed[index % len(self.fixed)], d, o, "scene-text")
        mode = self.g["seed_mode"]
        if mode == "file":
            d, o = build_details(self.cfg, self.presets, self.rng, allow_random=False)
            return self._compose(self.rng.choice(self.file_scenes), d, o, "file")
        d, o = build_details(self.cfg, self.presets, self.rng)
        if mode == "llm" and not dry_run and self.backend:
            idea = self._llm_idea(d)
            if idea:
                return self._compose(idea, d, o, "llm")
        return self._compose(None, d, o, "fields")

    def _llm_idea(self, details: dict) -> str | None:
        req = details_to_text(details) or "(none — anything goes)"
        ask = ("Invent one original, specific scene idea for a single image in 1-2 plain sentences. "
               "It must fit all of these details and must not contradict them:\n" + req +
               "\nOutput only the scene idea, nothing else.")
        try:
            res = self.backend.chat([{"role": "user", "content": ask}],
                                    temperature=min(1.3, float(self.backend.m["temperature"]) + 0.2), max_tokens=200)
            text = clean_response(res.text).replace("\n", " ").strip()
            if len(text) < 15:
                raise BackendError("scene idea too short")
            return text
        except BackendError as e:
            log.warning("LLM scene idea failed (%s); using the details only", e)
            return None


# =============================================================================
# Response cleaning & validation
# =============================================================================
THINK_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.S | re.I)
PREAMBLE_RE = re.compile(r"^\s*(sure|certainly|of course|okay|ok|here(?:'s| is| are)|below is|the final|"
                         r"final prompt|image prompt|prompt)\b[^\n]{0,120}[:.!]\s*$", re.I)
LABEL_RE = re.compile(r"^\s*(\*\*|__)?\s*(paragraph\s*\d+|para\s*\d+|p\d+)\s*[:.\-—]\s*(\*\*|__)?\s*", re.I)
BULLET_RE = re.compile(r"^\s*([-*•●▪]|\d+[.)])\s+")
POSTSCRIPT_RE = re.compile(r"^\s*(note|notes|explanation|i hope|let me know|this prompt)\b", re.I)


def clean_response(text: str, strip_reasoning: bool = True, clean: bool = True) -> str:
    if strip_reasoning:
        text = THINK_RE.sub("", text)
        if re.search(r"</think>", text, re.I):
            text = re.split(r"</think>", text, flags=re.I)[-1]
    text = text.replace("\r\n", "\n").strip()
    if not clean:
        return text
    text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text.strip()).strip()
    lines = text.split("\n")
    while lines and (PREAMBLE_RE.match(lines[0]) or not lines[0].strip()):
        lines.pop(0)
    out = []
    for ln in lines:
        if ln.lstrip().startswith("#"):
            continue
        ln = BULLET_RE.sub("", LABEL_RE.sub("", ln)).replace("**", "").replace("__", "")
        out.append(ln.rstrip())
    while out and (not out[-1].strip() or POSTSCRIPT_RE.match(out[-1])):
        out.pop()
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    if len(text) > 1 and text[0] == '"' and text[-1] == '"' and text.count('"') == 2:
        text = text[1:-1].strip()
    return text


def paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) == 1 and text.count("\n") >= 2:
        paras = [p.strip() for p in text.split("\n") if p.strip()]
    return paras


def validate_prompt(text: str, min_paragraphs: int, cfg: dict | None = None) -> list[str]:
    if not text.strip():
        return ["empty response"]
    problems = []
    if cfg is not None:
        cp = content_problem(text, cfg)
        if cp:
            return [cp]
        g = cfg["generation"]
        if comic_on(cfg):
            problems += validate_comic(text, cfg)
            if g["filler_check"]:
                hits = filler_hits(text)
                if len(hits) > g["max_filler_hits"]:
                    problems.append(f"mood/filler phrases: {', '.join(sorted(set(hits)))}")
            return problems
        lo, hi = word_range(cfg)
        wc = word_count(text)
        if wc > hi * 1.15:
            problems.append(f"too long ({wc} words, target {lo}-{hi})")
        elif wc < lo * 0.7:
            problems.append(f"too short ({wc} words, target {lo}-{hi})")
        if g["filler_check"]:
            hits = filler_hits(text)
            if len(hits) > g["max_filler_hits"]:
                problems.append(f"mood/filler phrases: {', '.join(sorted(set(hits)))}")
    n = len(paragraphs(text))
    if n < min_paragraphs:
        problems.append(f"only {n} paragraph(s), expected >= {min_paragraphs}")
    letters = [c for c in text if c.isalpha()]
    if letters and sum(ord(c) > 127 for c in letters) / len(letters) > 0.15:
        problems.append("does not look like English")
    if len(text) < 300:
        problems.append(f"too short ({len(text)} chars)")
    if re.search(r"^\s*([-*•]|\d+[.)])\s", text, re.M):
        problems.append("contains bullet points")
    return problems


# ---- scene detail check: does the prompt actually use the details it was given? ----
_DECADES = {"teens": 10, "twenties": 20, "thirties": 30, "forties": 40, "fifties": 50, "sixties": 60,
            "seventies": 70, "eighties": 80, "nineties": 90}
_AGE_RE = re.compile(
    r"\b(?:(early|mid|late)[\s-]*)?(?:(\d)0'?s|(" + "|".join(_DECADES) + r"))\b"
    r"|\b(\d{2})[\s-]*(?:years?|yrs?)[\s-]*old\b|\b(?:aged?|age)\s+(\d{2})\b", re.I)
_STOP = {"and", "the", "with", "for", "from", "into", "onto", "its", "his", "her", "their", "of", "in", "on",
         "at", "to", "a", "an", "or", "matching", "role", "view", "shot", "light", "lighting", "style"}
_DETAIL_SCENE_KEYS = ("camera_angle", "shot", "lighting", "time_weather", "location")
_DETAIL_PERSON_KEYS = ("hair_color", "hair_length", "ethnicity", "clothing")


def _age_ranges(text: str) -> list[tuple[int, int, str]]:
    """Every age mention in the text as (low, high, the words used)."""
    out = []
    for m in _AGE_RE.finditer(text):
        part, dig, word, years, aged = m.groups()
        if years or aged:
            n = int(years or aged)
            if 16 <= n <= 99:
                out.append((n, n, m.group(0)))
            continue
        d = int(dig) * 10 if dig else _DECADES[word.lower()]
        if d < 10:
            continue
        lo, hi = {"early": (d, d + 3), "mid": (d + 4, d + 6), "late": (d + 7, d + 9)}.get(
            (part or "").lower(), (d, d + 9))
        out.append((lo, hi, m.group(0)))
    return out


def _tokens(value: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", value.lower()) if len(t) >= 3 and t not in _STOP]


def _has_words(value: str, text_words: set[str], share: float) -> bool | None:
    """True if enough of value's words appear in the text (prefix match, so 'Latino' finds 'Latina').
    None = too long or vague to check. 'A/B' values pass if either alternative is used."""
    if "/" in value:
        alts = [_has_words(a, text_words, share) for a in value.split("/") if a.strip()]
        return None if all(x is None for x in alts) else any(alts)
    toks = _tokens(value)
    if not toks or len(toks) > 6:
        return None
    found = sum(any(w.startswith(t[:max(4, len(t) - 2)]) for w in text_words) for t in toks)
    return found >= max(1, math.ceil(share * len(toks)))


def detail_problems(text: str, details: dict | None) -> list[str]:
    """Scene details that the prompt ignored or contradicted (ages, hair, clothing, camera, lighting...)."""
    if not details:
        return []
    probs = []
    people = details.get("people") or []
    asked = [(p["age"], _age_ranges(p["age"])) for p in people if p.get("age")]
    asked = [(a, r[0]) for a, r in asked if r]
    if asked:
        found = _age_ranges(text)
        free = list(found)
        missing = []
        for a, (lo, hi, _) in asked:
            hit = next((f for f in free if f[0] <= hi and f[1] >= lo), None)
            if hit:
                free.remove(hit)
            else:
                missing.append(a)
        all_set = len(asked) == len(people)
        wrong = [f[2] for f in free if not any(f[0] <= hi and f[1] >= lo for _, (lo, hi, _) in asked)] if all_set else []
        if missing or wrong:
            probs.append("ages don't match (asked for " + ", ".join(a for a, _ in asked) + "; prompt says "
                         + (", ".join(f[2] for f in found) or "no ages") + ")")
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    for k in _DETAIL_SCENE_KEYS:
        v = details.get(k)
        if v and _has_words(v, words, 0.5) is False:
            probs.append(f"{SEED_LABELS[k].lower()} '{v}' not used")
    for i, p in enumerate(people, 1):
        for k in _DETAIL_PERSON_KEYS:
            v = p.get(k)
            if v and _has_words(v, words, 0.5) is False:
                probs.append(f"character {i} {k.replace('_', ' ')} '{v}' not used")
    return probs


def correction_message(probs: list[str]) -> str:
    return ("Your prompt does not follow these required details from the scene description:\n"
            + "\n".join(f"- {p}" for p in probs)
            + "\nRewrite the whole prompt so it uses every listed detail exactly as given (each character's "
              "stated age, hair, ethnicity and clothing; the camera angle, shot, lighting, location and time). "
              "Keep everything else, the same structure and length. Output only the corrected prompt.")


FILLER_RE = re.compile(
    r"\b(atmosphere|atmospheric|ambiance|ambience|a sense of|sense of (calm|wonder|intimacy|mystery|unity|scale|"
    r"tranquility|serenity|anticipation|connection)|evok\w+|poignant|dreamlike|dream-like|ethereal|"
    r"serenity|tranquil\w*|juxtaposition|symboli\w+|captur\w+ the essence|harmonious\w*|enchanting|"
    r"mesmeriz\w+|breathtaking|stunning|captivating|whimsical|timeless|a moment of|an air of|a hint of|"
    r"the viewer|invit\w+ the viewer|emotional|emotionally|mood of|overall mood|story of|narrative)\b", re.I)


def filler_hits(text: str) -> list[str]:
    return [m.group(0).lower() for m in FILLER_RE.finditer(text)]


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9'’-]+", text))


def fingerprint(text: str) -> str:
    norm = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", text.lower()))
    return hashlib.sha1(norm.encode()).hexdigest()


# =============================================================================
# Illustrious tag pass
# =============================================================================
TAG_SYSTEM = (
    "You convert an image description into a Danbooru-style tag prompt for an SDXL anime/illustration model "
    "(Illustrious). Output exactly one line of comma-separated lowercase tags and nothing else.\n"
    "Order: subject count tags (e.g. 1girl, 1boy, 2girls, 1girl 1boy, solo), then for each character: hair, eyes, "
    "build, clothing, pose and hands; then interaction tags; then camera tags (e.g. from above, from below, "
    "from side, dutch angle, close-up, upper body, cowboy shot, full body, wide shot); then setting, objects, "
    "lighting, time of day and weather. Use 25 to 50 real Danbooru tags. Use underscores only inside tags "
    "that need them; prefer spaces. Do not add quality tags (masterpiece, best quality, etc.).\n"
    "Every character is an adult: when the description contains characters, include the tag 'adult' and age tags "
    "such as 'mature female' or 'mature male' where they fit. Never use tags that imply a minor or a child-like "
    "body, and never school-related tags."
)
_QUALITY_TAGS = {"masterpiece", "best quality", "amazing quality", "high quality", "very aesthetic",
                 "absurdres", "highres", "newest"}


def clean_tags(text: str, prefix: str = "") -> str:
    text = clean_response(text).replace("\n", ", ")
    text = re.sub(r"^(tags|prompt)\s*:\s*", "", text.strip(), flags=re.I)
    seen, out = set(), []
    for t in (x.strip().strip('."\'`').lower() for x in text.split(",")):
        t = re.sub(r"\s+", " ", t)
        if not t or len(t) > 40 or t in seen or t in _QUALITY_TAGS:
            continue
        seen.add(t)
        out.append(t)
    pre = [x.strip() for x in prefix.split(",") if x.strip()]
    return ", ".join(pre + out)


def make_tags(backend: BaseBackend, prose: str, cfg: dict, rng: random.Random, res=None) -> str | None:
    system = TAG_SYSTEM
    if comic_on(cfg):
        system += ("\nThe description is a multi-panel comic page. Start with 'comic, multiple panels' and "
                   "page-level tags; describe the shared characters once; do not try to describe each panel.")
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": f"Image description:\n{prose}\n\nTags:"}]
    for attempt in range(1, 3):
        try:
            r = backend.chat(msgs, temperature=0.4, max_tokens=400, seed=rng.randint(1, 2**31 - 1))
        except FatalBackendError:
            raise
        except BackendError as e:
            log.warning("Tag pass attempt %d failed: %s", attempt, e)
            continue
        if res is not None:
            res.llm_calls += 1
            res.tokens += r.completion_tokens or 0
            res.llm_seconds += r.seconds
        tags = clean_tags(r.text, cfg["generation"]["tag_prefix"])
        n_tags = tags.count(",") + 1 if tags else 0
        if content_problem(tags, cfg):
            log.warning("Tag pass attempt %d rejected by the content rule", attempt)
            continue
        if n_tags >= 10:
            return tags
        log.warning("Tag pass attempt %d: only %d tags", attempt, n_tags)
    return None


# =============================================================================
# Output
# =============================================================================
class OutputWriter:
    """Writes each prompt immediately so stopping never loses work."""

    def __init__(self, cfg: dict, run_dir: Path, run_id: str):
        self.save = cfg["comfy"]["save_prompts"] or not cfg["comfy"]["enabled"]
        self.o = cfg["output"]
        self.dir = run_dir
        self.prefix = self.o["filename_prefix"] or "prompts"
        self.run_id = run_id
        self.count = 0
        self.dir.mkdir(parents=True, exist_ok=True)
        stem = self.prefix if self.o["timestamp_subfolder"] else f"{self.prefix}_{run_id}"
        self.paths = {"txt": self.dir / f"{stem}.txt", "lines": self.dir / f"{stem}_lines.txt",
                      "jsonl": self.dir / f"{stem}.jsonl", "separate": self.dir / f"{stem}_separate",
                      "tags": self.dir / f"{stem}_illustrious_tags.txt"}
        self.files = [self.paths[f] for f in VALID_FORMATS if f in self.o["formats"]] if self.save else []
        if "separate" in self.o["formats"] and self.save:
            self.paths["separate"].mkdir(parents=True, exist_ok=True)

    def write(self, index: int, prompt: str, record: dict) -> None:
        if not self.save:                 # ComfyUI-only mode: prompts are not written to disk
            self.count += 1
            return
        fm = self.o["formats"]
        if "txt" in fm:
            with self.paths["txt"].open("a", encoding="utf-8") as f:
                if self.count:
                    f.write(self.o["separator"])
                f.write(prompt.strip() + "\n")
        if "lines" in fm and (record.get("comic") or {}).get("format") != "bbox":
            with self.paths["lines"].open("a", encoding="utf-8") as f:
                f.write(" ".join(paragraphs(prompt)) + "\n")
        if "jsonl" in fm:
            with self.paths["jsonl"].open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if "separate" in fm:
            (self.paths["separate"] / f"{self.prefix}_{index:04d}.txt").write_text(prompt.strip() + "\n", encoding="utf-8")
        if "tags" in fm and record.get("tags"):
            with self.paths["tags"].open("a", encoding="utf-8") as f:
                f.write(record["tags"] + "\n")
        self.count += 1


# =============================================================================
# The job
# =============================================================================
@dataclass
class JobResult:
    ok: int = 0
    flagged: int = 0
    failed: int = 0
    llm_calls: int = 0
    tokens: int = 0
    llm_seconds: float = 0.0
    wall_seconds: float = 0.0
    stopped: bool = False
    files: list = field(default_factory=list)
    run_dir: Path | None = None
    main_log: Path | None = None
    debug_log: Path | None = None
    random_seed: int = 0
    image_seconds: float = 0.0          # ComfyUI phase (send mode 'end'); 'each' is part of wall_seconds
    total_seconds: float = 0.0
    comfy_sent: int = 0
    comfy_done: int = 0
    comfy_failed: int = 0
    image_files: list = field(default_factory=list)
    retry_reasons: dict = field(default_factory=dict)   # rejected-attempt reason -> count


def output_root(cfg: dict) -> Path:
    d = cfg["output"]["directory"]
    return (Path(d).expanduser() if d else Path.cwd()).resolve()


def run_job(cfg: dict, *, fixed_scenes: list[str] | None = None, system_prompt_text: str | None = None,
            stop_event: threading.Event | None = None,
            on_prompt: Callable[[int, int, str, SceneSeed, dict], None] | None = None,
            on_progress: Callable[[int, int, str], None] | None = None,
            log_handlers: list[logging.Handler] | None = None, console_log: bool = True,
            presets: dict | None = None,
            on_comfy: Callable[[int, dict], None] | None = None) -> JobResult:
    """Run a full generation job. Raises PDError for config/backend setup problems."""
    t_job = time.monotonic()
    cfg = validate_config(copy.deepcopy(cfg))
    if "illustrious" in cfg["generation"]["targets"] and "tags" not in cfg["output"]["formats"]:
        cfg["output"]["formats"].append("tags")
    presets = copy.deepcopy(presets) if presets is not None else load_presets(cfg)
    check_scene_inputs(cfg, fixed_scenes, presets)
    stop_event = stop_event or threading.Event()
    root = output_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    res = JobResult()
    res.main_log, res.debug_log = setup_logging(cfg, root, log_handlers, console_log)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    o, g = cfg["output"], cfg["generation"]
    compare_images = 0
    if cfg["comfy"]["enabled"] and cfg["comfy"].get("compare_models"):
        compare_images, g["count"] = g["count"], 1      # compare: ONE prompt, 'count' = number of images
    run_dir = root / f"{o['filename_prefix']}_{run_id}" if o["timestamp_subfolder"] else root
    if o["timestamp_subfolder"]:
        n = 2
        while run_dir.exists():          # two runs in the same second
            run_dir = root / f"{o['filename_prefix']}_{run_id}_{n}"
            n += 1

    log.info("=" * 64)
    log.info("promptdirector %s — run %s", __version__, run_id)
    managed = cfg["backend"]["type"] == "llamacpp" and cfg["llama_server"]["managed"]
    log.info("Backend: %s%s @ %s | model: %s", cfg["backend"]["type"], " (managed by the app)" if managed else "",
             cfg["backend"]["host"] or "(in-process)",
             cfg["model"]["path"] if cfg["backend"]["type"] == "llamacpp_python" or managed else cfg["model"]["name"])
    log.info("Count: %s | seed mode: %s | random: %s | content: %s | formats: %s",
             f"1 prompt -> {compare_images} images (Compare models)" if compare_images else g["count"],
             "scene text" if fixed_scenes else g["seed_mode"], "on" if cfg["scene"]["random_enabled"] else "off",
             g["content_rating"], ",".join(o["formats"]))
    filled = fixed_fields(cfg)
    if comic_on(cfg):
        cm = cfg["comic"]
        log.info("Comic mode: %s grid, %s, %s format, location %s, bubbles %s | suggested image size %s",
                 cm["layout"], cm["mode"], cm["format"], cm["location_mode"], "on" if cm["speech_bubbles"] else "off",
                 suggested_size(cm["layout"]))
        if cm["format"] == "bbox" and "lines" in o["formats"]:
            log.info("Note: the 'lines' output format is skipped for BBOX specs (they need their line breaks)")
    log.info("Targets: %s | length %d-%d words | filler check: %s | detail check: %s | compat rules: %s",
             ", ".join(g["targets"]), *word_range(cfg), "on" if g["filler_check"] else "off",
             "on" if g.get("detail_check", True) else "off", "on" if g["compat_rules"] else "off")
    log.info("Fixed fields: %s", ", ".join(f"{FIELD_LABELS[k]}={v!r}" for k, v in filled.items()) or "(none)")
    log.info("Log: %s%s", res.main_log, f" | debug log: {res.debug_log}" if res.debug_log else "")
    log.debug("Effective config: %s", json.dumps(redacted(cfg), indent=2))

    seed_val = g["random_seed"] or random.SystemRandom().randint(1, 2**31 - 1)
    res.random_seed = seed_val
    rng = random.Random(seed_val)
    log.info("Random seed: %d", seed_val)

    backend = make_backend(cfg, stop_event)
    try:
        if managed:
            try:
                log.info(lserver.MANAGER.ensure_running(cfg, res.main_log.parent, stop_event))
            except lserver.ServerError as e:
                raise ConfigError(f"llama-server: {e}") from e
        if cfg["backend"]["verify_connection"]:
            log.info("Checking backend ...")
            backend.check()
            log.info("Backend OK")
        system_prompt = prepare_system_prompt(cfg, system_prompt_text)
        _check_context_fits(cfg, backend, system_prompt)
        scenes = SceneGenerator(cfg, rng, backend, fixed_scenes, presets)
        writer = OutputWriter(cfg, run_dir, run_id)
        res.run_dir, res.files = run_dir, writer.files
        sender = None
        if cfg["comfy"]["enabled"]:
            cc = cfg["comfy"]
            try:
                sender = comfy.ComfySender(cc, run_dir, stop_event, random.Random(rng.random()))
                if cc["rotation"] != "off":
                    report = sender.set_plan(g["count"], comfy.load_model_cache(cc, APP_DIR).get("lists", {}))
                    log.info(report)
                    if sender.compare:
                        log.info(sender.set_compare_images(compare_images))
                log.info("ComfyUI: %s | workflow %s | send %s | %s", comfy._base(cc), cc["workflow"],
                         "after all prompts" if cc["send_mode"] == "end" else "after each prompt",
                         "prompt files saved too" if cc["save_prompts"] else "prompt files NOT saved (ComfyUI only)")
                log.info(comfy.test_connection(cc))
            except comfy.ComfyError as e:
                raise ConfigError(f"ComfyUI: {e}") from e
            if cc["workflow"] == "builtin_sdxl" and "illustrious" not in g["targets"] and cc["prompt_source"] != "prose":
                log.info("ComfyUI tip: tick the Illustrious target to send Danbooru tags to the SDXL workflow")
        pending: list[tuple] = []
        _generate_loop(cfg, backend, system_prompt, scenes, writer, rng, stop_event, res, on_prompt, on_progress,
                       sender, pending, on_comfy)
    finally:
        backend.close()
    if sender is not None:
        t_img = time.monotonic()
        _comfy_end_phase(cfg, backend, sender, pending, writer, stop_event, res, on_progress, on_comfy)
        res.image_seconds = time.monotonic() - t_img if pending else 0.0
    res.total_seconds = time.monotonic() - t_job

    log.info("-" * 64)
    log.info("Done: %d ok, %d saved with issues, %d failed (of %d)%s", res.ok, res.flagged, res.failed,
             g["count"], " — stopped early" if res.stopped else "")
    log.info("LLM calls: %d | tokens: %d | LLM time: %.1fs | prompt phase: %.1fs%s | total: %.1fs",
             res.llm_calls, res.tokens, res.llm_seconds, res.wall_seconds,
             f" | image phase: {res.image_seconds:.1f}s" if res.image_seconds else "", res.total_seconds)
    if res.retry_reasons:
        n_rej = sum(res.retry_reasons.values())
        log.info("Rejected attempts: %d (%s)", n_rej, ", ".join(
            f"{v} {k}" for k, v in sorted(res.retry_reasons.items(), key=lambda kv: -kv[1])))
    if sender is not None:
        for line in sender.timing_report():
            log.info("ComfyUI timing - %s", line)
    for f in res.files:
        log.info("Output: %s", f)
    if sender is not None:
        log.info("ComfyUI: %d sent, %d finished, %d failed%s", res.comfy_sent, res.comfy_done, res.comfy_failed,
                 f" | {len(res.image_files)} image(s) downloaded to {run_dir / 'images'}" if res.image_files else "")
    return res


def _comfy_send_one(cfg, sender, i, best, tags, seed_obj, res, on_comfy, model=None, variant=0) -> dict | None:
    """Send one prompt to ComfyUI (re-checking the content rule first)."""
    for txt in (best, tags or ""):
        cp = content_problem(txt, cfg) if txt else None
        if cp:
            log.warning("[%d] ComfyUI: not sent (%s)", i, cp)
            return None
    size = None
    comic = (seed_obj.details.get("comic") if seed_obj else None) or {}
    if comic.get("suggested_size"):
        w, h = comic["suggested_size"].split("x")
        size = (int(w), int(h))
    try:
        info = sender.send(i, best, tags, size, model=model, variant=variant)
        res.comfy_sent += 1
        if info.get("images") or not sender.cc.get("wait_for_images", True):
            res.comfy_done += 1 if info.get("images") else 0
        res.image_files = list(sender.image_files)
        if on_comfy:
            on_comfy(i, info)
        return info
    except comfy.ComfyError as e:
        res.comfy_failed += 1
        log.error("[%d] ComfyUI: %s", i, e)
        if on_comfy:
            on_comfy(i, {"error": str(e)})
        return None


def _safe_vram(cc: dict) -> float | None:
    try:
        return comfy.vram_free_gb(cc)
    except comfy.ComfyError:
        return None


def _comfy_end_phase(cfg, backend, sender, pending, writer, stop_event, res, on_progress, on_comfy):
    cc = cfg["comfy"]
    if not pending:
        sender.finish()
        return
    if stop_event.is_set():
        log.warning("ComfyUI: run was stopped; %d prompt(s) not sent", len(pending))
        return
    # Free the GPU for ComfyUI: LLM and image model rarely fit on one card together.
    if cc["unload_llm"]:
        t = cfg["backend"]["type"]
        if t == "ollama":
            try:
                comfy.unload_ollama(cfg["backend"]["host"], cfg["model"]["name"])
                log.info("Ollama: unloaded %s to free VRAM for ComfyUI", cfg["model"]["name"])
            except Exception as e:
                log.warning("Ollama: could not unload the model (%s)", e)
        elif t == "llamacpp_python":
            LlamaCppPythonBackend._cache.clear()
            import gc
            gc.collect()
            log.info("llama-cpp-python: model released to free VRAM for ComfyUI")
        elif t == "llamacpp" and cfg["llama_server"]["managed"]:
            if lserver.MANAGER.stop():
                log.info("llama-server stopped to free VRAM for ComfyUI (it restarts on the next run)")
        else:
            log.warning("%s keeps its model in VRAM; if ComfyUI runs out of memory, stop the LLM server first, "
                        "let the app manage llama-server (Backend & Model tab), or run ComfyUI on another GPU", t)
        waited_log = False
        if cc.get("vram_wait", 30) > 0:
            before, after, waited = comfy.wait_for_vram(cc, cc["vram_wait"], stop_event)
            if after is not None:
                log.info("ComfyUI: %.1f GB VRAM free after waiting %.0fs for the LLM to release it%s", after, waited,
                         f" (was {before:.1f} GB)" if before is not None and after - before > 0.2 else "")
                waited_log = True
        if not waited_log and (v := _safe_vram(cc)) is not None:
            log.info("ComfyUI: %.1f GB VRAM free before the first image", v)
    elif (v := _safe_vram(cc)) is not None:
        log.info("ComfyUI: %.1f GB VRAM free before the first image", v)
    # Compare mode: model by model (each model loads once), every prompt on each.
    jobs = ([(m, v, p) for m, v in sender.compare_jobs() for p in pending] if sender.compare
            else [(None, 0, p) for p in pending])
    n = len(jobs)
    for k, (model, variant, (i, best, tags, seed_obj)) in enumerate(jobs, 1):
        if stop_event.is_set():
            log.warning("ComfyUI: stopped; %d image job(s) not sent", n - k + 1)
            break
        if on_progress:
            on_progress(k - 1, n, f"ComfyUI {k}/{n} ..." + (f" ({Path(model).stem[:30]})" if model else ""))
        info = _comfy_send_one(cfg, sender, i, best, tags, seed_obj, res, on_comfy, model=model, variant=variant)
        if info and writer.save:
            with (writer.dir / "comfy_jobs.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"index": i, **info}, ensure_ascii=False) + "\n")
    if on_progress:
        on_progress(n, n, "Done")
    sender.finish()


def _check_context_fits(cfg: dict, backend: BaseBackend, system_prompt: str) -> None:
    """Warn early if the system prompt + output budget won't fit the context window."""
    est_prompt = int(len(system_prompt) / 3.5) + 300        # rough: ~3.5-4 chars per token + scene
    out = int(cfg["model"]["max_tokens"])
    if cfg["backend"]["type"] == "llamacpp":
        ctx, where = getattr(backend, "server_ctx", None), "llama-server (-c)"
    elif cfg["backend"]["type"] in ("ollama", "llamacpp_python"):
        ctx, where = int(cfg["model"]["n_ctx"]), "Context size (n_ctx)"
    else:
        return
    if not ctx:
        return
    trained = getattr(backend, "n_ctx_train", None)
    if trained and est_prompt + out > trained:
        log.warning("This model was trained for only %d tokens of context, but each request needs ~%d "
                    "(prompt + output). Expect degraded or repetitive output; a long-context model "
                    "(8K+) is recommended.", trained, est_prompt + out)
    if est_prompt + out > ctx:
        log.warning("Context may be too small: ~%d prompt tokens + %d max output tokens > %d (%s). "
                    "Raise %s to at least %d%s.", est_prompt, out, ctx, where, where,
                    ((est_prompt + out) // 1024 + 1) * 1024,
                    " — Ollama silently cuts the start of the prompt (your system prompt!) when it doesn't fit"
                    if cfg["backend"]["type"] == "ollama" else "")


def _generate_loop(cfg, backend, system_prompt, scenes, writer, rng, stop_event, res, on_prompt, on_progress,
                   sender=None, pending=None, on_comfy=None):
    g = cfg["generation"]
    total = g["count"]
    seen: set[str] = set()
    t0 = time.monotonic()
    for i in range(1, total + 1):
        if stop_event.is_set():
            res.stopped = True
            log.warning("Stopped after %d/%d prompts", i - 1, total)
            break
        seed = scenes.next(i - 1)
        log.info("[%d/%d] Scene (%s): %s", i, total, seed.source, seed.text.replace("\n", " | "))
        if on_progress:
            on_progress(i - 1, total, f"Generating {i}/{total} ...")
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": g["user_template"].format(scene=seed.text)
                     + ("\n\nUse every listed detail exactly as given, including each character's stated age."
                        if seed.details and seed.details.get("people") else "")}]
        best, best_problems, attempts = None, None, 0
        for attempt in range(1, g["max_attempts_per_prompt"] + 1):
            if stop_event.is_set():
                break
            attempts = attempt
            try:
                r = backend.chat(messages, seed=rng.randint(1, 2**31 - 1))
            except FatalBackendError:
                raise
            except BackendError as e:
                log.error("[%d/%d] attempt %d: %s", i, total, attempt, e)
                res.retry_reasons["backend error"] = res.retry_reasons.get("backend error", 0) + 1
                continue
            res.llm_calls += 1
            res.tokens += r.completion_tokens or 0
            res.llm_seconds += r.seconds
            log.debug("[%d/%d] raw response (%.1fs):\n%s", i, total, r.seconds, r.text)
            text = clean_response(r.text, g["strip_reasoning"], g["clean_output"])
            if comic_on(cfg) and cfg["comic"]["format"] == "bbox":
                text = normalize_bbox(text)
            problems = validate_prompt(text, g["min_paragraphs"], cfg)
            if problems and problems[0].startswith("content rule"):
                log.warning("[%d/%d] attempt %d rejected (%s); not saved", i, total, attempt, problems[0])
                res.retry_reasons["content rule"] = res.retry_reasons.get("content rule", 0) + 1
                continue                      # never kept, not even as a fallback
            if g["dedupe"] and text and fingerprint(text) in seen:
                problems.append("duplicate of an earlier prompt")
            dprobs = (detail_problems(text, seed.details)
                      if g.get("detail_check", True) and text and not comic_on(cfg) else [])
            problems += dprobs
            if not problems:
                best, best_problems = text, []
                break
            log.warning("[%d/%d] attempt %d rejected: %s", i, total, attempt, "; ".join(problems))
            for pr in problems:
                key = ("scene detail not used" if pr.endswith("' not used")
                       else re.split(r" \(|:", pr, maxsplit=1)[0].strip())
                res.retry_reasons[key] = res.retry_reasons.get(key, 0) + 1
            if "empty response" in problems and cfg["model"].get("thinking", "off") == "off" \
                    and "/no_think" not in messages[-1]["content"]:
                # Qwen3-style soft switch: some fine-tunes ignore enable_thinking but honor this.
                messages = [messages[0], {"role": "user", "content": messages[-1]["content"] + "\n\n/no_think"}]
                log.info("[%d/%d] empty answer: retrying with /no_think added", i, total)
            if text and (best is None or len(problems) < len(best_problems)):
                best, best_problems = text, problems
            if dprobs and len(dprobs) == len(problems):
                # only the details are off: show the model its draft and ask for a corrected version
                messages = messages[:2] + [{"role": "assistant", "content": text},
                                           {"role": "user", "content": correction_message(dprobs)}]
                log.info("[%d/%d] asking the model to fix: %s", i, total, "; ".join(dprobs))
            elif len(messages) > 2:
                messages = messages[:2]           # other problems: start over from the scene
        if not best:
            if stop_event.is_set():
                res.stopped = True
                break
            res.failed += 1
            log.error("[%d/%d] failed after %d attempt(s); skipped", i, total, attempts)
            continue
        seen.add(fingerprint(best))
        tags = None
        if "illustrious" in g["targets"] and not stop_event.is_set():
            tags = make_tags(scenes.backend, best, cfg, rng, res)
            if not tags:
                best_problems = (best_problems or []) + ["tag pass failed"]
        record = {
            "index": i, "prompt": best, "scene": seed.text, "seed_source": seed.source,
            "details": seed.details, "detail_origin": seed.origin,
            "words": word_count(best), "tags": tags,
            "comic": seed.details.get("comic"),
            "valid": not best_problems, "problems": best_problems, "attempts": attempts,
            "backend": cfg["backend"]["type"],
            "model": (cfg["model"]["path"] if cfg["backend"]["type"] == "llamacpp_python"
                      or (cfg["backend"]["type"] == "llamacpp" and cfg["llama_server"]["managed"])
                      else cfg["model"]["name"]),
            "temperature": cfg["model"]["temperature"], "content_rating": g["content_rating"],
            "created": datetime.now().isoformat(timespec="seconds"), "run_id": writer.run_id,
        }
        if sender is not None:
            if cfg["comfy"]["send_mode"] == "each":
                if sender.compare:
                    record["comfy"] = [_comfy_send_one(cfg, sender, i, best, tags, seed, res, on_comfy, model=m, variant=v)
                                       for m, v in sender.compare_jobs() if not stop_event.is_set()]
                else:
                    record["comfy"] = _comfy_send_one(cfg, sender, i, best, tags, seed, res, on_comfy)
            else:
                pending.append((i, best, tags, seed))
        writer.write(i, best, record)
        if best_problems:
            res.flagged += 1
            log.warning("[%d/%d] saved with issues: %s", i, total, "; ".join(best_problems))
        else:
            res.ok += 1
            log.info("[%d/%d] OK — %s%s", i, total,
                     (f"comic {cfg['comic']['layout']} ({cfg['comic']['format']}), {len(best)} chars"
                      if comic_on(cfg) else f"{len(paragraphs(best))} paragraphs, {word_count(best)} words"),
                     f", {tags.count(',') + 1} tags" if tags else "")
        if on_prompt:
            on_prompt(i, total, best, seed, record)
    if on_progress:
        on_progress(total if not res.stopped else res.ok + res.flagged, total, "Stopped" if res.stopped else
                    ("Prompts done" if pending else "Done"))
    if sender is not None and cfg["comfy"]["send_mode"] == "each":
        sender.finish()
    res.wall_seconds = time.monotonic() - t0


def preview_scenes(cfg: dict, n: int, fixed_scenes: list[str] | None = None,
                   presets: dict | None = None) -> list[SceneSeed]:
    """Scene seeds without any LLM calls (dry run / GUI preview)."""
    cfg = validate_config(copy.deepcopy(cfg))
    presets = copy.deepcopy(presets) if presets is not None else load_presets(cfg)
    check_scene_inputs(cfg, fixed_scenes, presets)
    seed_val = cfg["generation"]["random_seed"] or random.SystemRandom().randint(1, 2**31 - 1)
    gen = SceneGenerator(cfg, random.Random(seed_val), None, fixed_scenes, presets)
    return [gen.next(i, dry_run=True) for i in range(n)]
