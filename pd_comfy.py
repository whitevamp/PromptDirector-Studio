# PromptDirector Studio - Copyright (c) 2026 whitevamp
# SPDX-License-Identifier: CC-BY-NC-SA-4.0 (see LICENSE)
"""
pd_comfy.py — send generated prompts to a ComfyUI server and (optionally) wait for / download the images.

Workflows
---------
* builtin_sdxl : SDXL / Illustrious checkpoint (CheckpointLoaderSimple, CLIP skip, positive + negative, KSampler)
* builtin_flux : FLUX (UNETLoader + DualCLIPLoader[t5xxl, clip_l] + VAELoader, FluxGuidance, KSampler cfg 1)
* custom       : your own workflow exported from ComfyUI with "Export (API)" (use this for Krea 2 / Moody Krea 2
                 Mix / LoRA setups). Put %PROMPT% in your prompt text box before exporting (and optionally
                 %NEGATIVE%, %TAGS%). Seed, size, batch and file prefix are filled in automatically; model
                 drop-downs (if set) override the loader nodes.

Only the Python standard library is used.
"""
from __future__ import annotations

import copy
import json
from collections import Counter
import logging
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

log = logging.getLogger("promptdirector")

MODEL_FOLDERS = {                     # cache key -> ComfyUI folder names (new name first)
    "checkpoints": ["checkpoints"],
    "diffusion_models": ["diffusion_models", "unet"],
    "text_encoders": ["text_encoders", "clip"],
    "vae": ["vae"],
    "loras": ["loras"],
}
OBJECT_INFO_FALLBACK = {              # cache key -> (node class, input name)
    "checkpoints": ("CheckpointLoaderSimple", "ckpt_name"),
    "diffusion_models": ("UNETLoader", "unet_name"),
    "text_encoders": ("CLIPLoader", "clip_name"),
    "vae": ("VAELoader", "vae_name"),
    "loras": ("LoraLoader", "lora_name"),
}
MODEL_EXTS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft")
WORKFLOWS = ("builtin_sdxl", "builtin_flux", "custom")
SEND_MODES = ("end", "each")
SIZE_MODES = ("auto", "workflow", "custom")
ROTATION_MODES = ("off", "random", "list")
ROTATION_TARGETS = ("auto", "checkpoint", "diffusion_model")
TARGET_FOLDER = {"checkpoint": "checkpoints", "diffusion_model": "diffusion_models"}
PROMPT_SOURCES = ("auto", "prose", "tags")

DEFAULT_COMFY = {
    "enabled": False,                 # send prompts to ComfyUI
    "save_prompts": True,             # also write the prompt files (off = send only, don't save prompts)
    "url": "http://127.0.0.1:8188",
    "models_dir": "",                 # ComfyUI 'models' folder: used to list models when ComfyUI isn't running
    "cache_file": "comfy_models_cache.json",
    "workflow": "custom",             # builtin_sdxl | builtin_flux | custom
    "workflow_file": "",              # API-format JSON exported from ComfyUI (custom)
    "checkpoint": "",                 # builtin_sdxl (and overrides CheckpointLoaderSimple in custom)
    "diffusion_model": "",            # builtin_flux (and overrides UNETLoader in custom)
    "text_encoder": "",               # flux: t5xxl / krea2: qwen3vl (overrides CLIPLoader / DualCLIPLoader 1)
    "text_encoder_2": "",             # flux: clip_l (DualCLIPLoader 2)
    "vae": "",                        # empty = checkpoint's / workflow's VAE
    "prompt_source": "auto",          # auto (tags for builtin_sdxl if available) | prose | tags
    "negative_prompt": "lowres, worst quality, bad anatomy, bad hands, watermark, signature, text",
    # auto     = comic pages get the app's suggested size x comic_scale; otherwise keep the workflow's size
    # workflow = always keep the workflow's own size
    # custom   = always use width/height below
    "size_mode": "auto",
    "comic_scale": 1.0,               # e.g. 0.67 when your workflow upscales x1.5 in a second pass
    "width": 0,                       # custom size (also used in auto mode when set)
    "height": 0,
    "steps": 0,                       # 0 = workflow default (sdxl 28, flux 24) / keep custom workflow's
    "cfg": 0.0,                       # 0 = default (sdxl 5.5, flux 1.0) / keep custom's
    "guidance": 3.5,                  # FLUX guidance
    "sampler": "",                    # empty = default (sdxl euler_ancestral, flux euler) / keep custom's
    "scheduler": "",                  # empty = default (sdxl normal, flux simple) / keep custom's
    "clip_skip": 2,                   # builtin_sdxl (Illustrious usually 2)
    "seed": 0,                        # 0 = random per image
    "images_per_prompt": 1,
    "filename_prefix": "pds",
    "send_mode": "end",               # end = after all prompts (LLM unloaded first) | each = right after each prompt
    "wait_for_images": True,
    "download_images": False,         # copy finished images into the run folder
    "unload_llm": True,               # Ollama: unload the LLM before images (frees VRAM on a shared GPU)
    "free_vram_after": True,          # ask ComfyUI to unload its models when the run is done
    "timeout": 900,                   # seconds to wait per image job
    # Model rotation: use several models across the prompts of one run (consecutive prompts share a model,
    # so each model is loaded only once).
    "rotation": "off",                # off | random (N different cached models) | list (the models below)
    "rotation_target": "auto",        # auto (from the workflow) | checkpoint | diffusion_model
    "rotation_count": 2,              # random: how many different models (<= number of prompts)
    "rotation_models": [],            # list: models in order
    "rotation_compatible_only": True, # only use models with the same architecture as the workflow's model
    "compare_models": False,          # ON = every prompt is rendered on EVERY rotation model (same seed), to compare
    "vram_wait": 30,                  # seconds to wait for the LLM's VRAM to be released before images (0 = off)
    "missing_loras": "skip",          # LoRA file not in ComfyUI: skip = leave it out of the job (warned); stop = stop
    "model_search_dirs": "",          # extra folders (separated by ;) where model files live, for the check
}


class ComfyError(Exception):
    pass


# ---------------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------------
def _base(cc: dict) -> str:
    url = (cc.get("url") or DEFAULT_COMFY["url"]).strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def _request(cc: dict, method: str, path: str, payload=None, timeout: float = 30, raw: bool = False):
    url = _base(cc) + path
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            return body if raw else (json.loads(body.decode("utf-8")) if body else {})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise ComfyError(_explain_error(e.code, body, path)) from e
    except urllib.error.URLError as e:
        raise ComfyError(f"Cannot reach ComfyUI at {_base(cc)} ({e.reason}). Is it running? "
                         f"Check the ComfyUI URL (default http://127.0.0.1:8188).") from e
    except (TimeoutError, OSError) as e:
        raise ComfyError(f"ComfyUI request {path} failed: {e}") from e
    except json.JSONDecodeError as e:
        raise ComfyError(f"ComfyUI returned invalid JSON for {path}") from e


def _explain_error(code: int, body: str, path: str) -> str:
    msg = f"ComfyUI HTTP {code} on {path}"
    try:
        data = json.loads(body)
    except ValueError:
        return f"{msg}: {body[:300]}"
    err = data.get("error") or {}
    if isinstance(err, dict) and err.get("message"):
        msg += f": {err.get('message')}"
        if err.get("details"):
            msg += f" ({err['details']})"
    for nid, ne in (data.get("node_errors") or {}).items():
        for e in ne.get("errors", [])[:3]:
            msg += f"\n  node {nid} ({ne.get('class_type', '?')}): {e.get('message', '')} {e.get('details', '')}".rstrip()
    if "value_not_in_list" in body:
        msg += "\nHint: a model name in the workflow isn't installed in ComfyUI - pick one from the model lists."
    return msg


def test_connection(cc: dict) -> str:
    info = _request(cc, "GET", "/system_stats", timeout=10)
    sysd = info.get("system", {})
    devs = info.get("devices", [])
    gpu = ", ".join(f"{d.get('name', '?')} ({int(d.get('vram_free', 0) / 2**30)}/{int(d.get('vram_total', 0) / 2**30)} GB free)"
                    for d in devs) or "no GPU info"
    return f"ComfyUI {sysd.get('comfyui_version', '')} OK at {_base(cc)} - {gpu}".replace("  ", " ")


# ---------------------------------------------------------------------------------------------
# Model lists (API first, then folder scan) with a JSON cache
# ---------------------------------------------------------------------------------------------
def _list_via_api(cc: dict) -> dict[str, list[str]]:
    lists: dict[str, list[str]] = {}
    folders = []
    try:
        folders = _request(cc, "GET", "/models", timeout=15) or []
    except ComfyError:
        folders = []
    for key, names in MODEL_FOLDERS.items():
        for name in names:
            if folders and name not in folders:
                continue
            try:
                vals = _request(cc, "GET", f"/models/{name}", timeout=15)
            except ComfyError:
                continue
            if isinstance(vals, list):
                lists.setdefault(key, [])
                lists[key] += [v for v in vals if v not in lists[key]]
        if key not in lists:                       # older ComfyUI: read the loader node's choices
            cls, inp = OBJECT_INFO_FALLBACK[key]
            try:
                info = _request(cc, "GET", f"/object_info/{cls}", timeout=15)
                choices = info[cls]["input"]["required"][inp][0]
                if isinstance(choices, list):
                    lists[key] = choices
            except (ComfyError, KeyError, IndexError, TypeError):
                pass
    if not lists:
        raise ComfyError("ComfyUI did not report any models")
    return {k: sorted(v) for k, v in lists.items()}


def _list_via_folder(models_dir: str) -> dict[str, list[str]]:
    root = Path(models_dir).expanduser()
    if not root.is_dir():
        raise ComfyError(f"Models folder not found: {root}")
    known = {n for names in MODEL_FOLDERS.values() for n in names}
    if root.name in known:            # a subfolder like .../models/diffusion_models was picked: use its parent
        root = root.parent
    lists: dict[str, list[str]] = {}
    for key, names in MODEL_FOLDERS.items():
        found: list[str] = []
        for name in names:
            d = root / name
            if not d.is_dir():
                continue
            for f in d.rglob("*"):
                if f.is_file() and f.suffix.lower() in MODEL_EXTS:
                    rel = str(f.relative_to(d))
                    if rel not in found:
                        found.append(rel)
        lists[key] = sorted(found)
    if not any(lists.values()):
        raise ComfyError(f"No model files found under {root} (expected checkpoints/, diffusion_models/, ...)")
    return lists


def cache_path(cc: dict, app_dir: Path) -> Path:
    p = Path(cc.get("cache_file") or DEFAULT_COMFY["cache_file"]).expanduser()
    return p if p.is_absolute() else app_dir / p


def load_model_cache(cc: dict, app_dir: Path) -> dict:
    p = cache_path(cc, app_dir)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"lists": {}, "source": "", "updated": ""}


def refresh_model_cache(cc: dict, app_dir: Path) -> dict:
    """Ask ComfyUI for its models; if that fails, scan the models folder. Saves and returns the cache."""
    errors = []
    try:
        lists, source = _list_via_api(cc), f"ComfyUI API ({_base(cc)})"
    except ComfyError as e:
        errors.append(str(e).splitlines()[0])
        if not cc.get("models_dir"):
            raise ComfyError(errors[0] + "\nTip: set the ComfyUI models folder to list models while ComfyUI is off.")
        try:
            lists = _list_via_folder(cc["models_dir"])
        except ComfyError as e2:
            raise ComfyError(f"{errors[0]}\nModels folder: {e2}") from e2
        source = f"folder {cc['models_dir']}"
    cache = {"lists": lists, "source": source, "updated": datetime.now().isoformat(timespec="seconds")}
    p = cache_path(cc, app_dir)
    try:
        p.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except OSError as e:
        log.warning("Could not write model cache %s: %s", p, e)
    n = sum(len(v) for v in lists.values())
    log.info("ComfyUI model lists refreshed from %s: %d files", source, n)
    return cache


# ---------------------------------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------------------------------
def _sdxl_workflow(cc: dict, prompt: str, negative: str, seed: int, w: int, h: int, batch: int) -> dict:
    if not cc.get("checkpoint"):
        raise ComfyError("builtin_sdxl needs a checkpoint - pick one from the model list")
    vae_ref = ["1", 2]
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": cc["checkpoint"]}},
        "2": {"class_type": "CLIPSetLastLayer", "inputs": {"stop_at_clip_layer": -abs(int(cc.get("clip_skip") or 1)),
                                                            "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 0]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": batch}},
        "6": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": int(cc.get("steps") or 28), "cfg": float(cc.get("cfg") or 5.5),
            "sampler_name": cc.get("sampler") or "euler_ancestral", "scheduler": cc.get("scheduler") or "normal",
            "denoise": 1.0, "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"filename_prefix": cc.get("filename_prefix") or "pds",
                                                    "images": ["7", 0]}},
    }
    if cc.get("vae"):
        wf["9"] = {"class_type": "VAELoader", "inputs": {"vae_name": cc["vae"]}}
        vae_ref = ["9", 0]
    wf["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": vae_ref}}
    return wf


def _flux_workflow(cc: dict, prompt: str, seed: int, w: int, h: int, batch: int) -> dict:
    missing = [k for k in ("diffusion_model", "text_encoder", "text_encoder_2", "vae") if not cc.get(k)]
    if missing:
        raise ComfyError("builtin_flux needs: " + ", ".join(missing) +
                         " (diffusion model, t5xxl text encoder, clip_l text encoder, ae VAE)")
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": cc["diffusion_model"], "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": cc["text_encoder"],
                                                         "clip_name2": cc["text_encoder_2"], "type": "flux"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": cc["vae"]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "FluxGuidance", "inputs": {"guidance": float(cc.get("guidance") or 3.5),
                                                       "conditioning": ["4", 0]}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": batch}},
        "8": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": int(cc.get("steps") or 24), "cfg": float(cc.get("cfg") or 1.0),
            "sampler_name": cc.get("sampler") or "euler", "scheduler": cc.get("scheduler") or "simple",
            "denoise": 1.0, "model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"filename_prefix": cc.get("filename_prefix") or "pds",
                                                     "images": ["9", 0]}},
    }


def load_custom_workflow(path: str) -> dict:
    p = Path(path).expanduser()
    if not p.is_file():
        raise ComfyError(f"Workflow file not found: {p}")
    try:
        wf = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        raise ComfyError(f"Workflow file is not valid JSON: {e}") from e
    if isinstance(wf, dict) and "nodes" in wf and "links" in wf:
        raise ComfyError("This is a regular ComfyUI workflow file. In ComfyUI use Workflow > Export (API) and pick "
                         "that file instead (enable dev mode options in older versions).")
    if not isinstance(wf, dict) or not all(isinstance(v, dict) and "class_type" in v for v in wf.values()):
        raise ComfyError("Workflow file is not in ComfyUI API format (expected {node_id: {class_type, inputs}})")
    return wf


def _replace_placeholders(obj, mapping: dict) -> int:
    """Replace %PROMPT%/%NEGATIVE%/%TAGS% inside every string input. Returns number of replacements."""
    n = 0
    for node in obj.values():
        for k, v in list(node.get("inputs", {}).items()):
            if isinstance(v, str):
                new = v
                for ph, val in mapping.items():
                    if ph in new:
                        new = new.replace(ph, val)
                if new != v:
                    node["inputs"][k] = new
                    n += 1
    return n


def _auto_prompt_node(wf: dict) -> tuple[str, str] | None:
    """Find the positive prompt text input when no %PROMPT% placeholder is used."""
    negatives = set()
    for node in wf.values():
        neg = node.get("inputs", {}).get("negative")
        if isinstance(neg, list) and neg:
            negatives.add(str(neg[0]))
    candidates = []
    for nid, node in wf.items():
        cls = node.get("class_type", "")
        for key in ("text", "prompt", "text_g", "value", "string"):
            v = node.get("inputs", {}).get(key)
            if isinstance(v, str) and nid not in negatives and (
                    "TextEncode" in cls or "Prompt" in cls or "String" in cls or "Text" in cls):
                candidates.append((len(v), nid, key))
    if not candidates:
        return None
    _, nid, key = max(candidates)
    return nid, key


def _apply_overrides(wf: dict, cc: dict, seed: int, w: int, h: int, batch: int) -> list[str]:
    notes = []
    for nid, node in wf.items():
        cls, inp = node.get("class_type", ""), node.get("inputs", {})
        if "seed" in inp and not isinstance(inp["seed"], list):
            inp["seed"] = seed
        if "noise_seed" in inp and not isinstance(inp["noise_seed"], list):
            inp["noise_seed"] = seed
        if "Latent" in cls and "width" in inp and "height" in inp:
            if w and h and not isinstance(inp["width"], list):
                inp["width"], inp["height"] = w, h
            if "batch_size" in inp and not isinstance(inp["batch_size"], list):
                inp["batch_size"] = batch
        if "filename_prefix" in inp and isinstance(inp["filename_prefix"], str):   # any save node
            inp["filename_prefix"] = cc.get("filename_prefix") or "pds"
        if cc.get("steps") and "steps" in inp and not isinstance(inp["steps"], list):
            inp["steps"] = int(cc["steps"])
        if cc.get("cfg") and "cfg" in inp and not isinstance(inp["cfg"], list):
            inp["cfg"] = float(cc["cfg"])
        if cc.get("sampler") and "sampler_name" in inp and not isinstance(inp["sampler_name"], list):
            inp["sampler_name"] = cc["sampler"]
        if cc.get("scheduler") and "scheduler" in inp and not isinstance(inp["scheduler"], list):
            inp["scheduler"] = cc["scheduler"]
        # model drop-downs override loader nodes
        for key, field in (("checkpoint", "ckpt_name"), ("diffusion_model", "unet_name"), ("vae", "vae_name")):
            if cc.get(key) and field in inp and not isinstance(inp[field], list):
                inp[field] = cc[key]
                notes.append(f"{cls}.{field}")
        if cc.get("text_encoder"):
            for field in ("clip_name", "clip_name1"):
                if field in inp and not isinstance(inp[field], list):
                    inp[field] = cc["text_encoder"]
                    notes.append(f"{cls}.{field}")
        if cc.get("text_encoder_2") and "clip_name2" in inp and not isinstance(inp["clip_name2"], list):
            inp["clip_name2"] = cc["text_encoder_2"]
            notes.append(f"{cls}.clip_name2")
    return notes


def build_workflow(cc: dict, prose: str, tags: str | None, seed: int, size: tuple[int, int] | None,
                   custom_template: dict | None = None) -> tuple[dict, str]:
    """Returns (api_workflow, description of which text was sent)."""
    wf_type = cc.get("workflow", "custom")
    src = cc.get("prompt_source", "auto")
    use_tags = bool(tags) and (src == "tags" or (src == "auto" and wf_type == "builtin_sdxl"))
    text = tags if use_tags else prose
    sent = "tags" if use_tags else "prose"
    batch = max(1, int(cc.get("images_per_prompt") or 1))
    w = int(cc.get("width") or 0)
    h = int(cc.get("height") or 0)
    mode = cc.get("size_mode", "auto")
    if mode == "workflow":
        w = h = 0
    elif mode == "auto" and not (w and h) and size:
        sc = float(cc.get("comic_scale") or 1.0)
        w, h = (max(256, int(round(v * sc / 16.0)) * 16) for v in size)
    negative = cc.get("negative_prompt") or ""
    if wf_type == "builtin_sdxl":
        return _sdxl_workflow(cc, text, negative, seed, w or 1024, h or 1024, batch), sent
    if wf_type == "builtin_flux":
        return _flux_workflow(cc, text, seed, w or 1024, h or 1024, batch), sent
    if custom_template is None:
        if not cc.get("workflow_file"):
            raise ComfyError("Custom workflow selected but no workflow file is set "
                             "(ComfyUI: Workflow > Export (API))")
        custom_template = load_custom_workflow(cc["workflow_file"])
    wf = copy.deepcopy(custom_template)
    n = _replace_placeholders(wf, {"%PROMPT%": text, "%NEGATIVE%": negative, "%TAGS%": tags or text})
    if n == 0:
        found = _auto_prompt_node(wf)
        if not found:
            raise ComfyError("Could not find where the prompt goes in the custom workflow. Put %PROMPT% in your "
                             "prompt text box in ComfyUI and export the workflow again with Export (API).")
        nid, key = found
        wf[nid]["inputs"][key] = text
        sent += f" -> node {nid} ({wf[nid].get('class_type')}.{key}, auto-detected)"
    _apply_overrides(wf, cc, seed, w, h, batch)
    return wf, sent


# ---------------------------------------------------------------------------------------------
# Queue / wait / download
# ---------------------------------------------------------------------------------------------
def queue_prompt(cc: dict, workflow: dict, client_id: str) -> str:
    res = _request(cc, "POST", "/prompt", {"prompt": workflow, "client_id": client_id}, timeout=60)
    if res.get("node_errors"):
        raise ComfyError(_explain_error(400, json.dumps(res), "/prompt"))
    pid = res.get("prompt_id")
    if not pid:
        raise ComfyError(f"ComfyUI did not return a prompt_id: {str(res)[:200]}")
    return pid


def _exec_seconds(status: dict) -> float | None:
    """Execution time of a finished job from ComfyUI's execution_start/_success timestamps (ms)."""
    ts = {}
    for m in status.get("messages", []) or []:
        if isinstance(m, list) and len(m) > 1 and isinstance(m[1], dict) and "timestamp" in m[1]:
            ts[m[0]] = m[1]["timestamp"]
    if "execution_start" in ts and ("execution_success" in ts or "execution_error" in ts):
        end = ts.get("execution_success", ts.get("execution_error"))
        return max(0.0, (end - ts["execution_start"]) / 1000.0)
    return None


def wait_for(cc: dict, prompt_id: str, stop_event=None, timeout: float | None = None,
             timing: dict | None = None) -> list[dict]:
    """Poll /history until the job finishes. Returns the list of output image records.
    If `timing` is given, timing['exec'] gets ComfyUI's own execution time (seconds) when it reports one."""
    timeout = float(timeout or cc.get("timeout") or 900)
    t0, delay = time.monotonic(), 1.0
    while time.monotonic() - t0 < timeout:
        if stop_event is not None and stop_event.is_set():
            raise ComfyError("stopped while waiting for ComfyUI")
        hist = _request(cc, "GET", f"/history/{prompt_id}", timeout=30)
        job = hist.get(prompt_id)
        if job:
            status = job.get("status", {})
            if status.get("status_str") == "error":
                msgs = [m[1].get("exception_message", "") for m in status.get("messages", [])
                        if isinstance(m, list) and len(m) > 1 and m[0] == "execution_error"]
                raise ComfyError("ComfyUI job failed: " + ("; ".join(msgs) or "see the ComfyUI console"))
            if status.get("completed", True):
                if timing is not None:
                    timing["exec"] = _exec_seconds(status)
                images = []
                for out in job.get("outputs", {}).values():
                    images += [i for i in out.get("images", []) if i.get("type", "output") == "output"]
                return images
        time.sleep(delay)
        delay = min(delay * 1.3, 3.0)
    raise ComfyError(f"Timed out after {int(timeout)}s waiting for ComfyUI job {prompt_id}")


def download_image(cc: dict, img: dict, dest_dir: Path) -> Path:
    q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                                "type": img.get("type", "output")})
    data = _request(cc, "GET", f"/view?{q}", timeout=120, raw=True)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / img["filename"]
    out.write_bytes(data)
    return out


def vram_free_gb(cc: dict) -> float | None:
    """Free VRAM (GB) on ComfyUI's first GPU, or None if it doesn't say."""
    devs = _request(cc, "GET", "/system_stats", timeout=10).get("devices") or []
    for d in devs:
        if d.get("vram_total"):
            return d.get("vram_free", 0) / 2**30
    return None


def wait_for_vram(cc: dict, max_wait: float, stop_event=None) -> tuple[float | None, float | None, float]:
    """After the LLM unloads, wait until ComfyUI's free VRAM stops rising (the release has finished).
    Returns (free GB at start, free GB at end, seconds waited)."""
    t0 = time.monotonic()
    try:
        first = last = vram_free_gb(cc)
    except ComfyError:
        return None, None, 0.0
    if first is None or max_wait <= 0:
        return first, first, 0.0
    steady = 0
    while time.monotonic() - t0 < max_wait:
        if stop_event is not None and stop_event.is_set():
            break
        time.sleep(1.0)
        try:
            now = vram_free_gb(cc)
        except ComfyError:
            break
        if now is None:
            break
        steady = steady + 1 if now - last < 0.1 else 0     # under 100 MB change = settled
        last = now
        if steady >= 2:
            break
    return first, last, time.monotonic() - t0


def free_vram(cc: dict) -> None:
    _request(cc, "POST", "/free", {"unload_models": True, "free_memory": True}, timeout=30)


def unload_ollama(host: str, model: str) -> None:
    """Ask Ollama to unload a model now (keep_alive 0) so its VRAM is free for ComfyUI."""
    req = urllib.request.Request(host.rstrip("/") + "/api/generate", method="POST",
                                 data=json.dumps({"model": model, "keep_alive": 0}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        r.read()


# ---------------------------------------------------------------------------------------------
# Model rotation + compatibility check
# ---------------------------------------------------------------------------------------------
import re as _re
import struct as _struct

FAMILY_PATTERNS = [            # filename hints (fallback when the file can't be read)
    ("krea2", r"krea[ _-]?2|krea2"), ("flux", r"flux|chroma"), ("qwen_image", r"qwen[ _-]?image"),
    ("sd3", r"sd3|sd_3|stable[ _-]?diffusion[ _-]?3"), ("wan", r"\bwan[ _-]?2|wan2"), ("hidream", r"hidream"),
    ("sdxl", r"sdxl|illustrious|noob|pony|xl[_ .-]|_xl\b|animagine"), ("sd15", r"sd15|sd_1\.5|sd1\.5"),
]


def name_family(name: str) -> str | None:
    n = name.lower()
    for fam, pat in FAMILY_PATTERNS:
        if _re.search(pat, n):
            return fam
    return None


def rotation_target(cc: dict, template: dict | None) -> str:
    """Which loader the rotation swaps: 'checkpoint' (ckpt_name) or 'diffusion_model' (unet_name)."""
    t = cc.get("rotation_target", "auto")
    if t in TARGET_FOLDER:
        return t
    wf = cc.get("workflow")
    if wf == "builtin_sdxl":
        return "checkpoint"
    if wf == "builtin_flux":
        return "diffusion_model"
    classes = {n.get("class_type", "") for n in (template or {}).values()}
    if any("CheckpointLoader" in c for c in classes) and not any("UNETLoader" in c or "UnetLoader" in c
                                                                 for c in classes):
        return "checkpoint"
    return "diffusion_model"


def workflow_model(cc: dict, template: dict | None, target: str) -> str:
    """The model the workflow (or the drop-down override) currently uses for the rotation target."""
    if cc.get(target):
        return cc[target]
    field = "ckpt_name" if target == "checkpoint" else "unet_name"
    for n in (template or {}).values():
        v = n.get("inputs", {}).get(field)
        if isinstance(v, str):
            return v
    return ""


_FILE_INDEX: dict[str, dict[str, str]] = {}


def _search_roots(cc: dict) -> list[Path]:
    roots = []
    if cc.get("models_dir"):
        r = Path(cc["models_dir"]).expanduser()
        known = {n for names in MODEL_FOLDERS.values() for n in names}
        roots.append(r.parent if r.name in known else r)
    for part in str(cc.get("model_search_dirs") or "").replace("\n", ";").split(";"):
        if part.strip():
            roots.append(Path(part.strip()).expanduser())
    return [r for r in roots if r.is_dir()]


def find_model_file(name: str, folder_key: str, cc: dict) -> Path | None:
    roots = _search_roots(cc)
    rel = Path(name)
    for r in roots:
        for sub in [*MODEL_FOLDERS.get(folder_key, []), ""]:
            p = r / sub / rel if sub else r / rel
            if p.is_file():
                return p
    key = ";".join(str(r) for r in roots)
    if key not in _FILE_INDEX:                 # one-time index by file name over the search folders
        idx: dict[str, str] = {}
        for r in roots:
            for p in r.rglob("*"):
                if p.suffix.lower() in MODEL_EXTS and p.is_file():
                    idx.setdefault(p.name, str(p))
        _FILE_INDEX[key] = idx
    found = _FILE_INDEX[key].get(rel.name)
    return Path(found) if found else None


_NON_DIFFUSION = ("first_stage_model.", "cond_stage_model.", "conditioner.", "text_encoders.", "text_encoder.",
                  "text_model.", "vae.", "clip_l.", "clip_g.", "t5xxl.", "model_ema.", "embedding_manager.")


class ModelSignature:
    """Architecture fingerprint from a safetensors header (tensor names and shapes; no weights are read).

    names:  normalized module paths (block numbers replaced by #), for a rough layout comparison
    tops:   top-level module names
    blocks: number of repeated blocks per top-level module (e.g. layers -> 30)
    dim:    the most common tensor dimension of the repeated blocks (the model width, e.g. 3840)
    """

    def __init__(self, names: frozenset, tops: frozenset, blocks: dict, dim: int | None, n: int):
        self.names, self.tops, self.blocks, self.dim, self.n = names, tops, blocks, dim, n

    def compare(self, other: "ModelSignature") -> tuple[bool, str]:
        jac = len(self.names & other.names) / max(1, len(self.names | other.names))
        top_jac = len(self.tops & other.tops) / max(1, len(self.tops | other.tops))
        shared = set(self.blocks) & set(other.blocks)
        diff_blocks = [f"{k}: {self.blocks[k]} vs {other.blocks[k]}" for k in sorted(shared)
                       if self.blocks[k] != other.blocks[k]]
        if self.dim and other.dim:
            if self.dim != other.dim:
                return False, f"different model width ({other.dim}; the workflow's model is {self.dim})"
            if diff_blocks:
                return False, "different number of blocks (workflow vs this: " + "; ".join(diff_blocks[:2]) + ")"
            if shared or top_jac >= 0.5:
                # same width and same block counts: same architecture, even if some tensors are named
                # differently (diffusers vs original format, extra scale tensors, merged LoRA leftovers)
                return True, f"same layout (width {self.dim}, {jac:.0%} name match)"
            return False, f"different structure ({jac:.0%} name match)"
        return (jac >= 0.75), f"{'same' if jac >= 0.75 else 'different'} layout ({jac:.0%} name match)"


def safetensors_signature(path: Path) -> ModelSignature | None:
    if path.suffix.lower() not in (".safetensors", ".sft"):
        return None
    try:
        with open(path, "rb") as f:
            n = _struct.unpack("<Q", f.read(8))[0]
            if not 0 < n < 200_000_000:
                return None
            header = json.loads(f.read(n))
    except (OSError, ValueError, _struct.error):
        return None
    keys = [k for k in header if k != "__metadata__"]
    # all-in-one checkpoints: keep only the diffusion model's tensors
    if any(k.startswith("model.diffusion_model.") for k in keys):
        keys = [k for k in keys if k.startswith("model.diffusion_model.")]
    else:
        keys = [k for k in keys if not k.startswith(_NON_DIFFUSION)]
    names, tops, idx = set(), set(), {}
    dims = Counter()
    for k in keys:
        k2 = _re.sub(r"^(model\.diffusion_model\.|diffusion_model\.|model\.|unet\.|transformer\.)", "", k)
        parts = k2.split(".")
        tops.add(parts[0])
        if len(parts) > 1 and parts[1].isdigit():
            idx.setdefault(parts[0], set()).add(parts[1])
        norm = _re.sub(r"\d+", "#", k2).split(".")
        if len(norm) > 1:
            names.add(".".join(norm[:min(3, len(norm) - 1)]))
        shape = (header.get(k) or {}).get("shape") or []
        if len(shape) == 2 and len(parts) > 1 and parts[1].isdigit():
            dims.update(d for d in shape if d >= 64)
    if not names:
        return None
    blocks = {t: len(v) for t, v in idx.items() if len(v) > 1}
    return ModelSignature(frozenset(names), frozenset(tops), blocks,
                          dims.most_common(1)[0][0] if dims else None, len(keys))


def check_compat(cc: dict, reference: str, candidates: list[str], folder_key: str) -> dict[str, tuple[str, str]]:
    """name -> (verdict, reason) with verdict 'ok' | 'no' | 'unknown', compared to the reference model."""
    out: dict[str, tuple[str, str]] = {}
    ref_file = find_model_file(reference, folder_key, cc) if reference else None
    ref_sig = safetensors_signature(ref_file) if ref_file else None
    ref_fam = name_family(reference) if reference else None
    for c in candidates:
        if c == reference:
            out[c] = ("ok", "the workflow's own model")
            continue
        sig, f = None, None
        if ref_sig:
            f = find_model_file(c, folder_key, cc)
            sig = safetensors_signature(f) if f else None
        if ref_sig and sig:
            ok, why = ref_sig.compare(sig)
            out[c] = ("ok" if ok else "no", why)
            continue
        fam = name_family(c)
        if ref_fam and fam:
            out[c] = ("ok", f"name looks like {fam}") if fam == ref_fam else \
                     ("no", f"name looks like {fam}, workflow uses {ref_fam}")
        elif f is not None:
            out[c] = ("unknown", f"can't read the file's layout ({f.suffix or 'no extension'})")
        elif not ref_sig:
            out[c] = ("unknown", "the workflow's model file wasn't found in the search folders")
        else:
            out[c] = ("unknown", "file not found in the search folders and the name gives no hint")
    for c, (v, why) in out.items():
        log.debug("compat %s: %s - %s", c, v, why)
    return out


def plan_rotation(cc: dict, n_jobs: int, lists: dict, template: dict | None,
                  rng: random.Random, compare: bool | None = None) -> tuple[list[str] | None, str]:
    """Model per prompt (consecutive prompts share a model). Returns (assignments or None, report).
    Compare mode: returns the list of models, each of which renders every prompt."""
    mode = cc.get("rotation", "off")
    compare = cc.get("compare_models", False) if compare is None else compare
    if compare:
        n_jobs = max(n_jobs, 10**6)          # no "more models than prompts" limit when comparing
    if mode == "off" or n_jobs < 1:
        return None, ""
    target = rotation_target(cc, template)
    folder = TARGET_FOLDER[target]
    ref = workflow_model(cc, template, target)
    if mode == "random":
        want = int(cc.get("rotation_count") or 0)
        if want < 1:
            raise ComfyError("Model rotation: the number of random models must be at least 1")
        if want > n_jobs:
            raise ComfyError(f"Model rotation: {want} models but only {n_jobs} prompt(s) - use {n_jobs} or fewer")
        pool = list(lists.get(folder) or [])
        if not pool:
            raise ComfyError(f"Model rotation: no cached {folder} list - press 'Refresh model lists' first")
        if cc.get("rotation_compatible_only", True):
            verdicts = check_compat(cc, ref, pool, folder)
            ok = [m for m in pool if verdicts[m][0] == "ok"]
            if len(ok) < want:
                raise ComfyError(
                    f"Model rotation: only {len(ok)} model(s) are known to match the workflow's model "
                    f"({ref or 'unknown'}), {want} wanted. Add model search folders so the files can be checked, "
                    "lower the number, or turn off 'only compatible models'.")
            pool = ok
        models = rng.sample(pool, want)
    else:
        models = [m for m in (cc.get("rotation_models") or []) if str(m).strip()]
        if not models:
            raise ComfyError("Model rotation: the model list is empty")
        if cc.get("rotation_compatible_only", True):
            verdicts = check_compat(cc, ref, models, folder)
            bad = [m for m in models if verdicts[m][0] == "no"]
            for m in bad:
                log.warning("Model rotation: skipping %s (%s)", m, verdicts[m][1])
            models = [m for m in models if m not in bad]
            if not models:
                raise ComfyError("Model rotation: none of the listed models match the workflow's model")
        if len(models) > n_jobs:
            raise ComfyError(f"Model rotation: {len(models)} models but only {n_jobs} prompt(s) - "
                             "generate at least as many prompts as models")
    if compare:
        return models, (f"Compare models ({mode}, {target.replace('_', ' ')}): rendering on "
                        f"{len(models)} model(s): " + ", ".join(models))
    base, extra = divmod(n_jobs, len(models))
    plan: list[str] = []
    parts = []
    for i, m in enumerate(models):
        k = base + (1 if i < extra else 0)
        if k:
            parts.append(f"prompts {len(plan) + 1}-{len(plan) + k} -> {m}" if k > 1 else f"prompt {len(plan) + 1} -> {m}")
        plan += [m] * k
    report = (f"Model rotation ({mode}, {target.replace('_', ' ')}): {len(models)} model(s) over {n_jobs} prompt(s): "
              + "; ".join(parts))
    return plan, report


class ComfySender:
    """Sends prompts for one run. Use .send() per prompt; .finish() at the end."""

    def __init__(self, cc: dict, run_dir: Path | None, stop_event=None, seed_rng: random.Random | None = None):
        self.cc = cc
        self.run_dir = run_dir
        self.stop_event = stop_event
        self.rng = seed_rng or random.Random()
        self.client_id = uuid.uuid4().hex
        self.template = load_custom_workflow(cc["workflow_file"]) if cc.get("workflow") == "custom" else None
        self.sent = self.done = self.failed = 0
        self.image_files: list[str] = []
        self.plan: list[str] | None = None
        self.compare: list[str] | None = None      # compare mode: every prompt goes to each of these
        self.target = rotation_target(cc, self.template)
        self.timings: list[dict] = []              # one per finished image job
        self._prompt_seeds: dict = {}
        self.per_model = 1                         # compare mode: images per model (seed variations)
        self.compare_rows: list[dict] = []         # for compare.html

    def preflight(self) -> list[str]:
        """Before any prompt is written: check that ComfyUI has every model and LoRA the workflow uses.
        Missing LoRAs are left out of the jobs (or stop the run); a missing model always stops it.
        Returns the warnings that were logged."""
        if self.template is None:
            return []
        try:
            lists = _list_via_api(self.cc)
        except ComfyError as e:
            log.warning("ComfyUI: couldn't read its model lists, so the workflow's models and LoRAs weren't "
                        "checked (%s)", e)
            return []
        from pd_workflow import WorkflowDoc
        doc = WorkflowDoc(self.template)
        warnings = []
        skip = {"ckpt_name"} if self.target == "checkpoint" else {"unet_name"}
        if self.cc.get("rotation", "off") == "off":
            skip = set()
        missing = doc.missing_models(lists, skip_fields=skip)
        if missing:
            raise ComfyError("the workflow uses model file(s) that ComfyUI doesn't have: "
                             + "; ".join(f"{t} {f} = {v}" for t, f, v in missing)
                             + ". Pick the right file on the Workflow tab (Choose...) and save the workflow.")
        if "loras" in lists:
            gone = doc.missing_loras(lists["loras"])
            if gone:
                names = ", ".join(dict.fromkeys(e["name"] for e in gone))
                if self.cc.get("missing_loras", "skip") == "stop":
                    raise ComfyError(f"the workflow uses LoRA file(s) that ComfyUI doesn't have: {names}. "
                                     "Fix or remove them on the Workflow tab, or set 'Missing LoRAs' to skip.")
                for e in gone:
                    try:
                        doc.remove_lora(e)
                    except (KeyError, ValueError):
                        pass
                self.template = doc.nodes
                msg = (f"ComfyUI doesn't have these LoRA file(s), so they are left out of every image: {names}. "
                       "Fix or remove them on the Workflow tab.")
                log.warning("ComfyUI: %s", msg)
                warnings.append(msg)
        return warnings

    def set_plan(self, n_jobs: int, lists: dict) -> str:
        plan, report = plan_rotation(self.cc, n_jobs, lists, self.template, self.rng)
        if self.cc.get("compare_models"):
            self.compare = plan
        else:
            self.plan = plan
        return report

    def _seed(self) -> int:
        return int(self.cc.get("seed") or 0) or self.rng.randint(1, 2**48)

    def set_compare_images(self, n_images: int) -> str:
        """Compare mode: one prompt -> n_images, split evenly across the models (same seeds on each)."""
        n_models = max(1, len(self.compare or []))
        self.per_model = max(1, -(-n_images // n_models))          # ceil
        total = self.per_model * n_models
        note = f" (rounded up from {n_images} so every model gets the same seeds)" if total != n_images else ""
        return (f"Compare models: the LLM writes 1 prompt; it is rendered {total} times{note} = "
                f"{self.per_model} seed(s) x {n_models} model(s)")

    def compare_jobs(self) -> list[tuple[str, int]]:
        """(model, variant) pairs, model by model so each model loads once."""
        return [(m, v) for m in (self.compare or []) for v in range(self.per_model)]

    def prompt_seed(self, key) -> int:
        """One seed per (prompt, variant), shared by every model in compare mode (a fair comparison)."""
        if key not in self._prompt_seeds:
            fixed = int(self.cc.get("seed") or 0)
            v = key[1] if isinstance(key, tuple) else 0
            self._prompt_seeds[key] = fixed + v if fixed else self.rng.randint(1, 2**48)
        return self._prompt_seeds[key]

    def send(self, index: int, prose: str, tags: str | None, size: tuple[int, int] | None,
             model: str | None = None, variant: int = 0) -> dict:
        """Queue one prompt; waits/downloads if configured. Returns info for the record.
        `model` (compare mode) renders this prompt on that model with the prompt's shared seed."""
        cc = self.cc
        if model:
            seed = self.prompt_seed((index, variant))
            stem = Path(model.replace("\\", "/")).stem[:40]
            sv = f"_s{variant + 1:02d}" if self.per_model > 1 else ""
            cc = dict(self.cc, **{self.target: model,
                                  "filename_prefix": f"{self.cc.get('filename_prefix') or 'pds'}_{index:03d}{sv}_{stem}"})
        else:
            seed = self._seed()
            if self.plan and 1 <= index <= len(self.plan):
                model = self.plan[index - 1]
                cc = dict(self.cc, **{self.target: model})
        wf, what = build_workflow(cc, prose, tags, seed, size, self.template)
        if model:
            what += f", model {model}"
        t0 = time.monotonic()
        pid = queue_prompt(self.cc, wf, self.client_id)
        self.sent += 1
        info = {"prompt_id": pid, "seed": seed, "sent": what, "images": [], "model": model}
        log.info("[%d] ComfyUI: queued %s (seed %d, %s)", index, pid, seed, what)
        urls: list[str] = []
        if self.cc.get("wait_for_images", True):
            timing: dict = {}
            imgs = wait_for(self.cc, pid, self.stop_event, timing=timing)
            wall = time.monotonic() - t0
            self.done += 1
            names = [i.get("filename", "?") for i in imgs]
            info["images"] = names
            info["seconds"] = round(wall, 1)
            urls = [f"{_base(self.cc)}/view?" + urllib.parse.urlencode(
                {"filename": i.get("filename", ""), "subfolder": i.get("subfolder", ""), "type": i.get("type", "output")})
                for i in imgs]
            first_for_model = not any(t["model"] == model for t in self.timings)
            self.timings.append({"index": index, "model": model, "wall": wall, "exec": timing.get("exec"),
                                 "first": first_for_model})
            ex = f", ComfyUI ran it in {timing['exec']:.1f}s" if timing.get("exec") is not None else ""
            log.info("[%d] ComfyUI: finished in %.1fs%s%s -> %s", index, wall, ex,
                     " (first image with this model: includes loading it)" if first_for_model else "",
                     ", ".join(names) or "(no images reported)")
            if self.cc.get("download_images") and self.run_dir is not None:
                for img in imgs:
                    try:
                        p = download_image(self.cc, img, self.run_dir / "images")
                        self.image_files.append(str(p))
                    except ComfyError as e:
                        log.warning("[%d] ComfyUI: could not download %s: %s", index, img.get("filename"), e)
                        p = None
                    if p is not None:
                        info.setdefault("files", []).append(p.name)
        if self.compare:
            self.compare_rows.append({"index": index, "variant": variant, "model": model, "seed": seed, "prompt": prose,
                                      "files": info.get("files", []), "seconds": info.get("seconds"),
                                      "urls": urls if self.cc.get("wait_for_images", True) else []})
        return info

    def timing_report(self) -> list[str]:
        """Summary lines: first image (model load) vs the rest, per model."""
        if not self.timings:
            return []
        lines = []
        models = list(dict.fromkeys(t["model"] for t in self.timings))
        for m in models:
            ts = [t for t in self.timings if t["model"] == m]
            first = ts[0]["wall"]
            rest = [t["wall"] for t in ts[1:]]
            label = f"model {m}" if m else "images"
            line = f"{label}: {len(ts)} image(s), first {first:.0f}s"
            if rest:
                avg = sum(rest) / len(rest)
                line += f", the rest {avg:.0f}s on average"
                if first > avg * 2 + 15:
                    line += f" - loading the model cost about {first - avg:.0f}s"
            lines.append(line)
        return lines

    def write_compare_page(self) -> Path | None:
        """compare.html in the run folder: one row per prompt, one column per model."""
        if not self.compare or not self.compare_rows or self.run_dir is None:
            return None
        import html as _h
        prompts = list(dict.fromkeys((r["index"], r.get("variant", 0)) for r in self.compare_rows))
        cells = {(r["index"], r.get("variant", 0), r["model"]): r for r in self.compare_rows}
        head = "".join(f"<th>{_h.escape(m)}</th>" for m in self.compare)
        body = []

        def cell(r):
            if not r:
                return "<td class=miss>not rendered</td>"
            srcs = [f"images/{f}" for f in r["files"]] or r.get("urls") or []   # local copy, else from ComfyUI
            imgs = "".join(f'<a href="{_h.escape(u)}"><img src="{_h.escape(u)}" loading=lazy></a>'
                           for u in srcs) or "<div class=miss>no image</div>"
            tm = f"{r['seconds']:.0f}s" if r.get("seconds") is not None else ""
            return f"<td>{imgs}<div class=meta>seed {r['seed']} &middot; {tm}</div></td>"

        texts = list(dict.fromkeys((r["index"], r["prompt"]) for r in self.compare_rows))
        intro = "".join(f"<details{' open' if len(texts) == 1 else ''}><summary>Prompt {i}</summary>"
                        f"<pre>{_h.escape(t)}</pre></details>" for i, t in texts)
        if len(self.compare) == 1:
            # one model: a gallery of the seed variations
            m = self.compare[0]
            tiles = [cell(cells.get((i, v, m))) for i, v in prompts]
            rows = [tiles[k:k + 4] for k in range(0, len(tiles), 4)]
            body = ["<tr>" + "".join(r) + "</tr>" for r in rows]
            head = f"<th colspan=4>{_h.escape(m)}</th>"
        else:
            for i, v in prompts:
                body.append(f"<tr><th class=p><b>Prompt {i}</b>"
                            + (f" &middot; seed {v + 1}" if self.per_model > 1 else "") + "</th>"
                            + "".join(cell(cells.get((i, v, m))) for m in self.compare) + "</tr>")
            head = "<th></th>" + head
        page = f"""<!DOCTYPE html><html><head><meta charset=utf-8><title>Model comparison</title><style>
body{{font:14px system-ui,sans-serif;background:#1b1b1d;color:#ddd;margin:16px}}
table{{border-collapse:collapse}}td,th{{border:1px solid #333;padding:6px;vertical-align:top}}
thead th{{position:sticky;top:0;background:#26262a;font-weight:600;max-width:340px;word-break:break-all}}
th.p{{text-align:left;width:220px;font-weight:400}}img{{max-width:340px;max-height:420px;display:block}}
pre{{white-space:pre-wrap;font-size:13px;max-width:900px}}.meta{{color:#888;font-size:12px}}.miss{{color:#a66}}
</style></head><body><h2>Model comparison</h2><p>Same prompt, and the same seeds on every model. Click an image for full size. Images that weren't downloaded load from ComfyUI, so it has to be running.</p>
{intro}<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></body></html>"""
        out = self.run_dir / "compare.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(page, encoding="utf-8")
        return out

    def finish(self) -> None:
        try:
            page = self.write_compare_page()
            if page:
                log.info("ComfyUI: model comparison page -> %s", page)
        except OSError as e:
            log.warning("ComfyUI: could not write the comparison page: %s", e)
        if self.cc.get("free_vram_after") and self.sent and (self.cc.get("wait_for_images", True)):
            try:
                free_vram(self.cc)
                log.info("ComfyUI: asked to unload models and free VRAM")
            except ComfyError as e:
                log.warning("ComfyUI: could not free VRAM: %s", e)
