"""
pd_workflow.py — inspect and edit a ComfyUI API-format workflow (the file from Workflow > Export (API)).

Supports:
* LoRAs: rgthree "Power Lora Loader" entries (lora_1, lora_2, ...), LoraLoader and LoraLoaderModelOnly nodes.
  Edit name / strength / on-off, remove (Power entries are deleted; loader nodes are unwired and removed),
  add (a Power entry if the workflow has a Power Lora Loader, otherwise a new LoraLoaderModelOnly node
  wired in after the model source).
* Latent size (Empty*Latent* nodes), upscale passes (LatentUpscaleBy, ImageScaleBy, NNLatentUpscale,
  UltimateSDUpscale, ...) and a size suggestion that accounts for them.
* Samplers (KSampler, KSamplerAdvanced, SamplerCustom*, BasicScheduler ...): steps, cfg, sampler, scheduler, denoise.
* Model loaders: ckpt_name, unet_name, clip_name*, vae_name.
* Where the prompt goes (%PROMPT% placeholder or auto-detected box).
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

LOADER_FIELDS = ("ckpt_name", "unet_name", "clip_name", "clip_name1", "clip_name2", "clip_name3", "vae_name")
SAMPLER_FIELDS = ("steps", "cfg", "sampler_name", "scheduler", "denoise")
UPSCALE_SCALE_KEYS = ("scale_by", "upscale_by", "scale", "upscale_factor")


def _is_ref(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)


class WorkflowDoc:
    def __init__(self, nodes: dict, path: str | None = None):
        self.nodes = nodes
        self.path = path
        self.dirty = False

    # ------------------------------------------------------------------ load / save
    @classmethod
    def load(cls, path: str) -> "WorkflowDoc":
        p = Path(path).expanduser()
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "nodes" in data and "links" in data:
            raise ValueError("This is a regular ComfyUI workflow. Use Workflow > Export (API) in ComfyUI.")
        if not isinstance(data, dict) or not all(isinstance(v, dict) and "class_type" in v for v in data.values()):
            raise ValueError("Not a ComfyUI API-format workflow")
        return cls(data, str(p))

    def save(self, path: str) -> str:
        p = Path(path).expanduser()
        p.write_text(json.dumps(self.nodes, indent=2, ensure_ascii=False), encoding="utf-8")
        self.path = str(p)
        self.dirty = False
        return str(p)

    @staticmethod
    def edited_copy_path(original: str) -> str:
        p = Path(original).expanduser()
        stem = p.stem
        for tag in ("_pd", "_kpg"):                      # _kpg = copies made by the old app name
            if stem.endswith(tag):
                stem = stem[: -len(tag)]
        return str(p.with_name(f"{stem}_pd{p.suffix or '.json'}"))

    def title(self, nid: str) -> str:
        n = self.nodes[nid]
        t = (n.get("_meta") or {}).get("title") or n.get("class_type", "?")
        return f"{t} (#{nid})"

    def _new_id(self) -> str:
        nums = [int(k) for k in self.nodes if str(k).isdigit()]
        return str(max(nums, default=0) + 1)

    def _redirect(self, old: list, new: list, skip: str | None = None):
        """Point every input that references `old` [node, slot] to `new`."""
        for nid, n in self.nodes.items():
            if nid == skip:
                continue
            for k, v in n.get("inputs", {}).items():
                if _is_ref(v) and v[0] == old[0] and v[1] == old[1]:
                    n["inputs"][k] = list(new)

    # ------------------------------------------------------------------ LoRAs
    def loras(self) -> list[dict]:
        out = []
        for nid, n in self.nodes.items():
            cls, inp = n.get("class_type", ""), n.get("inputs", {})
            if "Power Lora Loader" in cls:
                for key in sorted((k for k in inp if k.lower().startswith("lora_") and isinstance(inp[k], dict)),
                                  key=lambda k: int(re.sub(r"\D", "", k) or 0)):
                    e = inp[key]
                    out.append({"node": nid, "kind": "power", "key": key, "name": e.get("lora", ""),
                                "strength": e.get("strength", 1.0), "on": bool(e.get("on", True))})
            elif cls in ("LoraLoader", "LoraLoaderModelOnly") and isinstance(inp.get("lora_name"), str):
                out.append({"node": nid, "kind": cls, "key": None, "name": inp["lora_name"],
                            "strength": inp.get("strength_model", 1.0),
                            "on": float(inp.get("strength_model", 1.0) or 0) != 0.0 or
                                  float(inp.get("strength_clip", 0) or 0) != 0.0})
        return out

    def set_lora(self, entry: dict, name: str | None = None, strength: float | None = None, on: bool | None = None):
        n = self.nodes[entry["node"]]["inputs"]
        if entry["kind"] == "power":
            e = n[entry["key"]]
            if name is not None:
                e["lora"] = name
            if strength is not None:
                e["strength"] = float(strength)
            if on is not None:
                e["on"] = bool(on)
        else:
            if name is not None:
                n["lora_name"] = name
            if strength is not None:
                n["strength_model"] = float(strength)
                if "strength_clip" in n and not _is_ref(n["strength_clip"]):
                    n["strength_clip"] = float(strength)
            if on is False:                       # loader nodes can't be bypassed in API format: zero them
                n["strength_model"] = 0.0
                if "strength_clip" in n:
                    n["strength_clip"] = 0.0
        self.dirty = True

    def remove_lora(self, entry: dict):
        nid = entry["node"]
        if entry["kind"] == "power":
            del self.nodes[nid]["inputs"][entry["key"]]
        else:
            inp = self.nodes[nid]["inputs"]
            if _is_ref(inp.get("model")):
                self._redirect([nid, 0], inp["model"], skip=nid)
            if _is_ref(inp.get("clip")):
                self._redirect([nid, 1], inp["clip"], skip=nid)
            del self.nodes[nid]
        self.dirty = True

    def _model_source(self) -> list | None:
        """The [node, slot] feeding the samplers' model input, followed up through LoRA-like nodes."""
        for n in self.nodes.values():
            if "model" in n.get("inputs", {}) and any(f in n["inputs"] for f in ("steps", "noise_seed", "seed")):
                if _is_ref(n["inputs"]["model"]):
                    return n["inputs"]["model"]
        for nid, n in self.nodes.items():
            if n.get("class_type") in ("UNETLoader", "CheckpointLoaderSimple", "UnetLoaderGGUF"):
                return [nid, 0]
        return None

    def add_lora(self, name: str, strength: float = 1.0) -> str:
        power = [nid for nid, n in self.nodes.items() if "Power Lora Loader" in n.get("class_type", "")]
        if power:
            inp = self.nodes[power[0]]["inputs"]
            nums = [int(re.sub(r"\D", "", k) or 0) for k in inp if k.lower().startswith("lora_")]
            key = f"lora_{max(nums, default=0) + 1}"
            inp[key] = {"on": True, "lora": name, "strength": float(strength), "strengthTwo": None}
            self.dirty = True
            return f"added to {self.title(power[0])} as {key}"
        src = self._model_source()
        if not src:
            raise ValueError("Could not find where the model is loaded, so the LoRA can't be wired in")
        nid = self._new_id()
        self.nodes[nid] = {"class_type": "LoraLoaderModelOnly",
                           "inputs": {"lora_name": name, "strength_model": float(strength), "model": list(src)},
                           "_meta": {"title": "LoRA (added by PromptDirector Studio)"}}
        self._redirect(src, [nid, 0], skip=nid)
        self.dirty = True
        return f"added as new node #{nid} after {self.title(src[0])}"

    # ------------------------------------------------------------------ size / upscale / samplers / loaders
    def latents(self) -> list[str]:
        return [nid for nid, n in self.nodes.items()
                if "Latent" in n.get("class_type", "") and "width" in n.get("inputs", {})
                and "height" in n.get("inputs", {}) and not _is_ref(n["inputs"]["width"])]

    def size(self) -> tuple[int, int] | None:
        lat = self.latents()
        if not lat:
            return None
        i = self.nodes[lat[0]]["inputs"]
        return int(i["width"]), int(i["height"])

    def set_size(self, w: int, h: int):
        for nid in self.latents():
            self.nodes[nid]["inputs"]["width"], self.nodes[nid]["inputs"]["height"] = int(w), int(h)
        self.dirty = True

    def upscale_factor(self) -> tuple[float, str] | None:
        """Total upscale applied after the first pass, if the workflow has one."""
        factor, where = 1.0, []
        base = self.size()
        for nid, n in self.nodes.items():
            cls, inp = n.get("class_type", ""), n.get("inputs", {})
            if "Upscale" not in cls and "Scale" not in cls:
                continue
            for k in UPSCALE_SCALE_KEYS:
                v = inp.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and 1.0 < v <= 8:
                    factor *= float(v)
                    where.append(f"{self.title(nid)} x{v:g}")
                    break
            else:
                if base and isinstance(inp.get("width"), (int, float)) and inp.get("width", 0) > base[0] \
                        and "Latent" in cls and "Empty" not in cls:
                    f = inp["width"] / base[0]
                    factor *= f
                    where.append(f"{self.title(nid)} to {inp['width']}x{inp.get('height')}")
        return (factor, ", ".join(where)) if factor > 1.0 else None

    def samplers(self) -> list[str]:
        return [nid for nid, n in self.nodes.items()
                if any(f in n.get("inputs", {}) for f in ("steps", "sampler_name", "denoise"))
                and ("Sampler" in n.get("class_type", "") or "Scheduler" in n.get("class_type", "")
                     or "steps" in n.get("inputs", {}))]

    def set_input(self, nid: str, key: str, value):
        self.nodes[nid]["inputs"][key] = value
        self.dirty = True

    def loaders(self) -> list[tuple[str, str, str]]:
        out = []
        for nid, n in self.nodes.items():
            for f in LOADER_FIELDS:
                v = n.get("inputs", {}).get(f)
                if isinstance(v, str):
                    out.append((nid, f, v))
        return out

    def prompt_location(self) -> str:
        for nid, n in self.nodes.items():
            for k, v in n.get("inputs", {}).items():
                if isinstance(v, str) and "%PROMPT%" in v:
                    return f"%PROMPT% in {self.title(nid)}.{k}"
        try:
            from pd_comfy import _auto_prompt_node
            found = _auto_prompt_node(self.nodes)
        except Exception:
            found = None
        if found:
            return f"no %PROMPT% - the prompt will go into {self.title(found[0])}.{found[1]} (auto-detected)"
        return "no prompt box found - put %PROMPT% in your prompt box and export again"


def round16(x: float) -> int:
    return max(256, int(round(x / 16.0)) * 16)


def suggest_size(doc: WorkflowDoc, cfg: dict, original_size: tuple[int, int] | None = None
                 ) -> tuple[tuple[int, int] | None, str]:
    """What the app thinks the base (first-pass) size should be, and why.
    original_size: the size in the untouched export, used for single images."""
    import pd_core as core
    cur = doc.size()
    up = doc.upscale_factor()
    comic = cfg.get("comic", {})
    if comic.get("enabled"):
        tw, th = (int(x) for x in core.suggested_size(comic["layout"]).split("x"))
        why = f"comic {comic['layout']} page should end near {tw}x{th}"
    elif cur:
        base = tuple(original_size) if original_size else cur
        why = f"single image: use your workflow's own size {base[0]}x{base[1]}"
        if original_size and tuple(cur) != base:
            why += (f" (it is currently {cur[0]}x{cur[1]}, probably from an earlier comic suggestion; "
                    f"your original export uses {base[0]}x{base[1]})")
        if up:
            why += f"; after your x{up[0]:.2g} upscale it ends near {round16(base[0] * up[0])}x{round16(base[1] * up[0])}"
        return base, why
    else:
        return None, "the workflow has no latent size node"
    if up:
        f, where = up
        bw, bh = round16(tw / f), round16(th / f)
        why += f"; your workflow upscales x{f:.2g} ({where}), so start at {bw}x{bh} -> ends near {round16(bw * f)}x{round16(bh * f)}"
        return (bw, bh), why
    return (round16(tw), round16(th)), why
