# PromptDirector Studio v2.7

Generates character-centric image prompts with a local LLM, following `system_prompt.txt`.
It has a GUI (CustomTkinter) and a CLI, which share the same engine and config file.

## Files

| File | What it is |
|---|---|
| `promptdirector_gui.py` | The GUI |
| `promptdirector.py` | The CLI |
| `pd_llamaserver.py` | Starts/stops llama.cpp's llama-server for the app (managed server) |
| `pd_workflow.py` | Reads and edits ComfyUI API workflows (LoRAs, size, samplers, models) |
| `pd_picker.py` | Searchable picker window for long model lists |
| `pd_comfy.py` | ComfyUI connection: model lists, workflows, sending, waiting, downloading |
| `pd_core.py` | Shared engine: config, backends, scene builder, logging, output. `DEFAULT_CONFIG` and `DEFAULT_PRESETS` live here |
| `system_prompt.txt` | Your system prompt (cleaned up, plus section 11 on the scene input format) |
| `presets.json` | The random-value lists (editable in the GUI's Presets tab) |
| `config.example.toml` | Every setting with its default value |
| `pd_gui_bench.py` | Measures CustomTkinter speed on your machine (for troubleshooting a slow GUI) |

## Setup

```bash
pip install customtkinter          # GUI only; the CLI needs nothing extra
# optional: pip install tomli (Python < 3.11), pip install llama-cpp-python (in-process .gguf)
cp config.example.toml config.toml # optional
ollama pull llama3.1:8b
python promptdirector_gui.py
```

On Linux, tkinter may need a system package: `sudo apt install python3-tk`.

The Comic, Backend & Model, Generation and Output & Logging tabs are built the first time you open them, which keeps startup fast. To see where any slowness comes from, start the GUI with `PD_GUI_TIMING=1 python promptdirector_gui.py`; it prints startup and tab-switch timings in the terminal.

On Linux the GUI starts with a private, minimal font setup (Roboto, DejaVu Sans/Mono, Noto Sans and CustomTkinter's shape font, linked into `~/.cache/promptdirector/fontconfig`). Tk sorts every installed font each time it opens a new font, which is very slow on systems with tens of thousands of fonts. Your system fonts are not changed. Disable with `PD_GUI_FONTCONFIG=0`; in that case the GUI falls back to fast canvas corners (see below).

Without the private font setup, the GUI draws rounded corners with plain canvas shapes (`circle_shapes`) instead of CustomTkinter's default font-based method, which is very slow on systems with large font collections. Override with `PD_GUI_DRAW=font_shapes|circle_shapes|polygon_shapes`. The toolbar and bottom bar keep the smooth font-drawn corners (they only hold a few buttons), and they repaint themselves after resizes and tab switches.

`PD_GUI_TIMING=1` also prints a profiler summary of the startup build and of the Generation tab build.

The GUI uses an installed font: Roboto if present, otherwise Noto Sans, DejaVu Sans or similar, because a missing theme font makes every CustomTkinter widget slow to create on Linux. To force a font: `PD_GUI_FONT="Noto Sans" python promptdirector_gui.py`. `python pd_gui_bench.py` measures font and widget creation speed on your machine.

## How scene fields work

The fields are style, number of characters, subject, action, camera angle, shot/distance, lighting, location, time/weather, composition and extra notes.

| Field state | Result |
|---|---|
| You typed a value | Used in **every** prompt, and the LLM is told it must not change it |
| Empty, its *random* box ticked, and the master **Random ON** switch on | A new random value for each prompt, taken from the presets |
| Empty, and random is off (for that field or overall) | Left out, so the LLM picks what fits the scene |

Lighting and composition are not random by default, so the LLM can match them to the scene (the file's "NO FIXED STYLE" rule).
If there are 3 or more characters, random shots never pick close-ups.
Free-text scene ideas (in the GUI box, or with `--scene` on the CLI) are combined with the fields you filled in; random fields are not added to them.

## Target image models

Choose one or more on the Generation tab, or with `--target` on the CLI:

| Target | Prose length | Notes |
|---|---|---|
| Krea 2 | 150–300 words | reads up to 512 tokens and handles long, detailed prompts |
| FLUX | 100–220 words | 512-token limit, so a bit shorter |
| Illustrious | 90–180 words, plus a tag line | SDXL-based, so a second pass writes Danbooru-style tags to `*_illustrious_tags.txt`, one line per prompt |

When several targets are ticked, the prose uses the range that suits all the prose models (Illustrious relies on its tag line, so it doesn't shorten the prose). Min/Max words override the range.

## Comic panel layout (for Krea 2)

Turn it on with the **Comic** tab checkbox, or `--comic 2x3` on the CLI. Each prompt then describes one comic page split into panels.

| Setting | Options |
|---|---|
| Grid (columns x rows) | 2x1, 1x2, 3x1, 1x3, 2x2, 2x3 (6 panels), 3x2, 3x3 |
| Panel mode | **vignettes**: separate moments with the same cast. **story**: consecutive beats |
| Prompt format | **prose**: `Panel 1 (Top Left): ...` paragraphs, for stock Krea 2 and mixes (FLUX handles 2-4 panels). **bbox**: grounded layout spec for the `krea2-turbo-bbox` finetune |
| Location per panel | auto (vignettes = different places, story = one place), same, varied |
| Speech bubbles | off / on (max 8 words each) |

- The cast (style, characters, lighting, weather) comes from the Scene tab and is described once. Camera angle and shot, plus location and action depending on the mode, are rolled separately for each panel, and the compatibility rules still apply.
- **Generate the image at the suggested size**, which is shown on the Comic tab, in the Results tab and in the JSONL, so the panels stay roughly square. For example, 2x3 uses 1248x1888.
- **BBOX format:** the script works out the panel boxes and character slots itself (0-1000 grid, x first), and checks the reply's boxes, panel count, character ids and length (max ~1800 characters, since Krea 2 reads 507 tokens). The `lines` output format is skipped for BBOX specs, because they need their line breaks.
- Krea 2 reads 512 tokens, so 6 panels get about 45 words each. Use fewer panels for more detail per panel.

## Workflow tab (edit the ComfyUI workflow)

This tab opens the workflow file set on the ComfyUI tab.

- **Prompt:** shows where the prompt goes (`%PROMPT%`, or the auto-detected box).
- **Models:** every model the workflow loads (checkpoint, diffusion model, text encoders, VAE), each changeable with **Choose...**.
- **LoRAs:** turn each on/off, change the file or strength, remove it, or **Add LoRA...**. Works with the rgthree Power Lora Loader and with LoraLoader / LoraLoaderModelOnly nodes. Removed nodes are unwired cleanly; added LoRAs go into the Power Lora Loader, or are wired in as a new node after the model loader.
- **Resolution:** the workflow's size, plus **App suggests** with the reasoning. For example, for a 2x3 comic page and a workflow that upscales ×1.5 in a second pass, it suggests starting at 832×1264 so the result ends near 1248×1888. **Use suggestion** applies it.
  - **Size used when sending:** `auto` (comic pages use the app's size × comic scale), `workflow` (keep the size above) or `custom` (the ComfyUI tab's Width/Height).
  - **Match upscale** sets the comic scale for you.
- **Sampling passes:** steps, CFG, sampler, scheduler and denoise for each pass, so you can trim an expensive second pass.

**Save** writes a copy named `..._pd.json` next to your export (the original is never overwritten), and the app switches to it. **Revert** goes back to the original. Unsaved edits are saved automatically when you press Generate.

## Managed llama-server (the app starts and stops llama.cpp)

Backend & Model tab, with Backend type set to `llamacpp`:

| Setting | Meaning |
|---|---|
| Let the app run llama-server | ON = the app starts it with the chosen GGUF and stops it when needed |
| llama.cpp folder | your custom build; `llama-server` is found in `build/bin`, `bin` or the folder itself (else on PATH) |
| GGUF folder / Model file | GGUFs are used in place (no importing); the drop-down lists the folder's models |
| GPU, GPU layers, Flash attention, Parallel slots, KV cache type, Extra arguments | become `CUDA_VISIBLE_DEVICES`, `-ngl`, `-fa`, `-np`, `-ctk/-ctv` and anything extra; Context size (n_ctx) becomes `-c` |

- The app waits until the model is loaded (`/health`) before generating, and keeps the server running between runs.
- Picking a different model or changing a flag restarts the server with the new settings.
- With ComfyUI, the server is **stopped automatically after the prompts are written**, so the image model gets the whole GPU.
- **Start server** / **Stop server** buttons are on the tab. The server stops when the app closes, unless you turn that off.
- Output goes to `logs/llama-server.log`. If startup fails (for example out of VRAM, or a flag an older build doesn't know), the error shows the server's last lines.
- CLI: `--manage-server --llama-cpp-dir ~/llama.cpp --gguf /path/model.gguf --gpu 0`, and `--list-ggufs FOLDER`.
- If the port is already in use (a llama-server you started yourself), stop that one or change the port in the Host URL.

## ComfyUI (render the prompts)

Use the **ComfyUI** tab, or `--comfy` / `--comfy-only` on the CLI.

| Setting | Meaning |
|---|---|
| Send prompts to ComfyUI | ON = every finished prompt is rendered |
| Also save prompt files | OFF = send only, no prompt files (logs are always kept) |
| ComfyUI URL | default `http://127.0.0.1:8188`; **Test** checks it |
| Models folder + Refresh model lists | lists come from ComfyUI's API and are cached in `comfy_models_cache.json`; the folder is used when ComfyUI is off |
| Workflow | **custom** (your own workflow: Krea 2, Moody mix, LoRAs), **builtin_sdxl** (Illustrious/SDXL), **builtin_flux** |
| Checkpoint / Diffusion model / Text encoder(s) / VAE | drop-downs from the cache; blank keeps the workflow's own choice |

**Custom workflow (recommended for Krea 2):**
1. Build or open your working workflow in ComfyUI.
2. Type `%PROMPT%` in the positive prompt box. `%NEGATIVE%` and `%TAGS%` are optional.
3. Use **Workflow > Export (API)** and select the exported file on the ComfyUI tab.

For each image, the script fills in the prompt, a seed, the size (the comic suggested size in comic mode), the batch size and the file prefix. If there's no `%PROMPT%` placeholder, it finds the positive prompt box automatically and logs which one it used.

**Picking models:** fields with long lists (ComfyUI models, the GGUF file, the Ollama/server model) have a **Choose...** button. It opens a window that stays open until you pick. Type to filter (words match in any order, e.g. `krea fp8`), use the arrow keys or mouse wheel, and press Enter or double-click to choose. Refresh, Browse... and "Use workflow's own" (clear) are in the same window. You can also type a value straight into the field.

**Sharing one GPU:** with *When to send = end* (the default), all prompts are written first. The LLM is then unloaded (automatic for Ollama and llama-cpp-python; **llama-server must be stopped by hand**), the images are rendered, and ComfyUI is asked to free its VRAM at the end. Use *each* only when the LLM and ComfyUI fit on the GPU(s) together.

Illustrious: tick the Illustrious target and use builtin_sdxl, and the tag line is sent (with negative prompt and CLIP skip 2). Every prompt is re-checked against the content rule before it is sent.

CLI: `--comfy-test`, `--comfy-models` (refresh and print the lists), `--comfy-workflow FILE`, `--comfy-builtin sdxl|flux`, `--comfy-send end|each`.

### Model rotation (several models in one run)

ComfyUI tab > **Model rotation**:

| Setting | Meaning |
|---|---|
| Rotate models | `off`, `random` (N different models from the cached list) or `list` (the models you add, in order) |
| Model type to rotate | `auto` picks from the workflow: the diffusion model for Krea 2/FLUX, the checkpoint for SDXL |
| Random: number of models | must be no more than the number of prompts |
| List | one model per line; **Add model...** opens the picker |
| Only compatible models | only uses models with the same architecture as the workflow's own model |
| Model search folders | where your model files live (separated by `;`), so they can be checked |

- **Grouping:** consecutive prompts share a model, so each model is loaded only once. 8 prompts with 4 models gives 2 each; 7 prompts with 3 models gives 3, 2, 2. The rotation counts per prompt: with Images per prompt above 1, a prompt's whole batch uses one model.
- **Compatibility check:** reads each model file's header (the tensor names; no weights are loaded) and compares its layout with the workflow's model. If a file can't be found, the model name is used as a hint (krea2, flux, sdxl/illustrious/pony, ...). Anything else is "unknown".
  - Random mode only picks compatible models.
  - List mode skips incompatible models with a warning.
- **Check models** shows the verdicts and the exact plan before you run. The plan is also written to the log at the start of each run.
- CLI: `--rotate-random N`, `--rotate-models A B C`, `--any-model`, `--model-dirs "/path/one;/path/two"`.

## Quality checks

- **Length:** replies far outside the word range are re-asked.
- **Filler check:** prompts that describe mood or meaning ("atmosphere", "poignant", "evoking", ...) instead of visible things are re-asked.
- **Compatibility rules:** random picks never contradict each other or your fixed fields. For example, no studio light outdoors, no close-up of a group, and no moonlight at midday.

## Content rule

With NSFW allowed, every character must be a clearly consenting adult. The tool refuses scene details that break this rule, and replies (prose or tags) that break it are discarded and never saved. This check can't be switched off.

### Characters (per-person details)

Below the scene fields, the **Characters** table shows one row per person. The number of rows follows **Number of characters** (1-6).

| Column | Notes |
|---|---|
| Role | e.g. "a knight". Filled-in roles build the Subject line |
| Age | adults only: 18+ (e.g. `30` or `early 30s`); random ages are adult ranges |
| Ethnicity | e.g. East Asian, Black, Hispanic/Latino; random is off by default (the LLM chooses) |
| Hair color / Hair length / style | random by default |
| Clothing | random is off by default, so the LLM matches the scene |

Scene fields and character boxes have a **...** button that opens the searchable picker for that preset list (it stays open until you choose; type to filter). You can also type straight into the box. Each column has a **random** checkbox. A filled-in box is used for every prompt; an empty box is randomized (if ticked and Random is ON) or left to the LLM. Each person is sent as a line like `Character 1: a knight; age 30; black hair (long straight); wearing plate armor`, and the system prompt tells the LLM to use each line for that character. Comic mode uses the same cast in every panel.

CLI: `--person "role=a knight;age=30;hair_color=black;clothing=plate armor"`, repeated once per person.

## GUI tabs

- **Comic**: comic panel layout on/off, grid, mode, format, suggested image size
- **Scene**: fields, per-field *random* boxes, 🎲 (roll a value now), ✕ (clear), master Random switch, free-text ideas
- **Results**: prompts appear as they are made; Copy all; Open output folder
- **Log**: live log, with warnings and errors colored
- **Backend & Model**: backend type, host, API key, model (Refresh list), .gguf path, sampling settings, Test connection
- **Generation**: seed mode, SFW, random seed, system prompt and presets files, quality checks
- **Output & Logging**: output folder, formats, log folder, debug log on/off
- **Presets**: edit any random list (repeat a line to make it more likely)
- **System Prompt**: edit the prompt; the text shown is what the next run uses

The bottom bar (count, Generate, Stop, Preview scenes) is on every tab. **Preview scenes** shows the scene details without calling the LLM.
**Save** writes everything, including your scene fields, to `config.toml`. Closing the window offers to save any changes.

## CLI

```bash
python promptdirector.py -n 20
python promptdirector.py -n 5 --style "anime illustration" --characters 2 --camera-angle "low angle"
python promptdirector.py -n 5 --subject "a knight and a sorceress" --no-random
python promptdirector.py -n 5 --random-fields location time_weather   # only these may be random
python promptdirector.py -n 10 --target krea2 illustrious             # Krea 2 length + Illustrious tags
python promptdirector.py -n 5 --comic 2x3 --comic-mode story         # 6-panel comic pages
python promptdirector.py -n 5 --comic 2x2 --comic-format bbox        # krea2-turbo-bbox layout spec
python promptdirector.py --dry-run -n 10                              # preview scenes
python promptdirector.py --check | --list-models | --show-config | --help
```

Settings are applied in this order, with later ones winning: `DEFAULT_CONFIG` → `config.toml` (or `--config`, or the `PD_CONFIG` environment variable) → CLI flags.

## Output and logs

Each run is saved to `<output>/prompts_YYYYmmdd_HHMMSS/` in any of these formats:
- `txt`: all prompts in one file
- `lines`: one prompt per line
- `jsonl`: each prompt plus its scene details, and which details were yours and which were random
- `separate`: one file per prompt
- `tags`: Illustrious tag lines (added automatically when Illustrious is a target)

Prompts are written as soon as each one is made, so stopping a run never loses work.
Logs go to `<output>/logs/`. The main log is always written. The debug log is a separate file with full requests and responses, turned on with `--debug` or the GUI switch.
