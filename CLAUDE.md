# CLAUDE.md: Hunyuan3D Chair Studio on Google Colab

Handoff for any Claude Code session working on this repo. Read this before changing anything.

## What this is

Omid (GitHub `Solamente0`) runs **Tencent Hunyuan3D 2.1** on **Google Colab Pro** to turn **photos of his chairs into realistic, PBR-textured 3D models** (`.glb` + `.obj` + 4K albedo/metallic/roughness maps).

His goals, in priority order:
1. **Best possible quality**, especially realistic textures and thin chair parts (legs, rungs, slats)
2. **As fast as possible at that quality**
3. **Simple to use**: a Gradio app (Chair Studio) to upload photos, generate, preview in 3D and download
4. **GitHub-driven**: the notebook is a thin launcher that `git pull`s this repo, so pushing to `main` is how changes ship

Repo: https://github.com/Solamente0/hunyuan3d-google-colab (public, branch `main`)
Open in Colab: https://colab.research.google.com/github/Solamente0/hunyuan3d-google-colab/blob/main/Hunyuan3D_Chairs.ipynb

## Target machine

Colab **A100 80 GB** with High-RAM (167 GB system RAM, ~235 GB disk). The code adapts to smaller GPUs, but 80 GB is what Omid uses.

## Status: important

- **Never run on a real GPU yet.** Everything was built from the upstream README, `requirements.txt`, `gradio_app.py` and `textureGenPipeline.py`, and verified only by syntax checks and the GPU-free stub test (`tests/stub_test.py`).
- The first real Colab run is the moment of truth. Expect possible fixes in `scripts/install.sh` (see "Known risks"). When Omid pastes an error, fix it here, push, and tell him which cell to rerun.

## Repo layout

| File | Runs where | Purpose |
|---|---|---|
| `Hunyuan3D_Chairs.ipynb` | Colab kernel (Python 3.12) | Thin launcher. Cell 1 clones/pulls this repo into `/content/hunyuan3d-google-colab` and runs `setup_env.py`; cell 2 runs `install.sh`; cell 3 runs `chair_studio.py`; cell 4 runs `official_ui.py`; then logs and stop |
| `setup_env.py` | Colab kernel via `%run -i` | GPU check, sets `LOW_VRAM`, `GPU_VRAM_GB`, `TORCH_CUDA_ARCH_LIST`, `HF_HOME` (optionally on Drive), `HF_HUB_ENABLE_HF_TRANSFER`, `HF_TOKEN` from Colab secrets |
| `scripts/install.sh` | bash | Clones `Tencent-Hunyuan/Hunyuan3D-2.1` to `/content/Hunyuan3D-2.1`, builds a **Python 3.10** venv at `/content/h3d-env` with `uv`, installs torch 2.5.1+cu124 and the requirements, compiles CUDA/C++ extensions, gets RealESRGAN, pre-downloads all weights |
| `chair_studio.py` | Colab kernel via `%run -i` | **Gradio 5 app** (`gradio>=5.30,<6`, installed into the Colab kernel). Upload, settings, launches workers, live log, `Model3D` preview, zip download, `share=True` link |
| `scripts/run_chairs.py` | `/content/h3d-env` Python, cwd=`/content/Hunyuan3D-2.1` | The worker. `--role shape` or `--role texture`; several run in parallel |
| `official_ui.py` | Colab kernel via `%run -i` | Optional: Tencent's own `gradio_app.py`, with slider defaults regex-patched to best quality, exposed via the Colab proxy and an optional cloudflared link |
| `tests/stub_test.py` | any Python 3.10+ with Pillow + gradio 5 | GPU-free end-to-end test with stubbed models |

### Why two Pythons
Colab ships Python 3.12/3.13, but upstream pins packages that only have 3.10-era wheels (`numpy==1.24.4`, `open3d==0.18.0`, `pymeshlab==2022.2.post3`). All model code therefore runs in `/content/h3d-env` (Python 3.10). The Gradio UI runs in the Colab kernel and only spawns subprocesses.

### `%run -i` contract
The notebook passes settings as kernel globals, and scripts read them with `globals().get(...)`: `TOOLS_DIR`, `CACHE_WEIGHTS_IN_DRIVE`, `GPU_VRAM_GB`, `LOW_VRAM`, `PUBLIC_LINK`, `FORCE_LOW_VRAM_MODE`. Keep these names stable, or update the notebook in the same commit.

## Pipeline architecture (Chair Studio)

```
Gradio generate()  ──spawns──►  run_chairs.py --role shape      (1 process, ~10 GB VRAM)
                   ──spawns──►  run_chairs.py --role texture ×N (N=2 on ≥70 GB, 1 on 40–70 GB; ~21 GB each)
```
- **File queue** in `<out>/<run_id>/.queue/`. The shape worker writes `<tag>.job` (JSON, written as `.tmp` then `os.replace`, so the write is atomic). Texture workers claim a job by `os.rename(job, job + ".w<id>")`; losing the race raises `FileNotFoundError`, and the worker moves on.
- **`SHAPE_DONE` flag**: written by the shape worker in a `finally` and **also** by `generate()` once the shape process exits (covers a SIGKILL/OOM). Texture workers exit when the flag exists and no `.job` files remain.
- **On GPUs under 40 GB**: no overlap. Shape runs first, then 1 texture worker, with views capped at 6 × 512.
- Each texture job runs in its own `_work_<tag>/` dir, because the paint pipeline writes temp files (e.g. `white_mesh_remesh.obj`) next to the input mesh, and parallel workers would otherwise clash. Work dirs and `.queue` are removed at the end.
- **Outputs**: `/content/outputs/<run_id>/<photo_stem>/` holds `input_clean.png` (after background removal), `<tag>_shape_highres.glb`, `<tag>_textured.{obj,glb}` and the `.jpg` maps. Each run is zipped to `/content/outputs/chairs_<run_id>.zip`, and copied to `MyDrive/` if Drive is mounted.
- **Own meshes**: if an uploaded `.obj/.glb/.gltf/.ply/.stl` shares a photo's stem, shape generation is skipped and the mesh is textured with `use_remesh=False` (keeps the user's geometry; reduced to 40k faces only if larger).
- **Background removal**: only for images without real transparency (a fully opaque alpha counts as no transparency), unless "Always remove background" is ticked.

## Quality settings and why

| Preset | Shape steps | Octree (mesh grid) | Texture views | View res |
|---|---|---|---|---|
| **Best** (default) | 50 | 512 (max) | 8 | 768 |
| Balanced | 30 | 384 | 8 | 768 |
| Fast | 30 | 256 | 6 | 512 |

- Guidance 5.0 (upstream UI default). Seeds `[1234, 42, 2025, 7][:n]`.
- Octree 512 because thin chair parts get lost or merged at lower grids.
- `num_chunks` is 200000 on ≥40 GB (speed only, no quality effect) and 20000 below.
- **FlashVDM** (`pipe.enable_flashvdm(mc_algo="mc")`) is on by default for much faster mesh extraction at the same grid. If enabling it or the first decode fails, the worker reloads without it and retries.
- Texture: 8 views at 768 px is the highest preset upstream ships (`gradio_app.py`). Paint config defaults already give `render_size` 2048 and `texture_size` 4096. Tested-quality limits were deliberately **not** exceeded (no >8 views, no >768 px); don't change that without evidence.
- `FaceReducer(max_facenum=40000)` before texturing, matching upstream.
- TF32 and `cudnn.benchmark` are enabled in the workers.

## Upstream facts relied on (Hunyuan3D-2.1)
- VRAM: shape ~10 GB, texture ~21 GB, both ~29 GB. Tested on A100 with Python 3.10 and torch 2.5.1+cu124.
- HF repo `tencent/Hunyuan3D-2.1` (~15 GB): `hunyuan3d-dit-v2-1/` (model.fp16.ckpt), `hunyuan3d-vae-v2-1/`, `hunyuan3d-paintpbr-v2-1/`. The paint pipeline also uses `facebook/dinov2-giant` and `hy3dpaint/ckpt/RealESRGAN_x4plus.pth`.
- Shape API: `Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(path, subfolder=..., use_safetensors=False, device="cuda")`, called with `output_type="mesh"`, then `export_to_trimesh(out)[0]`.
- Paint API: `Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view, resolution))` with `realesrgan_ckpt_path`, `multiview_cfg_path="hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"`, `custom_pipeline="hy3dpaint/hunyuanpaintpbr"`. It's called as `paint(mesh_path, image_path=PIL, output_mesh_path, use_remesh, save_glb=False)`, which returns the OBJ path. The GLB comes from `hy3dpaint.convert_utils.create_glb_with_pbr_materials`.
- `torchvision_fix.apply_fix()` must run before basicsr/RealESRGAN imports.
- `gradio_app.py` runs FastAPI + uvicorn (no `share`), with args `--host --port --low_vram_mode --enable_flashvdm --mc_algo`.

## Install decisions (`scripts/install.sh`)
- **uv only, always with `--python /content/h3d-env/bin/python`.** Colab presets `UV_SYSTEM_PYTHON` (and similar `UV_*` vars), which make a bare `uv pip install` ignore `VIRTUAL_ENV` and install into the system Python (3.13 on newer runtimes), where torch 2.5.1 has no wheels. `install.sh` unsets those vars, creates the env from a uv-managed Python 3.10 (`UV_PYTHON_PREFERENCE=only-managed`), rebuilds it if an earlier run left a non-3.10 env, and pins torch/numpy as constraints for the requirements step. `chair_studio.py` installs gradio with `uv pip install --python sys.executable` (falls back to pip only if uv is missing).
- Strips `--extra-index-url` lines (Tencent/Aliyun mirrors are slow from Colab, and uv would prefer them), `deepspeed` (training only) and `bpy` (`bpy==4.0` was removed from PyPI; it is only imported by `hy3dshape/tools/render/render.py`, the training-data renderer, never by inference). If it's ever needed, Blender hosts `bpy-4.0.0-cp310` at https://download.blender.org/pypi/bpy/.
- `basicsr` is built with `--no-build-isolation-package basicsr` because its setup imports torch.
- `custom_rasterizer`: `uv pip install --no-build-isolation .` with `CUDA_HOME` set to a CUDA **12.x** toolkit. If the system nvcc isn't 12.x, it uses `/usr/local/cuda-12*` or installs `cuda-toolkit-12-4` from NVIDIA's apt repo. `TORCH_CUDA_ARCH_LIST` is set to the GPU's compute capability.
- The mesh painter is compiled with the venv's python (`pybind11 --includes`, `sysconfig EXT_SUFFIX`), because upstream's `compile_mesh_painter.sh` calls `python3-config`, which uv venvs lack.
- Blackwell GPUs (compute capability ≥10) are rejected in `setup_env.py`, since torch 2.5.1/cu124 doesn't support them.

## Known risks: check these first when something breaks
1. `uv pip install -r requirements` on Python 3.10. Without `bpy`, the full set was dry-resolved for 3.10/manylinux_2_28 and succeeds; the remaining risks are sdist builds (`basicsr`).
2. CUDA extension build: nvcc/torch CUDA mismatch, missing headers, arch list.
3. X/GL libs for opencv/open3d; `install.sh` apt-installs `libxrender1 libxi6 libxkbcommon0 libsm6 libgl1 libegl1 libglib2.0-0`.
4. FlashVDM on the 2.1 VAE. The fallback exists, but confirm the speedup on a real run.
5. `hf_xet`/`hf_transfer` with the pinned `huggingface-hub==0.30.2`.
6. The model repo may need an HF token and accepted terms (README metadata has `extra_gated_eu_disallowed`). `setup_env.py` reads `HF_TOKEN` from Colab secrets.
7. `official_ui.py` regex patches may silently stop matching if upstream edits `gradio_app.py`; it prints ✔/⚠ for each patch.
8. Gradio 5 file access: `launch(allowed_paths=[OUT_ROOT])` is required for `Model3D` and the zip download.

## How to work on this repo
- **Always merge into `main`** (Omid's standing instruction). Whatever branch a session is told to work on, every finished change also lands on `main` (fast-forward, or merge if needed) and is pushed, without asking first. No pull requests unless he asks for one.
- **Workflow**: edit, run `python tests/stub_test.py` (it must pass), commit, push to `main`. Then tell Omid which cell to rerun:
  - `chair_studio.py`, `scripts/run_chairs.py`, `official_ui.py` → rerun **cell 3** (or 4), which pulls first
  - `setup_env.py` → rerun cell 1
  - `scripts/install.sh` → rerun cell 1, then cell 2 (idempotent; finished steps are skipped)
  - the `.ipynb` → he must reopen the Colab link
- Keep the notebook thin; put logic in the scripts.
- If you change `chair_studio.py`'s `REPO, PY, OUT_ROOT = ...` line or the pip-install line, update the string replacements in `tests/stub_test.py`.
- The stub test can't catch real model or API errors; only a Colab run can. Ask Omid for logs (cell 5 shows `/content/hunyuan3d.log` for the official UI; Chair Studio streams logs in its Progress box).
- Omid writes in English and Persian; reply in the language he uses.

## Ideas not yet built
- Filter the preview dropdown to show only textured models when they exist (it currently lists both shape-only and textured `.glb` files).
- Optional upscaled texture export (`render_size`/`texture_size` tuning), once real outputs are inspected.
- Text-to-3D (`--enable_t23d`) is unused; Omid works from photos.

## License
Hunyuan3D 2.1 is under the **Tencent Hunyuan Community License**, which has commercial-use terms and territory exclusions. Point this out before any commercial use (Omid runs a startup, Fidar).
