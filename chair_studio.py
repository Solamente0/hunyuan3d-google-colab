"""Chair Studio: a Gradio app to upload chair photos, generate textured 3D models, preview and download them.

Run from the notebook with `%run -i` so it can read the notebook's settings.
Part of https://github.com/Solamente0/hunyuan3d-google-colab
"""
import subprocess, sys, os
import shutil
_uv = shutil.which("uv") or shutil.which(os.path.expanduser("~/.local/bin/uv"))
subprocess.run([_uv, "pip", "install", "-q", "--python", sys.executable, "gradio>=5.30,<6"] if _uv
               else [sys.executable, "-m", "pip", "install", "-q", "gradio>=5.30,<6"], check=True)

import os, shutil, glob, threading, time, queue
import gradio as gr

REPO, PY, OUT_ROOT = "/content/Hunyuan3D-2.1", "/content/h3d-env/bin/python", "/content/outputs"
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")
MESH_EXT = (".obj", ".glb", ".gltf", ".ply", ".stl")
os.makedirs(OUT_ROOT, exist_ok=True)

PRESETS = {
    #            shape steps, mesh grid (thin legs/slats need a fine grid), texture views, view resolution
    "Best":     dict(steps=50, octree=512, views=8, view_res=768),
    "Balanced": dict(steps=30, octree=384, views=8, view_res=768),
    "Fast":     dict(steps=30, octree=256, views=6, view_res=512),
}
vram = globals().get("GPU_VRAM_GB", 0)
if not vram:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    vram = int(out.split()[0]) / 1024
# Shape model ~10 GB, each texture worker ~21 GB: on big GPUs they all run at the same time.
if vram >= 70:
    TEX_WORKERS, OVERLAP = 2, True
elif vram >= 40:
    TEX_WORKERS, OVERLAP = 1, True
else:
    TEX_WORKERS, OVERLAP = 1, False
CHUNKS = 200000 if vram >= 40 else 20000
GPU_NOTE = (f"GPU {vram:.0f} GB → {TEX_WORKERS} texture worker(s)"
            + (", running alongside shape generation" if OVERLAP else ", stages run one after another")
            + ". 4K PBR maps (albedo, metallic, roughness).")
print(GPU_NOTE)

TOOLS_DIR = globals().get("TOOLS_DIR", "/content/hunyuan3d-google-colab")
WORKER = os.path.join(TOOLS_DIR, "scripts", "run_chairs.py")


def list_results():
    found = sorted(glob.glob(f"{OUT_ROOT}/*/*/*_textured.glb"), key=os.path.getmtime, reverse=True)
    found += sorted(glob.glob(f"{OUT_ROOT}/*/*/*_shape_highres.glb"), key=os.path.getmtime, reverse=True)
    return [(os.path.relpath(p, OUT_ROOT), p) for p in found]


def generate(files_in, quality, n_seeds, fast_decode, force_rembg, texture):
    if not files_in:
        raise gr.Error("Upload at least one chair photo first.")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    in_dir, out_dir = f"/content/chair_inputs/{run_id}", f"{OUT_ROOT}/{run_id}"
    os.makedirs(in_dir)
    for f in files_in:
        shutil.copy(f, os.path.join(in_dir, os.path.basename(f)))
    images = sorted(p for p in glob.glob(f"{in_dir}/*") if p.lower().endswith(IMG_EXT))
    meshes = sorted(p for p in glob.glob(f"{in_dir}/*") if p.lower().endswith(MESH_EXT))
    if not images:
        raise gr.Error("No photos found. Upload PNG, JPG or WEBP images (3D models only come with a photo).")

    cfg = dict(PRESETS[quality])
    if not OVERLAP:
        cfg.update(views=min(cfg["views"], 6), view_res=512)
    seeds = [1234, 42, 2025, 7][:int(n_seeds)]

    subprocess.run(["pkill", "-f", "gradio_app.py"])   # free the GPU from Tencent's UI if it's running
    subprocess.run(["pkill", "-f", "run_chairs.py"])
    qdir = os.path.join(out_dir, ".queue")
    os.makedirs(qdir)

    base = [PY, "-u", WORKER, "--queue", qdir, "--out", out_dir,
            "--views", str(cfg["views"]), "--view_res", str(cfg["view_res"])]
    shape_cmd = base + ["--role", "shape", "--images", *images, "--meshes_dir", in_dir,
                        "--steps", str(cfg["steps"]), "--octree", str(cfg["octree"]),
                        "--chunks", str(CHUNKS), "--seeds", *map(str, seeds)]
    if fast_decode:
        shape_cmd.append("--flashvdm")
    if not texture:
        shape_cmd.append("--no_texture")
    if force_rembg:
        shape_cmd.append("--force_rembg")

    lines = queue.Queue()
    def launch(cmd, label):
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=dict(os.environ))
        def pump():
            for line in proc.stdout:
                lines.put(f"[{label}] {line.rstrip()}")
        threading.Thread(target=pump, daemon=True).start()
        return proc

    log = [f"{len(images)} photo(s), {len(meshes)} 3D model(s) | {quality}: {cfg} | seeds {seeds}",
           GPU_NOTE, "First run loads the models (about a minute)...", ""]
    start = time.time()
    shape_proc = launch(shape_cmd, "shape")
    workers = []
    def start_workers():
        for w in range(TEX_WORKERS):
            workers.append(launch(base + ["--role", "texture", "--worker", str(w)], f"texture{w + 1}"))
    if texture and OVERLAP:
        start_workers()

    shape_finished = False
    yield "\n".join(log), gr.update(), gr.update(), gr.update()
    while True:
        try:
            log.append(lines.get(timeout=1))
            while not lines.empty():
                log.append(lines.get_nowait())
        except queue.Empty:
            pass
        if not shape_finished and shape_proc.poll() is not None:
            shape_finished = True
            open(os.path.join(qdir, "SHAPE_DONE"), "w").close()   # lets texture workers finish
            if texture and not OVERLAP:
                start_workers()
        alive = (not shape_finished) or any(p.poll() is None for p in workers)
        elapsed = f"⏱ {(time.time() - start) / 60:.1f} min"
        yield "\n".join(log[-400:] + ["", elapsed]), gr.update(), gr.update(), gr.update()
        if not alive and lines.empty():
            break

    for d in glob.glob(f"{out_dir}/*/_work_*"):
        shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(qdir, ignore_errors=True)
    finals = sorted(glob.glob(f"{out_dir}/*/*_textured.glb" if texture else f"{out_dir}/*/*_shape_highres.glb"))
    done = sum("DONE " in l for l in log)
    failed = [l for l in log if "FAILED " in l]
    log += ["", f"✅ Finished in {(time.time() - start) / 60:.1f} min: {len(finals)} model(s)"
            + (f", {len(failed)} failed (see lines marked FAILED)" if failed else "")]
    if not finals:
        yield "\n".join(log[-400:]), gr.update(), gr.update(), gr.update()
        raise gr.Error("No models were produced; check the progress log.")
    zip_path = shutil.make_archive(f"{OUT_ROOT}/chairs_{run_id}", "zip", out_dir)
    if os.path.isdir("/content/drive/MyDrive"):
        shutil.copy(zip_path, "/content/drive/MyDrive/")
        log.append(f"📁 Also copied to Google Drive: MyDrive/{os.path.basename(zip_path)}")
    yield ("\n".join(log[-400:]), gr.update(choices=list_results(), value=finals[0]),
           finals[0], zip_path)


TIPS = """**Best photos:** one chair per photo, 3/4 front view, soft even light, sharp and high resolution.
A PNG with the background already removed works best; chairs with open backs or thin legs can confuse automatic removal.
Each result folder includes `input_clean.png`, which shows exactly what the model saw."""

with gr.Blocks(title="Chair Studio") as demo:
    gr.Markdown("# 🪑 Chair Studio: photos → realistic textured 3D (Hunyuan3D 2.1)")
    with gr.Row():
        with gr.Column(scale=1):
            files_in = gr.File(label="Chair photos (+ optional 3D models with the same names)",
                               file_count="multiple", type="filepath",
                               file_types=list(IMG_EXT + MESH_EXT))
            quality = gr.Radio(["Best", "Balanced", "Fast"], value="Best", label="Quality",
                               info="Best: 50 steps, 512 mesh grid, 8 texture views at 768 px")
            n_seeds = gr.Slider(1, 4, value=1, step=1, label="Shape variants per chair",
                                info="More than 1 lets you pick the best shape; each variant is fully textured")
            fast_decode = gr.Checkbox(value=True, label="Fast mesh decoding (FlashVDM)",
                                      info="Same detail, much faster; falls back automatically if it fails")
            force_rembg = gr.Checkbox(value=False, label="Always remove background",
                                      info="Off: transparent PNGs are used as they are")
            texture = gr.Checkbox(value=True, label="Generate PBR textures")
            go = gr.Button("Generate 3D chairs", variant="primary", size="lg")
            gr.Markdown(GPU_NOTE + "\n\n" + TIPS)
        with gr.Column(scale=2):
            viewer = gr.Model3D(label="3D preview (drag to rotate)", height=540)
            pick = gr.Dropdown(label="Choose a model to preview", choices=list_results(), interactive=True)
            zip_out = gr.File(label="Download this run (zip with .glb, .obj and 4K texture maps)")
            log_box = gr.Textbox(label="Progress", lines=14, max_lines=14, autoscroll=True)
    go.click(generate, [files_in, quality, n_seeds, fast_decode, force_rembg, texture],
             [log_box, pick, viewer, zip_out])
    pick.change(lambda p: p, pick, viewer)

demo.queue(default_concurrency_limit=1).launch(share=True, allowed_paths=[OUT_ROOT], height=1100)
