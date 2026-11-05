"""Chair Studio worker: --role shape generates meshes, --role texture paints them (several can run in parallel).
Runs inside /content/h3d-env with cwd=/content/Hunyuan3D-2.1."""
import sys, os, gc, time, json, argparse, traceback, glob
sys.path.insert(0, os.getcwd())   # run with cwd = the Hunyuan3D-2.1 repo
sys.path.insert(0, "./hy3dshape")
sys.path.insert(0, "./hy3dpaint")
try:
    from torchvision_fix import apply_fix
    apply_fix()
except Exception as e:
    print("torchvision fix skipped:", e)

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
from PIL import Image, ImageOps

p = argparse.ArgumentParser()
p.add_argument("--role", choices=["shape", "texture"], required=True)
p.add_argument("--worker", type=int, default=0)
p.add_argument("--queue", required=True)
p.add_argument("--images", nargs="*", default=[])
p.add_argument("--meshes_dir", default="")
p.add_argument("--out", default="/content/outputs")
p.add_argument("--steps", type=int, default=50)
p.add_argument("--guidance", type=float, default=5.0)
p.add_argument("--octree", type=int, default=512)
p.add_argument("--chunks", type=int, default=200000)
p.add_argument("--seeds", type=int, nargs="+", default=[1234])
p.add_argument("--max_faces", type=int, default=40000)
p.add_argument("--views", type=int, default=8)
p.add_argument("--view_res", type=int, default=768)
p.add_argument("--flashvdm", action="store_true")
p.add_argument("--no_texture", action="store_true")
p.add_argument("--force_rembg", action="store_true")
a = p.parse_args()
os.makedirs(a.queue, exist_ok=True)
DONE_FLAG = os.path.join(a.queue, "SHAPE_DONE")

def enqueue(job):
    tmp = os.path.join(a.queue, job["tag"] + ".tmp")
    with open(tmp, "w") as f:
        json.dump(job, f)
    os.replace(tmp, os.path.join(a.queue, job["tag"] + ".job"))   # atomic: workers never see half a file

def shape_role():
    import trimesh
    from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
    from hy3dshape.pipelines import export_to_trimesh
    from hy3dshape.rembg import BackgroundRemover
    MESH_EXT = (".obj", ".glb", ".gltf", ".ply", ".stl")
    rmbg = BackgroundRemover()
    pending = []
    for path in a.images:
        stem = os.path.splitext(os.path.basename(path))[0]
        folder = os.path.join(a.out, stem)
        os.makedirs(folder, exist_ok=True)
        try:
            img = ImageOps.exif_transpose(Image.open(path))
            opaque = img.mode != "RGBA" or img.getextrema()[3][0] == 255
            if a.force_rembg or opaque:
                img = rmbg(img.convert("RGB"))
            clean = os.path.join(folder, "input_clean.png")
            img.save(clean)
        except Exception:
            traceback.print_exc(); print(f"FAILED {stem} (photo)"); continue
        own = next((os.path.join(a.meshes_dir, stem + e) for e in MESH_EXT
                    if a.meshes_dir and os.path.exists(os.path.join(a.meshes_dir, stem + e))), None)
        if own:
            mesh = trimesh.load(own, force="mesh")
            if len(mesh.faces) > a.max_faces:
                mesh = FaceReducer()(mesh, max_facenum=a.max_faces)
            obj = os.path.join(folder, f"{stem}_own.obj")
            mesh.export(obj)
            enqueue(dict(tag=f"{stem}_own", folder=folder, image=clean, obj=obj, remesh=False))
            print(f"{stem}: queued your model {os.path.basename(own)} for texturing")
        else:
            pending.append((stem, folder, img))
    if not pending:
        return

    def load(flash):
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2.1", subfolder="hunyuan3d-dit-v2-1", use_safetensors=False, device="cuda")
        if flash:
            pipe.enable_flashvdm(mc_algo="mc")
        return pipe

    flash = a.flashvdm
    try:
        pipe = load(flash)
    except Exception:
        traceback.print_exc(); print("FlashVDM unavailable, using the standard decoder")
        flash = False; pipe = load(False)
    print(f"shape model ready (FlashVDM {'on' if flash else 'off'})")

    for stem, folder, img in pending:
        for seed in a.seeds:
            tag = stem if len(a.seeds) == 1 else f"{stem}_seed{seed}"
            t = time.time()
            for attempt in range(2):
                try:
                    out = pipe(image=img, num_inference_steps=a.steps, guidance_scale=a.guidance,
                               generator=torch.Generator().manual_seed(seed),
                               octree_resolution=a.octree, num_chunks=a.chunks, output_type="mesh")
                    mesh = export_to_trimesh(out)[0]
                    if mesh is None:
                        raise RuntimeError("empty mesh")
                    break
                except Exception:
                    traceback.print_exc()
                    mesh = None
                    if flash and attempt == 0:
                        print("retrying with the standard decoder")
                        del pipe; gc.collect(); torch.cuda.empty_cache()
                        flash = False; pipe = load(False)
                    else:
                        break
            if mesh is None:
                print(f"FAILED {tag} (shape)"); continue
            mesh = DegenerateFaceRemover()(FloaterRemover()(mesh))
            mesh.export(os.path.join(folder, f"{tag}_shape_highres.glb"))
            msg = f"{tag}: shape {len(mesh.faces):,} faces in {time.time() - t:.0f}s"
            if a.no_texture:
                print("DONE " + msg); continue
            reduced = FaceReducer()(mesh, max_facenum=a.max_faces)
            obj = os.path.join(folder, f"{tag}_shape.obj")
            reduced.export(obj)
            enqueue(dict(tag=tag, folder=folder, image=os.path.join(folder, "input_clean.png"), obj=obj, remesh=True))
            print(msg + " → queued for texturing")
            torch.cuda.empty_cache()

def texture_role():
    from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
    from hy3dpaint.convert_utils import create_glb_with_pbr_materials
    conf = Hunyuan3DPaintConfig(max_num_view=a.views, resolution=a.view_res)
    conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    paint = Hunyuan3DPaintPipeline(conf)
    print("texture model ready")
    while True:
        jobs = sorted(glob.glob(os.path.join(a.queue, "*.job")))
        if not jobs:
            if os.path.exists(DONE_FLAG) and not glob.glob(os.path.join(a.queue, "*.job")):
                return
            time.sleep(2); continue
        claimed = jobs[0] + f".w{a.worker}"
        try:
            os.rename(jobs[0], claimed)          # atomic claim; another worker may win the race
        except FileNotFoundError:
            continue
        job = json.load(open(claimed))
        tag, folder = job["tag"], job["folder"]
        t = time.time()
        try:
            # each job gets its own work dir so parallel workers never overwrite each other's temp meshes
            work = os.path.join(folder, f"_work_{tag}")
            os.makedirs(work, exist_ok=True)
            obj_in = os.path.join(work, os.path.basename(job["obj"]))
            os.replace(job["obj"], obj_in)
            out_obj = paint(mesh_path=obj_in, image_path=Image.open(job["image"]),
                            output_mesh_path=os.path.join(folder, f"{tag}_textured.obj"),
                            use_remesh=job["remesh"], save_glb=False)
            textures = {"albedo": out_obj.replace(".obj", ".jpg"),
                        "metallic": out_obj.replace(".obj", "_metallic.jpg"),
                        "roughness": out_obj.replace(".obj", "_roughness.jpg")}
            create_glb_with_pbr_materials(out_obj, textures, os.path.join(folder, f"{tag}_textured.glb"))
            print(f"DONE {tag}: textured in {time.time() - t:.0f}s")
        except Exception:
            traceback.print_exc(); print(f"FAILED {tag} (texture)")
        os.remove(claimed)
        torch.cuda.empty_cache()

if a.role == "shape":
    try:
        shape_role()
    finally:
        open(DONE_FLAG, "w").close()
else:
    texture_role()
