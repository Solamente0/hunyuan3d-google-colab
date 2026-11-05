"""GPU-free end-to-end test of Chair Studio and the parallel worker queue.

Replaces torch, hy3dshape, hy3dpaint and trimesh with tiny stubs, then drives
chair_studio.generate() exactly like a button click: upload copy, shape worker,
two parallel texture workers, FlashVDM failure + fallback, zip, preview list.

Run from the repo root (needs only Python 3.10+, Pillow and gradio>=5.30,<6):
    python tests/stub_test.py
"""
import os, re, sys, tempfile, textwrap, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tmp = tempfile.mkdtemp(prefix="chair_stub_")
fake = os.path.join(tmp, "fake_repo")

STUBS = {
    "torch/__init__.py": """
        import types
        backends = types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace()),
                                         cudnn=types.SimpleNamespace())
        cuda = types.SimpleNamespace(empty_cache=lambda: None)
        class Generator:
            def manual_seed(self, s): return self
    """,
    "hy3dshape/__init__.py": """
        import time
        class M:
            def __init__(s): s.faces = list(range(90000))
            def export(s, p): open(p, "w").write("mesh")
        class Hunyuan3DDiTFlowMatchingPipeline:
            calls = 0
            @classmethod
            def from_pretrained(cls, *a, **k): return cls()
            def enable_flashvdm(self, mc_algo): self.flash = True
            def __call__(self, **k):
                Hunyuan3DDiTFlowMatchingPipeline.calls += 1
                if getattr(self, "flash", False) and Hunyuan3DDiTFlowMatchingPipeline.calls == 1:
                    raise RuntimeError("simulated flashvdm failure")
                time.sleep(0.3); return [M()]
        class FaceReducer:
            def __call__(self, m, max_facenum=40000): m.faces = m.faces[:max_facenum]; return m
        class FloaterRemover:
            def __call__(self, m): return m
        class DegenerateFaceRemover:
            def __call__(self, m): return m
    """,
    "hy3dshape/pipelines.py": "def export_to_trimesh(o): return o\n",
    "hy3dshape/rembg.py": """
        class BackgroundRemover:
            def __call__(self, img): return img.convert("RGBA")
    """,
    "hy3dpaint/__init__.py": "",
    "hy3dpaint/textureGenPipeline.py": """
        import os, time
        class Hunyuan3DPaintConfig:
            def __init__(self, max_num_view, resolution): pass
        class Hunyuan3DPaintPipeline:
            def __init__(self, conf): pass
            def __call__(self, mesh_path, image_path, output_mesh_path, use_remesh, save_glb):
                assert os.path.exists(mesh_path), mesh_path
                time.sleep(1.0)
                for suf in ["", "_metallic", "_roughness"]:
                    open(output_mesh_path.replace(".obj", suf + ".jpg"), "w").write("x")
                open(output_mesh_path, "w").write("obj")
                return output_mesh_path
    """,
    "hy3dpaint/convert_utils.py": "def create_glb_with_pbr_materials(o, t, g): open(g, 'w').write('glb')\n",
    "torchvision_fix.py": "def apply_fix(): return True\n",
    "trimesh.py": "def load(p, force=None): raise NotImplementedError\n",
}
for rel, body in STUBS.items():
    path = os.path.join(fake, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(textwrap.dedent(body))

from PIL import Image
inputs = os.path.join(tmp, "inputs")
os.makedirs(inputs)
names = ["armchair", "stool", "rocking_chair"]
for n in names:
    Image.new("RGB", (64, 64), "brown").save(os.path.join(inputs, n + ".jpg"))

src = open(os.path.join(ROOT, "chair_studio.py")).read()
src = src.replace('subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio>=5.30,<6"], check=True)', "")
src = src.replace('REPO, PY, OUT_ROOT = "/content/Hunyuan3D-2.1", "/content/h3d-env/bin/python", "/content/outputs"',
                  f'REPO, PY, OUT_ROOT = {fake!r}, sys.executable, {os.path.join(tmp, "out")!r}')
src = src.replace("/content/chair_inputs", os.path.join(tmp, "chair_inputs"))
src = re.sub(r"demo\.queue\(.*\n?$", "", src)   # build the UI, don't launch it
assert fake in src and '"/content/outputs"' not in src, "path patch failed: chair_studio.py paths changed?"

g = {"GPU_VRAM_GB": 80, "TOOLS_DIR": ROOT, "__name__": "stub_test"}
exec(compile(src, "chair_studio.py", "exec"), g)

seeds = 2
t = time.time()
last = None
for last in g["generate"]([os.path.join(inputs, n + ".jpg") for n in names], "Best", seeds, True, False, True):
    pass
log, pick, viewer, zip_path = last
print("\n".join(l for l in log.splitlines() if any(k in l for k in ("DONE", "FAILED", "retrying", "Finished"))))

expected = len(names) * seeds
done = log.count("DONE ")
assert done == expected, f"expected {expected} textured models, got {done}"
assert "retrying with the standard decoder" in log, "FlashVDM fallback did not trigger"
assert "[texture1] DONE" in log and "[texture2] DONE" in log, "both texture workers should have done work"
assert viewer.endswith("_textured.glb") and os.path.exists(viewer)
assert os.path.exists(zip_path)
out_run = os.path.dirname(os.path.dirname(viewer))
assert not os.path.exists(os.path.join(out_run, ".queue")), "queue dir should be cleaned up"
print(f"\n✅ stub test passed: {done} models, 2 workers, fallback ok, {time.time() - t:.1f}s  (tmp: {tmp})")
