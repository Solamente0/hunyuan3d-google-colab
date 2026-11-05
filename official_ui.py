"""Launches Tencent's official Hunyuan3D 2.1 web UI with best-quality defaults.

Run from the notebook with `%run -i` so it can read the notebook's settings.
Part of https://github.com/Solamente0/hunyuan3d-google-colab
"""
PUBLIC_LINK = globals().get("PUBLIC_LINK", False)
FORCE_LOW_VRAM_MODE = globals().get("FORCE_LOW_VRAM_MODE", False)

import os, re, socket, subprocess, time

REPO, PY, PORT = "/content/Hunyuan3D-2.1", "/content/h3d-env/bin/python", 8080
LOG = "/content/hunyuan3d.log"
low = FORCE_LOW_VRAM_MODE or globals().get("LOW_VRAM", False)

# Raise the UI's default sliders: 50 steps, octree 512 (384 on small GPUs), big decode chunks.
# Texturing in the UI already uses the highest official preset (8 views at 768 px, 4K texture maps).
app_py = os.path.join(REPO, "gradio_app.py")
src = open(app_py, encoding="utf-8").read()
patches = [
    (r"(value=5 if 'turbo' in args\.subfolder else )\d+", r"\g<1>50", "inference steps → 50"),
    (r"(octree_resolution = gr\.Slider\(maximum=512,\s*minimum=16,\s*value=)\d+",
     r"\g<1>" + ("384" if low else "512"), "octree resolution → " + ("384" if low else "512")),
    (r"(maximum=5000000, minimum=1000, value=)\d+", r"\g<1>" + ("20000" if low else "200000"), "decode chunks raised"),
]
for pattern, repl, label in patches:
    src, n = re.subn(pattern, repl, src)
    print(("✔ " if n else "⚠ not found: ") + label)
open(app_py, "w", encoding="utf-8").write(src)

subprocess.run(["pkill", "-f", "gradio_app.py"])
time.sleep(2)

cmd = [PY, "gradio_app.py",
       "--model_path", "tencent/Hunyuan3D-2.1",
       "--subfolder", "hunyuan3d-dit-v2-1",
       "--texgen_model_path", "tencent/Hunyuan3D-2.1",
       "--host", "0.0.0.0", "--port", str(PORT)]
if low:
    cmd.append("--low_vram_mode")

env = dict(os.environ, PYTHONUNBUFFERED="1")
server = subprocess.Popen(cmd, cwd=REPO, stdout=open(LOG, "w"),
                          stderr=subprocess.STDOUT, env=env)

def port_open():
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", PORT)) == 0

print("⏳ Loading models. The first run downloads ~25 GB, so this can take 5–15 minutes...")
start, last_line = time.time(), ""
while not port_open():
    if server.poll() is not None:
        print(open(LOG).read()[-4000:])
        raise RuntimeError("❌ The server stopped. See the log above.")
    if time.time() - start > 45 * 60:
        raise TimeoutError("Still not up after 45 minutes. Check cell 4 (logs).")
    lines = [l for l in open(LOG).read().splitlines() if l.strip()]
    if lines and lines[-1] != last_line:
        last_line = lines[-1]
        print("   ", last_line[:150])
    time.sleep(15)

print(f"\n✅ Ready after {(time.time() - start) / 60:.1f} min")
from google.colab.output import eval_js
print("🔗 Open the UI (works in your browser only):", eval_js(f"google.colab.kernel.proxyPort({PORT})"))

if PUBLIC_LINK:
    if not os.path.exists("/content/cloudflared"):
        subprocess.run(["wget", "-q", "-O", "/content/cloudflared",
                        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"], check=True)
        os.chmod("/content/cloudflared", 0o755)
    subprocess.run(["pkill", "-f", "cloudflared tunnel"])
    CF_LOG = "/content/cloudflared.log"
    subprocess.Popen(["/content/cloudflared", "tunnel", "--url", f"http://localhost:{PORT}", "--no-autoupdate"],
                     stdout=open(CF_LOG, "w"), stderr=subprocess.STDOUT)
    for _ in range(30):
        time.sleep(2)
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open(CF_LOG).read())
        if m:
            print("🌍 Public link:", m.group(0))
            break
    else:
        print("Couldn't get a public link; see /content/cloudflared.log")

print("\nIn the UI: upload a chair photo → 'Gen Textured Shape'. Leave 'Remove Background' on.")
print("For several chairs at once, cell 5 is easier. Results: /content/Hunyuan3D-2.1/save_dir")
