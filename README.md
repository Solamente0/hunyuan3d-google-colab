# Hunyuan3D Chair Studio

Turn chair photos into realistic, PBR-textured 3D models with [Tencent Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) on Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Solamente0/hunyuan3d-google-colab/blob/main/Hunyuan3D_Chairs.ipynb)

## Use
1. Click **Open in Colab**, pick **A100 GPU + High-RAM**.
2. Run cells 1 → 2 → 3 and use the Chair Studio app.

## Files
| File | What it does |
|---|---|
| `Hunyuan3D_Chairs.ipynb` | Thin launcher: pulls this repo, runs the scripts |
| `setup_env.py` | GPU check, weight cache, HF token |
| `scripts/install.sh` | Python 3.10 env, PyTorch 2.5.1/CUDA 12.4, CUDA extensions, weights |
| `chair_studio.py` | Gradio app: upload, generate, 3D preview, zip download |
| `scripts/run_chairs.py` | Worker: shape generation and parallel PBR texturing |
| `official_ui.py` | Tencent's own UI with best-quality defaults |

## Updating
Edit and push to `main`. Colab picks up the change when you rerun cell 1 or 3, with no notebook re-upload needed.

## License
Hunyuan3D 2.1 is under the Tencent Hunyuan Community License; read it before commercial use.
