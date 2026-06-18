#!/usr/bin/env python3
"""Visualize synthetic blob dataset as a standalone HTML dashboard.

Examples
--------
    python tools/visualize_blobs.py
    python tools/visualize_blobs.py --seed 3 --H 50 --W 200
    python tools/visualize_blobs.py --seed 0 --output my_blobs.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generator import generate_dataset


def _mask_to_list(arr: np.ndarray) -> list:
    return arr.astype(int).tolist()


def render_html(truth_blob: np.ndarray, truth_out: np.ndarray, truth_full: np.ndarray,
                seed: int, output_path: Path) -> Path:
    _, H, W = truth_blob.shape
    n_layers = truth_blob.shape[0]

    layers = []
    for k in range(n_layers):
        blob_coverage = float(truth_blob[k].mean())
        outlier_count = int(truth_out[k].sum())
        layers.append({
            "blob": _mask_to_list(truth_blob[k]),
            "outlier": _mask_to_list(truth_out[k]),
            "full": _mask_to_list(truth_full[k]),
            "coverage": round(blob_coverage * 100, 1),
            "outlier_count": outlier_count,
        })

    payload = {
        "seed": seed,
        "shape": [H, W],
        "n_layers": n_layers,
        "layers": layers,
    }

    html = _html_template(json.dumps(payload, separators=(",", ":")))
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _html_template(payload_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blobchecker — Synthetic Dataset</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; background: #f8fafc; }}
header {{ padding: 20px 24px 14px; background: #ffffff; border-bottom: 1px solid #e5e7eb; }}
h1 {{ margin: 0 0 10px; font-size: 22px; font-weight: 700; }}
.meta {{ display: flex; gap: 20px; font-size: 13px; color: #6b7280; }}
.meta strong {{ color: #111827; }}
main {{ padding: 18px 24px 32px; }}
h2 {{ margin: 0 0 12px; font-size: 15px; font-weight: 700; }}
.layers {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
.layer {{ border: 1px solid #e5e7eb; border-radius: 8px; background: #ffffff; padding: 14px; }}
.layer-header {{ display: flex; align-items: center; gap: 16px; margin-bottom: 10px; font-size: 13px; }}
.layer-header strong {{ font-size: 15px; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; background: #f1f5f9; color: #475569; }}
.canvases {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
.canvas-block label {{ display: block; margin-bottom: 4px; font-size: 12px; color: #6b7280; font-weight: 500; }}
canvas {{ image-rendering: pixelated; border: 1px solid #e5e7eb; border-radius: 4px; width: 100%; height: auto; display: block; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 16px; padding: 10px 14px; border: 1px solid #e5e7eb; border-radius: 6px; background: #f8fafc; font-size: 12px; color: #4b5563; }}
.swatch {{ display: inline-block; width: 10px; height: 10px; margin-right: 5px; border-radius: 2px; vertical-align: -1px; border: 1px solid #d1d5db; }}
.coverage-bar-bg {{ height: 4px; background: #e5e7eb; border-radius: 2px; margin-top: 4px; width: 120px; }}
.coverage-bar {{ height: 4px; background: #2563eb; border-radius: 2px; }}
@media (max-width: 700px) {{ .canvases {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>Blobchecker — Synthetic Blob Dataset</h1>
  <div class="meta" id="meta"></div>
</header>
<main>
  <section>
    <h2>Layers</h2>
    <div class="layers" id="layers"></div>
    <div class="legend">
      <span><i class="swatch" style="background:#111827"></i>blob (foreground)</span>
      <span><i class="swatch" style="background:#f8fafc"></i>background</span>
      <span><i class="swatch" style="background:#2563eb"></i>outlier pixel</span>
      <span><i class="swatch" style="background:#7c3aed"></i>outlier on full mask</span>
    </div>
  </section>
</main>
<script>
const DATA = {payload_json};

function drawPixels(canvas, arr, colorMap) {{
  const h = arr.length, w = arr[0].length;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  for (let r = 0; r < h; r++) {{
    for (let c = 0; c < w; c++) {{
      const v = arr[r][c];
      const color = colorMap[v] || colorMap[0];
      const i = (r * w + c) * 4;
      img.data[i]     = color[0];
      img.data[i + 1] = color[1];
      img.data[i + 2] = color[2];
      img.data[i + 3] = 255;
    }}
  }}
  ctx.putImageData(img, 0, 0);
}}

function drawFull(canvas, blob, outlier) {{
  const h = blob.length, w = blob[0].length;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  for (let r = 0; r < h; r++) {{
    for (let c = 0; c < w; c++) {{
      const b = blob[r][c], o = outlier[r][c];
      let color;
      if (b === 1 && o === 0) color = [17, 24, 39];       // blob only → near-black
      else if (b === 0 && o === 1) color = [124, 58, 237]; // outlier only → purple
      else if (b === 1 && o === 1) color = [124, 58, 237]; // both (shouldn't happen) → purple
      else color = [248, 250, 252];                         // background
      const i = (r * w + c) * 4;
      img.data[i]     = color[0];
      img.data[i + 1] = color[1];
      img.data[i + 2] = color[2];
      img.data[i + 3] = 255;
    }}
  }}
  ctx.putImageData(img, 0, 0);
}}

function init() {{
  const [H, W] = DATA.shape;
  document.getElementById("meta").innerHTML = `
    <span>seed <strong>${{DATA.seed}}</strong></span>
    <span>grid <strong>${{H}} × ${{W}}</strong></span>
    <span>layers <strong>${{DATA.n_layers}}</strong></span>
  `;

  const container = document.getElementById("layers");
  const blobColors = {{0:[248,250,252],1:[17,24,39]}};
  const outColors  = {{0:[248,250,252],1:[37,99,235]}};

  DATA.layers.forEach((layer, k) => {{
    const hasOutliers = layer.outlier_count > 0;
    const div = document.createElement("div");
    div.className = "layer";
    div.innerHTML = `
      <div class="layer-header">
        <strong>Layer ${{k}}</strong>
        <span class="badge">coverage ${{layer.coverage}}%</span>
        ${{hasOutliers ? `<span class="badge" style="background:#ede9fe;color:#5b21b6">outliers ${{layer.outlier_count}}</span>` : `<span class="badge" style="background:#f0fdf4;color:#166534">no outliers</span>`}}
        <div>
          <div style="font-size:11px;color:#9ca3af;margin-bottom:2px">blob coverage</div>
          <div class="coverage-bar-bg"><div class="coverage-bar" style="width:${{layer.coverage}}%"></div></div>
        </div>
      </div>
      <div class="canvases">
        <div class="canvas-block"><label>truth_blob</label><canvas></canvas></div>
        <div class="canvas-block"><label>truth_outlier</label><canvas></canvas></div>
        <div class="canvas-block"><label>truth_full</label><canvas></canvas></div>
      </div>`;
    container.appendChild(div);

    const canvases = div.querySelectorAll("canvas");
    drawPixels(canvases[0], layer.blob, blobColors);
    drawPixels(canvases[1], layer.outlier, outColors);
    drawFull(canvases[2], layer.blob, layer.outlier);
  }});
}}

init();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize synthetic blob dataset.")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument("--H", type=int, default=50, help="grid height (default: 50)")
    parser.add_argument("--W", type=int, default=200, help="grid width (default: 200)")
    parser.add_argument("--output", type=Path, default=None,
                        help="output HTML path (default: artifacts/blobs_seed_<N>.html)")
    args = parser.parse_args()

    H, W = args.H, args.W
    print(f"generating dataset  seed={args.seed}  grid={H}×{W} …")
    truth_blob, truth_out, truth_full = generate_dataset(H, W, args.seed)

    for k in range(truth_blob.shape[0]):
        cov = truth_blob[k].mean() * 100
        n_out = truth_out[k].sum()
        print(f"  layer {k}: coverage={cov:.1f}%  outliers={n_out}")

    output_path = args.output
    if output_path is None:
        out_dir = ROOT / "artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"blobs_seed_{args.seed:03d}.html"

    path = render_html(truth_blob, truth_out, truth_full, args.seed, output_path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
