#!/usr/bin/env python3
"""Render a standalone HTML dashboard from a strategy diagnostic artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PHASE_COLORS = {
    0: "#6b7280",
    1: "#2563eb",
    2: "#f97316",
    3: "#16a34a",
    9: "#111827",
}


def _curve_json(curve: np.ndarray) -> str:
    step = max(1, len(curve) // 300)
    points = []
    for i in range(0, len(curve), step):
        row = curve[i]
        points.append([int(i + 1), float(row.min()), float(row.mean())])
    if points[-1][0] != len(curve):
        row = curve[-1]
        points.append([int(len(curve)), float(row.min()), float(row.mean())])
    return json.dumps(points, separators=(",", ":"))


def _load_summary(npz_path: Path) -> dict:
    summary_path = npz_path.with_name(npz_path.stem + "_summary.json")
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {"seed": None, "algo": npz_path.parent.name}


def render_html(npz_path: Path, output_path: Path | None = None) -> Path:
    data = np.load(npz_path)
    summary = _load_summary(npz_path)
    truth = data["truth_blob"]
    pred = data["predicted_final"]
    error = data["error_map"]
    rows = data["query_rows"].astype(int)
    cols = data["query_cols"].astype(int)
    phases = data["query_phases"].astype(int)
    curve = data["accuracy_curve"]
    _, H, W = truth.shape

    query_payload = [
        [int(r), int(c), int(p)]
        for r, c, p in zip(rows.tolist(), cols.tolist(), phases.tolist())
    ]
    layer_payload = []
    for k in range(truth.shape[0]):
        layer_payload.append(
            {
                "truth": truth[k].astype(int).tolist(),
                "pred": pred[k].astype(int).tolist(),
                "error": error[k].astype(int).tolist(),
            }
        )

    payload = {
        "summary": summary,
        "shape": [int(H), int(W)],
        "queries": query_payload,
        "phaseColors": PHASE_COLORS,
        "curve": json.loads(_curve_json(curve)),
        "layers": layer_payload,
    }

    html = _html_template(json.dumps(payload, separators=(",", ":")))
    if output_path is None:
        output_path = npz_path.with_name(npz_path.stem + "_dashboard.html")
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _html_template(payload_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blobchecker Strategy Dashboard</title>
<style>
body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; background: #f8fafc; }}
header {{ padding: 20px 24px 12px; background: #ffffff; border-bottom: 1px solid #e5e7eb; }}
h1 {{ margin: 0 0 12px; font-size: 22px; font-weight: 700; letter-spacing: 0; }}
.metrics {{ display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 8px; }}
.metric {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px 12px; background: #ffffff; }}
.metric span {{ display: block; color: #6b7280; font-size: 12px; }}
.metric strong {{ display: block; margin-top: 2px; font-size: 18px; }}
main {{ padding: 18px 24px 28px; }}
section {{ margin: 0 0 18px; }}
h2 {{ margin: 0 0 10px; font-size: 15px; font-weight: 700; }}
.panel {{ border: 1px solid #e5e7eb; border-radius: 8px; background: #ffffff; padding: 14px; overflow-x: auto; }}
.wide {{ width: 100%; max-width: 1200px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 8px; color: #4b5563; font-size: 12px; }}
.swatch {{ display: inline-block; width: 10px; height: 10px; margin-right: 5px; border-radius: 2px; vertical-align: -1px; }}
.layers {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
.layer {{ border: 1px solid #e5e7eb; border-radius: 8px; background: #ffffff; padding: 12px; }}
.layer-title {{ display: flex; justify-content: space-between; gap: 12px; margin-bottom: 8px; font-size: 13px; color: #374151; }}
.canvases {{ display: grid; grid-template-columns: repeat(4, minmax(220px, 1fr)); gap: 10px; }}
.canvas-block label {{ display: block; margin-bottom: 4px; font-size: 12px; color: #6b7280; }}
canvas {{ image-rendering: pixelated; border: 1px solid #e5e7eb; background: #ffffff; width: 100%; height: auto; }}
@media (max-width: 900px) {{ .metrics {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }} .canvases {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>Blobchecker Strategy Dashboard</h1>
  <div class="metrics" id="metrics"></div>
</header>
<main>
  <section>
    <h2>Accuracy Curve</h2>
    <div class="panel"><canvas id="curve" class="wide" width="1000" height="260"></canvas></div>
  </section>
  <section>
    <h2>Query Trajectory</h2>
    <div class="panel">
      <canvas id="queries" width="1000" height="250"></canvas>
      <div class="legend">
        <span><i class="swatch" style="background:#6b7280"></i>preplanned</span>
        <span><i class="swatch" style="background:#2563eb"></i>coarse</span>
        <span><i class="swatch" style="background:#f97316"></i>boundary</span>
        <span><i class="swatch" style="background:#16a34a"></i>entropy</span>
      </div>
    </div>
  </section>
  <section>
    <h2>Layer Diagnostics</h2>
    <div class="layers" id="layers"></div>
  </section>
</main>
<script>
const DATA = {payload_json};

function metric(label, value) {{
  return `<div class="metric"><span>${{label}}</span><strong>${{value}}</strong></div>`;
}}

function drawBinary(canvas, arr, colors) {{
  const h = arr.length, w = arr[0].length;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  for (let r = 0; r < h; r++) {{
    for (let c = 0; c < w; c++) {{
      const color = colors[arr[r][c]] || colors[0];
      const idx = (r * w + c) * 4;
      img.data[idx] = color[0];
      img.data[idx + 1] = color[1];
      img.data[idx + 2] = color[2];
      img.data[idx + 3] = 255;
    }}
  }}
  ctx.putImageData(img, 0, 0);
}}

function drawQueries() {{
  const canvas = document.getElementById("queries");
  const ctx = canvas.getContext("2d");
  const [h, w] = DATA.shape;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (const [r, c, phase] of DATA.queries) {{
    ctx.fillStyle = DATA.phaseColors[phase] || "#111827";
    const x = c / w * canvas.width;
    const y = r / h * canvas.height;
    ctx.fillRect(x, y, 3, 3);
  }}
}}

function drawCurve() {{
  const canvas = document.getElementById("curve");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const pad = 38;
  const w = canvas.width - pad * 2;
  const h = canvas.height - pad * 2;
  ctx.strokeStyle = "#e5e7eb";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {{
    const y = pad + h * i / 4;
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(pad + w, y); ctx.stroke();
  }}
  const maxIter = DATA.curve[DATA.curve.length - 1][0];
  function line(col, color) {{
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    DATA.curve.forEach((p, i) => {{
      const x = pad + (p[0] / maxIter) * w;
      const y = pad + (1 - p[col]) * h;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }});
    ctx.stroke();
  }}
  line(1, "#dc2626");
  line(2, "#2563eb");
  ctx.fillStyle = "#374151";
  ctx.font = "12px sans-serif";
  ctx.fillText("min accuracy", pad + 8, pad + 16);
  ctx.fillStyle = "#2563eb";
  ctx.fillText("mean accuracy", pad + 110, pad + 16);
}}

function init() {{
  const s = DATA.summary;
  document.getElementById("metrics").innerHTML = [
    metric("seed", s.seed),
    metric("algorithm", s.algo),
    metric("pass", String(s.overall_pass)),
    metric("min accuracy", Number(s.min_accuracy).toFixed(4)),
    metric("grid", DATA.shape[0] + " x " + DATA.shape[1]),
  ].join("");

  drawCurve();
  drawQueries();

  const layers = document.getElementById("layers");
  const binColors = {{0:[248,250,252],1:[17,24,39]}};
  const errColors = {{0:[248,250,252],1:[17,24,39],2:[220,38,38],3:[37,99,235]}};
  DATA.layers.forEach((layer, k) => {{
    const acc = s.per_layer_accuracy ? Number(s.per_layer_accuracy[k]).toFixed(4) : "";
    const div = document.createElement("div");
    div.className = "layer";
    div.innerHTML = `
      <div class="layer-title">
        <strong>Layer ${{k}}</strong>
        <span>acc ${{acc}} | h ${{s.height_pred?.[k]}}/${{s.height_truth?.[k]}} | w ${{s.width_pred?.[k]}}/${{s.width_truth?.[k]}}</span>
      </div>
      <div class="canvases">
        <div class="canvas-block"><label>truth</label><canvas></canvas></div>
        <div class="canvas-block"><label>prediction</label><canvas></canvas></div>
        <div class="canvas-block"><label>error</label><canvas></canvas></div>
        <div class="canvas-block"><label>query overlay</label><canvas></canvas></div>
      </div>`;
    layers.appendChild(div);
    const canvases = div.querySelectorAll("canvas");
    drawBinary(canvases[0], layer.truth, binColors);
    drawBinary(canvases[1], layer.pred, binColors);
    drawBinary(canvases[2], layer.error, errColors);
    drawBinary(canvases[3], layer.truth, binColors);
    const ctx = canvases[3].getContext("2d");
    for (const [r, c, phase] of DATA.queries) {{
      ctx.fillStyle = DATA.phaseColors[phase] || "#111827";
      ctx.fillRect(c, r, 1, 1);
    }}
  }});
}}

init();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = render_html(args.artifact, args.output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
