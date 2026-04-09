#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>FEM Surface Fluence</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #0b1020;
      color: #e8eefc;
    }}
    .wrap {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 16px;
    }}
    h1 {{ margin: 0 0 8px 0; }}
    .meta {{ color: #b7c4e0; margin-bottom: 14px; line-height: 1.6; }}
    .plot {{ width: 100%; height: 840px; background: #11172a; border-radius: 12px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>FEM Surface Fluence Viewer</h1>
    <div class="meta">
      <div>surface nodes: <code>{n_nodes}</code> | surface faces: <code>{n_faces}</code></div>
      <div>fluence range: <code>{fluence_min:.6e}</code> ~ <code>{fluence_max:.6e}</code></div>
    </div>
    <div id="plot" class="plot"></div>
  </div>
  <script>
    const dataObj = {payload_json};
    const traceSurface = {{
      type: 'mesh3d',
      x: dataObj.x,
      y: dataObj.y,
      z: dataObj.z,
      i: dataObj.i,
      j: dataObj.j,
      k: dataObj.k,
      intensity: dataObj.face_values,
      intensitymode: 'cell',
      colorscale: 'Viridis',
      opacity: 0.95,
      flatshading: true,
      colorbar: {{
        title: 'Surface Fluence',
        tickcolor: '#e8eefc',
        tickfont: {{color: '#e8eefc'}}
      }},
      contour: {{
        show: true,
        color: 'rgba(40,50,70,0.18)',
        width: 1
      }},
      hovertemplate: 'surface fluence=%{{intensity:.4e}}<extra></extra>'
    }};

    const tracePoints = {{
      type: 'scatter3d',
      mode: 'markers',
      x: dataObj.x,
      y: dataObj.y,
      z: dataObj.z,
      marker: {{
        size: 2.5,
        color: dataObj.node_values,
        colorscale: 'Viridis',
        opacity: 0.65,
        showscale: false
      }},
      hovertemplate: 'node fluence=%{{marker.color:.4e}}<extra></extra>',
      name: 'surface nodes'
    }};

    const layout = {{
      title: {{text: 'Surface Fluence', font: {{color: '#e8eefc', size: 24}}}},
      paper_bgcolor: '#11172a',
      plot_bgcolor: '#11172a',
      margin: {{l: 0, r: 0, b: 0, t: 48}},
      legend: {{font: {{color: '#e8eefc'}}}},
      scene: {{
        bgcolor: '#11172a',
        xaxis: {{title: 'X (mm)', color: '#c9d6f2', gridcolor: '#24304d'}},
        yaxis: {{title: 'Y (mm)', color: '#c9d6f2', gridcolor: '#24304d'}},
        zaxis: {{title: 'Z (mm)', color: '#c9d6f2', gridcolor: '#24304d'}},
        aspectmode: 'data'
      }}
    }};

    Plotly.newPlot('plot', [traceSurface, tracePoints], layout, {{responsive: true}});
  </script>
</body>
</html>
"""


def parse_args():
    parser = argparse.ArgumentParser(description="导出表面光通量交互HTML")
    parser.add_argument("--nodes", required=True, help="fem_surface_nodes.npy")
    parser.add_argument("--faces", required=True, help="fem_surface_faces.npy")
    parser.add_argument("--node-fluence", required=True, help="fem_surface_nodal_fluence.npy")
    parser.add_argument("--face-fluence", required=True, help="fem_surface_face_fluence.npy")
    parser.add_argument("--out", required=True, help="输出 HTML 文件路径")
    return parser.parse_args()


def main():
    args = parse_args()
    nodes = np.load(args.nodes)
    faces = np.load(args.faces)
    node_values = np.load(args.node_fluence)
    face_values = np.load(args.face_fluence)

    payload = {
        "x": nodes[:, 0].tolist(),
        "y": nodes[:, 1].tolist(),
        "z": nodes[:, 2].tolist(),
        "i": faces[:, 0].tolist(),
        "j": faces[:, 1].tolist(),
        "k": faces[:, 2].tolist(),
        "node_values": node_values.tolist(),
        "face_values": face_values.tolist(),
    }
    html = HTML_TEMPLATE.format(
        payload_json=json.dumps(payload, separators=(",", ":")),
        n_nodes=len(nodes),
        n_faces=len(faces),
        fluence_min=float(np.nanmin(node_values)),
        fluence_max=float(np.nanmax(node_values)),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"已生成表面光通量HTML: {out_path}")


if __name__ == "__main__":
    main()
