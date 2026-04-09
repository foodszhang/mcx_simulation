#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.fem_mesh import load_mesh


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>FEM 3D Viewer</title>
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
    h1 {{
      margin: 0 0 8px 0;
      font-size: 24px;
    }}
    .meta {{
      color: #b7c4e0;
      margin-bottom: 16px;
      line-height: 1.6;
    }}
    .controls {{
      margin-bottom: 12px;
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .plot {{
      width: 100%;
      height: 720px;
      background: #11172a;
      border-radius: 12px;
      margin-bottom: 18px;
    }}
    label {{
      cursor: pointer;
    }}
    code {{
      color: #9fd3ff;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>FEM Mesh & Fluence Viewer</h1>
    <div class="meta">
      <div>nodes: <code>{n_nodes}</code> | boundary faces: <code>{n_faces}</code> | boundary nodes: <code>{n_bnd_nodes}</code></div>
      <div>fluence range: <code>{fluence_min:.6e}</code> ~ <code>{fluence_max:.6e}</code></div>
      <div>HTML generated from local FEM result.</div>
    </div>

    <div class="controls">
      <label><input type="checkbox" id="togglePoints" checked /> show boundary node points</label>
      <label><input type="checkbox" id="toggleMeshEdges" checked /> show mesh edges</label>
    </div>

    <div id="meshPlot" class="plot"></div>
    <div id="fluencePlot" class="plot"></div>
  </div>

  <script>
    const meshData = {mesh_json};
    const fluenceData = {fluence_json};

    const edgeColor = 'rgba(40, 50, 70, 0.22)';
    const baseLayout = (title) => ({{
      title: {{text: title, font: {{color: '#e8eefc', size: 22}}}},
      paper_bgcolor: '#11172a',
      plot_bgcolor: '#11172a',
      margin: {{l: 0, r: 0, b: 0, t: 48}},
      scene: {{
        bgcolor: '#11172a',
        xaxis: {{title: 'X (mm)', color: '#c9d6f2', gridcolor: '#24304d'}},
        yaxis: {{title: 'Y (mm)', color: '#c9d6f2', gridcolor: '#24304d'}},
        zaxis: {{title: 'Z (mm)', color: '#c9d6f2', gridcolor: '#24304d'}},
        aspectmode: 'data',
      }},
      legend: {{
        font: {{color: '#e8eefc'}},
        bgcolor: 'rgba(17,23,42,0.6)'
      }}
    }});

    const meshSurface = {{
      type: 'mesh3d',
      name: 'structure surface',
      x: meshData.x,
      y: meshData.y,
      z: meshData.z,
      i: meshData.si,
      j: meshData.sj,
      k: meshData.sk,
      intensity: meshData.surface_region,
      intensitymode: 'cell',
      colorscale: 'Turbo',
      opacity: 0.88,
      flatshading: true,
      colorbar: {{
        title: 'Region ID',
        tickcolor: '#e8eefc',
        tickfont: {{color: '#e8eefc'}}
      }},
      hovertemplate: 'region=%{{intensity:.0f}}<extra></extra>',
      contour: {{
        show: true,
        color: edgeColor,
        width: 1
      }}
    }};

    const boundaryPoints = {{
      type: 'scatter3d',
      mode: 'markers',
      name: 'boundary nodes',
      x: meshData.boundary_x,
      y: meshData.boundary_y,
      z: meshData.boundary_z,
      marker: {{
        size: 2.8,
        color: '#ffd166',
        opacity: 0.8
      }},
      hovertemplate: 'x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>z=%{{z:.2f}}<extra></extra>'
    }};

    const fluenceSurface = {{
      type: 'mesh3d',
      name: 'fluence surface',
      x: meshData.x,
      y: meshData.y,
      z: meshData.z,
      i: meshData.i,
      j: meshData.j,
      k: meshData.k,
      intensity: fluenceData.face_values,
      intensitymode: 'cell',
      colorscale: 'Viridis',
      opacity: 0.92,
      flatshading: true,
      colorbar: {{
        title: 'Fluence',
        tickcolor: '#e8eefc',
        tickfont: {{color: '#e8eefc'}},
        titlefont: {{color: '#e8eefc'}}
      }},
      contour: {{
        show: true,
        color: edgeColor,
        width: 1
      }},
      hovertemplate: 'surface fluence=%{{intensity:.4e}}<extra></extra>'
    }};

    const fluencePoints = {{
      type: 'scatter3d',
      mode: 'markers',
      name: 'boundary node fluence',
      x: fluenceData.boundary_x,
      y: fluenceData.boundary_y,
      z: fluenceData.boundary_z,
      marker: {{
        size: 3.0,
        color: fluenceData.boundary_values,
        colorscale: 'Viridis',
        opacity: 0.85,
        colorbar: {{
          title: 'Node Fluence',
          tickcolor: '#e8eefc',
          tickfont: {{color: '#e8eefc'}},
          titlefont: {{color: '#e8eefc'}}
        }}
      }},
      hovertemplate: 'node fluence=%{{marker.color:.4e}}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>z=%{{z:.2f}}<extra></extra>'
    }};

    Plotly.newPlot('meshPlot', [meshSurface, boundaryPoints], baseLayout('FEM Structure Surface'), {{responsive: true}});
    Plotly.newPlot('fluencePlot', [fluenceSurface, fluencePoints], baseLayout('FEM Fluence Surface'), {{responsive: true}});

    const pointToggle = document.getElementById('togglePoints');
    const edgeToggle = document.getElementById('toggleMeshEdges');

    function updatePoints() {{
      const visible = pointToggle.checked ? true : 'legendonly';
      Plotly.restyle('meshPlot', {{visible: visible}}, [1]);
      Plotly.restyle('fluencePlot', {{visible: visible}}, [1]);
    }}

    function updateEdges() {{
      const show = edgeToggle.checked;
      const contour = show ? {{show: true, color: edgeColor, width: 1}} : {{show: false}};
      Plotly.restyle('meshPlot', {{contour: [contour]}}, [0]);
      Plotly.restyle('fluencePlot', {{contour: [contour]}}, [0]);
    }}

    pointToggle.addEventListener('change', updatePoints);
    edgeToggle.addEventListener('change', updateEdges);
  </script>
</body>
</html>
"""


def parse_args():
    parser = argparse.ArgumentParser(description="导出可交互HTML的FEM三维查看页面")
    parser.add_argument("--mesh", required=True, help="mesh .npz 文件路径")
    parser.add_argument("--fluence", required=True, help="节点光通量 .npy 文件路径")
    parser.add_argument("--out", required=True, help="输出 HTML 文件路径")
    return parser.parse_args()


def _to_mesh_payload(mesh, nodal_fluence):
    boundary_nodes = mesh.nodes[mesh.bndvtx > 0]
    face_values = nodal_fluence[mesh.boundary_faces].mean(axis=1)
    surface_faces = mesh.surface_faces if mesh.surface_faces is not None else mesh.boundary_faces
    surface_regions = (
        mesh.surface_face_regions[:, 0]
        if mesh.surface_face_regions is not None and len(mesh.surface_face_regions) > 0
        else np.ones(len(surface_faces), dtype=np.float64)
    )
    return {
        "mesh_json": {
            "x": mesh.nodes[:, 0].tolist(),
            "y": mesh.nodes[:, 1].tolist(),
            "z": mesh.nodes[:, 2].tolist(),
            "si": surface_faces[:, 0].tolist(),
            "sj": surface_faces[:, 1].tolist(),
            "sk": surface_faces[:, 2].tolist(),
            "surface_region": surface_regions.tolist(),
            "i": mesh.boundary_faces[:, 0].tolist(),
            "j": mesh.boundary_faces[:, 1].tolist(),
            "k": mesh.boundary_faces[:, 2].tolist(),
            "boundary_x": boundary_nodes[:, 0].tolist(),
            "boundary_y": boundary_nodes[:, 1].tolist(),
            "boundary_z": boundary_nodes[:, 2].tolist(),
        },
        "fluence_json": {
            "face_values": face_values.tolist(),
            "boundary_x": boundary_nodes[:, 0].tolist(),
            "boundary_y": boundary_nodes[:, 1].tolist(),
            "boundary_z": boundary_nodes[:, 2].tolist(),
            "boundary_values": nodal_fluence[mesh.bndvtx > 0].tolist(),
        },
        "n_nodes": len(mesh.nodes),
        "n_faces": len(mesh.boundary_faces),
        "n_bnd_nodes": int(np.sum(mesh.bndvtx > 0)),
        "fluence_min": float(np.nanmin(nodal_fluence)),
        "fluence_max": float(np.nanmax(nodal_fluence)),
    }


def main():
    args = parse_args()
    mesh = load_mesh(args.mesh)
    nodal_fluence = np.load(args.fluence)

    if len(nodal_fluence) != len(mesh.nodes):
        raise ValueError(
            f"节点光通量长度 {len(nodal_fluence)} 与 mesh 节点数 {len(mesh.nodes)} 不一致。"
        )

    payload = _to_mesh_payload(mesh, nodal_fluence)
    html = HTML_TEMPLATE.format(
        mesh_json=json.dumps(payload["mesh_json"], separators=(",", ":")),
        fluence_json=json.dumps(payload["fluence_json"], separators=(",", ":")),
        n_nodes=payload["n_nodes"],
        n_faces=payload["n_faces"],
        n_bnd_nodes=payload["n_bnd_nodes"],
        fluence_min=payload["fluence_min"],
        fluence_max=payload["fluence_max"],
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"已生成交互式HTML: {out_path}")


if __name__ == "__main__":
    main()
