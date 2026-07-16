#!/usr/bin/env python3
"""Create a small, shareable, offline 3D browser of all HDBSCAN cluster centers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


COORDS = ["flow_z1", "flow_z2", "flow_z3"]
LABEL_COLUMNS = ["final_relative_foil", "canonical_hole_family", "relative_hole_family"]


def mode_or_none(s: pd.Series):
    s = s.dropna()
    return None if s.empty else s.mode().iloc[0]


def to_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return str(value)


def make_html(centers: list[dict], title: str, plotly_js: str) -> str:
    data = json.dumps(centers, separators=(",", ":"), allow_nan=False)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><script>{plotly_js.replace('</script', '<\\/script')}</script>
<style>
body{{margin:0;background:#f8fafc;color:#172033;font-family:system-ui,-apple-system,Segoe UI,sans-serif}} header{{padding:14px 24px 10px;background:#fff;border-bottom:1px solid #d9e1eb}}h1{{font-size:21px;margin:0 0 4px}}#subtitle{{font-size:13px;color:#526174}}#controls{{display:flex;gap:10px;flex-wrap:wrap;align-items:end;padding:9px 24px;background:#fff;border-bottom:1px solid #d9e1eb}}label{{font-size:12px;font-weight:650;display:grid;gap:3px;color:#3f4b5b}}select,input{{font:14px system-ui;padding:5px 7px;border:1px solid #bbc7d5;border-radius:5px;background:#fff;min-width:150px}}input{{min-width:100px}}button{{font:14px system-ui;padding:6px 11px;border:1px solid #2c6fb3;border-radius:5px;background:#1675c1;color:#fff;cursor:pointer}}#status{{padding:6px 24px;background:#fff;font-size:12px;color:#536174}}#plot{{height:calc(100vh - 164px);min-height:620px}}#scale{{position:fixed;right:28px;bottom:28px;z-index:5;padding:9px 11px;background:rgba(255,255,255,.93);border:1px solid #cbd5e1;border-radius:7px;box-shadow:0 2px 8px rgba(15,23,42,.12);font-size:11px;color:#334155}}#scale b{{display:block;margin-bottom:5px;font-size:12px}}.row{{display:flex;align-items:center;gap:7px;height:24px}}.ball{{display:inline-block;border-radius:50%;background:#64748b;border:1px solid #172033;flex:none}}
</style></head><body>
<header><h1>{title}</h1><div id="subtitle">Each marker is one HDBSCAN cluster center, calculated from the full input sample. Rotate to inspect separation in flow latent space.</div></header>
<div id="controls"><label>Colour by<select id="colour"><option value="foil">inferred foil</option><option value="hole">canonical hole family</option><option value="cluster">cluster ID</option></select></label><label>Show labels<select id="labels"><option value="off">off</option><option value="on">cluster ID</option></select></label><label>Cluster filter<input id="filter" placeholder="e.g. 219,220"></label><button id="reset">Reset camera</button></div>
<div id="status"></div><div id="plot"></div><div id="scale"></div>
<script>
const D={data}; const $=id=>document.getElementById(id); const fields={{foil:'foil',hole:'hole',cluster:'cluster'}};
function hash(s){{let h=2166136261;for(let i=0;i<s.length;i++)h=Math.imul(h^s.charCodeAt(i),16777619);return h>>>0}} function color(v){{if(v===null||String(v)==='-1')return '#9aa3ad';return `hsl(${{hash(String(v))%360}},63%,47%)`;}}
function size(n,max){{return 3+8*Math.cbrt(n/max)}} function filterData(){{let q=$('filter').value.trim();if(!q)return D;let set=new Set(q.split(/[ ,]+/));return D.filter(d=>set.has(String(d.cluster)));}}
function scaleBox(a){{let ns=a.map(d=>d.n).sort((x,y)=>x-y), max=Math.max(...ns), vals=[ns[0],ns[Math.floor(ns.length/2)],ns[ns.length-1]];vals=[...new Set(vals)];$('scale').innerHTML='<b>Marker size: full cluster N</b>'+vals.map(n=>`<div class="row"><i class="ball" style="width:${{size(n,max)}}px;height:${{size(n,max)}}px"></i><span>N = ${{n.toLocaleString()}}</span></div>`).join('')+'<div style="margin-top:4px">diameter ∝ N<sup>1/3</sup></div>';}}
function render(){{let a=filterData(), mode=$('colour').value, f=fields[mode], groups={{}};a.forEach(d=>{{let k=d[f]===null?'unassigned':String(d[f]);(groups[k]??=[]).push(d)}});let max=Math.max(1,...a.map(d=>d.n));let textOn=$('labels').value==='on';let traces=Object.entries(groups).sort((a,b)=>a[0].localeCompare(b[0],undefined,{{numeric:true}})).map(([k,g])=>({{type:'scatter3d',mode:textOn?'markers+text':'markers',name:mode+' '+k,x:g.map(d=>d.z1),y:g.map(d=>d.z2),z:g.map(d=>d.z3),text:g.map(d=>String(d.cluster)),textposition:'top center',textfont:{{size:10,color:'#182433'}},hovertext:g.map(d=>`<b>cluster ${{d.cluster}}</b><br>events: ${{d.n.toLocaleString()}}<br>inferred foil: ${{d.foil??'unassigned'}}<br>canonical hole: ${{d.hole??'unassigned'}}<br>mean sieve: (${{d.sieve_x.toFixed(2)}}, ${{d.sieve_y.toFixed(2)}})<br>mean y_tar: ${{d.ytar.toFixed(3)}}<br>mean HDBSCAN probability: ${{d.prob.toFixed(3)}}`),hovertemplate:'%{{hovertext}}<extra></extra>',marker:{{size:g.map(d=>size(d.n,max)),color:color(k),opacity:1,line:{{color:'#172033',width:.8}}}}}}));
let layout={{margin:{{l:0,r:0,t:8,b:0}},paper_bgcolor:'#f8fafc',showlegend:true,legend:{{itemsizing:'constant'}},scene:{{xaxis:{{title:'flow z1',backgroundcolor:'#fff',gridcolor:'#e2e8f0'}},yaxis:{{title:'flow z2',backgroundcolor:'#fff',gridcolor:'#e2e8f0'}},zaxis:{{title:'flow z3',backgroundcolor:'#fff',gridcolor:'#e2e8f0'}},aspectmode:'data'}},uirevision:'flow-z123-centers'}};Plotly.react('plot',traces,layout,{{responsive:true,displaylogo:false,scrollZoom:true}});$('status').textContent=`${{a.length}} of ${{D.length}} cluster centers · colour: ${{mode}} · opaque markers use WebGL depth occlusion`;scaleBox(a);}}
['colour','labels'].forEach(id=>$(id).addEventListener('change',render));$('filter').addEventListener('input',render);$('reset').addEventListener('click',()=>Plotly.relayout('plot',{{'scene.camera':null}}));render();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("flow-z123-cluster-centers.html"))
    parser.add_argument("--cluster-column", default="flow_hdbscan_cluster")
    parser.add_argument("--plotly-js", type=Path, required=True, help="local plotly.min.js embedded in output")
    args = parser.parse_args()
    usecols = list(dict.fromkeys(COORDS + [args.cluster_column, "sieve_x", "sieve_y", "P.gtr.y", "hdbscan_probability"] + LABEL_COLUMNS))
    df = pd.read_csv(args.input, usecols=lambda c: c in usecols)
    missing = [c for c in COORDS + [args.cluster_column] if c not in df]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    rows = []
    for cluster, g in df[df[args.cluster_column] >= 0].groupby(args.cluster_column, sort=True):
        row = {"cluster": int(cluster), "n": int(len(g)), "z1": float(g["flow_z1"].mean()), "z2": float(g["flow_z2"].mean()), "z3": float(g["flow_z3"].mean()), "sieve_x": float(g["sieve_x"].mean()), "sieve_y": float(g["sieve_y"].mean()), "ytar": float(g["P.gtr.y"].mean()), "prob": float(g["hdbscan_probability"].mean())}
        for col, out in [("final_relative_foil", "foil"), ("canonical_hole_family", "hole")]:
            row[out] = to_value(mode_or_none(g[col])) if col in g else None
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(make_html(rows, "FP5D flow: all HDBSCAN cluster centers in (z1, z2, z3)", args.plotly_js.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"Wrote {args.output} with {len(rows)} cluster centers; {args.output.stat().st_size / 1024 / 1024:.2f} MiB.")


if __name__ == "__main__":
    main()
