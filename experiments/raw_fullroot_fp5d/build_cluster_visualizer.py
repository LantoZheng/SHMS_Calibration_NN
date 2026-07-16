#!/usr/bin/env python3
"""Build a self-contained interactive viewer for an already-clustered CSV.

Example
-------
python build_cluster_visualizer.py results/global_foil_flattened_z3_labels.csv \
    --output results/foil-clusters-3d.html --sample-per-cluster 220

Pass ``--plotly-js path/to/plotly.min.js`` to produce a fully self-contained,
offline HTML.  Without that option the output loads a pinned Plotly release
from the official CDN; all event data are embedded in either case.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DISPLAY_NAMES = {
    "P.dc.x_fp": "x_fp", "P.dc.y_fp": "y_fp", "P.dc.xp_fp": "x'_fp",
    "P.dc.yp_fp": "y'_fp", "P.rb.raster.frybRawAdc": "fr_ybpm",
    "P.gtr.y": "reconstructed y_tar", "sieve_x": "reconstructed x_sieve",
    "sieve_y": "reconstructed y_sieve", "flow_hdbscan_cluster": "HDBSCAN cluster",
    "linear_local_field_cluster": "local-field cluster",
    "final_relative_foil": "relative foil", "canonical_hole_family": "hole family",
    "global_foil_flattened_z3": "flattened global z3",
    "global_continuous_z3": "continuous global z3",
}

PREFERRED_COORDS = [
    "flow_z1", "flow_z2", "flow_z3", "flow_z4", "flow_z5",
    "global_continuous_z3", "global_foil_flattened_z3",
    "sieve_x", "sieve_y", "P.gtr.y",
    "P.dc.x_fp", "P.dc.y_fp", "P.dc.xp_fp", "P.dc.yp_fp", "P.rb.raster.frybRawAdc",
]
PREFERRED_COLOURS = [
    "flow_hdbscan_cluster", "linear_local_field_cluster", "final_relative_foil",
    "relative_foil_layer", "canonical_foil_layer", "canonical_hole_family",
    "relative_hole_family", "local_field_component",
]


def display_name(column: str) -> str:
    return DISPLAY_NAMES.get(column, column)


def choose_sample(frame: pd.DataFrame, group_col: str, limit: int, seed: int) -> pd.DataFrame:
    """Balanced deterministic downsampling, preserving small clusters intact."""
    if limit <= 0 or group_col not in frame or len(frame) <= limit:
        return frame
    rng = np.random.default_rng(seed)
    # A per-cluster cap prevents the largest holes from hiding small clusters.
    n_groups = max(1, frame[group_col].nunique(dropna=False))
    cap = max(20, int(np.ceil(limit / n_groups)))
    indices: list[np.ndarray] = []
    for _, group in frame.groupby(group_col, dropna=False, sort=False):
        take = min(len(group), cap)
        indices.append(rng.choice(group.index.to_numpy(), take, replace=False))
    chosen = np.concatenate(indices)
    if len(chosen) > limit:
        chosen = rng.choice(chosen, limit, replace=False)
    return frame.loc[chosen].sort_index()


def build_html(data: dict, title: str, plotly_js: str | None = None) -> str:
    packed = json.dumps(data, separators=(",", ":"), allow_nan=False)
    safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    plotly_tag = (
        f"<script>{plotly_js.replace('</script', '<\\/script')}</script>"
        if plotly_js else '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{safe_title}</title>{plotly_tag}
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:0;background:#f8fafc;color:#172033}}
header{{padding:14px 22px 10px;background:#fff;border-bottom:1px solid #dce3ec}} h1{{font-size:20px;margin:0 0 4px}}
#meta{{font-size:13px;color:#536174}} #controls{{display:flex;flex-wrap:wrap;gap:10px;padding:10px 22px;background:#fff;border-bottom:1px solid #dce3ec;align-items:end}}
label{{font-size:12px;font-weight:650;display:grid;gap:3px;color:#3b4758}} select,input{{font:14px system-ui;padding:5px 7px;border:1px solid #bbc7d5;border-radius:5px;background:white;min-width:145px}}
input{{min-width:80px;width:80px}} button{{font:14px system-ui;padding:6px 11px;border:1px solid #2c6fb3;border-radius:5px;background:#1675c1;color:#fff;cursor:pointer}}
#plot{{height:calc(100vh - 150px);min-height:620px}} #status{{font-size:12px;color:#536174;padding:0 22px 8px;background:#fff}}
#size-scale{{position:fixed;right:28px;bottom:30px;z-index:5;padding:9px 11px;background:rgba(255,255,255,.92);border:1px solid #cbd5e1;border-radius:7px;box-shadow:0 2px 8px rgba(15,23,42,.12);font-size:11px;color:#334155;display:none;min-width:125px}}
#size-scale b{{display:block;margin-bottom:5px;font-size:12px}} .scale-row{{display:flex;align-items:center;gap:7px;height:25px}} .scale-ball{{display:inline-block;border-radius:50%;background:#64748b;border:1px solid #172033;flex:none}}
</style></head><body>
<header><h1>{safe_title}</h1><div id="meta"></div></header>
<div id="controls"><label>View<select id="view"><option value="3d">3D rotate / zoom</option><option value="2d">2D projection</option></select></label>
<label>Show<select id="objects"><option value="both">Points + centroids</option><option value="points">Cluster points only</option><option value="centroids">Cluster centroids only</option></select></label>
<label>X axis<select id="x"></select></label><label>Y axis<select id="y"></select></label><label>Z axis<select id="z"></select></label>
<label>Colour by<select id="colour"></select></label><label>Cluster / category filter<input id="filter" placeholder="e.g. 219,220"></label><button id="reset">Reset camera</button></div>
<div id="status"></div><div id="plot"></div><div id="size-scale"></div>
<script>
const DATA={packed};
const $=id=>document.getElementById(id), nice=c=>DATA.names[c]||c;
function fill(id, cols, selected){{ const s=$(id); cols.forEach(c=>{{let o=document.createElement('option');o.value=c;o.textContent=nice(c);s.append(o)}}); s.value=selected||cols[0]; }}
fill('x',DATA.coords,DATA.defaults.x); fill('y',DATA.coords,DATA.defaults.y); fill('z',DATA.coords,DATA.defaults.z); fill('colour',DATA.colours,DATA.defaults.colour);
$('meta').textContent=`${{DATA.rows.toLocaleString()}} displayed events from ${{DATA.source_rows.toLocaleString()}} input rows · ${{DATA.cluster_col ? 'balanced sample by '+nice(DATA.cluster_col) : 'random sample'}}`;
function hash(s){{let h=2166136261;for(let i=0;i<s.length;i++)h=Math.imul(h^s.charCodeAt(i),16777619);return h>>>0}}
function color(v){{if(String(v)==='-1'||String(v).toLowerCase()==='nan')return '#a0a7b0';let h=hash(String(v))%360;return `hsl(${{h}},63%,47%)`;}}
function filteredIndices(c){{let query=$('filter').value.trim(); if(!query)return DATA.values[c].map((_,i)=>i); let wanted=new Set(query.split(/[ ,]+/));return DATA.values[c].map((v,i)=>wanted.has(String(v))?i:-1).filter(i=>i>=0)}}
function centroidDiameter(n,maxN,is3){{return (is3?3.0:5.0)+(is3?7.0:12.0)*Math.cbrt(n/Math.max(1,maxN));}}
function updateScale(counts,maxN,is3,visible){{const box=$('size-scale');if(!visible||!counts.length){{box.style.display='none';return;}}let s=[Math.min(...counts),counts.slice().sort((a,b)=>a-b)[Math.floor(counts.length/2)],Math.max(...counts)];s=[...new Set(s)];box.innerHTML='<b>Centroid size scale</b>'+s.map(n=>{{let d=centroidDiameter(n,maxN,is3);return `<div class="scale-row"><i class="scale-ball" style="width:${{d}}px;height:${{d}}px"></i><span>N = ${{n.toLocaleString()}}</span></div>`;}}).join('');box.style.display='block';}}
function render(){{
 if(typeof Plotly==='undefined'){{$('status').textContent='Plotly failed to load. This file should be self-contained; regenerate it with build_cluster_visualizer.py.';return;}}
 const x=$('x').value,y=$('y').value,z=$('z').value,c=$('colour').value,objects=$('objects').value,indices=filteredIndices(c), groups={{}};
 indices.forEach(i=>{{let k=String(DATA.values[c][i]);(groups[k]??=[]).push(i)}});
 const is3=$('view').value==='3d'; let traces=[]; updateScale([],1,is3,false);
 if(objects!=='centroids') traces=Object.entries(groups).sort((a,b)=>a[0].localeCompare(b[0],undefined,{{numeric:true}})).map(([k,ids])=>{{
  let base={{name:nice(c)+' '+k,mode:'markers',type:is3?'scatter3d':'scattergl',x:ids.map(i=>DATA.values[x][i]),y:ids.map(i=>DATA.values[y][i]),
  text:ids.map(i=>DATA.hover.map(h=>`${{nice(h)}}: ${{DATA.values[h][i]}}`).join('<br>')),hovertemplate:'%{{text}}<extra></extra>',marker:{{size:is3?1.35:2.1,color:color(k),opacity:0.24}}}};
  if(is3)base.z=ids.map(i=>DATA.values[z][i]); return base; }});
 if(objects!=='points'){{
   const clusterCol=DATA.cluster_col, centroids={{}};
   indices.forEach(i=>{{let key=String(DATA.values[clusterCol][i]); let q=centroids[key]??={{n:0,sx:0,sy:0,sz:0,nx:0,ny:0,nz:0,colours:{{}}}}; let vx=DATA.values[x][i],vy=DATA.values[y][i],vz=DATA.values[z][i]; q.n++; if(vx!==null){{q.sx+=vx;q.nx++}} if(vy!==null){{q.sy+=vy;q.ny++}} if(vz!==null){{q.sz+=vz;q.nz++}} let ck=String(DATA.values[c][i]);q.colours[ck]=(q.colours[ck]||0)+1; }});
   const cs=Object.entries(centroids).filter(([,q])=>q.nx&&q.ny&&(!is3||q.nz)); const totalCounts=cs.map(([id,q])=>DATA.cluster_counts[id]||q.n); const maxN=Math.max(1,...totalCounts); updateScale(totalCounts,maxN,is3,true);
   const byColour={{}}; cs.forEach(([id,q])=>{{let ck=Object.entries(q.colours).sort((a,b)=>b[1]-a[1])[0][0];(byColour[ck]??=[]).push([id,q])}});
   Object.entries(byColour).forEach(([ck,items])=>{{let trace={{name:'centroids: '+nice(c)+' '+ck,mode:'markers',type:is3?'scatter3d':'scattergl',x:items.map(([,q])=>q.sx/q.nx),y:items.map(([,q])=>q.sy/q.ny),text:items.map(([id,q])=>`cluster: ${{id}}<br>events (full input): ${{(DATA.cluster_counts[id]||q.n).toLocaleString()}}<br>events displayed: ${{q.n}}<br>centroid colour: ${{nice(c)}} ${{ck}}`),hovertemplate:'%{{text}}<extra></extra>',marker:{{symbol:'circle',size:items.map(([id,q])=>centroidDiameter(DATA.cluster_counts[id]||q.n,maxN,is3)),color:color(ck),opacity:1,line:{{color:'#172033',width:0.8}}}}}};if(is3)trace.z=items.map(([,q])=>q.sz/q.nz);traces.push(trace);}});
 }}
 const layout={{margin:{{l:55,r:20,t:20,b:55}},showlegend:Object.keys(groups).length<=24,legend:{{itemsizing:'constant'}},uirevision:'axes',paper_bgcolor:'#f8fafc',plot_bgcolor:'#fff'}};
 if(is3)layout.scene={{xaxis:{{title:nice(x),backgroundcolor:'#fff',gridcolor:'#e2e8f0'}},yaxis:{{title:nice(y),backgroundcolor:'#fff',gridcolor:'#e2e8f0'}},zaxis:{{title:nice(z),backgroundcolor:'#fff',gridcolor:'#e2e8f0'}},aspectmode:'data'}};
 else {{layout.xaxis={{title:nice(x),gridcolor:'#e2e8f0',zerolinecolor:'#cbd5e1'}};layout.yaxis={{title:nice(y),gridcolor:'#e2e8f0',zerolinecolor:'#cbd5e1'}};}}
 $('status').textContent=`${{indices.length.toLocaleString()}} events · ${{Object.keys(groups).length}} ${{nice(c)}} categories · ${{objects}} · centroid diameter ∝ N^(1/3), using full-input N · opaque centroids use 3D depth occlusion · coordinates: ${{nice(x)}}, ${{nice(y)}}${{is3?', '+nice(z):''}}`;
 Plotly.react('plot',traces,layout,{{responsive:true,displaylogo:false,scrollZoom:true}});
}}
['view','objects','x','y','z','colour'].forEach(id=>$(id).addEventListener('change',render)); $('filter').addEventListener('input',render);
$('reset').addEventListener('click',()=>{{Plotly.relayout('plot',{{'scene.camera':null,'xaxis.autorange':true,'yaxis.autorange':true}})}}); render();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV containing coordinates and a cluster-label column")
    parser.add_argument("--output", type=Path, default=Path("foil-clusters-3d.html"))
    parser.add_argument("--cluster-column", default="flow_hdbscan_cluster", help="used for balanced sampling")
    parser.add_argument("--sample-per-cluster", type=int, default=220, help="maximum points retained from each cluster")
    parser.add_argument("--max-events", type=int, default=55000, help="hard cap on rendered points (0 disables)")
    parser.add_argument("--plotly-js", type=Path, help="local plotly.min.js to embed for offline use")
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    # Keep coordinates which are available for at least half the data.  Some
    # derived local-field quantities are intentionally undefined for rejected
    # events; Plotly handles those null points per view, so do not discard the
    # rest of the event sample merely because one optional coordinate is null.
    usable_numeric = [c for c in numeric if frame[c].notna().mean() >= 0.50]
    coords = [c for c in PREFERRED_COORDS if c in usable_numeric]
    coords += [c for c in usable_numeric if c not in coords and frame[c].nunique(dropna=True) > 8]
    colours = [c for c in PREFERRED_COLOURS if c in frame.columns]
    colours += [c for c in frame.columns if c.endswith("cluster") and c not in colours]
    if not coords:
        raise ValueError("No numeric coordinate columns found.")
    if not colours:
        raise ValueError("No categorical/cluster columns found; pass a CSV with labels.")
    source_rows = len(frame)
    cluster_counts = (frame[args.cluster_column].value_counts(dropna=False).to_dict()
                      if args.cluster_column in frame else {})
    sampled = frame
    if args.sample_per_cluster > 0 and args.cluster_column in frame:
        # Per-cluster cap is the primary control; max-events remains a safety cap.
        rng = np.random.default_rng(args.seed)
        selected = []
        for _, group in frame.groupby(args.cluster_column, dropna=False, sort=False):
            selected.append(rng.choice(group.index.to_numpy(), min(len(group), args.sample_per_cluster), replace=False))
        sampled = frame.loc[np.concatenate(selected)]
    if args.max_events > 0 and len(sampled) > args.max_events:
        sampled = choose_sample(sampled, args.cluster_column, args.max_events, args.seed)
    sampled = sampled.replace([np.inf, -np.inf], np.nan)
    hover = list(dict.fromkeys([args.cluster_column, "final_relative_foil", "canonical_hole_family", "sieve_x", "sieve_y", "P.gtr.y"] + coords))
    hover = [c for c in hover if c in sampled]
    used = list(dict.fromkeys(coords + colours + hover))
    values = {}
    for col in used:
        series = sampled[col]
        values[col] = [None if pd.isna(v) else (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else str(v)) for v in series]
    defaults = {
        "x": "flow_z1" if "flow_z1" in coords else coords[0],
        "y": "flow_z2" if "flow_z2" in coords else coords[min(1, len(coords)-1)],
        "z": "global_foil_flattened_z3" if "global_foil_flattened_z3" in coords else ("flow_z3" if "flow_z3" in coords else coords[min(2, len(coords)-1)]),
        "colour": args.cluster_column if args.cluster_column in colours else colours[0],
    }
    data = {"values": values, "coords": coords, "colours": colours, "hover": hover, "names": {c: display_name(c) for c in used}, "defaults": defaults, "rows": len(sampled), "source_rows": source_rows, "cluster_col": args.cluster_column if args.cluster_column in sampled else None, "cluster_counts": {str(k): int(v) for k, v in cluster_counts.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plotly_js = args.plotly_js.read_text(encoding="utf-8") if args.plotly_js else None
    args.output.write_text(build_html(data, f"Cluster explorer: {args.input.stem}", plotly_js), encoding="utf-8")
    print(f"Wrote {args.output} with {len(sampled):,} displayed events from {source_rows:,} input rows.")


if __name__ == "__main__":
    main()
