"""Infer foil/hole correspondence from repeated relative displacements in flow Z.

No sieve coordinate enters inference.  Cluster prototypes are matched by a
recurrent near-depth translation in Z=(flow_z1,flow_z2,flow_z3); connected
translation chains define relative hole families and their ordered foil layers.
Sieve coordinates appear only in the final blind diagnostic plot.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

HERE=Path(__file__).resolve().parent; OUT=HERE/'results'; SRC=OUT/'linear_local_field_pipeline_labels.csv'
ZCOL=['flow_z1','flow_z2','flow_z3']; EPS=.22; RNG=np.random.default_rng(25521)

def main():
 d=pd.read_csv(SRC); active=d[d.linear_local_field_cluster>=0].copy()
 c=active.groupby('linear_local_field_cluster').agg(events=('linear_local_field_cluster','size'),sieve_x=('sieve_x','median'),sieve_y=('sieve_y','median'),ytar=('P.gtr.y','median'),**{z:(z,'median') for z in ZCOL})
 ids=c.index.to_numpy(int); raw=c[ZCOL].to_numpy(); x=(raw-np.median(raw,0))/np.std(raw,0); tree=cKDTree(x)
 # Hough-style search: repeated, nearly depth-parallel translations are foil candidates.
 dif=(x[None,:,:]-x[:,None,:]).reshape(-1,3); norm=np.linalg.norm(dif,axis=1)
 cand=dif[(norm>.5)&(dif[:,2]>0)&(np.abs(dif[:,2])/norm>.75)]
 cand=cand[RNG.choice(len(cand),min(12000,len(cand)),replace=False)]
 modes=[]
 for t in cand:
  dist,_=tree.query(x+t); score=int((dist<EPS).sum())
  if all(np.linalg.norm(t-u[1])>.16 for u in modes): modes.append((score,t))
 modes=sorted(modes,key=lambda a:a[0],reverse=True)
 score,t=modes[0]
 # One-to-one correspondence and robust translation refinement.
 for _ in range(2):
  dist,j=tree.query(x+t); order=np.argsort(dist); used=set(); edges=[]
  for i in order:
   if dist[i]<EPS and int(j[i]) not in used:
    edges.append((int(i),int(j[i]),float(dist[i]))); used.add(int(j[i]))
  t=np.median(np.array([x[j]-x[i] for i,j,_ in edges]),axis=0)
 # Connected components of the translation graph; no sieve values used here.
 adj={i:set() for i in range(len(ids))}; out={i:set() for i in range(len(ids))}
 for i,j,_ in edges: adj[i].add(j);adj[j].add(i);out[i].add(j)
 seen=set(); comps=[]
 for start in range(len(ids)):
  if start in seen or not adj[start]: continue
  stack=[start];seen.add(start);cc=[]
  while stack:
   u=stack.pop();cc.append(u)
   for v in adj[u]:
    if v not in seen:seen.add(v);stack.append(v)
  comps.append(sorted(cc))
 hole=np.full(len(ids),-1,int); foil=np.full(len(ids),-1,int)
 valid=[]
 for h,cc in enumerate(comps):
  if len(cc)>3: continue
  proj=x[cc]@t; rank=np.argsort(np.argsort(proj)); hole[cc]=h; foil[cc]=rank; valid.append(cc)
 lookup=pd.DataFrame({'linear_local_field_cluster':ids,'relative_hole_family':hole,'relative_foil_layer':foil})
 d=d.merge(lookup,on='linear_local_field_cluster',how='left'); d.to_csv(OUT/'relative_z_lattice_mapping_labels.csv',index=False)
 rows=[]
 for ci,cc in enumerate(valid):
  rows.append({'relative_hole_family':ci,'clusters':';'.join(map(str,ids[cc])),'matched_foil_layers':';'.join(map(str,foil[cc])),'members':len(cc)})
 pd.DataFrame(rows).to_csv(OUT/'relative_z_lattice_hole_families.csv',index=False)
 summary={'inference_input':'cluster median flow_z1,z2,z3 only; sieve_x/sieve_y excluded','z_normalization':'robust centre standard deviation','translation_score':score,'translation_normalized':t.tolist(),'matching_tolerance':EPS,'one_to_one_translation_edges':len(edges),'translation_components':len(comps),'valid_relative_hole_families_size_2_or_3':len(valid),'clusters_in_relative_families':int((hole>=0).sum())}
 (OUT/'relative_z_lattice_mapping_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 # Blind display: Z graph first; sieve projection only validates the inferred graph.
 fig,ax=plt.subplots(1,2,figsize=(16,7),constrained_layout=True)
 for i,j,_ in edges:
  if hole[i]>=0 and hole[j]>=0: ax[0].plot([x[i,0],x[j,0]],[x[i,2],x[j,2]],c='0.65',lw=.45,zorder=1)
 cols=np.where(foil>=0,foil,3); pal=['#356bb4','#e98b27','#35a16f','#bdbdbd']
 for f in range(4):
  m=cols==f;ax[0].scatter(x[m,0],x[m,2],s=np.clip(c.events.to_numpy()[m]/5,15,120),c=pal[f],label=('relative foil '+str(f) if f<3 else 'unmatched'),alpha=.82,edgecolors='none')
 ax[0].set(xlabel='normalised $Z_1$',ylabel='normalised $Z_3$',title='Relative translation graph in flow $Z$ space');ax[0].legend(frameon=False);ax[0].grid(alpha=.15)
 for f in range(4):
  m=cols==f;ax[1].scatter(c.sieve_x.to_numpy()[m],c.sieve_y.to_numpy()[m],s=np.clip(c.events.to_numpy()[m]/5,15,120),c=pal[f],label=('relative foil '+str(f) if f<3 else 'unmatched'),alpha=.82,edgecolors='black',linewidths=.15)
 ax[1].set(xlabel='reconstructed $x_{sieve}$',ylabel='reconstructed $y_{sieve}$',title='Blind evaluation in reconstructed sieve plane');ax[1].grid(alpha=.15)
 fig.suptitle('Relative Z-space lattice matching: foil/hole identities from repeated cluster displacements',fontsize=15,fontweight='bold')
 fig.savefig(OUT/'relative_z_lattice_mapping.png',dpi=220)
if __name__=='__main__':main()
