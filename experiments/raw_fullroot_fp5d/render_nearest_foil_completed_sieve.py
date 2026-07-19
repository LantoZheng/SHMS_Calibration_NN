"""Complete foil assignment from local cross-foil relations in flow Z.

Canonical families with two or more already-established foil layers act as
local anchors.  For an unmatched cluster, candidate anchors are compared in a
plane transverse to the learned foil-translation vector; its layer is inferred
from its *relative* signed displacement along that vector.  Thus an assignment
uses nearby same-position clusters on other foil sheets rather than an absolute
location or a generic nearest neighbour in all of Z.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent; OUT=HERE/'results'
SRC=OUT/'nonlinear_canonical_z_mapping_labels.csv'; Z=['flow_z1','flow_z2','flow_z3']; SEED=25521

def colour(label:int): return plt.get_cmap('hsv')((int(label)*0.61803398875)%1.0)


def relational_completion(c: pd.DataFrame, x: np.ndarray, known: np.ndarray):
 """Assign unmatched clusters by local multi-foil anchor families.

 ``t`` is the data-driven approximate foil direction discovered by the
 relative-Z Hough/translation stage.  A family is eligible only when it
 contains anchors on at least two distinct, already-classified foil layers.
 """
 summary=json.loads((OUT/'relative_z_lattice_mapping_summary.json').read_text(encoding='utf-8'))
 t=np.asarray(summary['translation_normalized'],float); t2=float(t@t)
 if not np.isfinite(t2) or t2 <= 0: raise ValueError('Invalid relative foil translation vector')
 layers=c.canonical_layer.to_numpy(int); families=c.canonical_hole.to_numpy()
 ids=c.linear_local_field_cluster.to_numpy(int)
 family_indices={}
 for family in np.unique(families[known]):
  idx=np.flatnonzero(known & (families==family))
  if len(np.unique(layers[idx])) >= 2: family_indices[int(family)]=idx

 final=layers.copy(); source=np.full(len(c),'canonical_relative_mapping',object)
 reference=np.full(len(c),'',object); relation_distance=np.full(len(c),np.nan)
 relation_family=np.full(len(c),-1,int); fallback_distance=np.full(len(c),np.nan)
 unresolved=[]
 for u in np.flatnonzero(~known):
  candidates=[]
  for family, anchors in family_indices.items():
   delta=x[u]-x[anchors]
   steps=(delta@t)/t2
   # Each anchor votes for the layer reached by moving from it toward u.
   votes=layers[anchors]+np.rint(steps).astype(int)
   valid=(votes>=0)&(votes<=2)&(np.abs(steps-np.rint(steps))<.42)
   if valid.sum()<2: continue
   vote_values, vote_counts=np.unique(votes[valid],return_counts=True)
   foil=int(vote_values[np.argmax(vote_counts)])
   if vote_counts.max()<2: continue
   transverse=np.linalg.norm(delta-steps[:,None]*t,axis=1)
   step_error=np.abs(steps-np.rint(steps))
   # A robust family score: same transverse position is primary; consistency
   # of the discrete cross-foil displacement regularises the decision.
   score=float(np.median(transverse[valid])+.35*np.median(step_error[valid]))
   agreeing=anchors[valid & (votes==foil)]
   candidates.append((score,foil,family,agreeing))
  if candidates:
   score,foil,family,agreeing=min(candidates,key=lambda item:item[0])
   final[u]=foil; source[u]='local_cross_foil_relative_relation'
   relation_distance[u]=score; relation_family[u]=family
   reference[u]=';'.join(map(str,ids[agreeing]))
  else:
   unresolved.append(u)

 # Rare edge/outlier fallback retains full coverage, but is explicitly marked
 # as such so it cannot be confused with the cross-foil relational result.
 if unresolved:
  tree=cKDTree(x[known]); dist, idx=tree.query(x[unresolved],k=1); known_idx=np.flatnonzero(known); target=known_idx[idx]
  final[unresolved]=layers[target]; source[unresolved]='nearest_classified_cluster_fallback'
  reference[unresolved]=ids[target].astype(str); fallback_distance[unresolved]=dist
 return final,source,reference,relation_distance,relation_family,fallback_distance,len(family_indices)

def main():
 d=pd.read_csv(SRC); active=d[d.linear_local_field_cluster>=0].copy()
 c=active.groupby('linear_local_field_cluster').agg(events=('linear_local_field_cluster','size'),sieve_x=('sieve_x','median'),sieve_y=('sieve_y','median'),ytar=('P.gtr.y','median'),canonical_hole=('canonical_hole_family','median'),canonical_layer=('canonical_foil_layer','median'),**{z:(z,'median') for z in Z}).reset_index()
 raw=c[Z].to_numpy(); x=StandardScaler().fit_transform(raw); known=np.isfinite(c.canonical_hole)&(c.canonical_hole>=0); unknown=~known
 final,source,reference,relation_distance,relation_family,fallback_distance,n_families=relational_completion(c,x,known)
 assign=pd.DataFrame({'linear_local_field_cluster':c.linear_local_field_cluster.astype(int),'final_relative_foil':final,'foil_assignment_method':source,'cross_foil_reference_clusters':reference,'cross_foil_relation_distance':relation_distance,'cross_foil_anchor_family':relation_family,'nearest_fallback_distance_Z':fallback_distance})
 d=d.drop(columns=['final_relative_foil','foil_assignment_method','nearest_reference_cluster','nearest_reference_distance_Z','cross_foil_reference_clusters','cross_foil_relation_distance','cross_foil_anchor_family','nearest_fallback_distance_Z'],errors='ignore').merge(assign,on='linear_local_field_cluster',how='left');d.to_csv(OUT/'nearest_foil_completed_labels.csv',index=False)
 relational=(source=='local_cross_foil_relative_relation'); fallback=(source=='nearest_classified_cluster_fallback')
 summary={'classified_clusters':int(len(c)),'canonical_relative_mapping_clusters':int(known.sum()),'local_cross_foil_completed_clusters':int(relational.sum()),'nearest_fallback_completed_clusters':int(fallback.sum()),'eligible_multi_foil_anchor_families':int(n_families),'remaining_unclassified_clusters':0,'cross_foil_relation_distance_median':float(np.nanmedian(relation_distance)) if relational.any() else None,'cross_foil_relation_distance_max':float(np.nanmax(relation_distance)) if relational.any() else None,'relational_assignments':[{'cluster':int(c.linear_local_field_cluster.iloc[u]),'foil':int(final[u]),'anchor_family':int(relation_family[u]),'anchor_clusters':reference[u],'distance':float(relation_distance[u])} for u in np.flatnonzero(relational)]}
 (OUT/'nearest_foil_completed_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 fig,axes=plt.subplots(1,3,figsize=(22,7.4),sharex=True,sharey=True,constrained_layout=True)
 xmin,xmax=active.sieve_x.quantile([.001,.999]);ymin,ymax=active.sieve_y.quantile([.001,.999])
 for foil,ax in enumerate(axes):
  cluster_ids=c.loc[final==foil,'linear_local_field_cluster'].to_numpy(int);part=active[active.linear_local_field_cluster.isin(cluster_ids)]
  draw=part.sample(min(50000,len(part)),random_state=SEED+foil);labs=draw.linear_local_field_cluster.to_numpy(int)
  ax.scatter(draw.sieve_x,draw.sieve_y,c=[colour(v) for v in labs],s=.62,alpha=.62,linewidths=0,rasterized=True)
  centers=part.groupby('linear_local_field_cluster').agg(x=('sieve_x','median'),y=('sieve_y','median'),n=('linear_local_field_cluster','size'))
  ax.scatter(centers.x,centers.y,c='red',s=7,zorder=4,linewidths=0)
  for label,row in centers.iterrows(): ax.text(row.x+.08,row.y+.07,str(int(label)),color='red',fontsize=5.2,fontweight='bold',zorder=5)
  completed=int(((final==foil)&relational).sum()); fallback_completed=int(((final==foil)&fallback).sum())
  ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),xlabel=r'reconstructed $x_{sieve}$',title=f'relative foil {foil}: {len(centers)} clusters, {len(part):,} events\nlocal cross-foil completion: {completed}; fallback: {fallback_completed}');ax.grid(alpha=.15)
 axes[0].set_ylabel(r'reconstructed $y_{sieve}$')
 fig.suptitle('Final foil-separated clusters: relative Z mapping + local cross-foil completion',fontsize=16,fontweight='bold')
 fig.savefig(OUT/'nearest_foil_completed_three_panel_sieve.png',dpi=220)
if __name__=='__main__':main()
