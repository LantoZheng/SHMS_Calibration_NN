"""Complete foil assignment by nearest classified cluster in Z and plot sieve panels."""
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

def main():
 d=pd.read_csv(SRC); active=d[d.linear_local_field_cluster>=0].copy()
 c=active.groupby('linear_local_field_cluster').agg(events=('linear_local_field_cluster','size'),sieve_x=('sieve_x','median'),sieve_y=('sieve_y','median'),ytar=('P.gtr.y','median'),canonical_hole=('canonical_hole_family','median'),canonical_layer=('canonical_foil_layer','median'),**{z:(z,'median') for z in Z}).reset_index()
 raw=c[Z].to_numpy(); x=StandardScaler().fit_transform(raw); known=np.isfinite(c.canonical_hole)&(c.canonical_hole>=0); unknown=~known
 final=c.canonical_layer.to_numpy(int); source=np.full(len(c),'canonical_relative_mapping',object); nearest_cluster=np.full(len(c),-1,int); nearest_distance=np.full(len(c),np.nan)
 tree=cKDTree(x[known]); dist,idx=tree.query(x[unknown],k=1); known_idx=np.flatnonzero(known); target=known_idx[idx]
 final[unknown]=c.canonical_layer.to_numpy(int)[target]; source[unknown]='nearest_classified_cluster_in_Z'; nearest_cluster[unknown]=c.linear_local_field_cluster.to_numpy(int)[target];nearest_distance[unknown]=dist
 assign=pd.DataFrame({'linear_local_field_cluster':c.linear_local_field_cluster.astype(int),'final_relative_foil':final,'foil_assignment_method':source,'nearest_reference_cluster':nearest_cluster,'nearest_reference_distance_Z':nearest_distance})
 d=d.drop(columns=['final_relative_foil','foil_assignment_method','nearest_reference_cluster','nearest_reference_distance_Z'],errors='ignore').merge(assign,on='linear_local_field_cluster',how='left');d.to_csv(OUT/'nearest_foil_completed_labels.csv',index=False)
 summary={'classified_clusters':int(len(c)),'canonical_relative_mapping_clusters':int(known.sum()),'nearest_neighbour_completed_clusters':int(unknown.sum()),'remaining_unclassified_clusters':0,'nearest_distance_median':float(np.median(dist)),'nearest_distance_max':float(np.max(dist)),'nearest_assignments':[{'cluster':int(c.linear_local_field_cluster.iloc[u]),'foil':int(final[u]),'nearest_cluster':int(nearest_cluster[u]),'distance_Z':float(nearest_distance[u])} for u in np.flatnonzero(unknown)]}
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
  completed=int(((final==foil)&unknown).sum())
  ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),xlabel=r'reconstructed $x_{sieve}$',title=f'relative foil {foil}: {len(centers)} clusters, {len(part):,} events\nnearest-Z completion: {completed} clusters');ax.grid(alpha=.15)
 axes[0].set_ylabel(r'reconstructed $y_{sieve}$')
 fig.suptitle('Final foil-separated linear-local-field clusters: relative Z mapping + nearest-neighbour completion',fontsize=16,fontweight='bold')
 fig.savefig(OUT/'nearest_foil_completed_three_panel_sieve.png',dpi=220)
if __name__=='__main__':main()
