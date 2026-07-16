"""Warp foil layers into a shared canonical hole plane and rematch clusters.

The initial relative-Z translation graph supplies seed correspondences.  A
quadratic, regularized map removes position-dependent optical warp for foil
layers 0 and 2 relative to layer 1.  Hole identities are then obtained by
global one-to-one assignment in the canonical plane.  Sieve coordinates are
strictly excluded from inference and used only in the diagnostic plot.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.semi_supervised import LabelSpreading
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

HERE=Path(__file__).resolve().parent; OUT=HERE/'results'
SRC=OUT/'relative_z_lattice_mapping_labels.csv'; Z=['flow_z1','flow_z2','flow_z3']; SEED=25521

class UF:
 def __init__(self,n): self.p=list(range(n))
 def find(self,x):
  while self.p[x]!=x: self.p[x]=self.p[self.p[x]];x=self.p[x]
  return x
 def union(self,a,b):
  a,b=self.find(a),self.find(b)
  if a!=b:self.p[max(a,b)]=min(a,b)

def main():
 d=pd.read_csv(SRC); a=d[d.linear_local_field_cluster>=0].copy()
 c=a.groupby('linear_local_field_cluster').agg(events=('linear_local_field_cluster','size'),sieve_x=('sieve_x','median'),sieve_y=('sieve_y','median'),ytar=('P.gtr.y','median'),old_hole=('relative_hole_family','median'),old_layer=('relative_foil_layer','median'),**{z:(z,'median') for z in Z}).reset_index()
 ids=c.linear_local_field_cluster.to_numpy(int); raw=c[Z].to_numpy(); scaler=StandardScaler().fit(raw); x=scaler.transform(raw); n=len(c)
 seed=np.isfinite(c.old_layer)&(c.old_layer>=0); y=c.loc[seed,'old_layer'].to_numpy(int)
 layer_model=LabelSpreading(kernel='knn',n_neighbors=15,alpha=.20,max_iter=100); cv=StratifiedKFold(5,shuffle=True,random_state=SEED)
 oof=cross_val_predict(layer_model,x[seed],y,cv=cv); layer_cv=float(balanced_accuracy_score(y,oof))
 semi_labels=np.full(n,-1,int);semi_labels[seed]=y;layer_model.fit(x,semi_labels)
 prob=layer_model.label_distributions_;layer=layer_model.classes_[prob.argmax(1)].astype(int);layer_conf=prob.max(1)
 # Keep relationally established layers immutable; infer only previous unmatched nodes.
 layer[seed]=y
 xy=x[:,:2]; canonical=xy.copy(); seed_family=c.old_hole.to_numpy()
 models={}; training_errors=[]
 for source_layer in (0,2):
  src=[];ref=[]
  for family in np.unique(seed_family[np.isfinite(seed_family)]):
   members=np.flatnonzero(seed_family==family)
   s=members[layer[members]==source_layer]; r=members[layer[members]==1]
   if len(s)==1 and len(r)==1: src.append(xy[s[0]]);ref.append(xy[r[0]])
  src=np.asarray(src);ref=np.asarray(ref)
  model=make_pipeline(PolynomialFeatures(2,include_bias=False),StandardScaler(),Ridge(alpha=.35)).fit(src,ref)
  models[source_layer]=model; prediction=model.predict(src);training_errors.extend(np.linalg.norm(prediction-ref,axis=1))
  mask=layer==source_layer;canonical[mask]=model.predict(xy[mask])
 # Adaptive scale comes from the canonical lattice spacing, not an absolute coordinate window.
 ref_idx=np.flatnonzero(layer==1); spacing=float(np.median(cKDTree(canonical[ref_idx]).query(canonical[ref_idx],k=2)[0][:,1]))
 seed_error=float(np.quantile(training_errors,.90)); threshold=max(seed_error*2.5,spacing*.62)
 uf=UF(n); matched_edges=[]; used=set()
 def assign(left,right,tag):
  if not len(left) or not len(right):return
  cost=np.linalg.norm(canonical[left,None,:]-canonical[right][None,:,:],axis=2); rr,cc=linear_sum_assignment(cost)
  for i,j in zip(rr,cc):
   if cost[i,j]<=threshold:
    u,v=int(left[i]),int(right[j]);uf.union(u,v);matched_edges.append((u,v,float(cost[i,j]),tag));used.add(u);used.add(v)
 assign(np.flatnonzero(layer==0),ref_idx,'0-1');assign(np.flatnonzero(layer==2),ref_idx,'2-1')
 # Recover holes absent in the reference layer by matching remaining outer layers.
 left=np.array([i for i in np.flatnonzero(layer==0) if i not in used]);right=np.array([i for i in np.flatnonzero(layer==2) if i not in used]);assign(left,right,'0-2')
 groups={}
 for i in range(n):groups.setdefault(uf.find(i),[]).append(i)
 families=[g for g in groups.values() if len(g)>=2 and len(set(layer[g]))==len(g)]
 hole=np.full(n,-1,int)
 for h,g in enumerate(families):hole[g]=h
 lookup=pd.DataFrame({'linear_local_field_cluster':ids,'canonical_foil_layer':layer,'foil_layer_probability':layer_conf,'canonical_z1':canonical[:,0],'canonical_z2':canonical[:,1],'canonical_hole_family':hole})
 d=d.drop(columns=['canonical_foil_layer','foil_layer_probability','canonical_z1','canonical_z2','canonical_hole_family'],errors='ignore').merge(lookup,on='linear_local_field_cluster',how='left');d.to_csv(OUT/'nonlinear_canonical_z_mapping_labels.csv',index=False)
 fam=[]
 for h,g in enumerate(families):fam.append({'canonical_hole_family':h,'clusters':';'.join(map(str,ids[g])),'layers':';'.join(map(str,layer[g])),'members':len(g),'canonical_spread':float(np.max(np.linalg.norm(canonical[g]-canonical[g].mean(0),axis=1)))})
 pd.DataFrame(fam).to_csv(OUT/'nonlinear_canonical_z_hole_families.csv',index=False)
 summary={'inference':'relative-Z seeds -> graph label spreading -> quadratic layer warps -> global one-to-one canonical matching','sieve_coordinates_used_in_inference':False,'previous_matched_clusters':int(seed.sum()),'previous_unmatched_clusters':int((~seed).sum()),'foil_layer_classifier':'15-neighbour LabelSpreading in Z','foil_layer_classifier_oof_balanced_accuracy':layer_cv,'canonical_reference_layer':1,'quadratic_seed_pair_error_q90':seed_error,'canonical_median_lattice_spacing':spacing,'adaptive_match_threshold':threshold,'accepted_edges':len(matched_edges),'canonical_hole_families':len(families),'canonical_matched_clusters':int((hole>=0).sum()),'canonical_unmatched_clusters':int((hole<0).sum())}
 (OUT/'nonlinear_canonical_z_mapping_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 pal=['#356bb4','#e98b27','#35a16f'];fig,ax=plt.subplots(2,2,figsize=(16,13),constrained_layout=True)
 old=np.isfinite(c.old_layer)&(c.old_layer>=0)
 for f in range(3):
  m=old&(c.old_layer.to_numpy()==f);ax[0,0].scatter(x[m,0],x[m,1],c=pal[f],s=np.clip(c.events.to_numpy()[m]/5,14,100),alpha=.8,label=f'layer {f}')
 ax[0,0].scatter(x[~old,0],x[~old,1],c='0.72',s=np.clip(c.events.to_numpy()[~old]/5,14,100),alpha=.75,label=f'unmatched ({(~old).sum()})');ax[0,0].set(title='Before: global-translation matching in Z',xlabel='normalised $Z_1$',ylabel='normalised $Z_2$');ax[0,0].legend(frameon=False);ax[0,0].grid(alpha=.15)
 for u,v,_,_ in matched_edges:ax[0,1].plot(canonical[[u,v],0],canonical[[u,v],1],c='.72',lw=.45,zorder=1)
 for f in range(3):
  m=(layer==f)&(hole>=0);ax[0,1].scatter(canonical[m,0],canonical[m,1],c=pal[f],s=np.clip(c.events.to_numpy()[m]/5,14,100),alpha=.82,label=f'layer {f}')
 ax[0,1].scatter(canonical[hole<0,0],canonical[hole<0,1],c='0.72',s=np.clip(c.events.to_numpy()[hole<0]/5,14,100),label=f'unmatched ({(hole<0).sum()})');ax[0,1].set(title='After: common canonical hole plane',xlabel='canonical $Z_1$',ylabel='canonical $Z_2$');ax[0,1].legend(frameon=False);ax[0,1].grid(alpha=.15)
 for panel,mask,title in [(ax[1,0],old,'Blind sieve check before remapping'),(ax[1,1],hole>=0,'Blind sieve check after remapping')]:
  for f in range(3):
   m=mask&((c.old_layer.to_numpy()==f) if panel is ax[1,0] else (layer==f));panel.scatter(c.sieve_x[m],c.sieve_y[m],c=pal[f],s=np.clip(c.events.to_numpy()[m]/5,14,100),alpha=.82)
  panel.scatter(c.sieve_x[~mask],c.sieve_y[~mask],c='.72',s=np.clip(c.events.to_numpy()[~mask]/5,14,100),alpha=.75);panel.set(title=title,xlabel='reconstructed $x_{sieve}$',ylabel='reconstructed $y_{sieve}$');panel.grid(alpha=.15)
 fig.suptitle('Nonlinear canonical Z remapping recovers edge clusters without absolute sieve cuts',fontsize=16,fontweight='bold');fig.savefig(OUT/'nonlinear_canonical_z_remapping.png',dpi=220)
if __name__=='__main__':main()
