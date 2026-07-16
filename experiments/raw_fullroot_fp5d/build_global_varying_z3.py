"""Construct a continuous global z3 with coefficients varying over (z1,z2).

The model is linear in robust-scaled FP5D at every location, but its intercept
and normal vector are a smooth partition-of-unity blend of RBF experts over the
global flow (z1,z2) plane.  Relative foil labels provide only the chart gauge;
no reconstructed ytar value is used in fitting.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, StandardScaler

HERE=Path(__file__).resolve().parent; OUT=HERE/'results'; SRC=OUT/'nearest_foil_completed_labels.csv'
FP=['P.dc.x_fp','P.dc.y_fp','P.dc.xp_fp','P.dc.yp_fp','P.rb.raster.frybRawAdc']; U=['flow_z1','flow_z2']; SEED=25521

def basis(u,centres,width):
 d2=((u[:,None,:]-centres[None,:,:])**2).sum(2);p=np.exp(-.5*d2/(width**2));return p/(p.sum(1,keepdims=True)+1e-12)
def design(u,f,centres,width):
 p=basis(u,centres,width);aug=np.column_stack([np.ones(len(f)),f]);return (p[:,:,None]*aug[:,None,:]).reshape(len(f),-1)
def dprime(v,y):
 a,b=v[y==0],v[y==1];return float(abs(a.mean()-b.mean())/np.sqrt(.5*(a.var(ddof=1)+b.var(ddof=1))+1e-12))

def main():
 d=pd.read_csv(SRC);a=d[d.linear_local_field_cluster>=0].copy();cluster='linear_local_field_cluster'
 fp_scaler=RobustScaler(quantile_range=(5,95)).fit(a[FP]);a_fp=fp_scaler.transform(a[FP]);u_scaler=StandardScaler().fit(a[U]);a_u=u_scaler.transform(a[U])
 # Equal-weight cluster prototypes prevent large clusters from defining the gauge.
 proto=a.assign(**{f'F{i}':a_fp[:,i] for i in range(5)},U0=a_u[:,0],U1=a_u[:,1]).groupby(cluster).agg(**{f'F{i}':(f'F{i}','median') for i in range(5)},U0=('U0','median'),U1=('U1','median'),foil=('final_relative_foil','median'),oldz=('flow_z3','median'),events=(cluster,'size')).reset_index()
 pu=proto[['U0','U1']].to_numpy();pf=proto[[f'F{i}' for i in range(5)]].to_numpy();target=proto.foil.to_numpy(float)-1.0
 folds=KFold(5,shuffle=True,random_state=SEED);trials=[]
 for k in (9,16,25,36):
  centres=KMeans(k,random_state=SEED,n_init=20).fit(pu).cluster_centers_;width=float(np.median(cKDTree(centres).query(centres,k=2)[0][:,1])*1.45);X=design(pu,pf,centres,width)
  for alpha in (.03,.1,.3,1.,3.,10.):
   pred=np.empty(len(proto))
   for tr,te in folds.split(X):pred[te]=Ridge(alpha=alpha).fit(X[tr],target[tr]).predict(X[te])
   cls=np.clip(np.rint(pred+1),0,2);trials.append((float(np.mean(abs(pred-target))),float(accuracy_score(proto.foil,cls)),k,alpha,width,centres))
 best=min(trials,key=lambda q:(q[0],-q[1]));mae,cv_acc,k,alpha,width,centres=best;model=Ridge(alpha=alpha).fit(design(pu,pf,centres,width),target)
 # Event-level application in chunks keeps memory bounded.
 zg=np.empty(len(a))
 for lo in range(0,len(a),20000):
  hi=min(lo+20000,len(a));zg[lo:hi]=model.predict(design(a_u[lo:hi],a_fp[lo:hi],centres,width))
 a['global_varying_z3']=zg
 out=d.merge(a[[cluster,'global_varying_z3']].reset_index().rename(columns={'index':'_event_index'}),left_index=True,right_on='_event_index',how='left').sort_values('_event_index').drop(columns='_event_index')
 out.to_csv(OUT/'global_varying_z3_labels.csv',index=False)
 # Evaluate close cross-foil cluster pairs using event distributions.
 pc=a.groupby(cluster).agg(u1=('flow_z1','median'),u2=('flow_z2','median'),foil=('final_relative_foil','median'),events=(cluster,'size'))
 uu=StandardScaler().fit_transform(pc[['u1','u2']]);pairs=[]
 for i in range(len(pc)):
  for j in range(i+1,len(pc)):
   if pc.foil.iloc[i]==pc.foil.iloc[j]:continue
   du=float(np.linalg.norm(uu[i]-uu[j]))
   if du>.30:continue
   ca,cb=int(pc.index[i]),int(pc.index[j]);part=a[a[cluster].isin([ca,cb])];yy=(part[cluster].to_numpy()==cb).astype(int)
   pairs.append({'cluster_a':ca,'cluster_b':cb,'distance_z1z2':du,'events':len(part),'dprime_original_flow_z3':dprime(part.flow_z3.to_numpy(),yy),'dprime_global_varying_z3':dprime(part.global_varying_z3.to_numpy(),yy)})
 pairdf=pd.DataFrame(pairs).sort_values('distance_z1z2');pairdf.to_csv(OUT/'global_varying_z3_close_pair_metrics.csv',index=False)
 summary={'definition':'z3(u,f)=beta0(u)+beta(u)^T robust_scaled(FP5D), beta is normalized-RBF partition-of-unity over u=(flow_z1,flow_z2)','training_target':'relative foil layer minus one; reconstructed ytar excluded','rbf_experts':int(k),'rbf_width':width,'ridge_alpha':alpha,'cluster_level_5fold_mae':mae,'cluster_level_5fold_nearest_layer_accuracy':cv_acc,'close_cross_foil_pairs':len(pairdf),'median_dprime_original_flow_z3':float(pairdf.dprime_original_flow_z3.median()),'median_dprime_global_varying_z3':float(pairdf.dprime_global_varying_z3.median()),'fraction_pairs_improved':float((pairdf.dprime_global_varying_z3>pairdf.dprime_original_flow_z3).mean())}
 (OUT/'global_varying_z3_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 # Compact diagnostics: global continuity, side view, and pairwise gain.
 protog=a.groupby(cluster).agg(z1=('flow_z1','median'),z2=('flow_z2','median'),z3=('global_varying_z3','median'),foil=('final_relative_foil','median'),events=(cluster,'size'))
 pal=['#356bb4','#e98b27','#35a16f'];fig=plt.figure(figsize=(17,12),constrained_layout=True);ax3=fig.add_subplot(2,2,1,projection='3d');ax2=fig.add_subplot(2,2,2);axh=fig.add_subplot(2,2,3);axp=fig.add_subplot(2,2,4)
 for f in range(3):
  q=protog[protog.foil==f];s=np.clip(q.events/5,14,100);ax3.scatter(q.z1,q.z2,q.z3,s=s,c=pal[f],alpha=.82,label=f'foil {f}');ax2.scatter(q.z1,q.z3,s=s,c=pal[f],alpha=.82);vals=a.loc[a.final_relative_foil==f,'global_varying_z3'];axh.hist(vals,bins=60,density=True,histtype='step',lw=1.4,color=pal[f],label=f'foil {f}')
 ax3.set(xlabel='$z_1$',ylabel='$z_2$',zlabel='$z_3^{global}$',title='Global continuous chart');ax3.view_init(22,-58);ax3.legend(frameon=False)
 ax2.set(xlabel='$z_1$',ylabel='$z_3^{global}$',title='Side view');ax2.grid(alpha=.15)
 axh.set(xlabel='$z_3^{global}$',ylabel='density',title='Event-level global coordinate');axh.legend(frameon=False);axh.grid(alpha=.15)
 if len(pairdf):
  axp.scatter(pairdf.dprime_original_flow_z3,pairdf.dprime_global_varying_z3,c=pairdf.distance_z1z2,cmap='viridis',s=22,alpha=.78);lim=max(1,float(pairdf[['dprime_original_flow_z3','dprime_global_varying_z3']].quantile(.98).max()));axp.plot([0,lim],[0,lim],c='crimson',lw=1);axp.set(xlim=(0,lim),ylim=(0,lim))
 axp.set(xlabel="Fisher d' in original flow $z_3$",ylabel="Fisher d' in global varying $z_3$",title=f'Close cross-foil pairs ($\\Delta z_{{1,2}}<0.30$), n={len(pairdf)}');axp.grid(alpha=.15)
 fig.suptitle('A continuous global $z_3$ with a smoothly varying local FP5D normal field',fontsize=16,fontweight='bold');fig.savefig(OUT/'global_varying_z3_diagnostics.png',dpi=220)
if __name__=='__main__':main()
