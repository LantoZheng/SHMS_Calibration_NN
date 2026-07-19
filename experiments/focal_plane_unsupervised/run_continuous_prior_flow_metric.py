"""Full-5D invertible metric with raster conditioning and continuous weak optics targets.

No existing cluster, mechanical-hole centre, foil centre, row, or column is
used in fitting.  The only weak targets are per-event continuous sieve_x/y and
P_gtr_y from the current optics reconstruction.  Complete reference holes are
held out solely for evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import HDBSCAN
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, mean_squared_error, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, SplineTransformer, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

OPTICAL = ["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp"]
RASTER = "P_rb_raster_frybRawAdc"
FEATURES = OPTICAL + [RASTER]
TARGET = ["sieve_x", "sieve_y", "P_gtr_y"]
SEED = 25521


class Coupling(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 10))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, inverse=False):
        kept = x * self.mask
        log_s, t = self.net(kept).chunk(2, dim=1)
        log_s = .65 * torch.tanh(log_s) * (1-self.mask); t = t * (1-self.mask)
        return kept + (1-self.mask) * ((x-t)*torch.exp(-log_s) if inverse else x*torch.exp(log_s)+t)


class Flow(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Coupling(m) for m in ([1,0,1,0,1], [0,1,0,1,0], [1,1,0,0,1], [0,0,1,1,0], [1,0,0,1,1], [0,1,1,0,0])])
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x
    def inverse(self, z):
        for layer in reversed(self.layers): z = layer(z, inverse=True)
        return z


def geometry(a, target, labels):
    nn = NearestNeighbors(n_neighbors=21, n_jobs=-1).fit(a).kneighbors(return_distance=False)[:,1:]
    yn = NearestNeighbors(n_neighbors=21, n_jobs=-1).fit(target).kneighbors(return_distance=False)[:,1:]
    overlap = np.mean([len(set(i)&set(j))/20 for i,j in zip(nn,yn)])
    take = np.linspace(0, len(a)-1, min(5000,len(a)), dtype=int)
    return {"target_knn_overlap_k20":float(overlap), "same_reference_cluster_fraction_k20":float((labels[:,None]==labels[nn]).mean()), "reference_cluster_silhouette":float(silhouette_score(a[take], labels[take]))}


def hdbscan_scan(spaces, labels):
    rows=[]
    for name, x in spaces.items():
        for size in (15,30,60):
            for select in ("eom","leaf"):
                pred=HDBSCAN(min_cluster_size=size, min_samples=max(5,size//3), cluster_selection_method=select).fit_predict(x)
                active=pred>=0; counts=np.bincount(pred[active]) if active.any() else np.array([])
                rows.append({"space":name,"min_cluster_size":size,"selection":select,"clusters":int(len(counts)),"noise_fraction":float(1-active.mean()),"median_cluster_size":float(np.median(counts)) if len(counts) else 0.,"max_cluster_fraction":float(counts.max()/len(x)) if len(counts) else 0.,"reference_hole_ami":float(adjusted_mutual_info_score(labels,pred)),"reference_hole_ari":float(adjusted_rand_score(labels,pred))})
    return pd.DataFrame(rows)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    root=Path(__file__).resolve().parents[2]
    path=root/"dataset"/"stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv"
    out=Path(__file__).parent/"results"; out.mkdir(exist_ok=True)
    read=FEATURES+TARGET+["foil_position","cluster"]
    df=pd.read_csv(path,usecols=read).dropna()
    rng=np.random.default_rng(SEED); df=df.iloc[rng.choice(len(df),60000,replace=False)].reset_index(drop=True)
    labels=(df.foil_position.astype(int)*1000+df.cluster.astype(int)).to_numpy()
    holes=np.unique(labels); held=rng.choice(holes,size=int(np.ceil(.2*len(holes))),replace=False)
    test=np.isin(labels,held); train=~test
    # Raster is retained as a fifth coordinate, but its smooth within-hole
    # broadening of optical FP coordinates is removed before the metric is fit.
    conditional=make_pipeline(SplineTransformer(n_knots=8, degree=3, extrapolation="linear"), Ridge(alpha=2.0))
    conditional.fit(df.loc[train,[RASTER]], df.loc[train,OPTICAL])
    residual=df[OPTICAL].to_numpy()-conditional.predict(df[[RASTER]])
    corrected=np.column_stack([residual, df[RASTER].to_numpy()])
    xs=RobustScaler(quantile_range=(5,95)).fit(corrected[train])
    ys=StandardScaler().fit(df.loc[train,TARGET])
    x=xs.transform(corrected).astype("float32"); y=ys.transform(df[TARGET]).astype("float32")
    model=Flow(); opt=torch.optim.AdamW(model.parameters(),lr=1.5e-3,weight_decay=1e-5)
    loader=DataLoader(TensorDataset(torch.from_numpy(x[train]),torch.from_numpy(y[train])),batch_size=512,shuffle=True)
    history=[]
    for epoch in range(100):
        total=0.
        for xb,yb in loader:
            z=model(xb)
            loss=((z[:,:3]-yb)**2).mean()+.02*((z[:,3:]-xb[:,3:])**2).mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();total+=float(loss.detach())*len(xb)
        if epoch in (0,24,49,99): history.append({"epoch":epoch+1,"loss":total/train.sum()})
    with torch.no_grad():
        z=model(torch.from_numpy(x)).numpy()
        roundtrip=torch.max(torch.abs(model.inverse(model(torch.from_numpy(x[:1024])))-torch.from_numpy(x[:1024]))).item()
    weighted=z.copy();weighted[:,3:]*=np.sqrt(.10)
    metrics={"raster_conditioned_raw_5d":geometry(x[test],y[test],labels[test]),"flow_full_5d_equal":geometry(z[test],y[test],labels[test]),"flow_full_5d_residual_weight_0p10":geometry(weighted[test],y[test],labels[test])}
    metrics["flow_full_5d_equal"]["target_rmse_scaled"]=float(np.sqrt(mean_squared_error(y[test],z[test,:3])))
    metrics["flow_full_5d_residual_weight_0p10"]["target_rmse_scaled"]=metrics["flow_full_5d_equal"]["target_rmse_scaled"]
    scan=hdbscan_scan({"raster_conditioned_raw_5d":x[test],"flow_full_5d_equal":z[test],"flow_full_5d_residual_weight_0p10":weighted[test]},labels[test])
    scan.to_csv(out/"continuous_prior_flow_hdbscan_holeholdout_scan.csv",index=False)
    good=scan[(scan.clusters>=20)&(scan.noise_fraction<.5)&(scan.max_cluster_fraction<.1)]
    report={"n_events":int(len(df)),"train_events":int(train.sum()),"test_events":int(test.sum()),"held_out_complete_reference_holes":int(len(held)),"input_features":FEATURES,"weak_continuous_targets":TARGET,"not_used_for_fitting":["cluster_center_x","cluster_center_y","foil_ytar_center","hole_id","hole_row","hole_col","cluster","foil_position"],"raster_conditioning":"train-only cubic-spline Ridge prediction of four optical coordinates from fr_ybpm; residuals plus fr_ybpm remain five input dimensions","model":"6-coupling RealNVP 5D-to-5D; first 3 axes weakly supervised, final 2 residual axes retained","training_checkpoints":history,"round_trip_max_abs_error":float(roundtrip),"metrics":metrics,"hdbscan":{"selection_rule":"clusters>=20, noise<0.50, largest cluster<10%; then maximal reference-hole AMI","best":good.sort_values(["reference_hole_ami","reference_hole_ari"],ascending=False).head(1).to_dict(orient="records")[0] if not good.empty else None},"caution":"Continuous targets are still outputs of current optics reconstruction; this tests weaker supervision without discrete hole centres, not a fully reconstruction-free method."}
    (out/"continuous_prior_flow_metric_holeholdout_summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
