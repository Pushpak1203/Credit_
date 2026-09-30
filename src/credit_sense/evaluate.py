from pathlib import Path
import json, joblib, yaml, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from .features import feature_matrix
from .train import split
ROOT=Path(__file__).resolve().parents[2]; CFG=yaml.safe_load((ROOT/'config/config.yaml').read_text())

def rule_pd(df):
    score=np.nan_to_num(df.credit_score.to_numpy(),nan=610)
    z=-2.5-.006*(score-650)+.8*df.dti.to_numpy()+.22*df.delinquencies_24m.to_numpy()-.5*(df.on_time_ratio.to_numpy()-.9)+.2*df.thin_file.to_numpy()
    return 1/(1+np.exp(-z))
def ks(y,p):
    fpr,tpr,_=roc_curve(y,p); return float(np.max(tpr-fpr))
def metric(y,p): return {'auc':float(roc_auc_score(y,p)),'ks':ks(y,p),'brier':float(brier_score_loss(y,p))}
def main():
    data=pd.read_csv(ROOT/CFG['paths']['data'],parse_dates=['application_date']); _,_,te=split(data)
    b=joblib.load(ROOT/CFG['paths']['models']/ 'credit_sense_models.joblib'); y=te.default_12m
    preds={'rule_baseline':rule_pd(te),'logistic_traditional':b['logistic_traditional'].predict_proba(feature_matrix(te,'traditional'))[:,1], 'gbdt_traditional':b['gbdt_traditional'].predict_proba(feature_matrix(te,'traditional'))[:,1], 'gbdt_alternative':b['gbdt_alternative'].predict_proba(feature_matrix(te,'alternative'))[:,1]}
    m={k:metric(y,v) for k,v in preds.items()}; m['incremental_alt_auc']=m['gbdt_alternative']['auc']-m['gbdt_traditional']['auc']; m['incremental_alt_ks']=m['gbdt_alternative']['ks']-m['gbdt_traditional']['ks']
    out=ROOT/CFG['paths']['outputs']; (out/'plots').mkdir(exist_ok=True)
    f=plt.figure(figsize=(7,5));
    for k,p in preds.items():
        fpr,tpr,_=roc_curve(y,p); plt.plot(fpr,tpr,label=f'{k} AUC={m[k]["auc"]:.3f}')
    plt.plot([0,1],[0,1],'--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.tight_layout(); f.savefig(out/'plots/roc_curves.png',dpi=150); plt.close(f)
    f=plt.figure(figsize=(7,5));
    for k,p in preds.items():
        bins=pd.qcut(p,10,duplicates='drop'); g=pd.DataFrame({'p':p,'y':y.to_numpy()}).groupby(bins,observed=True).mean(); plt.plot(g.p,g.y,marker='o',label=k)
    plt.plot([0,1],[0,1],'--'); plt.xlabel('Mean predicted PD'); plt.ylabel('Observed bad rate'); plt.legend(fontsize=7); plt.tight_layout(); f.savefig(out/'plots/calibration.png',dpi=150); plt.close(f)
    # Approval tradeoff for alt model
    rows=[]
    for c in np.arange(.08,.31,.02):
        approve=preds['gbdt_alternative']<c; rows.append({'cutoff':round(float(c),3),'approval_rate':float(approve.mean()),'bad_rate_approved':float(y[approve].mean()) if approve.any() else np.nan})
    pd.DataFrame(rows).to_csv(out/'approval_tradeoff.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(m,indent=2))
    print(json.dumps(m,indent=2))
if __name__=='__main__': main()
