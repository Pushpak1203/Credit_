from pathlib import Path
import numpy as np,pandas as pd,yaml
from .features import feature_matrix
ROOT=Path(__file__).resolve().parents[2]; CFG=yaml.safe_load((ROOT/'config/config.yaml').read_text())
def psi(expected,actual,bins=10):
    e=np.asarray(expected); a=np.asarray(actual); qs=np.unique(np.quantile(e,np.linspace(0,1,bins+1))); 
    if len(qs)<3:return 0.0
    eb=np.histogram(e,qs)[0]/len(e)+1e-6; ab=np.histogram(a,qs)[0]/len(a)+1e-6; return float(np.sum((ab-eb)*np.log(ab/eb)))
def main():
    df=pd.read_csv(ROOT/CFG['paths']['data']); base=df.iloc[:int(len(df)*.7)]; recent=df.iloc[int(len(df)*.7):]; rows=[]
    for c in feature_matrix(base).columns: rows.append({'feature':c,'psi':psi(base[c].fillna(base[c].median()),recent[c].fillna(base[c].median()))})
    out=pd.DataFrame(rows); out.to_csv(ROOT/'outputs/drift_baseline.csv',index=False)
    drift=recent.copy(); rng=np.random.default_rng(CFG['seed']); drift['dti']=np.clip(drift.dti+.12,0,1); drift['income_volatility']=np.clip(drift.income_volatility+.12,0,1)
    rows2=[{'feature':c,'psi':psi(base[c].fillna(base[c].median()),drift[c].fillna(base[c].median()))} for c in feature_matrix(base).columns]; pd.DataFrame(rows2).to_csv(ROOT/'outputs/drift_simulated.csv',index=False)
    print('simulated drift written')
if __name__=='__main__': main()
