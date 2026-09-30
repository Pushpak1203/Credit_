from pathlib import Path
import json, joblib, yaml, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from .features import feature_matrix, TRADITIONAL, ALTERNATIVE
ROOT=Path(__file__).resolve().parents[2]; CFG=yaml.safe_load((ROOT/'config/config.yaml').read_text()); SEED=CFG['seed']

def split(df):
    q=df['application_date'].quantile([.70,.85]).values
    return df[df.application_date<=q[0]].copy(), df[(df.application_date>q[0])&(df.application_date<=q[1])].copy(), df[df.application_date>q[1]].copy()

def calibrate(base,X,y): return CalibratedClassifierCV(base, method='isotonic', cv=3).fit(X,y)

def main():
    data=pd.read_csv(ROOT/CFG['paths']['data'],parse_dates=['application_date']); tr,va,te=split(data)
    bundle={'seed':SEED,'features':{'traditional':TRADITIONAL,'alternative':TRADITIONAL+ALTERNATIVE},'splits':{'train':len(tr),'validation':len(va),'test':len(te)}}
    # Logistic traditional
    log=Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler()),('clf',LogisticRegression(max_iter=1000,C=.5,random_state=SEED))])
    log.fit(feature_matrix(tr,'traditional'),tr.default_12m)
    bundle['logistic_traditional']=log
    # GBDT traditional
    mono_tr=[-1,-1,-1,1,-1,1,1,1]
    gtr=HistGradientBoostingClassifier(max_iter=220,max_leaf_nodes=15,learning_rate=.045,l2_regularization=1.0,random_state=SEED,monotonic_cst=mono_tr)
    bundle['gbdt_traditional']=calibrate(gtr,feature_matrix(tr,'traditional'),tr.default_12m)
    # GBDT alt
    mono_alt=mono_tr+[-1,1,-1,1,-1,1,-1]
    galt=HistGradientBoostingClassifier(max_iter=260,max_leaf_nodes=17,learning_rate=.04,l2_regularization=1.0,random_state=SEED,monotonic_cst=mono_alt)
    bundle['gbdt_alternative']=calibrate(galt,feature_matrix(tr,'alternative'),tr.default_12m)
    # rule baseline is a pure function, stored as metadata
    bundle['rule_baseline']={'approval_cutoff':.18,'formula':'approve if synthetic rule PD < cutoff'}
    bundle['validation']=va[['application_date','default_12m']].to_dict('list')
    bundle['model_version']=CFG['model_version']
    out=ROOT/CFG['paths']['models']; out.mkdir(exist_ok=True)
    joblib.dump(bundle,out/'credit_sense_models.joblib')
    tr.to_csv(ROOT/'outputs/train.csv',index=False); va.to_csv(ROOT/'outputs/validation.csv',index=False); te.to_csv(ROOT/'outputs/test.csv',index=False)
    print(json.dumps(bundle['splits']))
if __name__=='__main__': main()
