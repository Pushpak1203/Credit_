from pathlib import Path
import yaml, joblib, numpy as np, pandas as pd
from .features import feature_matrix
from .explain import explain
ROOT=Path(__file__).resolve().parents[2]; CFG=yaml.safe_load((ROOT/'config/config.yaml').read_text())
BUNDLE=ROOT/CFG['paths']['models']/'credit_sense_models.joblib'

def load_bundle(): return joblib.load(BUNDLE)
def score_row(row, bundle=None):
    b=bundle or load_bundle(); df=pd.DataFrame([row])
    p=float(b['gbdt_alternative'].predict_proba(feature_matrix(df,'alternative'))[:,1][0])
    cutoff=CFG['thresholds']['approval_cutoff_pd']; low=CFG['thresholds']['review_low_pd']; high=CFG['thresholds']['review_high_pd']
    distance=min(abs(p-cutoff)/max(cutoff,1e-6),1.0); uncertainty=4*p*(1-p); confidence=float(np.clip(.55*distance+.45*(1-uncertainty),0,1))
    decision='approve' if p<cutoff else 'decline'
    if low<=p<=high or confidence<CFG['thresholds']['confidence_review_below']: decision='manual-review'
    exp=explain(b['gbdt_alternative'],df.iloc[0],'alternative',3)
    if decision=='decline' and not exp['reason_codes']: raise ValueError('Decline requires reason codes')
    return {'pd':p,'decision':decision,'confidence':confidence,'reason_codes':exp['reason_codes'],'borrower_message':exp['borrower_message'],'model_version':b['model_version']}
