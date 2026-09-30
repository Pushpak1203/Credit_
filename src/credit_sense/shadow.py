from pathlib import Path
import joblib,yaml,numpy as np,pandas as pd
from .evaluate import rule_pd
from .features import feature_matrix
from .train import split
ROOT=Path(__file__).resolve().parents[2]; CFG=yaml.safe_load((ROOT/'config/config.yaml').read_text())
def main():
    df=pd.read_csv(ROOT/CFG['paths']['data'],parse_dates=['application_date']); _,_,te=split(df); b=joblib.load(ROOT/CFG['paths']['models']/'credit_sense_models.joblib')
    incumbent=np.where(rule_pd(te)<.18,'approve','decline')
    p=b['gbdt_alternative'].predict_proba(feature_matrix(te,'alternative'))[:,1]; cutoff=CFG['thresholds']['approval_cutoff_pd']; low=CFG['thresholds']['review_low_pd']; high=CFG['thresholds']['review_high_pd']
    dist=np.minimum(np.abs(p-cutoff)/cutoff,1); conf=np.clip(.55*dist+.45*(1-4*p*(1-p)),0,1); cs=np.where((p>=low)&(p<=high)|(conf<CFG['thresholds']['confidence_review_below']),'manual-review',np.where(p<cutoff,'approve','decline'))
    out=pd.DataFrame({'incumbent':incumbent,'credit_sense':cs}); out['disagree']=out.incumbent!=out.credit_sense; summary=out.value_counts().reset_index(name='count'); summary.to_csv(ROOT/'outputs/shadow_disagreement.csv',index=False); print(summary.to_string(index=False))
if __name__=='__main__': main()
