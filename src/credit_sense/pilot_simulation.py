from pathlib import Path
import yaml,joblib,numpy as np,pandas as pd
from .evaluate import rule_pd
from .features import feature_matrix
from .train import split
ROOT=Path(__file__).resolve().parents[2]; CFG=yaml.safe_load((ROOT/'config/config.yaml').read_text())
def metrics(y,decision):
    dec=np.array(decision); appr=dec=='approve'; rev=dec=='manual-review'; return {'approval_rate':appr.mean(),'bad_rate':y[appr].mean() if appr.any() else np.nan,'risk_access_tradeoff':appr.mean()/(y[appr].mean()+1e-6),'manual_review_rate':rev.mean(),'estimated_review_cost_usd':rev.mean()*CFG['review_cost_usd']*len(y),'estimated_review_minutes':rev.mean()*CFG['review_minutes']*len(y)}
def main():
    df=pd.read_csv(ROOT/CFG['paths']['data'],parse_dates=['application_date']); _,_,te=split(df); y=te.default_12m.to_numpy(); b=joblib.load(ROOT/CFG['paths']['models']/'credit_sense_models.joblib'); Xtr=feature_matrix(te,'traditional'); Xalt=feature_matrix(te,'alternative')
    A=np.where(rule_pd(te)<.20,'approve','decline'); B=np.where((te.dti<.45)&(np.nan_to_num(te.credit_score,nan=610)>=600),'approve','decline'); C=np.where(b['logistic_traditional'].predict_proba(Xtr)[:,1]<.18,'approve','decline'); p=b['gbdt_alternative'].predict_proba(Xalt)[:,1]; D=np.where(p<.18,'approve','decline'); cutoff=CFG['thresholds']['approval_cutoff_pd']; low=CFG['thresholds']['review_low_pd']; high=CFG['thresholds']['review_high_pd']; dist=np.minimum(np.abs(p-cutoff)/cutoff,1); conf=np.clip(.55*dist+.45*(1-4*p*(1-p)),0,1); E=np.where((p>=low)&(p<=high)|(conf<CFG['thresholds']['confidence_review_below']),'manual-review',np.where(p<cutoff,'approve','decline'))
    rows=[{'stage':n,**metrics(y,d)} for n,d in [('A incumbent',A),('B rules',B),('C ML traditional',C),('D ML alternative',D),('E Credit Sense hybrid',E)]]; out=pd.DataFrame(rows); out.to_csv(ROOT/CFG['paths']['pilot'],index=False); out.to_html(ROOT/'outputs/pilot_comparison.html',index=False); print(out.to_string(index=False))
if __name__=='__main__': main()
