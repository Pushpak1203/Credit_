from pathlib import Path
import yaml,joblib,numpy as np,pandas as pd
from .features import feature_matrix
from .train import split
try:
    from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
    FAIRLEARN_AVAILABLE=True
except ImportError:
    FAIRLEARN_AVAILABLE=False
ROOT=Path(__file__).resolve().parents[2]; CFG=yaml.safe_load((ROOT/'config/config.yaml').read_text())
def main():
    df=pd.read_csv(ROOT/CFG['paths']['data'],parse_dates=['application_date']); _,_,te=split(df); b=joblib.load(ROOT/CFG['paths']['models']/'credit_sense_models.joblib'); p=b['gbdt_alternative'].predict_proba(feature_matrix(te,'alternative'))[:,1]; y=te.default_12m.to_numpy(); pred=(p<CFG['thresholds']['approval_cutoff_pd']).astype(int)
    lines=['# Fairness Report','','Prototype/synthetic data only. Protected attributes are audit-only and excluded from model features.','',f'Fairlearn available: **{FAIRLEARN_AVAILABLE}**','']; summary=[]
    for attr in CFG['protected_attributes']:
        g=te[attr]
        if FAIRLEARN_AVAILABLE:
            mf=MetricFrame(metrics={'selection_rate':selection_rate,'tpr':true_positive_rate,'fpr':false_positive_rate},y_true=y,y_pred=pred,sensitive_features=g)
            d=mf.by_group.reset_index().rename(columns={attr:'group'}); d['approved_bad_rate']=[float(y[(g.to_numpy()==grp)&(pred==1)].mean()) if ((g.to_numpy()==grp)&(pred==1)).any() else np.nan for grp in d['group']]
        else:
            rows=[]
            for val in sorted(pd.Series(g).dropna().unique()):
                ix=g.to_numpy()==val; yy=y[ix]; yp=pred[ix]; rows.append({'group':str(val),'selection_rate':yp.mean(),'tpr':yp[yy==0].mean() if (yy==0).any() else np.nan,'fpr':(1-yp[yy==1]).mean() if (yy==1).any() else np.nan,'approved_bad_rate':yy[yp==1].mean() if (yp==1).any() else np.nan})
            d=pd.DataFrame(rows)
        air=float(d.selection_rate.min()/d.selection_rate.max()) if d.selection_rate.max()>0 else np.nan
        lines += [f'## {attr}','',d.to_markdown(index=False),'',f'Adverse impact ratio (minimum/maximum selection rate): **{air:.3f}**','']
        summary.append({'attribute':attr,'adverse_impact_ratio':air})
    proxy=[]; nums=feature_matrix(te,'alternative')
    for attr in CFG['protected_attributes']:
        codes=pd.Series(te[attr]).astype('category').cat.codes.to_numpy()
        for col in nums.columns:
            x=nums[col].fillna(nums[col].median()).to_numpy(); corr=np.corrcoef(x,codes)[0,1]; proxy.append((attr,col,float(abs(corr))))
    pd.DataFrame(proxy,columns=['protected_attribute','feature','abs_corr']).sort_values('abs_corr',ascending=False).head(20).to_csv(ROOT/'outputs/proxy_correlations.csv',index=False)
    (ROOT/CFG['paths']['fairness']).write_text('\n'.join(lines)); pd.DataFrame(summary).to_csv(ROOT/'outputs/fairness_summary.csv',index=False); print('\n'.join(lines[:10]))
if __name__=='__main__': main()
