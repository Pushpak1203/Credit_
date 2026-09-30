from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT=Path(__file__).resolve().parents[3]
CFG=yaml.safe_load((ROOT/'config/config.yaml').read_text())
SEED=CFG['seed']

def sigmoid(x): return 1/(1+np.exp(-x))

def generate(n=20000, seed=42):
    rng=np.random.default_rng(seed)
    dates=pd.date_range('2022-01-01','2025-12-31',freq='D')
    application_date=rng.choice(dates,n)
    age=rng.integers(20,71,n)
    age_band=pd.cut(age,[0,24,34,44,54,64,200],labels=['18-24','25-34','35-44','45-54','55-64','65+']).astype(str)
    sex=rng.choice(['Female','Male','Non-binary/Other'],n,p=[.48,.48,.04])
    race=rng.choice(['White','Black','Hispanic/Latino','Asian','Other'],n,p=[.48,.16,.18,.10,.08])
    zip_tier=rng.choice(['Low','Middle','High'],n,p=[.30,.50,.20])
    thin=rng.random(n)<.38
    nofile=thin & (rng.random(n)<.28)
    credit_score=np.clip(rng.normal(650,62,n),500,820); credit_score[nofile]=np.nan
    hist=np.clip(rng.normal(7.0,5.0,n),0.2,30); hist[nofile]=rng.uniform(.1,1.5,nofile.sum())
    ontime=np.clip(rng.beta(18,2.2,n)-thin*rng.beta(2,8,n)*.25,0.35,1)
    delinq=np.clip(rng.poisson(0.9+1.1*thin,n),0,8)
    months_since=np.clip(rng.exponential(15,n),0,48); months_since[delinq==0]=48
    income=np.clip(rng.lognormal(8.5,.55,n),1800,35000)
    income_vol=np.clip(rng.beta(2.2,7,n),0.01,.85)
    inflow=income*rng.normal(1.05,.12,n)
    overdraft=rng.poisson(0.8+income_vol*2.8,n)
    savings=np.clip(rng.gamma(2.5,1.1,n),0,12)
    gig=np.clip(rng.beta(2,8,n)+thin*.10,nothing if False else None) if False else np.clip(rng.beta(2,8,n)+thin*.10,0,.9)
    tenure=np.clip(rng.lognormal(2.0,.65,n),0.5,20)
    utility=np.clip(rng.beta(15,2.2,n)-thin*.03,0.45,1)
    rent=np.clip(rng.beta(14,2.5,n),0.40,1)
    dti=np.clip(rng.beta(2.3,5.5,n)*.85,0.03,.95)
    amount=np.clip(rng.lognormal(8.0,.55,n),500,25000)
    term=rng.choice([6,9,12,18,24,36],n,p=[.08,.08,.24,.16,.24,.20])
    # Latent risk is deliberately transparent and nonlinear; protected attrs are not used.
    score=np.nan_to_num(credit_score,nan=610)
    latent=(-2.65 - .006*(score-650) - .65*(ontime-.9) + .25*delinq + .018*(12-months_since).clip(min=0)
            + .65*dti + .55*income_vol + .12*overdraft - .11*savings + .25*gig - .035*tenure
            - .75*(utility-.9) - .40*(rent-.9) + .000018*(amount-4000) + .012*(term-12)
            + .20*thin + .22*nofile)
    # small noise makes labels imperfectly deterministic
    pd_default=sigmoid(latent + rng.normal(0,.35,n))
    default=rng.binomial(1,pd_default)
    return pd.DataFrame({
        'application_date':pd.to_datetime(application_date),'age_band':age_band,'sex':sex,'race_ethnicity':race,'zip_income_tier':zip_tier,
        'thin_file':thin.astype(int),'no_file':nofile.astype(int),'credit_score':np.round(credit_score,1),'credit_history_years':np.round(hist,2),
        'on_time_ratio':np.round(ontime,4),'delinquencies_24m':delinq,'months_since_last_delinquency':np.round(months_since,2),
        'utility_rent_regularity':np.round((utility*.55+rent*.45),4),'income_volatility':np.round(income_vol,4),
        'avg_monthly_inflow':np.round(inflow,2),'overdraft_count':overdraft,'savings_buffer_months':np.round(savings,2),
        'gig_income_share':np.round(gig,4),'employment_tenure_years':np.round(tenure,2),'dti':np.round(dti,4),
        'requested_amount':np.round(amount,2),'term_months':term,'default_12m':default
    }).sort_values('application_date').reset_index(drop=True)

if __name__=='__main__':
    out=ROOT/CFG['paths']['data']; out.parent.mkdir(parents=True,exist_ok=True)
    df=generate(CFG['n_rows'],SEED); df.to_csv(out,index=False); print(f'generated {len(df):,} rows -> {out}')
