from dataclasses import dataclass
from typing import List
import pandas as pd

TRADITIONAL=['credit_score','credit_history_years','on_time_ratio','delinquencies_24m','months_since_last_delinquency','dti','requested_amount','term_months']
ALTERNATIVE=['utility_rent_regularity','income_volatility','avg_monthly_inflow','overdraft_count','savings_buffer_months','gig_income_share','employment_tenure_years']
PROTECTED=['age_band','sex','race_ethnicity','zip_income_tier']

@dataclass(frozen=True)
class FeatureMeta:
    name:str; source:str; description:str; permitted_use:bool; provenance:str

REGISTRY=[
 FeatureMeta('credit_score','traditional','Bureau-style credit score; nullable for thin/no-file.',True,'synthetic'),
 FeatureMeta('credit_history_years','traditional','Length of credit history.',True,'synthetic'),
 FeatureMeta('on_time_ratio','repayment','Share of payments made on time.',True,'synthetic'),
 FeatureMeta('delinquencies_24m','repayment','Count of delinquencies in 24 months.',True,'synthetic'),
 FeatureMeta('months_since_last_delinquency','repayment','Months since latest delinquency.',True,'synthetic'),
 FeatureMeta('dti','traditional','Debt-to-income ratio.',True,'synthetic'),
 FeatureMeta('requested_amount','application','Requested loan amount.',True,'synthetic'),
 FeatureMeta('term_months','application','Requested installment term.',True,'synthetic'),
 FeatureMeta('utility_rent_regularity','alternative','Permissioned utility/rent payment regularity.',True,'synthetic'),
 FeatureMeta('income_volatility','cash_flow','Volatility of bank-account inflows.',True,'synthetic'),
 FeatureMeta('avg_monthly_inflow','cash_flow','Average monthly bank-account inflow.',True,'synthetic'),
 FeatureMeta('overdraft_count','cash_flow','Recent overdraft count.',True,'synthetic'),
 FeatureMeta('savings_buffer_months','cash_flow','Savings buffer measured in months.',True,'synthetic'),
 FeatureMeta('gig_income_share','cash_flow','Share of income from gig work.',True,'synthetic'),
 FeatureMeta('employment_tenure_years','employment','Employment tenure.',True,'synthetic'),
]

def feature_matrix(df, kind='alternative'):
    cols=TRADITIONAL if kind=='traditional' else TRADITIONAL+ALTERNATIVE
    X=df[cols].copy()
    # Keep nullable credit score: HistGB supports NaNs; logistic gets median imputation downstream.
    return X

def registry_df(): return pd.DataFrame([m.__dict__ for m in REGISTRY]+[
    {'name':x,'source':'protected/proxy','description':'Audit-only attribute; prohibited from model features.','permitted_use':False,'provenance':'synthetic'} for x in PROTECTED])
