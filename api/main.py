from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from src.credit_sense.decision import score_row
from src.credit_sense.audit import append_decision
from src.credit_sense.appeals import submit
from pathlib import Path
import json
app=FastAPI(title='Credit Sense Prototype API',version='1.0.0',description='Synthetic-data-only offline prototype')
class Applicant(BaseModel):
    credit_score: Optional[float]=None; credit_history_years: float=Field(ge=0); on_time_ratio: float=Field(ge=0,le=1); delinquencies_24m:int=Field(ge=0); months_since_last_delinquency:float=Field(ge=0); utility_rent_regularity:float=Field(ge=0,le=1); income_volatility:float=Field(ge=0,le=1); avg_monthly_inflow:float=Field(gt=0); overdraft_count:int=Field(ge=0); savings_buffer_months:float=Field(ge=0); gig_income_share:float=Field(ge=0,le=1); employment_tenure_years:float=Field(ge=0); dti:float=Field(ge=0,le=2); requested_amount:float=Field(gt=0); term_months:int=Field(gt=0); thin_file:int=Field(ge=0,le=1); no_file:int=Field(ge=0,le=1)
class Appeal(BaseModel): applicant_ref:str; reason:str
@app.get('/health')
def health(): return {'status':'ok','prototype':'synthetic-data-only'}
@app.get('/model-card')
def model_card():
    p=Path('docs/MODEL_CARD.md'); return {'markdown':p.read_text() if p.exists() else 'Model card unavailable'}
@app.post('/score')
def score(a:Applicant):
    r=score_row(a.model_dump()); out={**r}; append_decision({**out,'raw_input':a.model_dump()}); return out
@app.post('/appeal')
def appeal(a:Appeal): return {'appeal_id':submit(a.applicant_ref,a.reason),'status':'submitted'}
