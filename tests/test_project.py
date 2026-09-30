from pathlib import Path
import pandas as pd, json, subprocess, sys
from src.credit_sense.data.generate_synthetic import generate
from src.credit_sense.features import TRADITIONAL,ALTERNATIVE,PROTECTED,feature_matrix
from src.credit_sense.decision import score_row
from src.credit_sense.audit import append_decision

def test_seed_reproducibility():
    a=generate(100,42); b=generate(100,42); pd.testing.assert_frame_equal(a,b)
def test_shape(): assert generate(200,42).shape[0]==200
def test_no_protected_features(): assert set(PROTECTED).isdisjoint(TRADITIONAL+ALTERNATIVE)
def test_decline_has_reasons():
    row=generate(1,123).iloc[0].to_dict(); row.update({'dti':.9,'delinquencies_24m':7,'on_time_ratio':.45,'credit_score':510})
    r=score_row(row); 
    if r['decision']=='decline': assert r['reason_codes']
def test_thresholds():
    row=generate(1,7).iloc[0].to_dict(); r=score_row(row); assert 0<=r['pd']<=1 and r['decision'] in {'approve','decline','manual-review'}
def test_audit_no_pii(tmp_path):
    p=tmp_path/'audit.jsonl'; append_decision({'decision':'decline','raw_input':{'name':'Jane Doe','ssn':'123-45-6789','dti':.5},'reason_codes':[]},p); s=p.read_text(); assert 'Jane Doe' not in s and '123-45-6789' not in s
