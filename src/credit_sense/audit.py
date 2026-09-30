from pathlib import Path
import hashlib,json,datetime
ROOT=Path(__file__).resolve().parents[2]
def append_decision(record,path=ROOT/'outputs/audit.jsonl'):
    path.parent.mkdir(exist_ok=True)
    safe={k:v for k,v in record.items() if k not in {'ssn','name','address','email','phone','raw_input'}}
    safe['input_hash']=hashlib.sha256(json.dumps(record.get('raw_input',record),sort_keys=True,default=str).encode()).hexdigest()
    safe['timestamp_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(safe,default=str)+'\n')
    return safe
