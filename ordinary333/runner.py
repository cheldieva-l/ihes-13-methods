"""Run a restartable GPU experiment in bounded chunks, never pretending local files persist."""
import argparse
import json
from pathlib import Path
import time
from .training import train,write_json
from .artifacts import publish_checkpoint
from .search import evaluate


def run(run_id,output,hours=11,seconds=1800):
    """Train, externally save, then evaluate; stop after a pilot if no durable upload exists."""
    config=json.loads((Path(__file__).parent/'configs'/f'{run_id}.json').read_text())
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+hours*3600;chunk=0
    while time.monotonic()<deadline:
        latest=output/'latest.pt'
        result=train(config,output,latest if latest.exists() else None,min(seconds,int(deadline-time.monotonic())))
        if result.get('status')!='checkpointed':return result
        saved=publish_checkpoint(latest,run_id,result['step'])
        chunk+=1;result.update(storage=saved,chunk=chunk)
        write_json(output/'run_status.json',result);print(json.dumps(result),flush=True)
        if saved['status']!='published':
            result['status']='pilot_finished_storage_required'
            print(json.dumps(result),flush=True)
            return result
        if chunk%2==0:
            evaluate(latest,output/f'eval_step{result["step"]}',width=16384,seconds=45)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--output',required=True)
    p.add_argument('--hours',type=float,default=11);p.add_argument('--chunk-seconds',type=int,default=1800)
    a=p.parse_args();run(a.run_id,a.output,a.hours,a.chunk_seconds)
