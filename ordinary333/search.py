"""Bounded Q-guided beam search; exact shallow splice and full native replay."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from .models import make_model
from .training import ASSETS,write_json


def replay(state,path,moves):
    """Return final labelled state after a sequence of action indices."""
    for action in path:state=state[moves[action]]
    return state


def endgame_index(device):
    """Hash shallow states for fast candidates; every hit is verified on all labels."""
    raw=np.load(ASSETS/'anchors_d4.npz')
    states=torch.tensor(raw['states'],dtype=torch.long,device=device)
    weights=torch.tensor(np.random.default_rng(333).integers(1,2**57,size=54,dtype=np.int64),device=device)
    # Wrapped signed-int64 arithmetic is used only for lookup, not for equality.
    hashes=(states*weights).sum(1)
    hashes,order=hashes.sort()
    return hashes,states[order],torch.tensor(raw['solutions'],device=device)[order],weights


@torch.inference_mode()
def beam_solve(initial,model,geometry,width=16384,max_depth=28,seconds=120,consistency=.3,reverse=False,endgame=None):
    """Return one replay-valid candidate and timing, or an explicit unsolved status."""
    device=next(model.parameters()).device
    moves=torch.tensor(geometry['moves'],device=device)
    inverse=torch.tensor(geometry['inverse_actions'],device=device)
    identity=torch.arange(54,device=device)
    state=torch.tensor(initial,device=device)
    original=state.clone()
    if reverse:state=state.argsort()
    beam=state[None];paths=torch.empty((1,0),device=device,dtype=torch.long)
    start=time.monotonic();index=endgame or endgame_index(device)
    hashes,known,suffix,weights=index
    model.eval()
    for depth in range(max_depth+1):
        h=(beam*weights).sum(1);pos=torch.searchsorted(hashes,h).clamp_max(len(hashes)-1)
        hit=(hashes[pos]==h)&(known[pos]==beam).all(1)
        found=torch.where(hit)[0]
        if len(found):
            candidates=[]
            for row in found[:256].tolist():
                tail=suffix[pos[row]];candidate=paths[row].tolist()+tail[tail>=0].tolist()
                if reverse:candidate=[int(inverse[a]) for a in candidate[::-1]]
                if torch.equal(replay(original,candidate,moves),identity):candidates.append(candidate)
            if candidates:
                word=min(candidates,key=len)
                return dict(status='solved',path=word,length=len(word),valid=True,seconds=time.monotonic()-start,beam=width,reverse=reverse)
        if depth==max_depth or time.monotonic()-start>=seconds:break
        scores=[]
        for chunk in beam.split(2048):
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                q,v=model(chunk)
            q=q.float();v=v.float()
            scores.append(q+consistency*(q-(v[:,None]-1)).abs())
        score=torch.cat(scores)
        if depth:
            score[torch.arange(len(beam),device=device),inverse[paths[:,-1]]]=torch.inf
        _,flat=score.flatten().topk(min(width*3,score.numel()),largest=False,sorted=True)
        parent=flat//18;action=flat%18
        children=beam[parent].gather(1,moves[action])
        # Exact deduplication; hash collisions cannot drop a different state.
        _,group=torch.unique(children,dim=0,return_inverse=True)
        rank=torch.arange(len(children),device=device)
        first=torch.full((len(children),),len(children),device=device,dtype=torch.long)
        first.scatter_reduce_(0,group,rank,reduce='amin',include_self=True)
        keep=rank[first[group]==rank][:width]
        beam=children[keep]
        paths=torch.cat((paths[parent[keep]],action[keep,None]),1)
    return dict(status='timeout_or_depth_limit',path=None,length=None,valid=None,seconds=time.monotonic()-start,beam=width,reverse=reverse)


def evaluate(checkpoint,output,width=16384,seconds=120,pids=None):
    """Evaluate identical fixed public cubes. Unsolved cases retain the fixed baseline length."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    geometry=json.loads((ASSETS/'geometry.json').read_text())
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    payload=torch.load(checkpoint,map_location=device,weights_only=False)
    model=make_model(geometry,payload['config']).to(device);model.load_state_dict(payload['model'])
    problems=json.loads((ASSETS/'benchmark50.json').read_text())
    endgame=endgame_index(device);rows=[]
    for item in problems:
        if pids is not None and item['puzzle_id'] not in pids:continue
        lam=.3 if payload['config'].get('value_weight',1.) else 0.
        attempts=[beam_solve(item['state'],model,geometry,width,28,seconds,consistency=lam,reverse=rev,endgame=endgame) for rev in (False,True)]
        solved=[a for a in attempts if a['valid']]
        chosen=min(solved,key=lambda x:x['length']) if solved else None
        rows.append(dict(puzzle_id=item['puzzle_id'],baseline_length=item['baseline_length'],
                         selected_length=min(item['baseline_length'],chosen['length']) if chosen else item['baseline_length'],
                         solved=bool(chosen),candidate_length=chosen['length'] if chosen else None,
                         valid=chosen['valid'] if chosen else None,path=chosen['path'] if chosen else None,
                         seconds=sum(a['seconds'] for a in attempts)))
        write_json(output/'benchmark.json',dict(config=payload['config'],step=payload['step'],beam=width,rows=rows))
        print(json.dumps(rows[-1]),flush=True)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--beam',type=int,default=16384);p.add_argument('--seconds',type=int,default=120)
    a=p.parse_args();evaluate(a.checkpoint,a.output,a.beam,a.seconds)
