"""Restartable sparse-Q + AZ-value training, exact anchors, conjugation augmentation.

Random-walk depths are surrogate upper bounds, NOT exact deep distances.
No test puzzle enters training. This is a port candidate, not a reproduced score.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from .models import make_model

ASSETS=Path(__file__).parent/'assets'


def write_json(path,value):
    """Atomically write a compact progress receipt; input path/payload, no return."""
    temp=path.with_suffix('.partial')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def sample_walk(moves,inverse,batch,k_max,generator,tilt=.5):
    """Return random pivot states, depths, undo/next actions from nonbacktracking walks."""
    device=moves.device
    length=torch.randint(2,k_max+1,(batch,),device=device,generator=generator)
    pivot=(torch.rand(batch,device=device,generator=generator).pow(1/(1+tilt))*(length-1)).long()+1
    state=torch.arange(54,device=device).expand(batch,-1).clone()
    result=torch.zeros_like(state);previous=torch.zeros(batch,dtype=torch.long,device=device)
    following=previous.clone();last=previous.clone()
    for step in range(k_max):
        if step==0: action=torch.randint(18,(batch,),device=device,generator=generator)
        else:
            proposal=torch.randint(17,(batch,),device=device,generator=generator)
            forbidden=inverse[last]
            action=proposal+(proposal>=forbidden).long()
        at=pivot==step
        result=torch.where(at[:,None],state,result)
        previous=torch.where(at,inverse[last],previous)
        following=torch.where(at,action,following)
        state=state.gather(1,moves[action]);last=action
    return result,pivot,previous,following


def augment(state,depth,undo,next_action,rotations,transport,rows,generator):
    """Apply verified conjugation; transport both sparse labels with the same frame."""
    b=len(state);device=state.device
    frames=torch.randint(len(rotations),(b,rows),device=device,generator=generator)
    frames[:,0]=0
    frames=frames.flatten()
    r=rotations[frames];ri=r.argsort(dim=1)
    source=state.repeat_interleave(rows,0)
    transformed=r.gather(1,source.gather(1,ri))
    a=transport[frames,undo.repeat_interleave(rows)]
    c=transport[frames,next_action.repeat_interleave(rows)]
    return transformed,depth.repeat_interleave(rows),a,c


def train(config,output,resume=None,seconds=1800,smoke_steps=None):
    """Train one bounded job; return checkpoint path, metrics and measured throughput."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    geometry=json.loads((ASSETS/'geometry.json').read_text())
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda' and smoke_steps is None:
        return {'status':'gpu_unavailable','checkpoint':None}
    torch.manual_seed(config['seed'])
    if device.type=='cuda':torch.backends.cuda.matmul.allow_tf32=True
    rng=torch.Generator(device=device).manual_seed(config['seed'])
    model=make_model(geometry,config).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=3e-3,fused=device.type=='cuda')
    moves=torch.tensor(geometry['moves'],device=device)
    inverse=torch.tensor(geometry['inverse_actions'],device=device)
    rotations=torch.tensor(geometry['rotations'],device=device)
    transport=torch.tensor(geometry['action_transport'],device=device)
    anchor=np.load(ASSETS/'anchors_d4.npz')
    ax=torch.tensor(anchor['states'],device=device,dtype=torch.long)
    aq=torch.tensor(anchor['q'],device=device,dtype=torch.float32)
    av=torch.tensor(anchor['depth'],device=device,dtype=torch.float32)
    step=0;fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if resume and Path(resume).is_file():
        saved=torch.load(resume,map_location=device,weights_only=False)
        if saved.get('fingerprint')!=fingerprint:return {'status':'resume_config_mismatch','checkpoint':str(resume)}
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);step=saved['step']
        rng.set_state(saved['rng'].cpu());torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type=='cuda' and 'cuda_rng' in saved:torch.cuda.set_rng_state_all(saved['cuda_rng'])
    start=time.monotonic();initial_step=step;last_save=start
    receipt=dict(status='running',run_id=config['run_id'],step=step,device=str(device),
                 gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
                 parameters=sum(p.numel() for p in model.parameters()),config=config,
                 anchor_depth=4,anchor_rows=len(ax),symmetry_frames=24,
                 recipe_deviations=['d4 anchors instead of d5','sampled symmetry rows rather than greedy full coverage','compile disabled during initial calibration'])
    print(json.dumps(receipt),flush=True)
    checkpoint=output/'latest.pt'

    def save():
        """Save model, optimizer, sampler RNG and metadata for exact resume."""
        payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),config=config,step=step,
                     fingerprint=fingerprint,rng=rng.get_state().cpu(),torch_rng=torch.get_rng_state())
        if device.type=='cuda':payload['cuda_rng']=torch.cuda.get_rng_state_all()
        temporary=output/'latest.partial';torch.save(payload,temporary);temporary.replace(checkpoint)
        receipt.update(step=step,elapsed_seconds=round(time.monotonic()-start,2),updated_unix=time.time(),checkpoint=str(checkpoint))
        write_json(output/'status.json',receipt)

    model.train()
    while (step-initial_step<smoke_steps if smoke_steps is not None else time.monotonic()-start<seconds):
        state,depth,undo,nxt=sample_walk(moves,inverse,config.get('batch',128),config['k_max'],rng)
        state,depth,undo,nxt=augment(state,depth,undo,nxt,rotations,transport,config.get('sym_rows',4),rng)
        ai=torch.randint(len(ax),(config.get('anchor_batch',128),),device=device,generator=rng)
        joined=torch.cat((state,ax[ai]),0);n=len(state);rows=torch.arange(n,device=device)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            q,v=model(joined);q=q.float();v=v.float()
            sparse=((q[rows,undo]-(depth-1))**2+(q[rows,nxt]-(depth+1))**2).mean()/2
            exact=(q[n:]-aq[ai]).square().mean()
            value=(v-torch.cat((depth.float(),av[ai]))).square().mean()
            loss=sparse+exact+config.get('value_weight',1.)*value
        if not bool(torch.isfinite(loss)):
            receipt['status']='nonfinite_loss';save();return receipt
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();step+=1
        if step%100==0 or smoke_steps is not None:
            receipt.update(loss=float(loss.detach()),sparse_mse=float(sparse.detach()),anchor_mse=float(exact.detach()),
                           value_mse=float(value.detach()),steps_per_second=round((step-initial_step)/max(.01,time.monotonic()-start),3))
        if time.monotonic()-last_save>120:
            save();print(json.dumps(receipt),flush=True);last_save=time.monotonic()
    receipt['status']='checkpointed';save();print(json.dumps(receipt),flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--seconds',type=int,default=1800);parser.add_argument('--smoke-steps',type=int)
    args=parser.parse_args();out=Path(args.output);resume=out/'latest.pt'
    train(json.loads(Path(args.config).read_text()),out,resume if resume.exists() else None,args.seconds,args.smoke_steps)


if __name__=='__main__':main()
