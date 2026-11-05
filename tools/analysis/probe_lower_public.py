"""Fit lower-only public-context controls against one frozen whole-game bank."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.algorithms.upper_probe import state_digest
from rl.policies.public_playout import PublicPlayoutPolicy, public_sketch
from rl.policies.small_playout import PreparedPlayoutSamples, SmallPlayoutPolicy
from rl.training.resource_monitor import ResourceMonitor


def fit(bank, source, output, rounds, device, seed):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic()
    with (source/'model.zip').open('rb') as f:
        model_sha = hashlib.file_digest(f,'sha256').hexdigest()
    with (bank/'samples.pt').open('rb') as f:
        data_sha = hashlib.file_digest(f,'sha256').hexdigest()
    if data_sha != json.loads((bank/'collection.json').read_text())['samples_sha256']:
        raise ValueError('frozen sample digest differs')
    data = torch.load(bank/'samples.pt',weights_only=False,map_location='cpu')
    if data['source_model_sha256'] != model_sha:
        raise ValueError('wrong mature checkpoint for this bank')
    samples = data['samples']
    games = np.asarray([s['env'] for s in samples])
    if not np.all(games[:-1] <= games[1:]):
        raise ValueError('bank must be sorted by game, preserving decision order')
    groups = np.unique(games)
    base = PreparedPlayoutSamples(samples,.2)
    feats = []
    bank_sha = hashlib.sha256()
    for i in groups:
        file = bank/'public'/f'{i:03d}.npz'
        with file.open('rb') as f:
            bank_sha.update(hashlib.file_digest(f,'sha256').digest())
        with np.load(file) as t:
            if not np.array_equal(t['alens'],base.lengths[games==i]):
                raise ValueError(f'misaligned public action rows: {i}')
            feats.append(public_sketch(t,t['lengths']))
    features = torch.as_tensor(np.concatenate(feats),device=device)
    del feats
    train, hold = np.flatnonzero(games%5!=0), np.flatnonzero(games%5==0)
    valid, test = np.flatnonzero(games[hold]%10==0), np.flatnonzero(games[hold]%10==5)
    assert len(valid) and len(test)
    model = AccumulatingMaskablePPO.load(source/'model.zip',device=device,amp_mode='off')
    mature = model.policy.small_playout
    def fixed():
        return state_digest({k:v for k,v in model.policy.state_dict().items() if not k.startswith('small_playout.')}),state_digest(model.policy.upper_optimizer.state_dict())
    frozen = fixed()
    def forward(net,indices):
        inputs,target,teacher = base.batch(indices,device)
        if isinstance(net,PublicPlayoutPolicy):
            inputs = (*inputs[:-1],features[indices],inputs[-1])
        logits = net.forward_packed(*inputs)
        logp = logits.log_softmax(-1).masked_fill(~inputs[-1],0)
        return -(target*logp).sum(-1), (logits.argmax(-1)==teacher), -(target*target.clamp_min(1e-30).log()).sum(-1)
    @torch.no_grad()
    def evaluate(net,indices):
        rows = [torch.stack(forward(net,indices[j:j+2048]),1).cpu().numpy() for j in range(0,len(indices),2048)]
        return np.concatenate(rows)
    def metric(rows):
        ce,top1,entropy = rows.mean(axis=0,dtype=np.float64)
        return dict(ce=ce,top1=top1,entropy=entropy,kl=ce-entropy)
    initial = evaluate(mature,hold)
    report = dict(seed=seed,source_sha256=model_sha,samples_sha256=data_sha,public_bank_sha256=bank_sha.hexdigest(),
        source_round=int(json.loads((source/'manifest.json').read_text())['rollout_round']),train_samples=len(train),held_samples=len(hold),
        train_games=len(np.unique(games[train])),held_games=len(np.unique(games[hold])),validation_games=len(np.unique(games[hold][valid])),
        test_games=len(np.unique(games[hold][test])),initial=metric(initial),initial_test=metric(initial[test]),
        settings=dict(batch=102400,microbatch=2048,epochs=8,rounds=rounds,weight_decay=.0001,optimizer='fresh AdamW in all arms',target_visit_fraction=.2),
        arms=[])
    best = (metric(initial[valid])['kl'],None,None)
    with ResourceMonitor(interval=1.,gpu=device.startswith('cuda'),torch_device=model.device) as mon:
        for name,public,lr in [('control',False,.001),('public001',True,.001),('public003',True,.003)]:
            rng = np.random.default_rng(seed)
            torch.manual_seed(seed)
            if public: net = PublicPlayoutPolicy.from_mature(mature)
            else:
                net = SmallPlayoutPolicy(mature.key_count).to(device)
                net.load_state_dict(mature.state_dict())
            before = evaluate(net,hold)
            if not np.array_equal(before,initial):
                raise AssertionError('migration did not preserve mature decisions and CE')
            opt = torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=.0001)
            arm = dict(name=name,params=sum(p.numel() for p in net.parameters()),lr=lr,rounds=[])
            report['arms'].append(arm)
            for r in range(1,rounds+1):
                for _ in range(8):
                    order = rng.permutation(train)
                    for start_idx in range(0,len(order),102400):
                        ids = order[start_idx:start_idx+102400]
                        opt.zero_grad(set_to_none=True)
                        for j in range(0,len(ids),2048):
                            losses,_,_ = forward(net,ids[j:j+2048])
                            (losses.sum()/len(ids)).backward()
                        opt.step()
                pred = evaluate(net,hold)
                row = dict(round=r,train=metric(evaluate(net,train)),validation=metric(pred[valid]),heldout=metric(pred))
                arm['rounds'].append(row)
                if public and row['validation']['kl'] < best[0]:
                    best=(row['validation']['kl'],name,r)
                    torch.save(dict(policy=net.state_dict(),optimizer=opt.state_dict(),round=r,name=name,source_sha256=model_sha,parent_lower_digest=state_digest(mature.state_dict())),output/'selected.pt')
                    np.save(output/'selected-held.npy',pred)
                print(json.dumps(dict(arm=name,**row)),flush=True)
                (output/'progress.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    report['selected']=best
    if best[1] is not None:
        pred=np.load(output/'selected-held.npy')
        report['selected_test']=metric(pred[test])
        sg=games[hold][test]; unique=np.unique(sg)
        delta=pred[test,0].astype(np.float64)-initial[test,0]
        sums=np.array([delta[sg==g].sum() for g in unique]); sizes=np.array([(sg==g).sum() for g in unique])
        boot=np.random.default_rng(seed+1).integers(len(unique),size=(10000,len(unique)))
        report['test_ce_delta_paired_games']=dict(delta=float(delta.mean()),ci95=np.quantile(sums[boot].sum(1)/sizes[boot].sum(1),[.025,.975]).tolist(),games=len(unique))
    assert fixed()==frozen
    report.update(resources=mon.summary,nonlower_policy_and_upper_adam_unchanged=True,seconds=time.monotonic()-start)
    (output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('bank','source','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--rounds',type=int,default=6,choices=range(1,25))
    p.add_argument('--seed',type=int,default=20261105)
    p.add_argument('--device',default='cuda:0')
    fit(**vars(p.parse_args()))
