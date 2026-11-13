"""Freeze064 inputs; reuse exact063 snapshots and successful demonstration trajectories."""
import copy,json,pickle
import numpy as np
from archive_project import ROOT
from diagnose_go2_update import read,sha,dump
from evaluate_go2_probe_value import verify_manifest

CONFIG='configs/go2-early-recovery-transfer.json'

def main():
    cfg=read(ROOT/'configs/go2-confirmed-selector.json');old=copy.deepcopy(cfg['recovery_specialist'])
    cfg.update(experiment_id='GO2-B-261026-064',run='artifacts/runs/20261026-early-recovery-train',
        screen_run='artifacts/runs/20261026-early-recovery-screen',public='docs/data/go2-early-recovery-transfer',
        predeclared_plan='docs/experiments/2026-10-26-early-recovery-transfer.md')
    spec=cfg['early_recovery']=dict(bank='artifacts/runs/20261026-early-recovery-bank-r2',
        parent=cfg['confirmed_selector']['specialist_train'],parent_checkpoint=131072,
        old_bank=old['bank'],new_bank='artifacts/runs/20261026-sustained-state-diagnosis',
        old_screen='artifacts/runs/20261026-recovery-specialist-screen',regression='artifacts/runs/20261026-recovery-regression',
        training_steps=131072,checkpoints=[65536,131072],seed=164001,learning_rate=3e-5,
        workers=8,n_steps=256,batch_size=256,n_epochs=5,auxiliary_weight=5.,auxiliary_batch=256,
        selector_sha256=sha(ROOT/'scripts/go2_confirmed_selector.py'),model_hashes=copy.deepcopy(cfg['confirmed_selector']['model_hashes']))
    cfg['recovery_specialist']['bank']=spec['bank'];cfg['recovery_specialist'].pop('screen_reuse',None)
    path=ROOT/CONFIG
    if path.exists():
        failed=ROOT/read(path)['early_recovery']['bank'];assert not (ROOT/cfg['run']).exists()
        (failed/'preparation-failed-config.json').write_bytes(path.read_bytes())
        dump(failed/'failure.json',dict(reason='r2 screen reuses snapshot-results from original060 screen, not copied locally',training_samples=0,new_physical_runs=0))
    dump(path,cfg)
    out=ROOT/spec['bank'];out.mkdir(exist_ok=False)
    inputs={n:sha(ROOT/n) for n in [CONFIG,cfg['predeclared_plan'],'docs/data/go2-current/selection.json',*spec['model_hashes']]}
    for root in [ROOT/spec['old_bank'],ROOT/spec['new_bank']]:
        verify_manifest(root)
        for name in ['bank.pkl','summary.json','manifest.json']:inputs[(root/name).relative_to(ROOT).as_posix()]=sha(root/name)
    oldbank=pickle.loads((ROOT/spec['old_bank']/'bank.pkl').read_bytes());newbank=pickle.loads((ROOT/spec['new_bank']/'bank.pkl').read_bytes())
    selected=[e for e in newbank if e['source'] in ['formal','D']]
    bank=[dict(copy.deepcopy(e),course='old30') for e in oldbank]+[dict(copy.deepcopy(e),course='new40') for e in selected]
    assert len(bank)==12 and len(selected)==6
    (out/'bank.pkl').write_bytes(pickle.dumps(bank,protocol=5))
    diag=read(ROOT/spec['new_bank']/'summary.json');oldscreen=read(ROOT/spec['old_screen']/'snapshot-results.json')
    previous=next(x for x in oldscreen if x['steps']==131072)
    formal=[diag['sources']['formal'],next(r for r in diag['branches'] if r['snapshot_id']=='D-step300' and r['controller']=='formal')]
    successful=[r for r in previous['results'] if r['passed']];assert len(successful)==5
    groups=[];sources=[]
    for records,kind in [(formal,'formal40'),(successful,'expert30')]:
        obs=[];actions=[]
        for r in records:
            p=ROOT/r['folder']/'states.npz';inputs[p.relative_to(ROOT).as_posix()]=sha(p)
            with np.load(p) as z:
                cut=300 if r in formal and r.get('cut_step')==0 else 0
                obs.append(z['observations'][cut:cut+150].copy());actions.append(z['actions'][cut:cut+150].copy())
            sources.append(dict(kind=kind,record=r,source_slice=[cut,cut+150]))
        x=np.concatenate(obs);y=np.concatenate(actions);ids=np.linspace(0,len(x)-1,300,dtype=int)
        groups.append((x[ids],np.clip(y[ids],-1,1)))
    assert all(x.shape==(300,50) and y.shape==(300,12) for x,y in groups)
    np.savez_compressed(out/'demonstrations.npz',observations=np.concatenate([x for x,y in groups]),
        actions=np.concatenate([y for x,y in groups]),group=np.repeat(['formal40','expert30'],300))
    dump(out/'demonstration-sources.json',sources);dump(out/'input-sha256.json',inputs)
    dump(out/'summary.json',dict(config=cfg,states=[dict(id=e['id'],course=e['course'],cut=e['snapshot']['control']['step_count']) for e in bank],
        demonstrations=600,formal40=300,expert30=300,old_successful_ids=[r['case'] for r in successful],new_physical_runs=0,
        snapshot_exactness_from_prior_audits=True,independent_generalization_claim=False))
    dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(dict(config=CONFIG,bank=str(out),states=len(bank),demonstrations=600)))

if __name__=='__main__':main()
