"""Freeze endpoint baselines and screen paired mass-course checkpoints, without fallback."""
import argparse,copy,hashlib,json,pickle,shutil,time,traceback
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
import evaluate_go2_robust as ev
from diagnose_go2_headroom import support_checks
from go2_parameter_probe import factory,save_probe
from go2_quality_guard import equal_state
from stress_go2_frozen import run_case
ROOT=ev.ROOT

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mass_key(scale):return f'{float(scale):.2f}'
def check_key(check):return check['segment']+'/'+check['metric']
def conditions():
    return [dict(id=f'F30-D0.6-S1-O{offset}',phase='straight',force=30,direction=[0,1],duration=.6,pulses=[6+offset],torque=0,friction=None) for offset in [0,.2,.4]]

def norm_state(norm):
    rms={}
    for key in ['obs_rms','ret_rms']:
        value=getattr(norm,key)
        if isinstance(value,dict):rms[key]={k:{n:copy.deepcopy(getattr(v,n)) for n in ['mean','var','count']} for k,v in value.items()}
        else:rms[key]={n:copy.deepcopy(getattr(value,n)) for n in ['mean','var','count']}
    rms['flags']={key:copy.deepcopy(getattr(norm,key,None)) for key in ['training','norm_obs','norm_reward','clip_obs','clip_reward','epsilon','gamma','norm_obs_keys']}
    return rms

def exact(a,b):
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):return isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(exact(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(exact(v,w) for v,w in zip(a,b))
    return bool(a==b)

def model_identity(folder,stem):
    model_path=folder/('final-model.zip' if stem=='final' else stem+'.zip')
    norm_path=folder/(stem+'-vecnormalize.pkl')
    model=PPO.load(model_path,device='cpu')
    with norm_path.open('rb') as f:norm=pickle.load(f)
    return dict(policy={k:v.detach().cpu().clone() for k,v in model.policy.state_dict().items()},normalization=norm_state(norm),
        files={model_path.relative_to(ROOT).as_posix():sha(model_path),norm_path.relative_to(ROOT).as_posix():sha(norm_path)})

def same_identity(a,b):
    return equal_state(a['policy'],b['policy']) and exact(a['normalization'],b['normalization'])

def adapter_policy(cfg,joint):
    folder=ROOT/'artifacts/runs'/cfg['working_stack']
    model=PPO.load(folder/'final-model.zip',device='cpu')
    norm=VecNormalize.load(folder/'final-vecnormalize.pkl',DummyVecEnv([lambda:ev.stack_factory(joint)('external')]))
    norm.training=False;norm.norm_reward=False
    return model,norm

def prepare(out,cfg,extra_inputs):
    out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
    for p in (ROOT/'scripts').glob('*.py'):shutil.copy2(p,out/'source'/p.name)
    ev.dump(out/'acceptance.json',cfg)
    names=['configs/go2-mass-round.json','docs/data/go2-current/selection.json','docs/data/go2-sustained-A/screen.json']+list(extra_inputs)
    for folder,stem in [(cfg['parent'],cfg['parent_checkpoint']),(cfg['working_stack'],'final'),('20261023-140238-938755-go2-transition-v3-mixed-ppo','final')]:
        names += [f'artifacts/runs/{folder}/'+name for name in ['config.json',('final-model.zip' if stem=='final' else stem+'.zip'),stem+'-vecnormalize.pkl']]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'third_party/unitree_go2').rglob('*') if p.is_file()]
    hashes={name:sha(ROOT/name) for name in sorted(set(names))}
    ev.dump(out/'input-sha256.json',hashes)
    return hashes

def finish(out,summary,hashes):
    for name,h in hashes.items():assert sha(ROOT/name)==h,name
    summary['input_hashes_unchanged']=True
    ev.dump(out/'summary.json',summary)
    ev.dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})

def evaluate_mass(out,label,scale,joint,adapter,cfg):
    spec=dict(id=label,factor='nominal' if scale==1 else 'mass',level=float(scale),scope='mass-course-evaluation')
    make=factory(joint,spec)
    normal=ev.normal_run(out,label,cfg['normal_seed'],make,adapter)
    audits={'normal':save_probe(make,ROOT/normal['folder'])};pressure=[]
    for condition in conditions():
        r,packed,rows=run_case(condition,cfg['development_seed'],make,adapter[0],adapter[1]);r['condition']=r.pop('case')
        r=ev.save_run(out,label,condition['id'],cfg['development_seed'],r,packed,rows)
        audits[condition['id']]=save_probe(make,ROOT/r['folder']);pressure.append(r)
    return dict(normal=normal,pressure=pressure,physics_audits=audits)

def annotate_mass(bundle,reference,cfg,working):
    checks=ev.normal_checks(bundle['normal'],working);oldchecks=ev.normal_checks(reference['normal'],working)
    oldpassed={check_key(c) for c in oldchecks if c['passed']}
    losses=[check_key(c) for c in checks if not c['passed'] and check_key(c) in oldpassed]
    supports=support_checks(bundle['normal'],reference['normal'],cfg['support'])
    previous={r['case']:r for r in reference['pressure']}
    pressure_losses=[r['case'] for r in bundle['pressure'] if previous[r['case']]['passed'] and not r['passed']]
    extra_terms=int(bundle['normal']['terminated'] and not reference['normal']['terminated'])+sum(int(r['terminated'] and not previous[r['case']]['terminated']) for r in bundle['pressure'])
    oldrl=reference['normal']['segments']['reverse right']['RL']['clearance_p95_m']
    rl=bundle['normal']['segments'].get('reverse right',{}).get('RL',{}).get('clearance_p95_m')
    rllimit=cfg['endpoint_existing_clearance_retention']*oldrl
    rlpassed=rl is not None and rl>=rllimit
    bundle.update(normal_checks=checks,normal_failures=sum(not c['passed'] for c in checks),new_normal_failures=losses,
        support_checks=supports,support_failures=sum(not c['passed'] for c in supports),rl_clearance_m=rl,rl_retention_limit_m=rllimit,
        rl_retention_pass=bool(rlpassed),pressure_pass_losses=pressure_losses,additional_terminations=extra_terms)
    bundle['retention_pass']=not losses and not pressure_losses and extra_terms==0 and bundle['support_failures']==0 and rlpassed
    return bundle

def aggregate(by_mass):
    endpoint=[r for key,b in by_mass.items() if key!='1.00' for r in b['pressure']]
    return dict(endpoint_passes=sum(r['passed'] for r in endpoint),endpoint_terminations=sum(r['terminated'] for r in endpoint),
        endpoint_mean_peak=float(np.mean([r['peak_tilt'] for r in endpoint])),
        total_normal_failures=sum(b.get('normal_failures',0) for b in by_mass.values()),
        retention_pass=all(b.get('retention_pass',False) for b in by_mass.values()))

def useful(candidate,reference,cfg):
    spec=cfg['useful_signal']
    return (candidate['endpoint_passes']-reference['endpoint_passes']>=spec['at_least_one_more_endpoint_pressure_pass'] or
        reference['endpoint_terminations']-candidate['endpoint_terminations']>=spec['or_one_fewer_endpoint_termination'] or
        reference['endpoint_mean_peak']-candidate['endpoint_mean_peak']>=spec['or_mean_peak_decrease_deg'])

def make_reference(cfg):
    out=ROOT/cfg['reference'];old=read(ROOT/'docs/data/go2-parameter-sensitivity/summary.json')
    for key in ['parent','parent_checkpoint','working_stack','normal_seed','development_seed','normal_budget']:
        assert old['config'][key]==cfg[key],key
    nominal=dict(normal=old['normals']['nominal'],pressure=[r for r in old['results'] if r['variant']=='nominal'],reused_from='GO2-B-261025-050',physics_audits={})
    assert len(nominal['pressure'])==3
    extra=['docs/data/go2-parameter-sensitivity/summary.json']
    for r in [nominal['normal']]+nominal['pressure']:
        extra += [p.relative_to(ROOT).as_posix() for p in (ROOT/r['folder']).iterdir() if p.is_file()]
    hashes=prepare(out,cfg,extra);started=time.perf_counter()
    joint=ev.load_policy(ROOT/'artifacts/runs'/cfg['parent'],cfg['parent_checkpoint']);adapter=adapter_policy(cfg,joint)
    before=[norm_state(n) for n in [joint[1],adapter[1]]]
    by_mass={'1.00':nominal}
    for scale in cfg['mass_scales']:
        if scale==1:continue
        by_mass[mass_key(scale)]=evaluate_mass(out,'D-mass-'+mass_key(scale),scale,joint,adapter,cfg)
    working=read(ROOT/'docs/data/go2-sustained-A/screen.json')['normals']['working']
    for key,bundle in by_mass.items():annotate_mass(bundle,copy.deepcopy(bundle),cfg,working)
    assert all(exact(a,norm_state(n)) for a,n in zip(before,[joint[1],adapter[1]]))
    joint[1].close();adapter[1].close()
    summary=dict(config=cfg,by_mass=by_mass,physical_runs=8,reused_runs=4,normalization_unchanged=True,reused_nominal_configuration_matches=True,elapsed_seconds=time.perf_counter()-started,**aggregate(by_mass))
    finish(out,summary,hashes)
    public=ROOT/cfg['public'];public.mkdir(parents=True,exist_ok=True);path=public/'reference.json';assert not path.exists();ev.dump(path,summary)
    print(json.dumps(dict(reference=str(out),physical_runs=8,endpoint_passes=summary['endpoint_passes'],endpoint_terminations=summary['endpoint_terminations'])),flush=True)

def screen(cfg):
    refroot=ROOT/cfg['reference'];reference=read(refroot/'summary.json')
    for name,h in read(refroot/'manifest.json').items():assert sha(refroot/name)==h,name
    descriptors=[];extra=[cfg['reference']+'/summary.json',cfg['reference']+'/manifest.json']
    for arm in cfg['arms']:
        folder=ROOT/'artifacts/runs'/(cfg['run_prefix']+arm)
        for steps in cfg['checkpoints']:
            stem='final' if steps==max(cfg['checkpoints']) else f'checkpoint-{steps}'
            identity=model_identity(folder,stem);descriptors.append(dict(arm=arm,added_steps=steps,label=f'{arm}-{steps}',folder=folder,stem=stem,identity=identity))
            extra+=list(identity['files'])
    out=ROOT/cfg['screen'];hashes=prepare(out,cfg,extra);started=time.perf_counter()
    working=read(ROOT/'docs/data/go2-sustained-A/screen.json')['normals']['working']
    identities=[];entries=[];aliases=[];actual_runs=0
    for descriptor in descriptors:
        previous=next((item for item in identities if same_identity(item['identity'],descriptor['identity'])),None)
        if previous is not None:
            entry=copy.deepcopy(previous['entry']);entry.update(arm=descriptor['arm'],added_steps=descriptor['added_steps'],label=descriptor['label'],reused_from=previous['entry']['label'])
            aliases.append(dict(label=descriptor['label'],source=previous['entry']['label'],reason='policy tensors and normalization statistics/flags exactly equal',files=descriptor['identity']['files']))
        else:
            joint=ev.load_policy(descriptor['folder'],descriptor['stem']);adapter=adapter_policy(cfg,joint)
            before=[norm_state(n) for n in [joint[1],adapter[1]]];by_mass={}
            for scale in cfg['mass_scales']:
                key=mass_key(scale);bundle=evaluate_mass(out,descriptor['label']+'-mass-'+key,scale,joint,adapter,cfg)
                by_mass[key]=annotate_mass(bundle,reference['by_mass'][key],cfg,working);actual_runs+=4
            assert all(exact(a,norm_state(n)) for a,n in zip(before,[joint[1],adapter[1]]))
            joint[1].close();adapter[1].close()
            entry=dict(arm=descriptor['arm'],added_steps=descriptor['added_steps'],label=descriptor['label'],by_mass=by_mass,**aggregate(by_mass))
            identities.append(dict(identity=descriptor['identity'],entry=copy.deepcopy(entry)))
        nominal=entry['by_mass']['1.00'];entry['nominal_stage_pass']=nominal['normal_failures']==0 and all(r['passed'] for r in nominal['pressure'])
        entries.append(entry);ev.dump(out/'progress.json',dict(entries=entries,aliases=aliases,actual_new_runs=actual_runs))
        print(json.dumps({k:v for k,v in entry.items() if k!='by_mass'}),flush=True)
    for entry in entries:
        entry['useful_against_D']=useful(entry,reference,cfg)
        control=next(e for e in entries if e['arm']=='control' and e['added_steps']==entry['added_steps'])
        entry['useful_against_control']=useful(entry,control,cfg)
        entry['extension_eligible']=entry['arm']=='randomized' and entry['retention_pass'] and entry['useful_against_D'] and entry['useful_against_control']
    candidates=sorted([e for e in entries if e['extension_eligible']],key=lambda e:(e['endpoint_terminations'],-e['endpoint_passes'],e['total_normal_failures'],e['endpoint_mean_peak'],e['added_steps']))
    selected=candidates[0]['label'] if candidates else None
    summary=dict(config=cfg,reference=reference,entries=entries,ranking=entries,model_aliases=aliases,unique_evaluation_count=len(identities),
        actual_new_runs=actual_runs,physical_runs=actual_runs,reference_new_runs=8,reference_reused_runs=4,needs_extension=bool(selected),selected=selected,
        normalization_unchanged=True,selection_changed=False,held_out_executed=False,visual_acceptance='pending',elapsed_seconds=time.perf_counter()-started)
    finish(out,summary,hashes)
    public=ROOT/cfg['public'];public.mkdir(parents=True,exist_ok=True);path=public/'summary.json';assert not path.exists();ev.dump(path,summary)
    print(json.dumps(dict(actual_new_runs=actual_runs,unique_evaluation_count=len(identities),selected=selected,needs_extension=bool(selected))),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);cfg=read(ROOT/'configs/go2-mass-round.json');ev.CONFIG=cfg
    if args.reference:make_reference(cfg)
    else:screen(cfg)

if __name__=='__main__':main()
