"""249 one finite handoff mechanism; reuse248 simulation and scoring."""
from pathlib import Path
import math,pickle,shutil
import numpy as np
import torch
from stable_baselines3 import PPO
from go2_157 import ROOT,RUNTIME,read,sha,write_new,load_stack
from go2_speed_hold import candidate_inputs
from go2_variable_speed_heading_hold import attach as attach212
from run_go2_full_heading_feedback import attach as attach197
from go2_dual_speed import attach as attach248
from go2_dual_heading_handoff import attach
from go2_action_transfer import norm_load
from go2_sequence import validate
from run_go2_dual_speed_handoff import run,metrics,PARENT

RUN=ROOT/'artifacts/runs/20261108-dual-heading-handoff'
PUBLIC=ROOT/'docs/data/go2-dual-heading-handoff'
CONFIG=ROOT/'configs/go2-dual-heading-handoff.json'
PLAN=ROOT/'docs/experiments/2026-11-08-dual-heading-handoff.md'
OLD=ROOT/'artifacts/runs/20261108-dual-speed-handoff'

def main():
    torch.set_num_threads(1)
    inputs,mechanism=candidate_inputs()
    inputs.update(read(ROOT/'configs/go2-dual-speed-handoff.json')['inputs'])
    for p in (Path(__file__),PLAN,ROOT/'scripts/go2_dual_heading_handoff.py',ROOT/'docs/data/go2-dual-speed-handoff/summary.json'):
        inputs[p.relative_to(ROOT).as_posix()]=sha(p)
    for name,digest in inputs.items():assert sha(ROOT/name)==digest,name
    write_new(CONFIG,dict(experiment=249,inputs=inputs,max_physical=2,training=0,gain=.4,limit=.06))
    RUN.mkdir(exist_ok=False);PUBLIC.mkdir(exist_ok=False)
    for p in (Path(__file__),PLAN,CONFIG,ROOT/'scripts/go2_dual_heading_handoff.py'):shutil.copy2(p,RUN/p.name)
    cfg,stack,factory,_=load_stack(read(RUNTIME));actor=PPO.load(PARENT/'checkpoint-65536.zip',device='cpu');norm=norm_load(PARENT/'checkpoint-65536-norm.pkl')
    reference=read(ROOT/read(RUNTIME)['cases']['normal']['reference_record'])
    mixed=[(0,100,[0.,0.,0.],'stand'),(100,300,[.4,0.,0.],'low'),(300,600,[.8,0.,0.],'high'),(600,800,[.4,0.,0.],'low'),(800,1100,[0.,0.,0.],'stop')]
    task=read(ROOT/'configs/go2-variable-speed-new-conditions.json')['task']
    records=[]
    try:
        for name,schedule,seed in (('mixed',mixed,648101),('low-replay',validate(task['sequence']),513101)):
            _,make=factory(reference)
            env=attach(attach248(attach212(attach197(make('external',episode_seconds=math.nextafter(22.,math.inf)),stack),mechanism),actor,norm))
            try:
                # Capture the actual initial state and fixed model before the runner resets to the same seed.
                env.reset(seed=seed,options={'command':schedule[0][2]})
                initial={k:getattr(env.data,k).copy() for k in ('qpos','qvel','ctrl','qacc_warmstart')}
                physics={k:getattr(env.model,k).copy() for k in ('body_mass','body_inertia','geom_friction','jnt_range','actuator_gainprm','actuator_biasprm','actuator_forcerange')}
                d,t=run(env,stack,schedule,seed,RUN/name)
                np.savez_compressed(RUN/name/'initial-full.npz',**initial);np.savez_compressed(RUN/name/'physics.npz',**physics)
                with (RUN/name/'handoff-calls.pkl').open('xb') as f:pickle.dump(env.handoff_calls,f)
                first=next((i for i,c in enumerate(env.handoff_calls) if c['extra']!=0),1100)
                with np.load(OLD/name/'states.npz') as old:
                    for k in old.files:np.testing.assert_array_equal(d[k][:first],old[k][:first])
                    if name=='low-replay':assert first==1100
                result=metrics(d,t) if name=='mixed' else dict(passed=True)
                result.update(name=name,folder=(RUN/name).relative_to(ROOT).as_posix(),steps=len(d['qpos']),prefix_exact_steps=first,
                    correction_frames=sum(c['extra']!=0 for c in env.handoff_calls),max_extra=max(abs(c['extra']) for c in env.handoff_calls))
                write_new(RUN/name/'result.json',result);records.append(result)
                print('249',result,flush=True)
            finally:env.close()
    finally:stack.close()
    old=read(ROOT/'docs/data/go2-dual-speed-handoff/summary.json');new=records[0]
    retained=all(new['checks'][k] for k,v in old['checks'].items() if v)
    local=new['checks']['path'] and old['all_heading_max_deg']-new['all_heading_max_deg']>=.5 and retained and records[1]['passed']
    write_new(PUBLIC/'summary.json',dict(experiment=249,records=records,local_progress=bool(local),full_task_qualified=new['passed'],new_physical=2,new_training=0,G3_used=4))
    for name,digest in inputs.items():assert sha(ROOT/name)==digest,name

if __name__=='__main__':main()
