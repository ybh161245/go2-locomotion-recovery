"""Prospectively budgeted full-stack screen for full-joint robustness finetuning."""
import argparse
import copy
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from go2_robust_env import CONFIG, ROOT
from go2_transition_env_v3 import Go2TransitionEnvV3
from go2_turn_residual_env import configured
from go2_motion_metrics import MotionMeter, sample_row, summarize, LEGS
from go2_transition_protocol import SCHEDULE, command_at
from stress_go2_frozen import cases, run_case


def dump(path,obj): path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')


def load_policy(folder,stem='final'):
    model=PPO.load(folder/('final-model.zip' if stem=='final' else f'{stem}.zip'),device='cpu')
    norm=VecNormalize.load(folder/f'{stem}-vecnormalize.pkl',DummyVecEnv([lambda:Go2TransitionEnvV3('external')]))
    norm.training=False;norm.norm_reward=False
    return model,norm


def stack_factory(joint_policy=None):
    cfg=json.loads((ROOT/'artifacts/runs'/CONFIG['working_stack']/'config.json').read_text())
    cls=configured(cfg)
    def make(*args,**kwargs):
        env=cls(*args,**kwargs)
        # Avoid cached-model cross-environment contamination during friction tests.
        env.model=copy.copy(env.model);env.data=mujoco.MjData(env.model);env.meter=MotionMeter(env.model)
        if joint_policy is not None:env.teacher,env.teacher_norm=joint_policy
        return env
    return make


def normal_run(out,label,seed,cls,policy):
    env=cls('external',episode_seconds=46);env.reset(seed=seed,options={'command':[0,0,0]})
    rows=[];states={k:[] for k in ['qpos','qvel','targets','observations','actions','world_wrench','feet_world','heading']}
    for i in range(2300):
        cmd,phase=command_at(i*.02);env.command[:]=cmd;obs=env.observe()
        action,_=policy[0].predict(policy[1].normalize_obs(obs.copy()),deterministic=True)
        _,_,term,trunc,info=env.step(action)
        for k,v in [('qpos',env.data.qpos),('qvel',env.data.qvel),('targets',env.data.ctrl),('observations',obs),('actions',action),('world_wrench',np.zeros(6)),('feet_world',env.data.geom_xpos[env.meter.feet]),('heading',env.data.xmat[env.base].reshape(3,3)[:,0])]:states[k].append(v.copy())
        row=sample_row(env,env.meter,info,env.data.time,phase);row.update(vy=info['vy'],terminated=term,command_vx=cmd[0],command_yaw=cmd[2]);rows.append(row)
        if term or trunc:break
    env.close();packed={k:np.array(v) for k,v in states.items()}
    result=dict(variant=label,seed=seed,case='normal',steps=len(rows),terminated=bool(term),segments={})
    for start,end,cmd,phase in SCHEDULE:
        ids=[i for i,r in enumerate(rows) if start+1<r['time']<=end+1e-8]
        group=[rows[i] for i in ids];s=summarize(group)
        s['command']=cmd
        if group:
            s['velocity_rmse']=float(np.sqrt(np.mean([(r['vx']-cmd[0])**2+r['vy']**2 for r in group])))
            s['yaw_rmse']=float(np.sqrt(np.mean([(r['yaw_rate']-cmd[2])**2 for r in group])))
            s['peak_tilt']=max(max(abs(r['roll_deg']),abs(r['pitch_deg'])) for r in rows if start<r['time']<=end+1e-8)
            # Touchdown-to-touchdown displacement along mean heading; descriptive.
            lengths=[]
            for leg in range(4):
                armed=False;last_contact=True;streak=0;last=None
                for index in ids:
                    contact=bool(rows[index][LEGS[leg]+'_contact'])
                    streak=streak+1 if contact==last_contact else 1
                    if not contact and streak>=2:armed=True
                    if contact and streak==2 and armed:
                        if last is not None and .2<(index-last)*.02<1.5:
                            direction=packed['heading'][index]+packed['heading'][last];direction[2]=0;direction/=np.linalg.norm(direction)
                            lengths.append(float(np.dot(packed['feet_world'][index,leg]-packed['feet_world'][last,leg],direction)))
                        last=index;armed=False
                    last_contact=contact
            s['stride_median_m']=float(np.median(lengths)) if lengths else None
        result['segments'][phase]=s
    q=packed['qpos'][:,3:7];w,x,y,z=q.T;yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    result['turn_angles']=[float(np.degrees(yaw[round(end/.02)-1]-yaw[round(start/.02)-1])) for start,end in [(17,23),(23,29),(29,35)]] if len(rows)==2300 else None
    return save_run(out,label,'normal',seed,result,packed,rows)


def save_run(out,label,case,seed,result,packed,rows):
    folder=out/f'{label}-{case}-seed-{seed}';folder.mkdir()
    np.savez_compressed(folder/'states.npz',**packed)
    with (folder/'steps.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result.update(variant=label,case=case,seed=seed,folder=folder.relative_to(ROOT).as_posix())
    dump(folder/'result.json',result)
    print(json.dumps({k:result[k] for k in ['variant','case','steps','terminated']}),flush=True)
    return result


def normal_checks(candidate,baseline):
    b=CONFIG['normal_budget'];checks=[]
    def add(segment,metric,value,limit,passed):checks.append(dict(segment=segment,metric=metric,value=value,limit=limit,passed=bool(passed)))
    add('all','full_episode',candidate['steps'],2300,candidate['steps']==2300 and not candidate['terminated'])
    for name,s in candidate['segments'].items():
        old=baseline['segments'][name]
        if s['samples']==0:add(name,'segment_present',0,1,False);continue
        add(name,'peak_tilt',s['peak_tilt'],8,s['peak_tilt']<=8)
        for metric,cap in [('velocity_rmse',b['per_segment_velocity_rmse_max']),('yaw_rmse',b['per_segment_yaw_rmse_max'])]:
            limit=min(cap,old[metric]+max(old[metric]*b['rmse_relative_increase'],b['rmse_absolute_floor_increment']))
            add(name,metric,s[metric],limit,s[metric]<=limit)
        if np.linalg.norm(s['command'])==0:
            for metric,cap,inc in [('joint_speed_rms_mean',b['stop_joint_speed_rms_max'],b['stop_joint_speed_increase']),('roll_deg_std',b['stop_roll_pitch_std_deg_max'],b['stop_angle_std_increase_deg']),('pitch_deg_std',b['stop_roll_pitch_std_deg_max'],b['stop_angle_std_increase_deg'])]:
                limit=min(cap,old[metric]+inc);add(name,metric,s[metric],limit,s[metric]<=limit)
        else:
            if 'straight_stride_preserved_ratio' in b and name in ['start 0.2','accelerate 0.4']:
                value=s.get('stride_median_m');old_stride=old.get('stride_median_m')
                limit=old_stride*b['straight_stride_preserved_ratio'] if old_stride is not None else None
                add(name,'stride_median_m',value,limit,value is not None and limit is not None and value>=limit)
            limit=old['height_std']+max(old['height_std']*b['moving_height_std_increase_ratio'],b['moving_height_std_increment_floor_m'])
            add(name,'height_std',s['height_std'],limit,s['height_std']<=limit)
            for leg in LEGS:
                value=s[leg]['clearance_p95_m'];limit=max(b['foot_clearance_p95_min_m'],old[leg]['clearance_p95_m']*b['foot_clearance_preserved_ratio'])
                add(name,leg+'_clearance',value,limit,value>=limit)
        for leg in LEGS:
            value=s[leg]['loaded_foot_speed_mean_m_s'];oldval=old[leg]['loaded_foot_speed_mean_m_s']
            limit=oldval+max(oldval*b['foot_loaded_speed_increase_ratio'],b['foot_loaded_speed_increment_floor_m_s']) if oldval is not None else 0
            add(name,leg+'_loaded_speed',value,limit,value is not None and value<=limit)
    if candidate['turn_angles'] is not None:
        for i,(value,old,sign) in enumerate(zip(candidate['turn_angles'],baseline['turn_angles'],[1,-1,1])):
            target=sign*np.degrees(3);error=abs(value-target)
            limit=min(b['turn_angle_abs_error_max_deg'],abs(old-target)+b['turn_angle_error_increase_max_deg'])
            add(f'turn-{i}','angle_error',error,limit,error<=limit)
    else:add('turns','angles_present',None,True,False)
    return checks


def development_cases():
    original={c['id']:c for c in cases()}
    selected=[original[k] for k in CONFIG['core_cases']]
    for sign in CONFIG['repeat_signs']:
        for offset in CONFIG['repeat_offsets']:
            selected.append(dict(id=f'repeat-{sign}-offset-{offset}',phase='straight',force=40,direction=[0,sign],duration=.2,pulses=[6+offset,7+offset,8+offset],torque=0,friction=None))
    return selected


def main():
    global CONFIG
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/go2-robust-finetune.json');p.add_argument('--public',default='go2-robust-finetune');args=p.parse_args()
    CONFIG=json.loads(args.config.read_text(encoding='utf-8'))
    run=args.run.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
    for f in (ROOT/'scripts').glob('*.py'):shutil.copy2(f,out/'source'/f.name)
    dump(out/'acceptance.json',CONFIG);torch.set_num_threads(1)
    stack=ROOT/'artifacts/runs'/CONFIG['working_stack'];adapter=PPO.load(stack/'final-model.zip',device='cpu')
    adapter_norm=VecNormalize.load(stack/'final-vecnormalize.pkl',DummyVecEnv([lambda:stack_factory()('external')]))
    adapter_norm.training=False;adapter_norm.norm_reward=False
    parent=load_policy(ROOT/'artifacts/runs'/CONFIG['parent'],CONFIG.get('parent_checkpoint','final'))
    parent_label='previous-stack' if CONFIG.get('parent_eval_mode')=='stack' else 'parent-only'
    variants=[('working',None),(parent_label,parent)]+[(f'joint-{added}',None) for added in CONFIG['checkpoints']]
    results=[];normals={}
    loaded=[]
    for label,joint in variants:
        if label.startswith('joint-'):
            added=int(label.split('-')[1]);stem='final' if added==CONFIG['steps'] else f'checkpoint-{added}'
            ready=run/('summary.json' if stem=='final' else f'{stem}-vecnormalize.pkl')
            deadline=time.monotonic()+900
            while not ready.exists():
                if (run/'failure.txt').exists() or time.monotonic()>deadline:raise RuntimeError('Training failed or checkpoint wait timed out')
                time.sleep(3)
            joint=load_policy(run,stem);loaded.append(joint)
        cls=Go2TransitionEnvV3 if label=='parent-only' else stack_factory(joint)
        policy=joint if label=='parent-only' else (adapter,adapter_norm)
        normals[label]=normal_run(out,label,CONFIG['normal_seed'],cls,policy)
        for case in development_cases():
            result,packed,rows=run_case(case,CONFIG['development_seed'],cls,*policy)
            result['condition']=result.pop('case')
            results.append(save_run(out,label,case['id'],CONFIG['development_seed'],result,packed,rows))
        dump(out/'progress.json',dict(results=results,normals=normals))
    baseline={r['case']:r for r in results if r['variant']=='working'};ranking=[]
    for label,_ in variants:
        group=[r for r in results if r['variant']==label];checks=normal_checks(normals[label],normals['working'])
        passes=sum(r['passed'] for r in group);terminations=sum(r['terminated'] for r in group)
        bp=sum(r['passed'] for r in baseline.values());bt=sum(r['terminated'] for r in baseline.values())
        losses=[r['case'] for r in group if baseline[r['case']]['passed'] and not r['passed']]
        new_terminations=[r['case'] for r in group if not baseline[r['case']]['terminated'] and r['terminated']]
        benefit=passes>=bp+2 or (terminations<=bt-1 and passes>=bp)
        eligible=label.startswith('joint-') and all(c['passed'] for c in checks) and not losses and not new_terminations and benefit
        ranking.append(dict(variant=label,passes=passes,terminations=terminations,worst_peak=max(r['peak_tilt'] for r in group),normal_pass=all(c['passed'] for c in checks),normal_checks=checks,losses=losses,new_terminations=new_terminations,benefit=benefit,eligible=eligible))
    eligible=sorted([r for r in ranking if r['eligible']],key=lambda r:(r['terminations'],-r['passes'],r['worst_peak'],int(r['variant'].split('-')[1])))
    summary=dict(results=results,normals=normals,ranking=ranking,eligible=[r['variant'] for r in eligible],next_stage='full-regression-required' if eligible else 'stop-no-qualified-candidate',held_out_executed=False,selection_changed=False,visual_acceptance='pending')
    dump(out/'summary.json',summary)
    public=ROOT/'docs/data'/args.public;public.mkdir(exist_ok=False);dump(public/'screen.json',summary)
    for joint in [parent]+loaded:joint[1].close()
    adapter_norm.close()
    dump(out/'manifest.json',{f.relative_to(out).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file()})
    print(json.dumps(dict(eligible=summary['eligible'],ranking=[{k:v for k,v in r.items() if k!='normal_checks'} for r in ranking])),flush=True)


if __name__=='__main__':main()
