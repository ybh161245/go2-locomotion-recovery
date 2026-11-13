"""Frozen model stress matrix: direction, magnitude, timing, pulses and friction."""
import csv,hashlib,json,shutil,sys,argparse
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize,DummyVecEnv
from go2_turn_residual_env import configured
from go2_motion_metrics import sample_row
from evaluate_go2_motion import render_views
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/runs/20261023-frozen-stress-friction-corrected-v2'

def cases():
    c=[]
    def add(phase,force=0,direction=(0,1),duration=.2,pulses=(6.,),torque=0,friction=None):
        c.append(dict(id=f'case-{len(c):03}',phase=phase,force=force,direction=direction,duration=duration,pulses=pulses,torque=torque,friction=friction))
    for phase in ['straight','turn','brake','stand']:add(phase)
    for phase in ['straight','turn']:
        for force in [20,40,60,80]:
            for direction in [(1,0),(-1,0),(0,1),(0,-1)]:add(phase,force,direction)
    for phase in ['brake','stand']:
        for force in [40,80]:
            for direction in [(0,1),(0,-1)]:add(phase,force,direction)
    for force in [40,80]:
        for direction in [(0,1),(0,-1)]:add('straight',force,direction,.6)
    for phase in ['straight','turn','brake']:
        add(phase,40,(0,1),.2,(6.,7.,8.))
        add(phase,60,(.7071067812,.7071067812),.3,(6.,7.5),3.)
        for friction in [.4,.2]:add(phase,40,(0,-1),.3,(6.,7.5),0,friction)
    return c

def command(phase,t):
    return np.array([0,0,0] if phase=='stand' or phase=='brake' and t>=6 else [.2,0,.5 if phase=='turn' else 0.])

def run_case(c,seed,env_class,model,norm,video=False):
    env=env_class('external',episode_seconds=12);env.reset(seed=seed,options={'command':command(c['phase'],0)})
    original_friction=float(env.model.geom_friction[env.floor,0])
    if c['friction'] is not None:
        env.model.geom_friction[env.floor,0]=c['friction']
        env.model.geom_friction[list(env.feet),0]=c['friction']
    mass=float(env.model.body_mass.sum());states={k:[] for k in ['observations','actions','qpos','qvel','targets','world_wrench']};rows=[]
    writer=None
    if video:
        folder=OUT/'videos'/c['id'];folder.mkdir(parents=True)
        writer=imageio.get_writer(folder/'video.mp4',fps=25,codec='libx264',macro_block_size=16)
    for i in range(600):
        t=i*.02;env.command[:]=command(c['phase'],t);obs=env.observe()
        active=any(round(start/.02)<=i<round((start+c['duration'])/.02) for start in c['pulses'])
        local=np.array([c['direction'][0]*c['force'],c['direction'][1]*c['force'],0.]) if active else np.zeros(3)
        rotation=env.data.xmat[env.base].reshape(3,3);wrench=np.r_[rotation@local,rotation@np.array([0,0,c['torque'] if active else 0.])]
        env.data.xfrc_applied[:]=0;env.data.xfrc_applied[env.base]=wrench
        action,_=model.predict(norm.normalize_obs(obs.copy()),deterministic=True)
        states['observations'].append(obs.copy());states['actions'].append(action.copy());states['world_wrench'].append(wrench.copy())
        obs,r,term,trunc,info=env.step(action)
        for k,v in [('qpos',env.data.qpos),('qvel',env.data.qvel),('targets',env.data.ctrl)]:states[k].append(v.copy())
        row=sample_row(env,env.meter,info,env.data.time,c['phase']);row.update(vy=float(info['vy']),terminated=term,command_vx=float(env.command[0]),command_yaw=float(env.command[2]))
        row['floor_contact_friction_min']=min([float(cn.friction[0]) for cn in env.data.contact[:env.data.ncon] if env.floor in [int(cn.geom1),int(cn.geom2)]] or [-1])
        rows.append(row)
        if writer and (i+1)%2==0:
            pic=render_views(env);draw=ImageDraw.Draw(pic)
            draw.text((15,15),f"STRESS {c['id']} | {c['phase']} | {c['force']}N | {c['duration']}s | t={t+.02:.2f}",fill='white')
            draw.text((15,45),f"ACTIVE={active} | torque={c['torque']}Nm | FROZEN POLICY | NO TRAINING",fill='#59dfcb')
            if term:draw.text((15,75),'TERMINATED: fall / non-foot contact / unsafe pose; no reset',fill='#ff8b75')
            writer.append_data(np.asarray(pic))
            if i==349 or term:pic.save(folder/'snapshot.png')
        if term or trunc:break
    if writer:writer.close()
    env.data.xfrc_applied[:]=0;env.close()
    packed={k:np.array(v) for k,v in states.items()}
    last=max(c['pulses'])+c['duration'];start=round(last/.02);after=rows[start:]
    band=[np.hypot(r['vx']-r['command_vx'],r['vy'])<=.08 and abs(r['yaw_rate']-r['command_yaw'])<=.15 for r in after]
    recovery=next((i*.02 for i in range(len(band)-24) if all(band[i:i+25])),None)
    peak=max([max(abs(r['roll_deg']),abs(r['pitch_deg'])) for r in rows[300:]] or [999])
    result=dict(case=c,seed=seed,steps=len(rows),terminated=bool(rows[-1]['terminated']),peak_tilt=peak,recovery_seconds=recovery,
        passed=len(rows)==600 and not rows[-1]['terminated'] and peak<=8 and recovery is not None and recovery<=2,
        mass_kg=mass,original_floor_friction=original_friction,force_impulse_norm_Ns=float(np.linalg.norm(packed['world_wrench'][:,:3],axis=1).sum()*.02))
    return result,packed,rows

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True,help='New output folder; existing evidence is never overwritten')
    parser.add_argument('--mode',choices=['full','friction-only'],default='full');args=parser.parse_args()
    OUT=args.out.resolve()
    torch.set_num_threads(1);cfg=json.loads((ROOT/'configs/go2-stress-acceptance.json').read_text());selected=ROOT/'artifacts/runs/20261023-turn-recovery-0.4'
    OUT.mkdir();(OUT/'source').mkdir()
    for f in (ROOT/'scripts').glob('*.py'):shutil.copy2(f,OUT/'source'/f.name)
    cfg['cases']=cases() if args.mode=='full' else [c for c in cases() if c['force']==0 or c['friction'] is not None]
    cfg['friction_correction']='Both floor and foot sliding friction changed; original foot priority1 overrode floor-only variation.'
    (OUT/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    env_class=configured(json.loads((selected/'config.json').read_text()));model=PPO.load(selected/'final-model.zip',device='cpu')
    norm=VecNormalize.load(selected/'final-vecnormalize.pkl',DummyVecEnv([lambda:env_class('external')]))
    norm.training=False;norm.norm_reward=False;mean=norm.obs_rms.mean.copy();count=norm.obs_rms.count
    results=[];controls={}
    for seed in cfg['seeds']:
        for c in cfg['cases']:
            result,packed,rows=run_case(c,seed,env_class,model,norm);folder=OUT/f"{c['id']}-seed-{seed}";folder.mkdir()
            np.savez_compressed(folder/'states.npz',**packed)
            with (folder/'steps.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            key=(seed,c['phase'])
            if c['force']==0:controls[key]=packed['qpos']
            else:
                if c['friction'] is None:np.testing.assert_array_equal(controls[key][:300],packed['qpos'][:300])
                n=min(len(controls[key]),len(packed['qpos']));result['max_horizontal_deviation_m']=float(np.linalg.norm(packed['qpos'][:n,:2]-controls[key][:n,:2],axis=1).max())
            results.append(result);print(json.dumps(dict(id=c['id'],seed=seed,passed=result['passed'],terminated=result['terminated'])),flush=True)
    # Select one successful nonzero-force case and one termination/failure; retain failed videos.
    candidates=[r for r in results if r['seed']==cfg['seeds'][0] and r['case']['force']>0]
    good=next((r for r in candidates if r['passed'] and r['case']['force']>=40),candidates[0])
    bad=next((r for r in candidates if r['terminated']),next((r for r in candidates if not r['passed']),candidates[-1]))
    for r in [good,bad]:
        if (OUT/'videos'/r['case']['id']).exists():continue
        result,packed,_=run_case(r['case'],r['seed'],env_class,model,norm,True)
        with np.load(OUT/f"{r['case']['id']}-seed-{r['seed']}"/'states.npz') as z:np.testing.assert_array_equal(z['qpos'],packed['qpos'])
    np.testing.assert_array_equal(mean,norm.obs_rms.mean);assert count==norm.obs_rms.count;norm.close()
    summary=dict(results=results,total=len(results),passed=sum(r['passed'] for r in results),terminated=sum(r['terminated'] for r in results),
        video_cases=[good['case']['id'],bad['case']['id']],training_steps_added=0,
        note='Terminated means environment safety/contact/pose termination, not automatically classified as a fall. Fixed seed matrix, not broad robustness proof.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    out=ROOT/'docs/data'/OUT.name;out.mkdir(parents=True,exist_ok=False);(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (OUT/'manifest.json').write_text(json.dumps({f.relative_to(OUT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.rglob('*') if f.is_file()},indent=2),encoding='utf-8')

if __name__=='__main__':main()
