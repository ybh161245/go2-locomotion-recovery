"""Archive continuous motion-quality measurements and optional two-view video."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
import torch
from go2_motion_env import Go2MotionEnv
from go2_motion_metrics import MotionMeter,LEGS,sample_row,summarize
from render_go2_policy import SCHEDULE,command_at


def render_views(env):
    if env.renderer is None:
        env.renderer=mujoco.Renderer(env.model,height=480,width=640)
    frames=[]
    for azimuth in (135,45):
        camera=mujoco.MjvCamera()
        camera.type=mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid=env.base
        camera.distance,camera.azimuth,camera.elevation=1.25,azimuth,-18
        option=mujoco.MjvOption();option.geomgroup[3]=0
        env.renderer.update_scene(env.data,camera=camera,scene_option=option)
        frames.append(env.renderer.render().copy())
    picture=Image.new('RGB',(1280,608),(12,22,38))
    picture.paste(Image.fromarray(frames[0]),(0,128))
    picture.paste(Image.fromarray(frames[1]),(640,128))
    return picture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',default='final')
    p.add_argument('--seed',type=int,default=71001)
    p.add_argument('--video',action='store_true')
    p.add_argument('--schedule',choices=['standard','transitions'],default='standard')
    p.add_argument('--out',type=Path)
    args=p.parse_args();torch.set_num_threads(2)
    schedule=SCHEDULE;get_command=command_at
    if args.schedule=='transitions':
        from go2_transition_protocol import SCHEDULE as transition_schedule,command_at as transition_command
        schedule=transition_schedule;get_command=transition_command
    seconds=schedule[-1][1];total_steps=round(seconds/.02)
    out=args.out or args.run/'motion-evaluation'
    if (out/'motion-summary.json').exists():
        raise FileExistsError(f'Existing trajectory evidence at {out}; use a fresh --out directory')
    out.mkdir(parents=True,exist_ok=True)
    stem='final-model' if args.checkpoint=='final' else args.checkpoint
    norm_stem='final' if args.checkpoint=='final' else args.checkpoint
    config=json.loads((args.run/'config.json').read_text(encoding='utf-8'))
    env_class=Go2MotionEnv
    if config['environment']['experiment']=='motion-quality-v2':
        from go2_motion_env_v2 import Go2MotionEnvV2
        env_class=Go2MotionEnvV2
    if config['environment']['experiment']=='comfortable-trot-v1':
        from go2_comfort_env import Go2ComfortEnv
        env_class=Go2ComfortEnv
    if config['environment']['experiment']=='turn-transition-v1':
        from go2_transition_env import Go2TransitionEnv
        env_class=Go2TransitionEnv
    if config['environment']['experiment']=='turn-transition-v2':
        from go2_transition_env_v2 import Go2TransitionEnvV2
        env_class=Go2TransitionEnvV2
    if config['environment']['experiment']=='turn-transition-v3':
        from go2_transition_env_v3 import Go2TransitionEnvV3
        env_class=Go2TransitionEnvV3
    if config['environment']['experiment']=='heading-feedback-v1':
        from functools import partial
        from go2_heading_env import Go2HeadingEnv
        env_class=partial(Go2HeadingEnv,gain=config['environment']['heading_gain'],max_correction=config['environment']['max_heading_correction'])
    if config['environment']['experiment']=='heading-balance-v1':
        from go2_balance_env import Go2BalanceEnv
        env_class=Go2BalanceEnv
    if config['environment']['experiment']=='heading-balance-v2':
        from go2_balance_env_v2 import Go2BalanceEnvV2
        env_class=Go2BalanceEnvV2
    if config['environment']['experiment']=='turn-residual-v1':
        from go2_turn_residual_env import configured
        env_class=configured(config)
    model=PPO.load(args.run/f'{stem}.zip',device='cpu')
    norm=VecNormalize.load(args.run/f'{norm_stem}-vecnormalize.pkl',DummyVecEnv([lambda:env_class('external')]))
    norm.training=False;norm.norm_reward=False
    norm_mean=norm.obs_rms.mean.copy();norm_count=norm.obs_rms.count
    env=env_class('external',episode_seconds=seconds)
    meter=MotionMeter(env.model);obs,_=env.reset(seed=args.seed,options={'command':[0,0,0]})
    rows=[];archive={k:[] for k in ('observations','actions','qpos','qvel','targets')}
    if config['environment']['experiment']=='turn-residual-v1':
        archive.update(joint_actions=[],yaw_command_correction=[],heading_error_rad=[])
    try:font=ImageFont.truetype('arial.ttf',20)
    except OSError:font=ImageFont.load_default()
    writer=imageio.get_writer(str(out/'motion-demo.mp4'),fps=25,codec='libx264',quality=8,macro_block_size=16) if args.video else None
    try:
        for step in range(total_steps):
            cmd,label=get_command(step*.02);env.command[:]=cmd;obs=env.observe()
            action,_=model.predict(norm.normalize_obs(obs.copy()),deterministic=True)
            archive['observations'].append(obs.copy());archive['actions'].append(action.copy())
            obs,reward,term,trunc,info=env.step(action)
            if 'joint_actions' in archive:
                archive['joint_actions'].append(info['joint_action'])
                archive['yaw_command_correction'].append(info['yaw_command_correction'])
                archive['heading_error_rad'].append(info['heading_error_rad'])
            for key,value in [('qpos',env.data.qpos),('qvel',env.data.qvel),('targets',env.data.ctrl)]:
                archive[key].append(value.copy())
            row=sample_row(env,meter,info,env.data.time,label)
            row.update(command_vx=float(cmd[0]),command_yaw=float(cmd[2]),terminated=term,
                       velocity_error=info['velocity_error'],yaw_error=info['yaw_error'])
            rows.append(row)
            if writer and (step+1)%2==0:
                picture=render_views(env);d=ImageDraw.Draw(picture)
                d.text((15,8),f'MOTION QUALITY | {label} | t={env.data.time:.2f}s | seed={args.seed}',font=font,fill='white')
                d.text((15,36),f'COMMAND vx={cmd[0]:+.2f}, yaw={cmd[2]:+.2f} | ACTUAL vx={info["vx"]:+.2f}, yaw={info["yaw_rate"]:+.2f}',font=font,fill='#59dfcb')
                d.text((15,65),'LOADED FEET (>5N): '+ '   '.join(f'{leg}: {"ON" if row[leg+"_contact"] else "--"}' for leg in LEGS),font=font,fill='white')
                controller_label='Frozen joint policy + trained yaw adapter' if 'joint_actions' in archive else 'Learned joint targets | contact and clearance reward experiment'
                d.text((15,94),'TERMINATED - no reset' if term else 'Complete - no fall / no reset' if trunc else controller_label,font=font,fill='#ffae7b' if term else '#59dfcb')
                writer.append_data(np.array(picture))
                if step in (499,849,1149,1449):picture.save(out/f'frame-{step+1}.png')
            if term or trunc:break
    finally:
        if writer:writer.close()
        env.close();norm.close()
    np.testing.assert_array_equal(norm_mean,norm.obs_rms.mean)
    assert norm_count==norm.obs_rms.count
    with (out/'motion-steps.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(out/'motion-transitions.npz',**{k:np.array(v) for k,v in archive.items()})
    segments={}
    for start,end,cmd,label in schedule:
        group=[r for r in rows if start+1<r['time']<=end]
        summary=summarize(group)
        if group:
            summary['velocity_rmse']=float(np.sqrt(np.mean([r['velocity_error']**2 for r in group])))
            summary['yaw_rmse']=float(np.sqrt(np.mean([r['yaw_error']**2 for r in group])))
        segments[f'{start}-{end} {label}']=summary
    report=dict(seed=args.seed,checkpoint=args.checkpoint,steps=len(rows),schedule=args.schedule,completed=len(rows)==total_steps and not rows[-1]['terminated'],
                normalization_frozen=True,segments=segments)
    (out/'motion-summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()}
    (out/'manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
