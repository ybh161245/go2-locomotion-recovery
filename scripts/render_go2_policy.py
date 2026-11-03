"""Record fixed speed/yaw commands, actual motion and a saved learned policy."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from go2_env import Go2VelocityEnv,ENV_CONFIG
from stand_go2 import roll_pitch

SCHEDULE=[(0,2,[0,0,0],'stand'),(2,8,[.2,0,0],'forward 0.2 m/s'),
          (8,14,[.4,0,0],'forward 0.4 m/s'),(14,18,[0,0,0],'stop'),
          (18,24,[.2,0,.5],'left turn'),(24,30,[.2,0,-.5],'right turn'),
          (30,34,[0,0,0],'stop')]


def command_at(t):
    for start,end,cmd,label in SCHEDULE:
        if start<=t<end:return np.array(cmd,dtype=float),label
    return np.zeros(3),'stop'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--seed',type=int,default=71001)
    p.add_argument('--checkpoint',default='final')
    args=p.parse_args();torch.set_num_threads(2)
    out=args.run
    model=PPO.load(out/f'{args.checkpoint}-model.zip',device='cpu')
    norm=VecNormalize.load(out/f'{args.checkpoint}-vecnormalize.pkl',DummyVecEnv([lambda:Go2VelocityEnv('external')]))
    norm.training=False;norm.norm_reward=False
    before=norm.obs_rms.mean.copy();count=norm.obs_rms.count
    env=Go2VelocityEnv('external',render_mode='rgb_array',episode_seconds=34)
    obs,_=env.reset(seed=args.seed,options={'command':[0,0,0]})
    rows=[];archive={k:[] for k in ('observations','actions','qpos','qvel','targets')}
    try:
        try:font=ImageFont.truetype('arial.ttf',17)
        except OSError:font=ImageFont.load_default()
        writer=imageio.get_writer(str(out/'command-demo.mp4'),fps=25,codec='libx264',quality=8,macro_block_size=16,pixelformat='yuv420p')
        ended=False
        for step in range(1700):
            time=step*ENV_CONFIG['control_dt'];cmd,label=command_at(time)
            if not ended:
                env.command[:]=cmd;obs=env.observe()
                action,_=model.predict(norm.normalize_obs(obs.copy()),deterministic=True)
                archive['observations'].append(obs.copy());archive['actions'].append(action.copy())
                obs,reward,term,trunc,info=env.step(action)
                archive['qpos'].append(env.data.qpos.copy());archive['qvel'].append(env.data.qvel.copy())
                archive['targets'].append(env.data.ctrl.copy())
                roll,pitch=roll_pitch(env.data.qpos[3:7])
                rows.append(dict(step=step,time=float(env.data.time),phase=label,reward=reward,
                                 roll_deg=float(roll),pitch_deg=float(pitch),terminated=term,truncated=trunc,**info))
                ended=term or trunc
                image=env.render()
            if (step+1)%2==0:
                picture=Image.fromarray(image.copy());d=ImageDraw.Draw(picture)
                d.rectangle((0,0,960,130),fill=(12,22,38))
                d.text((16,9),f'GO2 | PPO POLICY | {label} | t={time+.02:.2f}s | seed={args.seed}',font=font,fill='white')
                d.text((16,37),f'COMMAND vx={cmd[0]:+.2f} m/s, yaw={cmd[2]:+.2f} rad/s',font=font,fill='#59dfcb')
                d.text((16,65),f'ACTUAL  vx={info["vx"]:+.2f} m/s, yaw={info["yaw_rate"]:+.2f} rad/s | height={info["height"]:.2f}m',font=font,fill='white')
                status='Learned joint targets -> torque-limited position servos'
                if ended:
                    status=('TERMINATED - last frame held; no reset' if term else
                            'Command schedule complete | no falls / no resets' if len(rows)==1700 else
                            'TIME LIMIT - last frame held; no reset')
                d.text((16,94),status,font=font,fill='#ffae7b' if ended and term else '#59dfcb' if ended else 'white')
                writer.append_data(np.asarray(picture))
                if step==499:picture.save(out/'command-demo.png')
        writer.close()
    finally:
        env.close();norm.close()
    np.testing.assert_array_equal(before,norm.obs_rms.mean)
    if count!=norm.obs_rms.count:raise RuntimeError('Evaluation normalization changed')
    np.savez_compressed(out/'demo-transitions.npz',**{k:np.array(v) for k,v in archive.items()})
    with (out/'demo-steps.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with imageio.get_reader(str(out/'command-demo.mp4')) as r:frames=sum(1 for _ in r)
    if frames!=850:raise RuntimeError('Incomplete demo video')
    report=dict(seed=args.seed,checkpoint=args.checkpoint,seconds=34,frames=frames,
                physics_control_steps_recorded=len(rows),schedule=SCHEDULE,
                completed=not any(r['terminated'] for r in rows),normalization_frozen=True,
                files={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in
                       ('command-demo.mp4','command-demo.png','demo-transitions.npz','demo-steps.csv')})
    (out/'demo-manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
