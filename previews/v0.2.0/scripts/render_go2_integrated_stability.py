"""253 existing poses only, baseline versus last checkpoint."""
from unittest.mock import patch
import shutil
import mujoco,numpy as np,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from go2_env import Go2VelocityEnv
from go2_157 import ROOT,sha,write_new
def main():
    out=ROOT/'artifacts/integrated-stability-video-253';out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'renderer.py')
    paths=[ROOT/'artifacts/runs/20261108-dual-heading-handoff/mixed/states.npz',ROOT/'artifacts/runs/20261111-integrated-stability/65536-mixed/states.npz']
    data=[]
    for p in paths:
        with np.load(p) as z:data.append({k:z[k] for k in ('qpos','qvel','targets','terminated')})
    env=Go2VelocityEnv('external');r=mujoco.Renderer(env.model,height=336,width=560)
    cam=mujoco.MjvCamera();cam.distance,cam.azimuth,cam.elevation=1.4,135,-25
    font=ImageFont.truetype('msyh.ttc',18)
    try:
        with patch.object(mujoco,'mj_step',side_effect=AssertionError('render only')),imageio.get_writer(str(out/'comparison.mp4'),fps=25,codec='libx264',quality=8,macro_block_size=16) as writer:
            for i in range(1,1100,2):
                frame=Image.new('RGB',(1120,432),'#17242d');draw=ImageDraw.Draw(frame)
                draw.text((12,5),f'实际整合稳定训练 · {(i+1)*.02:.2f}s · 资格以完整数值记录为准',font=font,fill='white')
                for j,d in enumerate(data):
                    k=min(i,len(d['qpos'])-1);env.data.qpos[:]=d['qpos'][k];env.data.qvel[:]=d['qvel'][k];env.data.ctrl[:]=d['targets'][k];env.data.time=(k+1)*.02
                    mujoco.mj_forward(env.model,env.data);cam.lookat[:]=(*env.data.qpos[:2],.23);r.update_scene(env.data,camera=cam)
                    frame.paste(Image.fromarray(r.render()),(j*560,60))
                    draw.text((j*560+12,33),['249 基线','253 65536步检查点'][j],font=font,fill='#7ce0bc')
                    label=f"提前终止于 {len(d['qpos'])*.02:.2f}s，末帧保持" if d['terminated'][-1] and i>=len(d['qpos']) else '原始保存轨迹 · 新视觉待验'
                    draw.text((j*560+12,403),label,font=font,fill='white')
                writer.append_data(np.asarray(frame))
    finally:r.close();env.close()
    with imageio.get_reader(str(out/'comparison.mp4')) as v:assert v.count_frames()==550
    write_new(out/'manifest.json',dict(experiment=253,frames=550,new_physical=0,inputs={str(p.relative_to(ROOT)):sha(p) for p in paths},visual='pending'))
if __name__=='__main__':main()
