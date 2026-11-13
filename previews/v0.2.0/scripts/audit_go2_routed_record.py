"""Independent filtered-command, teacher blend, domain and recovery state-machine audit."""
import numpy as np
import mujoco
from archive_project import ROOT
from diagnose_go2_update import read
from audit_go2_recovery_screen import rows
from go2_mass_course_env import Go2MassCourseEnv
from go2_recovery_state_env import max_tilt

def audit_routed_record(cfg,r,mass):
    folder=ROOT/r['folder'];rs=rows(folder);logs=read(folder/'selector.json');spec=cfg['confirmed_selector']['selector']
    assert len(logs)==len(rs)==r['steps'];env=Go2MassCourseEnv('external',config=cfg)
    env.reset(seed=r['seed'],options={'command':[rs[0]['command_vx'],0,rs[0]['command_yaw']],'events':[],'mass_scale':mass})
    initialq=env.data.qpos.copy();initialv=env.data.qvel.copy();st=dict(active=False,risk=0,safe=0,age=0,cooldown=0,weight=0.,fw=0.);filtered=np.zeros(3)
    try:
        with np.load(folder/'states.npz') as z,np.load(folder/'physics.npz') as physics:
            q=np.vstack([initialq[None],z['qpos'][:-1]]);v=np.vstack([initialv[None],z['qvel'][:-1]]);applied=[]
            for i,l in enumerate(logs):
                cmd=np.array([rs[i]['command_vx'],0.,rs[i]['command_yaw']]);np.testing.assert_array_equal(l['requested_command'],cmd)
                np.testing.assert_allclose(l['filtered_command'],filtered,atol=1e-12,rtol=0)
                target=abs(cmd[2])>spec['max_requested_yaw'];st['fw']+=float(np.clip(float(target)-st['fw'],-.25,.25));assert l['formal_weight']==st['fw']
                D=np.asarray(l['D_action'],dtype=np.float64);formal=np.asarray(l['formal_action'],dtype=np.float32)
                routed=(1-st['fw'])*np.clip(D,-1,1)+st['fw']*np.clip(formal,-1,1) if st['fw'] else D
                np.testing.assert_array_equal(np.clip(routed,-1,1),l['base_action'])
                env.data.qpos[:]=q[i];env.data.qvel[:]=v[i];mujoco.mj_forward(env.model,env.data);lin,ang=env.velocity();tilt=np.degrees(max_tilt(env))
                assert abs(l['tilt']-tilt)<1e-10 and abs(l['vy']-lin[1])<1e-10 and abs(l['angular_xy']-np.linalg.norm(ang[:2]))<1e-10
                old=abs(cmd[0]-.2)<1e-8 and abs(cmd[1])<1e-8 and abs(cmd[2])<=spec['max_requested_yaw'];domain=st['fw']==0 and abs(filtered[2])<=spec['max_requested_yaw'];permitted=old and domain
                risk=bool(permitted and (tilt>=spec['tilt_on_deg'] or abs(lin[1])>=spec['vy_on'] or np.linalg.norm(ang[:2])>=spec['angular_xy_on']) and abs(lin[1])>=spec['vy_confirmation'])
                assert l['old_permitted']==old and l['domain']==domain and l['permitted']==permitted and l['risk']==risk
                st['cooldown']=max(0,st['cooldown']-1);st['risk']=st['risk']+1 if risk else 0
                if not st['active'] and not st['cooldown'] and st['risk']>=spec['risk_steps']:st.update(active=True,age=0,safe=0)
                if st['active']:
                    st['age']+=1;safe=tilt<=spec['safe_tilt_deg'] and np.linalg.norm(lin[:2]-cmd[:2])<=spec['safe_planar_error'] and abs(ang[2]-cmd[2])<=spec['safe_yaw_error'] and np.linalg.norm(ang[:2])<=spec['safe_angular_xy']
                    st['safe']=st['safe']+1 if safe else 0
                    if not permitted or st['age']>=spec['max_active_steps'] or (st['age']>=spec['min_active_steps'] and st['safe']>=spec['safe_steps']):st.update(active=False,risk=0,cooldown=spec['cooldown_steps'])
                st['weight']+=float(np.clip(float(st['active'])-st['weight'],-.25,.25));assert l['active']==st['active'] and l['weight']==st['weight']
                base=np.asarray(l['base_action']);expert=np.asarray(l['expert_action'],dtype=np.float32)
                result=(1-st['weight'])*base+st['weight']*expert if st['weight'] else base
                np.testing.assert_array_equal(result,l['applied_action']);applied.append(result)
                if 'observations' in z:
                    raw=z['observations'][i,:50].copy();raw[11]+=np.float32(.25*l['last_correction']);np.testing.assert_array_equal(raw,np.asarray(l['teacher_observation'],dtype=np.float32))
                filtered+=np.clip(cmd-filtered,-np.array([.8,.8,2.])*.02,np.array([.8,.8,2.])*.02)
            targets=np.clip(np.tile([0.,.8,-1.6],4)+.35*np.asarray(applied),physics['after_actuator_ctrlrange'][:,0],physics['after_actuator_ctrlrange'][:,1]);np.testing.assert_array_equal(targets,z['targets'])
    finally:env.close()
    return dict(folder=r['folder'],steps=len(rs),router_filtered_command_native_targets_and_recovery_state_machine_exact=True,
        formal_weight_steps=sum(l['formal_weight']>0 for l in logs),expert_weight_steps=sum(l['weight']>0 for l in logs))
