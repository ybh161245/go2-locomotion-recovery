"""Recompute 072/073 gates, native inputs, forces, routing and teacher inference from saved data."""
import argparse, copy, json, shutil
import numpy as np
import torch
from stable_baselines3 import PPO
import evaluate_go2_robust as ev
from archive_project import ROOT
from diagnose_go2_update import read, sha, dump
from evaluate_go2_probe_value import verify_manifest
from audit_go2_recovery_screen import rows
from audit_go2_routed_record import audit_routed_record
from audit_go2_transfer_regression import vector_force_audit
from go2_action_transfer import norm_load
from go2_env import servo_model
from go2_motion_metrics import MotionMeter
from go2_parameter_probe import snapshot, expected_inputs, assert_snapshot
from evaluate_go2_confirmed_selector import trajectory_protection


def gate(r):
    rs=rows(ROOT/r['folder']);assert len(rs)==r['steps']
    start=round((max(r['condition']['pulses'])+r['condition']['duration'])/.02)
    band=[np.hypot(x['vx']-x['command_vx'],x['vy'])<=.08 and abs(x['yaw_rate']-x['command_yaw'])<=.15 for x in rs[start:]]
    recovery=next((i*.02 for i in range(len(band)-24) if all(band[i:i+25])),None)
    peak=max([max(abs(x['roll_deg']),abs(x['pitch_deg'])) for x in rs[300:]] or [999])
    assert recovery==r['recovery_seconds'] and peak==r['peak_tilt']
    assert r['passed']==(len(rs)==600 and not r['terminated'] and peak<=8 and recovery is not None and recovery<=2)
    if 'recorded_peak_tilt_all_times' in r:
        assert r['recorded_peak_tilt_all_times']==max(max(abs(x['roll_deg']),abs(x['pitch_deg'])) for x in rs)
        assert r['post6_peak_available']==(len(rs)>300)
    return rs


def physics(r):
    folder=ROOT/r['folder'];report=read(folder/'probe-audit.json');spec=report['spec'];model=servo_model();meter=MotionMeter(model)
    with np.load(folder/'physics.npz') as p,np.load(folder/'telemetry.npz') as t,np.load(folder/'states.npz') as z:
        before={k[7:]:p[k] for k in p.files if k.startswith('before_')};after={k[6:]:p[k] for k in p.files if k.startswith('after_')}
        assert_snapshot(before,snapshot(model));assert_snapshot(after,expected_inputs(before,spec,meter.floor,meter.feet))
        assert r['mass_kg']==float(after['body_mass'].sum())
        np.testing.assert_array_equal(t['targets'],z['targets']);assert len(t['targets'])==r['steps']
        assert (t['actuator_force']>=after['actuator_forcerange'][:,0]-1e-9).all() and (t['actuator_force']<=after['actuator_forcerange'][:,1]+1e-9).all()
        assert (t['targets']>=after['actuator_ctrlrange'][:,0]-1e-12).all() and (t['targets']<=after['actuator_ctrlrange'][:,1]+1e-12).all()
        present=t['contact_count']>0;assert present.any();mu=spec['level'] if spec['factor']=='friction' else .8
        for k in ['effective_sliding_friction_min','effective_sliding_friction_max']:np.testing.assert_allclose(t[k][present],mu,atol=1e-12,rtol=0)
        impulse=float(np.linalg.norm(z['world_wrench'][:,:3],axis=1).sum()*.02);assert impulse==r['force_impulse_norm_Ns']
        if r['condition']['friction'] is not None:assert spec['factor']=='friction' and spec['level']==r['condition']['friction']
        else:assert spec['factor'] in ['nominal','mass']
    return mu


def infer(model,norm,obs):
    return model.predict(norm.normalize_obs(np.asarray(obs,dtype=np.float32).copy()),deterministic=True)[0]


def inference(r,policies,adapter_config):
    folder=ROOT/r['folder'];errors=[]
    with np.load(folder/'states.npz') as z:
        if (folder/'selector.json').exists():
            logs=read(folder/'selector.json');obs=np.asarray([x['teacher_observation'] for x in logs],dtype=np.float32)
            raw=np.asarray([x['base_observation'] for x in logs],dtype=np.float32);np.testing.assert_array_equal(raw,z['observations'][:,:50])
            checks=[('D',np.arange(len(logs)),'D_action',obs),('formal',np.array([i for i,x in enumerate(logs) if x['formal_weight']>0],dtype=int),'formal_action',obs),
                    ('expert',np.array([i for i,x in enumerate(logs) if x['weight']>0],dtype=int),'expert_action',raw)]
            for name,ids,key,source in checks:
                if not len(ids):continue
                expected=np.asarray([logs[i][key] for i in ids]);error=float(np.max(np.abs(infer(*policies[name],source[ids])-expected)));assert error<=2e-6,(r['folder'],name,error);errors.append(error)
        else:
            obs=z['observations'][:,:50].copy();age=np.arange(len(obs))*.02
            correction=z['actions'][:,0]*adapter_config['environment']['correction_limit']*np.minimum(age/.3,1.) if r['condition']['phase']=='turn' else np.zeros(len(obs))
            # These pressure protocols have constant turn sign: no reversal term.
            obs[:,11]+=np.asarray(.25*correction,dtype=np.float32)
            action=np.clip(infer(*policies[r['mode']],obs),-1,1).astype(np.float64)
            with np.load(folder/'physics.npz') as p:target=np.clip(np.tile([0.,.8,-1.6],4)+.35*action,p['after_actuator_ctrlrange'][:,0],p['after_actuator_ctrlrange'][:,1])
            error=float(np.max(np.abs(target-z['targets'])));assert error<=1e-6,(r['folder'],error);errors.append(error/.35)
    return max(errors or [0.])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    cfg=read(ROOT/args.config);ev.CONFIG=cfg;run=ROOT/cfg['run'];s=read(run/'summary.json');verify_manifest(run);torch.set_num_threads(2)
    assert not (run/'pressure-audit.json').exists()
    paths={'D':(ROOT/'artifacts/runs'/cfg['parent'],cfg['parent_checkpoint']),
           'formal':(ROOT/cfg['motion_composition']['formal_joint'],'final'),
           'expert':((ROOT/cfg['motion_composition']['expert']['model']).parent,'checkpoint-250')}
    policies={name:(PPO.load(folder/('final-model.zip' if stem=='final' else stem+'.zip'),device='cpu'),norm_load(folder/(stem+'-vecnormalize.pkl'))) for name,(folder,stem) in paths.items()}
    adapter_cfg=read(ROOT/'artifacts/runs'/cfg['working_stack']/'config.json');records=[r for rs in s['results'].values() for r in rs]
    if 'zero_controls' in s:records += [r for z in s['zero_controls'].values() for k,r in z.items() if k in ['D','routed']]
    assert len(records)==s['new_physical_runs'];forces=[];errors=[];audits=[];frictions={}
    for index,r in enumerate(records):
        gate(r);mu=physics(r);frictions[str(mu)]=frictions.get(str(mu),0)+1;forces.append(vector_force_audit(cfg,r,1.));errors.append(inference(r,policies,adapter_cfg))
        if r['mode']=='routed':audits.append(audit_routed_record(cfg,r,1.))
        if (index+1)%40==0:print('AUDIT_PROGRESS '+str(index+1),flush=True)
    if 'zero_controls' in s:
        for proof in s['zero_controls'].values():
            with np.load(ROOT/proof['D']['folder']/'states.npz') as x,np.load(ROOT/proof['routed']['folder']/'states.npz') as y:
                assert set(x.files)==set(y.files)
                for k in x.files:np.testing.assert_array_equal(x[k],y[k])
            logs=read(ROOT/proof['routed']['folder']/'selector.json');assert all(x['weight']==x['formal_weight']==0 for x in logs)
        d={r['case']:r for r in s['results']['D']}
        for r in s['results']['routed']:assert trajectory_protection(r,d[r['case']])==s['pre_force_protections'][r['case']]
        assert s['all16_pass'] and s['reserved_qualification_pass'] and all(r['passed'] for r in s['results']['routed'])
    for n,h in read(run/'input-sha256.json').items():assert sha(ROOT/n)==h,n
    for n,h in cfg['command_router']['model_hashes'].items():assert sha(ROOT/n)==h,n
    report=dict(raw_records=len(records),router_records=len(audits),gates_and_recovery_recomputed=True,all_native_physics_and_contact_friction_verified=True,
        records_by_effective_friction=frictions,full_world_force_and_torque_vectors_verified=True,maximum_force_vector_error=max(forces),
        teacher_observations_and_batch_inference_verified=True,maximum_batch_action_error=max(errors),routed_filters_weights_native_targets_state_machines_verified=True,
        no_new_physics_integration=True,models_inputs_unchanged=True,formal_selection_changed=False,visual_acceptance='pending')
    shutil.copy2(ROOT/'scripts/audit_go2_routed_pressure.py',run/'pressure-audit-runtime-source.py')
    dump(run/'pressure-audit.json',report);dump(ROOT/cfg['public']/'pressure-audit.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
