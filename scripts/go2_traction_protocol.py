"""093 environmental driver: time-varying global contact material, never a controller input."""
import numpy as np
from archive_project import ROOT
from diagnose_go2_update import dump
from go2_env import servo_model
from go2_parameter_probe import snapshot,assert_snapshot,expected_inputs

def attach_traction_schedule(env,protocol):
    schedule=protocol['friction_schedule'];assert schedule[0]['step']==0 and schedule[0]['mu']==protocol['friction'];native_step=env.step;native_reset=env.reset
    state=dict(i=0);geom_ids=[env.floor]+sorted(env.feet);env.traction_trace=[];env.traction_changes=[]
    def reset(*args,**kwargs):
        state['i']=0;env.traction_trace=[];env.traction_changes=[];env.model.geom_friction[geom_ids,0]=schedule[0]['mu'];return native_reset(*args,**kwargs)
    def step(action):
        i=state['i'];point=next(x for x in reversed(schedule) if x['step']<=i);mu=point['mu'];env.model.geom_friction[geom_ids,0]=mu
        expected=expected_inputs(env.probe_physics_before,dict(factor='friction',level=mu),env.floor,env.feet);assert_snapshot(snapshot(env.model),expected)
        if i==point['step']:env.traction_changes.append(dict(step=i,mu=mu,physics=snapshot(env.model)))
        actual=env.model.geom_friction[geom_ids,0].copy();transition=native_step(action);env.traction_trace.append(dict(step=i,planned_mu=mu,geometry_mu=actual));state['i']+=1;return transition
    env.step=step;env.reset=reset;return env

def save_traction_probe(plain,folder,protocol):
    env=plain.instances[-1];before=env.probe_physics_before;initial=env.probe_physics_after;after=snapshot(env.model);trace=env.traction_trace;telemetry=env.probe_telemetry;assert len(trace)==len(telemetry)>0
    assert_snapshot(before,snapshot(servo_model()));assert_snapshot(snapshot(servo_model()),env.probe_cached_snapshot)
    assert_snapshot(initial,expected_inputs(before,dict(factor='friction',level=protocol['friction']),env.floor,env.feet));assert_snapshot(after,expected_inputs(before,dict(factor='friction',level=trace[-1]['planned_mu']),env.floor,env.feet))
    physics={prefix+key:value for prefix,obj in [('before_',before),('initial_',initial),('after_',after)] for key,value in obj.items()}
    for index,point in enumerate(env.traction_changes):
        assert_snapshot(point['physics'],expected_inputs(before,dict(factor='friction',level=point['mu']),env.floor,env.feet));physics.update({f'change{index}_'+key:value for key,value in point['physics'].items()})
    data={key:np.asarray([row[key] for row in telemetry]) for key in telemetry[0]};data.update(planned_mu=np.asarray([x['planned_mu'] for x in trace]),geometry_mu=np.asarray([x['geometry_mu'] for x in trace]),control_step=np.asarray([x['step'] for x in trace]))
    np.testing.assert_array_equal(data['geometry_mu'],np.repeat(data['planned_mu'][:,None],5,axis=1));present=data['contact_count']>0
    for key in ['effective_sliding_friction_min','effective_sliding_friction_max']:np.testing.assert_allclose(data[key][present],data['planned_mu'][present],rtol=0,atol=1e-12)
    limits=before['actuator_forcerange'];ctrl=before['actuator_ctrlrange'];assert (data['actuator_force']>=limits[:,0]-1e-9).all() and (data['actuator_force']<=limits[:,1]+1e-9).all();assert (data['targets']>=ctrl[:,0]-1e-12).all() and (data['targets']<=ctrl[:,1]+1e-12).all()
    with np.load(folder/'states.npz') as z:np.testing.assert_array_equal(z['targets'],data['targets']);assert len(z['qpos'])==len(trace)
    for name in ['physics.npz','telemetry.npz','probe-audit.json']:assert not (folder/name).exists()
    np.savez_compressed(folder/'physics.npz',**physics);np.savez_compressed(folder/'telemetry.npz',**data)
    dump(folder/'probe-audit.json',dict(spec=env.probe_spec,dynamic_friction_schedule=protocol['friction_schedule'],observed_changes=[dict(step=x['step'],mu=x['mu']) for x in env.traction_changes],steps=len(trace),geometry_ids=[env.floor]+sorted(env.feet),only_declared_sliding_friction_changes=True,total_mass_kg=float(after['body_mass'].sum()),contact_control_steps=int(present.sum()),all_sampled_motor_forces_within_native_limits=True,all_targets_within_native_limits=True,cached_model_unmodified=True,no_extra_forward_at_material_changes=True,max_sampled_motor_limit_ratio=float((np.abs(data['actuator_force'])/np.maximum(np.abs(limits[:,0]),np.abs(limits[:,1]))).max()),caveat='global time-varying material, not spatial terrain; force sampling50Hz, not500Hz'))

def friction_event_metrics(rows,protocol,zero):
    results=[];changes=protocol['friction_schedule'][1:]
    for j,point in enumerate(changes):
        start=point['step'];end=min(len(rows),changes[j+1]['step'] if j+1<len(changes) else len(rows))
        if not zero:
            future=[round(e['start']/.02) for e in protocol['events'] if round(e['start']/.02)>=start];end=min([end]+future)
        band=[np.hypot(r['vx']-r['command_vx'],r['vy'])<=.08 and abs(r['yaw_rate']-r['command_yaw'])<=.15 for r in rows[start:end]];recovery=next((i*.02 for i in range(len(band)-24) if all(band[i:i+25])),None)
        results.append(dict(step=start,time_s=start*.02,mu=point['mu'],window_end_step=end,recovery_seconds=recovery,passed=bool(recovery is not None and recovery<=2.),interrupted_by_later_event=bool(end<len(rows))))
    return results
