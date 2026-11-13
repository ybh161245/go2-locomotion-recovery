"""Replay the archived first rollout/update and instrument learning signals exactly."""
import copy,csv,hashlib,json,pickle,random,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_schedule_fn
from archive_project import ROOT
from go2_exposure_course_env import Go2ExposureCourseEnv
from go2_quality_guard import equal_state

ACTIVE=None
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def csv_write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def describe(x):
    x=np.asarray(x);return dict(mean=float(x.mean()),median=float(np.median(x)),min=float(x.min()),max=float(x.max()),std=float(x.std()))


def replay_physics(source,cfg,first,out):
    groups=cfg['groups'];course=read(source/'acceptance.json');b=first['buffer']
    assert b.full and not b.generator_ready and b.observations.shape[:2]==(512,8)
    with (source/'first-rollout-vecnormalize.pkl').open('rb') as f:norm=pickle.load(f)
    arrays=[];rows=[];max_component_error=0.
    for worker in range(8):
        env=Go2ExposureCourseEnv(worker=worker,config=course)
        by_worker=[];count=0
        try:
            for meta_path in sorted((source/'episodes'/f'worker-{worker}').glob('*.json')):
                if count==512:break
                meta=read(meta_path);env.reset(seed=meta['reset_seed'])
                assert env.episode==meta['episode'] and env.mass_scale==meta['mass_scale']
                with np.load(meta_path.with_suffix('.npz')) as z:
                    for i in range(min(len(z['action']),512-count)):
                        obs=env.observe().copy();np.testing.assert_array_equal(obs,z['observation'][i])
                        np.testing.assert_array_equal(norm.normalize_obs(obs),b.observations[count,worker])
                        np.testing.assert_array_equal(np.clip(b.actions[count,worker],-1,1),z['action'][i])
                        _,reward,term,trunc,info=env.step(z['action'][i])
                        for key,current in [('qpos',env.data.qpos),('qvel',env.data.qvel),('target',env.data.ctrl)]:np.testing.assert_array_equal(current,z[key][i])
                        assert reward==z['reward'][i] and term==z['terminated'][i] and trunc==z['truncated'][i]
                        force=bool(np.linalg.norm(z['world_wrench'][i,:3])>0)
                        ends=[e['start_step']+e['duration_steps'] for e in meta['events']]
                        recovery=any(end<=i<end+cfg['recovery_control_steps'] for end in ends) and not force
                        if worker>=4:group='normal'
                        elif force:group='force'
                        elif recovery:group='recovery'
                        elif meta['events'] and i<min(e['start_step'] for e in meta['events']):group='pressure_before'
                        else:group='pressure_other'
                        lin,ang=env.velocity();cmd=env.filtered_command
                        moving=np.linalg.norm(cmd)>.025
                        err=float(np.square(lin[:2]-cmd[:2]).sum());ey=float((ang[2]-cmd[2])**2)
                        components=dict(base_tracking=info['tracking_reward'],alive=.25,base_cost=-info['costs'],termination=-5. if term else 0.,
                            motion_quality=info['quality_reward'],comfort_tracking=float(np.exp(-err/.16)) if moving else 0.,
                            yaw_tracking=float(2*np.exp(-ey/.08)),linear_tracking=float(3*np.exp(-err/.04)),
                            tracking_square_cost=-float(.5*(lin[0]-cmd[0])**2+.15*ey),vertical_cost=-info['vertical_finetune_cost'])
                        error=abs(sum(components.values())-reward);max_component_error=max(max_component_error,error);assert error<1e-10
                        grav=env.gravity();tilt=float(np.degrees(np.arccos(np.clip(-grav[2],-1,1))))
                        row=dict(worker=worker,rollout_step=count,episode=meta['episode'],episode_step=i,group=group,
                            force_N=float(np.linalg.norm(z['world_wrench'][i,:3])),mass_scale=env.mass_scale,tilt_deg=tilt,
                            reward=reward,normalized_reward=float(b.rewards[count,worker]),advantage=float(b.advantages[count,worker]),
                            value=float(b.values[count,worker]),return_target=float(b.returns[count,worker]),**components)
                        rows.append(row);by_worker.append(groups.index(group));count+=1
            assert count==512
        finally:env.close()
        arrays.append(np.array(by_worker))
        print(f'EXACT_PHYSICS_REPLAY worker={worker} transitions=512',flush=True)
    labels=np.stack(arrays,axis=1);csv_write(out/'transitions.csv',rows)
    np.savez_compressed(out/'derived-first-rollout.npz',group=labels,advantages=b.advantages,returns=b.returns,values=b.values)
    report=[]
    for group in groups:
        items=[r for r in rows if r['group']==group]
        entry=dict(group=group,count=len(items),fraction=len(items)/4096)
        if items:
            entry.update(reward=describe([r['reward'] for r in items]),advantage=describe([r['advantage'] for r in items]),
                positive_advantage_fraction=float(np.mean([r['advantage']>0 for r in items])),
                mean_components={k:float(np.mean([r[k] for r in items])) for k in components})
        report.append(entry)
    return labels,dict(transitions=4096,groups=report,reward_component_max_error=max_component_error,
        raw_obs_normalized_obs_clipped_action_states_targets_rewards_exact=True)


def vector(parameters):return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten() for p in parameters])
def cosine(a,b):
    denominator=float(a.norm()*b.norm())
    return float(torch.dot(a,b)/denominator) if denominator>1e-20 else None


def replay_optimizer(source,cfg,first,labels,out):
    course=read(source/'acceptance.json');parent=ROOT/'artifacts/runs'/cfg['parent']
    model=PPO.load(parent/(cfg['parent_checkpoint']+'.zip'),device='cpu')
    model.learning_rate=course['learning_rate'];model.lr_schedule=get_schedule_fn(course['learning_rate'])
    model.set_logger(configure(str(out/'optimizer-log'),['csv']))
    model.rollout_buffer=copy.deepcopy(first['buffer'])
    for k,v in first['context'].items():setattr(model,k,copy.deepcopy(v))
    assert model.ent_coef==0 and model.clip_range_vf is None
    named=list(model.policy.named_parameters())
    actor=[p for n,p in named if n=='log_std' or n.startswith(('mlp_extractor.policy_net.','action_net.'))]
    critic=[p for n,p in named if n.startswith(('mlp_extractor.value_net.','value_net.'))]
    assert len(actor)+len(critic)==len(named),'Shared feature parameters require an explicit new decomposition.'
    group_flat=model.rollout_buffer.swap_and_flatten(labels[...,None]).flatten()
    original_samples=model.rollout_buffer._get_samples;context={}
    def get_samples(ids,env=None):
        result=original_samples(ids,env);context.update(ids=ids.copy(),batch=result);return result
    model.rollout_buffer._get_samples=get_samples
    clip=torch.nn.utils.clip_grad_norm_;logs=[];max_sum_error=0.
    def instrument(parameters,max_norm,*args,**kwargs):
        nonlocal max_sum_error
        parameters=list(parameters);a=vector(actor);v=vector(critic)
        data=context['batch'];ids=context['ids'];rng=torch.get_rng_state().clone()
        _,logp,_=model.policy.evaluate_actions(data.observations,data.actions)
        adv=(data.advantages-data.advantages.mean())/(data.advantages.std()+1e-8)
        ratio=torch.exp(logp-data.old_log_prob);eps=model.clip_range(model._current_progress_remaining)
        individual=-torch.min(adv*ratio,adv*torch.clamp(ratio,1-eps,1+eps))
        contributions={};counts={}
        for j,group in enumerate(cfg['groups']):
            mask=torch.as_tensor(group_flat[ids]==j);counts[group]=int(mask.sum())
            if counts[group]:
                grads=torch.autograd.grad(individual[mask].sum()/len(ids),actor,retain_graph=True,allow_unused=True)
                contributions[group]=torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten() for g,p in zip(grads,actor)])
            else:contributions[group]=torch.zeros_like(a)
        total=sum(contributions.values());error=float((total-a).abs().max());max_sum_error=max(max_sum_error,error)
        assert error<=cfg['gradient_sum_max_absolute_error'],error
        assert torch.equal(torch.get_rng_state(),rng)
        total_norm=float(torch.sqrt(a.square().sum()+v.square().sum()))
        active=contributions['force']+contributions['recovery']
        entry=dict(minibatch=len(logs)+1,actor_norm=float(a.norm()),weighted_critic_norm=float(v.norm()),
            critic_energy_fraction=float(v.square().sum()/(a.square().sum()+v.square().sum())),
            global_grad_norm=total_norm,clip_coefficient=min(1.,max_norm/(total_norm+1e-6)),
            normal_active_cosine=cosine(contributions['normal'],active),
            gradient_group_sum_max_error=error)
        for group in cfg['groups']:
            entry[group+'_count']=counts[group];entry[group+'_gradient_norm']=float(contributions[group].norm())
            entry[group+'_full_gradient_cosine']=cosine(contributions[group],a)
        logs.append(entry)
        return clip(parameters,max_norm,*args,**kwargs)
    torch.nn.utils.clip_grad_norm_=instrument
    random.setstate(first['rng'][0]);np.random.set_state(first['rng'][1]);torch.set_rng_state(first['rng'][2])
    try:PPO.train(model)
    finally:torch.nn.utils.clip_grad_norm_=clip
    candidate=PPO.load(source/'update-candidates/update-01.zip',device='cpu')
    assert equal_state(model.policy.state_dict(),candidate.policy.state_dict()),'Instrumented policy differs from archived candidate'
    assert equal_state(model.policy.optimizer.state_dict(),candidate.policy.optimizer.state_dict()),'Instrumented Adam differs'
    csv_write(out/'minibatches.csv',logs)
    applicable=[r['normal_active_cosine'] for r in logs if r['normal_active_cosine'] is not None]
    ce=describe([r['critic_energy_fraction'] for r in logs]);cc=describe([r['clip_coefficient'] for r in logs])
    conflict=float(np.mean(np.asarray(applicable)<0)) if applicable else None
    aggregate={group:dict(mean_minibatch_gradient_norm=float(np.mean([r[group+'_gradient_norm'] for r in logs])),
        data_present_minibatches=sum(r[group+'_count']>0 for r in logs)) for group in cfg['groups']}
    return dict(optimizer_steps=len(logs),epochs=model.n_epochs,policy_and_Adam_exact_archived_candidate=True,
        actor_parameters=sum(p.numel() for p in actor),critic_parameters=sum(p.numel() for p in critic),shared_parameters=0,
        critic_gradient_energy_fraction=ce,global_clip_coefficient=cc,normal_active_cosine=describe(applicable) if applicable else None,
        normal_active_conflict_fraction=conflict,gradient_groups=aggregate,gradient_group_sum_max_error=max_sum_error,
        critic_coupling_flag=ce['median']>=cfg['critic_coupling_flag']['median_critic_energy_fraction_min'] and cc['median']<=cfg['critic_coupling_flag']['median_global_clip_coefficient_max'],
        gradient_conflict_flag=conflict is not None and conflict>=cfg['conflict_flag']['negative_cosine_minibatch_fraction_min'])


def main():
    global ACTIVE
    cfg=read(ROOT/'configs/go2-update-diagnosis.json');out=ROOT/cfg['run'];out.mkdir(exist_ok=False);ACTIVE=out
    public=ROOT/cfg['public'];public.mkdir(exist_ok=False);(out/'source').mkdir()
    for p in (ROOT/'scripts').glob('*.py'):shutil.copy2(p,out/'source'/p.name)
    dump(out/'acceptance.json',cfg);shutil.copy2(ROOT/cfg['plan'],out/'predeclared-plan.md')
    source=ROOT/cfg['first_buffer_run'];train=ROOT/cfg['training_run']
    manifest=read(source/'manifest.json')
    for n,h in manifest.items():assert sha(source/n)==h,n
    assert read(train/'post-training-audit.json')['first_update_vs052']['policy_exact']
    assert read(train/'post-training-audit.json')['first_update_vs052']['Adam_exact']
    names={cfg['plan'],'configs/go2-update-diagnosis.json',cfg['training_run']+'/post-training-audit.json',
        cfg['training_run']+'/summary.json','docs/data/go2-current/selection.json',
        '.venv/Lib/site-packages/stable_baselines3/ppo/ppo.py','.venv/Lib/site-packages/stable_baselines3/common/buffers.py'}
    names.update((source/n).relative_to(ROOT).as_posix() for n in manifest)
    names.add(cfg['first_buffer_run']+'/manifest.json')
    names.update(f'artifacts/runs/{cfg["parent"]}/'+n for n in [cfg['parent_checkpoint']+'.zip',cfg['parent_checkpoint']+'-vecnormalize.pkl'])
    for filename in ['go2_env.py','go2_comfort_env.py','go2_transition_env_v2.py','go2_transition_env_v3.py','go2_motion_metrics.py','go2_robust_env.py','go2_sequence_env.py','go2_mass_course_env.py','go2_exposure_course_env.py']:
        assert (ROOT/'scripts'/filename).read_bytes()==(source/'source'/filename).read_bytes(),filename
    inputs={n:sha(ROOT/n) for n in sorted(names)};dump(out/'input-sha256.json',inputs)
    torch.set_num_threads(2);torch.set_num_interop_threads(1);begin=time.perf_counter()
    with (source/'first-rollout.pkl').open('rb') as f:first=pickle.load(f)
    labels,physical=replay_physics(source,cfg,first,out);dump(out/'physical-audit.json',physical)
    optimizer=replay_optimizer(source,cfg,first,labels,out)
    for n,h in inputs.items():assert sha(ROOT/n)==h,n
    if optimizer['critic_coupling_flag']:decision='test_separate_actor_critic_clipping_next_round'
    elif optimizer['gradient_conflict_flag']:decision='inspect_pressure_objective_or_update_allocation_next_round'
    else:decision='no_predeclared_optimizer_bottleneck_proven_capture_all_rollout_buffers_next_round'
    summary=dict(config=cfg,physical=physical,optimizer=optimizer,decision=decision,elapsed_seconds=time.perf_counter()-begin,
        newly_sampled_training_steps=0,archived_transitions_replayed=4096,archived_optimizer_updates_replayed=1,
        new_candidates=0,formal_selection_changed=False,thresholds_unchanged=True,held_out_executed=False,
        limitations='First rejected update only. Group gradients use exact production minibatch advantage normalization. Repeated epoch batches are not independent. Flags do not establish a causal capability limit.')
    dump(out/'summary.json',summary);dump(public/'summary.json',summary)
    dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(dict(decision=decision,optimizer=optimizer,seconds=summary['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        if ACTIVE is not None:(ACTIVE/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
