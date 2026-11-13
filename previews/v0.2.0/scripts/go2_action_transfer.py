"""065 paired supervised objective test; no additional PPO/environment training."""
import argparse,copy,csv,hashlib,json,pickle,random,shutil,subprocess,time
from zipfile import ZipFile
import numpy as np
import torch
from stable_baselines3 import PPO
from archive_project import ROOT,write_archive
from diagnose_go2_update import read,sha,dump
from evaluate_go2_probe_value import verify_manifest
from go2_quality_guard import equal_state
from go2_env import servo_model
CONFIG='configs/go2-action-transfer-objective.json'
ACTOR=('mlp_extractor.policy_net.','action_net.')

def freeze_inputs(out,cfg):
    out.mkdir(exist_ok=False);(out/'source').mkdir()
    for p in (ROOT/'scripts').glob('*.py'):shutil.copy2(p,out/'source'/p.name)
    dump(out/'acceptance.json',cfg);shutil.copy2(ROOT/cfg['predeclared_plan'],out/'predeclared-plan.md')
    names={CONFIG,cfg['predeclared_plan'],'docs/data/go2-current/selection.json',
        (out/'source/go2_action_transfer.py').relative_to(ROOT).as_posix(),*cfg['action_transfer']['models']}
    inputs={n:sha(ROOT/n) for n in names};dump(out/'input-sha256.json',inputs)
    return inputs

def seal(out,inputs):
    verify_inputs(out,inputs)
    dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})

def verify_inputs(out,inputs):
    for n,h in inputs.items():
        p=ROOT/n
        # The first prepare snapshot used a live script hash. Its retained runtime
        # copy, rather than later compatibility fixes, is authoritative for that step.
        if n=='scripts/go2_action_transfer.py' and sha(p)!=h:p=out/'source/go2_action_transfer.py'
        assert sha(p)==h,n

def norm_load(path):
    norm=pickle.loads(path.read_bytes());norm.training=False;norm.norm_reward=False
    return norm

def seed_audit(model,seed):
    model.set_random_seed(seed)
    assert random.getstate()==random.Random(seed).getstate()
    nr=np.random.RandomState(seed).get_state();actual=np.random.get_state()
    assert np.array_equal(actual[1],nr[1]) and actual[2:]==nr[2:]
    expected=torch.Generator(device='cpu').manual_seed(seed).get_state()
    assert torch.equal(torch.get_rng_state(),expected)
    return dict(seed=seed,Python_NumPy_Torch_explicit_seed_exact=True)

def prepare():
    cfg=read(ROOT/'configs/go2-early-recovery-transfer.json')
    cfg.update(experiment_id='GO2-B-261026-065',run='artifacts/runs/20261026-action-transfer-train',
        screen_run='artifacts/runs/20261026-action-transfer-screen',public='docs/data/go2-action-transfer-objective',
        predeclared_plan='docs/experiments/2026-10-26-action-transfer-objective.md')
    spec=cfg['action_transfer']=dict(bank='artifacts/runs/20261026-action-transfer-bank',
        old_bank=cfg['early_recovery']['bank'],parent=cfg['early_recovery']['parent'],parent_checkpoint=131072,
        formal_teacher='artifacts/runs/20261023-140238-938755-go2-transition-v3-mixed-ppo',
        seed=165001,batch_seed=165002,updates=2000,checkpoints=[1000,2000],batch_size=256,learning_rate=.0003,
        gradient_clip=.5,reference_RMSE_deg_max=.5,teacher_action_atol=2e-6,
        feasible_states=['formal-step300','formal-step306','formal-step312','D-step300'],
        full_cases=['case-045','F30-D0.6-S1-O0.2','repeat-1-offset-0.25','repeat--1-offset-0.25'],
        models=copy.deepcopy(cfg['early_recovery']['model_hashes']))
    for name in ['final-model.zip','final-vecnormalize.pkl']:
        n=spec['formal_teacher']+'/'+name;spec['models'][n]=sha(ROOT/n)
    assert not (ROOT/CONFIG).exists();dump(ROOT/CONFIG,cfg)
    old=ROOT/spec['old_bank'];verify_manifest(old);out=ROOT/spec['bank'];inputs=freeze_inputs(out,cfg)
    for name in ['bank.pkl','demonstrations.npz','demonstration-sources.json','summary.json','manifest.json']:
        p=old/name;inputs[p.relative_to(ROOT).as_posix()]=sha(p);shutil.copy2(p,out/('original-'+name))
    with np.load(old/'demonstrations.npz') as z:obs=z['observations'].copy();actions=z['actions'].copy();groups=z['group'].copy()
    formal=PPO.load(ROOT/spec['formal_teacher']/'final-model.zip',device='cpu');fnorm=norm_load(ROOT/spec['formal_teacher']/'final-vecnormalize.pkl')
    stem=f'checkpoint-{spec["parent_checkpoint"]}';parent=PPO.load(ROOT/spec['parent']/(stem+'.zip'),device='cpu');pnorm=norm_load(ROOT/spec['parent']/(stem+'-vecnormalize.pkl'))
    means=[];torch.set_num_threads(2)
    for sl,model,norm in [(slice(0,300),formal,fnorm),(slice(300,600),parent,pnorm)]:
        with torch.no_grad():means.append(model.policy.get_distribution(torch.as_tensor(norm.normalize_obs(obs[sl].copy()))).distribution.mean.numpy())
    means=np.concatenate(means);error=float(np.max(np.abs(np.clip(means,-1,1)-actions)))
    assert error<=spec['teacher_action_atol'],error
    with torch.no_grad():self_mean=parent.policy.get_distribution(torch.as_tensor(pnorm.normalize_obs(obs[300:].copy()))).distribution.mean.numpy()
    np.testing.assert_array_equal(self_mean,means[300:])
    np.savez_compressed(out/'reference.npz',observations=obs,normalized_student_observations=pnorm.normalize_obs(obs.copy()),
        executed_actions=actions,teacher_means=means,groups=groups)
    batch_ids=np.random.default_rng(spec['batch_seed']).integers(0,600,size=(spec['updates'],spec['batch_size']))
    np.savez_compressed(out/'batch-indices.npz',indices=batch_ids)
    dump(out/'interface-audit.json',dict(teacher_clipped_mean_matches_original_actions=True,max_absolute_error=error,tolerance=spec['teacher_action_atol'],
        old_expert_raw_mean_labels_initially_exact=True,unique_observations=600,batch_table_shape=list(batch_ids.shape),batch_seed=spec['batch_seed'],
        teacher_normalizations_separate_and_frozen=True,new_physical_runs=0))
    dump(out/'input-sha256.json',inputs);seal(out,inputs);print(json.dumps(read(out/'interface-audit.json')),flush=True)

def reference_metrics(model,z,cfg):
    native=servo_model();nominal=np.tile([0.,.8,-1.6],4);bounds=native.actuator_ctrlrange
    with torch.no_grad():mean=model.policy.get_distribution(torch.as_tensor(z['normalized_student_observations'])).distribution.mean.numpy()
    clipped=np.clip(mean,-1,1);target=np.clip(nominal+.35*clipped,bounds[:,0],bounds[:,1])
    labels=z['teacher_means'];truth=np.clip(nominal+.35*np.clip(labels,-1,1),bounds[:,0],bounds[:,1]);metrics={}
    for name,sl in [('formal40',slice(0,300)),('expert30',slice(300,600))]:
        metrics[name]=dict(mean_MSE_raw_teacher=float(np.mean((mean[sl]-labels[sl])**2)),
            mean_MSE_clipped_teacher=float(np.mean((mean[sl]-np.clip(labels[sl],-1,1))**2)),
            executed_action_MSE=float(np.mean((clipped[sl]-np.clip(labels[sl],-1,1))**2)),
            native_joint_target_RMSE_deg=float(np.sqrt(np.mean((target[sl]-truth[sl])**2))*180/np.pi),
            mean_saturated_fraction=float(np.mean(np.abs(mean[sl])>=1)))
    metrics['reference_gate']=all(metrics[k]['native_joint_target_RMSE_deg']<=cfg['action_transfer']['reference_RMSE_deg_max'] for k in ['formal40','expert30'])
    return metrics

def train():
    cfg=read(ROOT/CONFIG);spec=cfg['action_transfer'];bank=ROOT/spec['bank'];verify_manifest(bank)
    out=ROOT/cfg['run'];inputs=freeze_inputs(out,cfg)
    for p in bank.rglob('*'):
        if p.is_file():inputs[p.relative_to(ROOT).as_posix()]=sha(p)
    dump(out/'input-sha256.json',inputs)
    (out/'requirements-lock.txt').write_text(subprocess.check_output([str(ROOT/'.venv/Scripts/python.exe'),'-m','pip','freeze'],text=True),encoding='utf-8')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    with np.load(bank/'reference.npz') as archive:z={k:archive[k].copy() for k in archive.files}
    with np.load(bank/'batch-indices.npz') as archive:indices=archive['indices'].copy()
    parentpath=ROOT/spec['parent']/f'checkpoint-{spec["parent_checkpoint"]}.zip';parent=PPO.load(parentpath,device='cpu')
    original=copy.deepcopy(parent.policy.state_dict());results={};started=time.perf_counter()
    for arm in ['A','B']:
        folder=out/arm;folder.mkdir();dump(folder/'acceptance.json',cfg)
        model=PPO.load(parentpath,device='cpu');assert equal_state(original,model.policy.state_dict())
        for name,p in model.policy.named_parameters():p.requires_grad_(name.startswith(ACTOR))
        model.policy.optimizer=torch.optim.Adam(model.policy.parameters(),lr=spec['learning_rate'])
        assert not model.policy.optimizer.state_dict()['state']
        rng_receipt=seed_audit(model,spec['seed']);initial_rng=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
        (folder/'initial-RNG.pkl').write_bytes(pickle.dumps(initial_rng,protocol=5));dump(folder/'seed-audit.json',rng_receipt)
        shutil.copy2(ROOT/spec['parent']/f'checkpoint-{spec["parent_checkpoint"]}-vecnormalize.pkl',folder/'initial-vecnormalize.pkl')
        model.supervised_arm=arm;model.supervised_updates=0;model.supervised_only_actor=True;model.save(folder/'initial-model')
        observations=torch.as_tensor(z['normalized_student_observations']);labels=torch.as_tensor(np.clip(z['teacher_means'],-1,1) if arm=='A' else z['teacher_means'])
        log=[];metrics={0:reference_metrics(model,z,cfg)}
        for i,ids in enumerate(indices,1):
            model.policy.set_training_mode(True);mean=model.policy.get_distribution(observations[ids]).distribution.mean
            loss=torch.nn.functional.mse_loss(mean,labels[ids]);model.policy.optimizer.zero_grad();loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.policy.parameters(),spec['gradient_clip']);model.policy.optimizer.step()
            assert torch.isfinite(loss) and torch.isfinite(grad)
            log.append([i,float(loss.detach()),float(grad)])
            if i in spec['checkpoints']:
                model.supervised_updates=i;model.save(folder/f'checkpoint-{i}')
                shutil.copy2(folder/'initial-vecnormalize.pkl',folder/f'checkpoint-{i}-vecnormalize.pkl')
                torch.save(dict(policy=copy.deepcopy(model.policy.state_dict()),Adam=copy.deepcopy(model.policy.optimizer.state_dict()),
                    RNG=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone()),batch_index=i),folder/f'checkpoint-{i}-optimizer-RNG.pt')
                model.policy.set_training_mode(False);metrics[i]=reference_metrics(model,z,cfg)
                print(json.dumps(dict(arm=arm,updates=i,reference=metrics[i])),flush=True)
        for name,value in original.items():
            if not name.startswith(ACTOR):assert torch.equal(value,model.policy.state_dict()[name]),name
        assert random.getstate()==initial_rng[0] and np.array_equal(np.random.get_state()[1],initial_rng[1][1]) and np.random.get_state()[2:]==initial_rng[1][2:]
        assert torch.equal(torch.get_rng_state(),initial_rng[2])
        with (folder/'updates.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f);writer.writerow(['update','MSE_training_target','gradient_norm_before_clip']);writer.writerows(log)
        dump(folder/'reference-metrics.json',metrics);dump(folder/'frozen-components-audit.json',dict(critic_and_logstd_exact_parent=True,global_RNG_unchanged_after_explicit_seed=True,fresh_Adam_actor_only=True))
        results[arm]=dict(metrics=metrics,updates=len(log),sample_presentations=len(log)*spec['batch_size'])
    for n,h in spec['models'].items():assert sha(ROOT/n)==h,n
    summary=dict(config=cfg,arms=results,wall_seconds=time.perf_counter()-started,unique_observations=600,
        environment_training_steps=0,PPO_updates=0,new_physical_runs=0,batch_table_shared=True,seed_audits_passed=True,
        selected=False,selection_changed=False,held_out_executed=False)
    dump(out/'summary.json',summary);seal(out,inputs);print(json.dumps(dict(complete=True,wall_seconds=summary['wall_seconds'],environment_training_steps=0)),flush=True)

def screen():
    import evaluate_go2_robust as ev
    from go2_mass_course_env import Go2MassCourseEnv
    from go2_recovery_state_env import restore_state,row_and_state
    from screen_go2_early_recovery import snapshot_metrics
    from screen_go2_recovery_specialist import load_expert
    from screen_go2_mass_course import factory,save_probe,adapter_policy,norm_state,exact
    from go2_confirmed_selector import attach_confirmed_selector
    from stress_go2_frozen import run_case
    from evaluate_go2_confirmed_selector import trajectory_protection
    cfg=read(ROOT/CONFIG);spec=cfg['action_transfer'];ev.CONFIG=cfg;torch.set_num_threads(1)
    train=ROOT/cfg['run'];verify_manifest(train);out=ROOT/cfg['screen_run'];inputs=freeze_inputs(out,cfg)
    for p in train.rglob('*'):
        if p.is_file():inputs[p.relative_to(ROOT).as_posix()]=sha(p)
    bank=pickle.loads((ROOT/spec['bank']/'original-bank.pkl').read_bytes());oldids=set(read(ROOT/spec['bank']/'original-summary.json')['old_successful_ids'])
    reference=read(ROOT/spec['old_bank']/'summary.json');reg=read(ROOT/cfg['early_recovery']['regression']/'summary.json')
    old061=read(ROOT/'artifacts/runs/20261026-confirmed-selector/summary.json');D={r['case']:r for r in reg['reference_D']};prev={r['case']:r for r in reg['results']}
    for n in [spec['bank']+'/original-bank.pkl',cfg['early_recovery']['regression']+'/summary.json','artifacts/runs/20261026-confirmed-selector/summary.json']:
        inputs[n]=sha(ROOT/n)
    dump(out/'input-sha256.json',inputs);all_results={};actual=0
    joint=ev.load_policy(ROOT/'artifacts/runs'/cfg['parent'],cfg['parent_checkpoint']);adapter=adapter_policy(cfg,joint)
    frozen_norms=[norm_state(n) for n in [joint[1],adapter[1]]];frozen_weights=[copy.deepcopy(m.policy.state_dict()) for m in [joint[0],adapter[0]]]
    try:
        for arm in ['A','B']:
            checkpoints=[]
            for updates in spec['checkpoints']:
                expert,enorm=load_expert(train/arm,updates);norm_before=norm_state(enorm);weights=copy.deepcopy(expert.policy.state_dict());records=[]
                try:
                    for entry in bank:
                        env=Go2MassCourseEnv('external',config=cfg);restore_state(env,entry['snapshot']);rows=[];arrays={}
                        try:
                            for _ in range(150):
                                obs=env.observe();action,_=expert.predict(enorm.normalize_obs(obs.copy()),deterministic=True)
                                _,reward,term,trunc,info=env.step(action);row,values=row_and_state(env,obs,action,info,term)
                                values.update(rewards=reward,actuator_force=env.data.actuator_force.copy());rows.append(row)
                                for k,v in values.items():arrays.setdefault(k,[]).append(v)
                                if term or trunc:break
                        finally:env.close()
                        r=snapshot_metrics(rows,entry);r['seed']=cfg['development_seed'];r['expert_weight']=1.
                        records.append(ev.save_run(out,f'{arm}-bc{updates}',entry['id'],cfg['development_seed'],r,{k:np.asarray(v) for k,v in arrays.items()},rows));actual+=1
                    assert exact(norm_before,norm_state(enorm)) and equal_state(weights,expert.policy.state_dict())
                finally:enorm.close()
                fits=read(train/arm/'reference-metrics.json')[str(updates)];feasible=sum(r['passed'] for r in records if r['bank_id'] in spec['feasible_states'])
                losses=[r['bank_id'] for r in records if r['bank_id'] in oldids and not r['passed']]
                item=dict(arm=arm,updates=updates,results=records,reference=fits,feasible_passes=feasible,old_pass_losses=losses,
                    terminations=sum(r['terminated'] for r in records),passes=sum(r['passed'] for r in records),
                    skill_gate=feasible==4 and not losses and fits['reference_gate'])
                checkpoints.append(item);dump(out/f'{arm}-checkpoints.json',checkpoints)
                print(json.dumps({k:v for k,v in item.items() if k not in ['reference','results']}),flush=True)
            ranking=lambda e:(not e['skill_gate'],-e['feasible_passes'],len(e['old_pass_losses']),e['terminations'],e['reference']['formal40']['native_joint_target_RMSE_deg'],e['updates'])
            selected=min(checkpoints,key=ranking);expert,enorm=load_expert(train/arm,selected['updates']);enorm_before=norm_state(enorm)
            def make_factory(label):
                plain=factory(joint,dict(id=label,factor='nominal',level=1.,scope='action-transfer-objective'))
                def make(*args,**kwargs):return attach_confirmed_selector(plain(*args,**kwargs),expert,enorm,cfg['confirmed_selector']['selector'])
                return plain,make
            full=[];proofs={}
            try:
                for cid in spec['full_cases']:
                    c=next(c for c in reg['config']['recovery_regression']['known_cases'] if c['id']==cid);plain,make=make_factory(arm+'-'+cid)
                    r,packed,rows=run_case(c,cfg['development_seed'],make,adapter[0],adapter[1]);r['condition']=r.pop('case')
                    r=ev.save_run(out,arm+'-full',cid,cfg['development_seed'],r,packed,rows);save_probe(plain,ROOT/r['folder'])
                    dump(ROOT/r['folder']/'selector.json',plain.instances[-1].recovery_selection_log);actual+=1;full.append(r)
                    proofs[cid]=trajectory_protection(r,D[cid])
                plain,make=make_factory(arm+'-normal');normal=ev.normal_run(out,arm+'-full',cfg['normal_seed'],make,adapter);save_probe(plain,ROOT/normal['folder'])
                dump(ROOT/normal['folder']/'selector.json',plain.instances[-1].recovery_selection_log);actual+=1
                oldnormal=old061['by_mass']['1.00']['normal'];proofs['normal']=trajectory_protection(normal,oldnormal,True)
                assert exact(enorm_before,norm_state(enorm))
            finally:enorm.close()
            lossfull=[r['case'] for r in full if prev[r['case']]['passed'] and not r['passed']]
            extra=[r['case'] for r in full if not prev[r['case']]['terminated'] and r['terminated']]
            all_results[arm]=dict(checkpoints=checkpoints,selected_updates=selected['updates'],selected_skill_gate=selected['skill_gate'],full_pressure=full,
                normal=normal,protections=proofs,previous061_pass_losses=lossfull,extra_terminations_vs061=extra,
                all_normal_and_pre_force_exact_D=all(p['passed'] for p in proofs.values()),
                limited_development_gate=selected['skill_gate'] and all(r['passed'] for r in full) and all(p['passed'] for p in proofs.values()) and not lossfull and not extra)
            dump(out/'progress.json',dict(arms=all_results,new_physical_runs=actual))
        assert all(exact(a,norm_state(n)) for a,n in zip(frozen_norms,[joint[1],adapter[1]]))
        assert all(equal_state(w,m.policy.state_dict()) for w,m in zip(frozen_weights,[joint[0],adapter[0]]))
    finally:joint[1].close();adapter[1].close()
    assert actual==58
    summary=dict(config=cfg,arms=all_results,new_physical_runs=actual,environment_training_steps=0,PPO_updates=0,
        strict_seed_compliance=True,selection_changed=False,held_out_executed=False,independent_generalization_claim=False,visual_acceptance='pending')
    dump(out/'summary.json',summary);seal(out,inputs);public=ROOT/cfg['public'];public.mkdir(exist_ok=False);dump(public/'summary.json',summary)
    print(json.dumps(dict(new_physical_runs=actual,gates={k:b['limited_development_gate'] for k,b in all_results.items()})),flush=True)

def audit():
    from audit_go2_confirmed_selector import audit_record
    from audit_go2_recovery_screen import rows
    from screen_go2_early_recovery import snapshot_metrics
    cfg=read(ROOT/CONFIG);spec=cfg['action_transfer'];train=ROOT/cfg['run'];screen=ROOT/cfg['screen_run'];verify_manifest(train);verify_manifest(screen)
    bank=pickle.loads((ROOT/spec['bank']/'original-bank.pkl').read_bytes());s=read(screen/'summary.json');reg=read(ROOT/cfg['early_recovery']['regression']/'summary.json')
    D={r['case']:r for r in reg['reference_D']};old061=read(ROOT/'artifacts/runs/20261026-confirmed-selector/summary.json')
    snapshot_count=0;fullaudits=[];frozen={};torch.set_num_threads(2)
    parent=PPO.load(ROOT/spec['parent']/f'checkpoint-{spec["parent_checkpoint"]}.zip',device='cpu');original=parent.policy.state_dict()
    for arm,b in s['arms'].items():
        for cp in b['checkpoints']:
            for r in cp['results']:
                again=snapshot_metrics(rows(ROOT/r['folder']),next(e for e in bank if e['id']==r['bank_id']))
                assert all(r[k]==v for k,v in again.items());snapshot_count+=1
            model=PPO.load(train/arm/f'checkpoint-{cp["updates"]}.zip',device='cpu')
            assert all(torch.equal(v,model.policy.state_dict()[k]) for k,v in original.items() if not k.startswith(ACTOR))
            assert read(train/arm/'seed-audit.json')['Python_NumPy_Torch_explicit_seed_exact']
        for r in b['full_pressure']:fullaudits.append(audit_record(cfg,r,1.,D[r['case']],False))
        fullaudits.append(audit_record(cfg,b['normal'],1.,old061['by_mass']['1.00']['normal'],True))
        frozen[arm]=True
    # Replay every supervised update and both saved policy/Adam endpoints.
    with np.load(ROOT/spec['bank']/'reference.npz') as a:z={k:a[k].copy() for k in a.files}
    with np.load(ROOT/spec['bank']/'batch-indices.npz') as a:ids=a['indices'].copy()
    loss_replays={}
    for arm in ['A','B']:
        model=PPO.load(train/arm/'initial-model.zip',device='cpu')
        for name,p in model.policy.named_parameters():p.requires_grad_(name.startswith(ACTOR))
        optimizer=torch.optim.Adam(model.policy.parameters(),lr=spec['learning_rate'])
        obs=torch.as_tensor(z['normalized_student_observations']);target=torch.as_tensor(np.clip(z['teacher_means'],-1,1) if arm=='A' else z['teacher_means'])
        with (train/arm/'updates.csv').open(encoding='utf-8') as f:logged=list(csv.DictReader(f))
        errors=[]
        endpoint_checks=[]
        for i in range(spec['updates']):
            mean=model.policy.get_distribution(obs[ids[i]]).distribution.mean;loss=torch.nn.functional.mse_loss(mean,target[ids[i]])
            errors.append(abs(float(loss.detach())-float(logged[i]['MSE_training_target'])));optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(),spec['gradient_clip']);optimizer.step()
            if i+1 in spec['checkpoints']:
                saved=PPO.load(train/arm/f'checkpoint-{i+1}.zip',device='cpu')
                assert equal_state(model.policy.state_dict(),saved.policy.state_dict())
                assert equal_state(optimizer.state_dict(),saved.policy.optimizer.state_dict())
                endpoint_checks.append(i+1)
        assert max(errors)==0.;loss_replays[arm]=dict(updates=spec['updates'],maximum_loss_replay_error=max(errors),policy_Adam_exact_at=endpoint_checks)
    for root in [train,screen,ROOT/spec['bank']]:
        verify_inputs(root,read(root/'input-sha256.json'))
    report=dict(snapshot_gates_recomputed=snapshot_count,complete_selector_records_audited=len(fullaudits),
        raw_sensors_selector_states_actions_native_targets_and_original_gates_verified=True,
        normal_and_pre_force_exact_D=all(a['trajectory_protection']['passed'] for a in fullaudits),
        non_actor_and_normalizations_unchanged=True,paired_batches_identical=True,strict_seed_audits_passed=True,
        all_supervised_optimization_updates_replayed_exact=loss_replays,environment_training_steps=0,PPO_updates=0,new_physical_runs=0,
        formal_selection_changed=False,visual_acceptance='pending')
    shutil.copy2(ROOT/'scripts/go2_action_transfer.py',screen/'audit-runtime-source.py')
    report['audit_runtime_source_sha256']=sha(screen/'audit-runtime-source.py')
    dump(screen/'audit.json',report);dump(ROOT/cfg['public']/'audit.json',report);print(json.dumps(report),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('step',choices=['prepare','train','screen','audit']);args=parser.parse_args()
    globals()[args.step]()

if __name__=='__main__':main()
