"""155 one contact-observed residual PPO chain, two fixed four-case screens."""
import argparse,copy,json,pickle,random,shutil,time,traceback
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from archive_project import ROOT
from diagnose_go2_update import read,sha,dump
from go2_bilateral_pilot import Stack
from go2_action_transfer import norm_load
from go2_random_feature_recovery import RandomFeatureActor
from run_go2_random_feature_recovery import make_factory as parent_factory
from run_go2_random_feature_new_conditions import costs
from run_go2_single_wrench_pulse import runtime_files
from go2_native_capture_learning import cache,restore,prepare_inputs
from go2_training_tape_replay import pack_native_cache
from go2_contact_residual_learning import contact_observation,attach_contact_residual,drive,packed,ContactResidualEnv,save_calls
from go2_complete_recovery_ppo import CompletePPO,AllValueMaskedActorBuffer
from validate_go2_context_states import run_transient,pressure_metrics
from go2_traction_protocol import save_traction_probe
from evaluate_go2_release_regression import raw_equal
from audit_go2_recovery_screen import rows
import evaluate_go2_robust as ev

CONFIG='configs/go2-contact-residual-learning.json'


def prepare():
    cfg=read(ROOT/'configs/go2-contact-posture-feedback.json');old=cfg.pop('contact_posture_feedback')
    cfg.update(experiment_id='GO2-B-261030-155',run='artifacts/runs/20261030-contact-residual-learning',public='docs/data/go2-contact-residual-learning',
               predeclared_plan='docs/experiments/2026-10-30-contact-residual-learning.md')
    cfg['contact_residual_learning']=dict(sources=old['sources'],actor=old['actor'],training_steps=65536,checkpoints=[32768,65536],
        seed=255101,n_steps=1024,batch_size=256,n_epochs=4,learning_rate=1e-4,residual_scale=2.,initial_std=.03,
        gamma=.99,gae_lambda=.95,clip_range=.15,target_kl=.02,ent_coef=0.,vf_coef=.5,max_grad_norm=.5,net_arch=[64,64],
        max_new_evaluations=8,prefix_constructions=4,new_normal_runs=0,
        observation='frozen-normalized raw50 + contact4(force>5N) + flat-floor foot clearance4/.1 + body-frame absolute foot velocity12/.5; extra continuous features clipped to +/-10',
        mechanism='Frozen145 action + zero-initialized12 residual; original eligibility/hold; actual=clip(parent_float64+2*residual,-1,1). Zero residual returns original dtype/copy. Residual Gaussian initial std .03.',
        qualification='Each checkpoint all4 original12s/no termination/whole episode8deg/after lastwithdrawal2s planar.08yaw.15continuous25. Earlier qualified checkpoint preferred. Report all old pass losses/new terminations.',
        stop='One65536 chain/two fixed checkpoints/four opened sources, no extra updates/parameter scan. Equal actual-step quotas. No new condition/normal/full matrix/ZIP until net benefit.',
        reward='Original13part complete recovery sum*.1; fresh velocity training/legacyinfo evaluation; no new reward weights; all-value critic with only actually eligible actor samples.')
    assert not (ROOT/CONFIG).exists();dump(ROOT/CONFIG,cfg)
    p=ROOT/cfg['predeclared_plan'];assert not p.exists()
    p.write_text('''# 155 接触观测残差交互训练（事前）

单一机制包：冻结145父动作，以70维观测学习零初始化12维关节残差，检验明确接触信息和真实交互能否在两失败/两旧成功上取得净收益。不是接触观测的单因素消融，也不证明它是唯一原因。所有新增超参和预算以随附config为准。

原50维与原formal冻结归一化不变，追加FL/FR/RL/RR接触4(原>5N)、足底相对平地高度4/.1米、绝对世界足速度旋转到机身坐标12/.5米每秒；新增连续特征clip±10。仿真接触/地面高度估计属于仿真传感信息，不宣称硬件直接可用。不输入真实推力、摩擦、协议步号或奖励稳定计数。

冻结145的float64原输出，仅原learner_eligible/hold允许时加入2倍residual并clip原[-1,1]，原native nominal±.35/PD/力矩界限保持。残差恰零直接copy父输出/dtype，保证接口身份。MLP pi/vf各64x64 Tanh、actor末层权重和bias零，初始Gaussian std .03，logstd可训练。seed255101；Adam LR1e-4，nsteps1024/batch256/epochs4，gamma.99/GAE.95/clip.15/KL.02/entropy0/value.5/grad.5。新网络是残差actor+critic，冻结父链不训练。

训练从146phase626/duration070/phase634/duration050四个原历史完整缓存开始，保留完整MjData包括接触派生量、warmstart和控制闭包。4次前缀构造实际物理，各运行到首次学习入口并做1步零残差身份检查，不算完整独立基线。缓存复位选实际交互步最少的源，不按回合数等比例；记录真实步和actor份额。原13项完整恢复奖励不改，训练fresh速度与原info评估双口径保存。actor仅实际资格内样本，critic/GAE包括交还后到原12秒/物理终止。未训练组件不取代145。

恰最多65536真实交互，一个新随机流；固定32768/65536两个训练后检查点，各四原完整协议，共最多8新评估，0新正常/留出。评估原info全程8度、最后撤力后2秒内平面.08/yaw.15连续25步、完整12秒无终止。两检查点都评，不挑单个源/seed。4/4才开发资格；有净收益才能下一轮旧质量保护，不能把已见条件称泛化。两者无净收益关闭此训练包，不追加更新、调探索/奖励/权重或现场补点。

保存实际源码/配置/输入SHA、初始和两检查点policy/Adam/RNG，每轮更新前buffer/随机状态及前后参数SHA、逐episode真实动作/原父影子/奖励/状态/源元数据。精简验证只做零端点、缓存前缀、原指标重算及实际供给接口，省去重复整栈推理/ZIP/复制树/总览。全部退步和未做验证如实报告，正式selection不换，新恢复视觉待用户。
''',encoding='utf8')
    print('PREPARED155',sha(ROOT/CONFIG),sha(p),flush=True)


def rng():return dict(Python=random.getstate(),NumPy=np.random.get_state(),Torch=torch.get_rng_state().clone())


def policy_hash(model):
    import hashlib
    h=hashlib.sha256()
    for k,v in sorted(model.policy.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class LeanPPO(CompletePPO):
    def train(self):
        b=self.rollout_buffer;folder=self.evidence_out/'updates';folder.mkdir(exist_ok=True);i=self.num_timesteps//self.n_steps
        path=folder/f'update-{i:03d}';assert not path.with_suffix('.npz').exists()
        np.savez_compressed(path.with_suffix('.npz'),**{k:np.asarray(getattr(b,k)) for k in ['observations','actions','rewards','values','log_probs','advantages','returns','episode_starts','actor_applied']})
        torch.save(rng(),folder/f'update-{i:03d}-rng.pt');before=policy_hash(self)
        self.train_without_capture()
        dump(path.with_suffix('.json'),dict(timesteps=self.num_timesteps,policy_before=before,policy_after=policy_hash(self),
             losses={k:float(v) for k,v in self.logger.name_to_value.items() if k.startswith('train/')},log_std=self.policy.log_std.detach().cpu().numpy().tolist()))
        if self.num_timesteps in self.fixed_checkpoints:
            self.save(self.evidence_out/f'checkpoint-{self.num_timesteps}')
            torch.save(dict(policy=self.policy.state_dict(),Adam=self.policy.optimizer.state_dict(),rng=rng(),steps=self.num_timesteps),self.evidence_out/f'checkpoint-{self.num_timesteps}-state.pt')
            print('CHECKPOINT155',self.num_timesteps,flush=True)


class Progress(BaseCallback):
    def __init__(self,out,env):super().__init__();self.out=out;self.env=env;self.last=time.perf_counter();self.started=self.last
    def _on_step(self):
        b=self.model.rollout_buffer;b.actor_applied[b.pos]=[i['actor_applied'] for i in self.locals['infos']]
        if time.perf_counter()-self.last>=25:
            r=dict(steps=self.num_timesteps,seconds=time.perf_counter()-self.started,source_steps=self.env.counts.tolist(),actor_steps=self.env.applied.tolist())
            dump(self.out/'training-progress.json',r);print('TRAIN155',json.dumps(r),flush=True);self.last=time.perf_counter()
        return True


def preflight(cfg,out,stack,student,actor,norm):
    task=cfg['contact_residual_learning'];envs=[];caches=[];checks=[]
    for source in task['sources']:
        plain,base=parent_factory(cfg,stack,student,actor,source)
        env=attach_contact_residual(base('external',episode_seconds=source['protocol']['seconds']),norm,task['residual_scale'])
        env.reset(seed=source['seed'],options={'command':source['protocol']['schedule'][0]['command']})
        entry=source['first_learner_step'];actual={};saved=None
        for i in range(entry+1):
            prepare_inputs(env,source)
            if i==entry:
                saved=cache(env);folder=out/'prefixes'/source['case'];folder.mkdir(parents=True)
                (folder/'native-entry.pkl').write_bytes(pickle.dumps(pack_native_cache(env),protocol=5))
                # Derived fields must survive RAM resets; full prefix permits exact reconstruction.
                np.savez_compressed(folder/'derived-entry.npz',qacc_warmstart=env.data.qacc_warmstart,contact_obs=contact_observation(env,norm),
                    qacc=env.data.qacc,actuator_force=env.data.actuator_force,geom_xpos=env.data.geom_xpos)
            obs,a,(_,_,term,trunc,info)=drive(env,source,stack,np.zeros(12,np.float32))
            for k,v in packed(env,obs,a).items():actual.setdefault(k,[]).append(np.asarray(v).copy())
            assert not(term or trunc)
        with np.load(ROOT/source['folder']/'states.npz') as z:
            for k,v in actual.items():np.testing.assert_array_equal(np.asarray(v),z[k][:entry+1],err_msg=source['case']+k)
        last=env.contact_calls[-1];assert last['actor_applied'] and not last['changed'];np.testing.assert_array_equal(last['parent'],last['actual'])
        np.savez_compressed(folder/'prefix-plus-identity.npz',**{k:np.asarray(v) for k,v in actual.items()})
        restore(env,saved);env.contact_calls=[];env.residual_override=None
        envs.append(env);caches.append(saved);checks.append(dict(case=source['case'],entry=entry,physical_steps=entry+1,prefix8_exact=True,zero_residual_native_step_exact=True,closure_dicts=len(saved['values'])))
        print('IDENTITY155',source['case'],entry,'EXACT',flush=True)
    dump(out/'preflight.json',dict(passed=True,partial_prefix_physical_runs=4,partial_prefix_control_steps=sum(c['physical_steps'] for c in checks),full_baseline_runs=0,checks=checks))
    return envs,caches


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');args=ap.parse_args()
    if args.prepare:prepare();return
    cfg=read(ROOT/CONFIG);task=cfg['contact_residual_learning'];ev.CONFIG=cfg;torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    out=ROOT/cfg['run'];out.mkdir(exist_ok=False);(out/'source').mkdir();start=time.perf_counter()
    stack=Stack(cfg);student=PPO.load(ROOT/cfg['observed_entry_burst']['student_model'],device='cpu')
    actor=RandomFeatureActor(ROOT/cfg['random_feature_recovery']['parent_model'],ROOT/task['actor']);norm=norm_load(ROOT/cfg['random_feature_recovery']['parent_norm'])
    code=runtime_files();names=set(code)|{CONFIG,cfg['predeclared_plan'],'docs/data/go2-current/selection.json'}
    names.update(read(ROOT/'docs/data/go2-retention-protected-recovery/frozen-candidate.json')['models'])
    for s in task['sources']:names.add(s['folder']+'/states.npz')
    hashes={n:sha(ROOT/n) for n in sorted(names)}
    for n in code:shutil.copy2(ROOT/n,out/'source'/n.split('/')[-1])
    dump(out/'input-sha256.json',hashes);dump(out/'acceptance.json',cfg);shutil.copy2(ROOT/cfg['predeclared_plan'],out/'predeclared-plan.md')
    dump(out/'runtime-freeze.json',dict(config_sha256=sha(ROOT/CONFIG),plan_sha256=sha(ROOT/cfg['predeclared_plan']),runtime_modules=len(code),saved_before_physics_and_training=True,validation_level='lean development'))
    try:
        envs,caches=preflight(cfg,out,stack,student,actor,norm)
        native=ContactResidualEnv(envs,task['sources'],caches,stack,norm,out/'episodes');vec=DummyVecEnv([lambda:native])
        model=LeanPPO('MlpPolicy',vec,learning_rate=task['learning_rate'],n_steps=task['n_steps'],batch_size=task['batch_size'],n_epochs=task['n_epochs'],
            gamma=task['gamma'],gae_lambda=task['gae_lambda'],clip_range=task['clip_range'],target_kl=task['target_kl'],ent_coef=task['ent_coef'],vf_coef=task['vf_coef'],max_grad_norm=task['max_grad_norm'],
            policy_kwargs=dict(net_arch=dict(pi=task['net_arch'],vf=task['net_arch']),activation_fn=torch.nn.Tanh,log_std_init=float(np.log(task['initial_std']))),
            rollout_buffer_class=AllValueMaskedActorBuffer,seed=task['seed'],device='cpu')
        with torch.no_grad():model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_()
        for env,saved in zip(envs,caches):
            restore(env,saved);env.contact_calls=[];env.residual_override=None
            np.testing.assert_array_equal(model.predict(contact_observation(env,norm),deterministic=True)[0],np.zeros(12,np.float32))
        model.evidence_out=out;model.fixed_checkpoints=task['checkpoints'];model.set_logger(configure(str(out/'logs'),['csv']))
        model.save(out/'initial-model');torch.save(dict(policy=model.policy.state_dict(),Adam=model.policy.optimizer.state_dict(),rng=rng()),out/'initial-state.pt')
        dump(out/'initialization.json',dict(zero_mean_all_four_entries=True,empty_Adam=not bool(model.policy.optimizer.state_dict()['state']),std=task['initial_std'],seed=task['seed'],obs_dim=70,base145_frozen=True))
        started=time.perf_counter();model.learn(total_timesteps=task['training_steps'],callback=Progress(out,native));native.flush()
        exposure=[dict(case=s['case'],interactions=int(n),actor_interactions=int(a),fraction=float(n/native.counts.sum())) for s,n,a in zip(task['sources'],native.counts,native.applied)]
        dump(out/'training-summary.json',dict(PPO_interactions=model.num_timesteps,new_training_random_streams=1,exposure=exposure,seconds=time.perf_counter()-started,
             rollout_updates=model.num_timesteps//task['n_steps'],new_supervised_updates=0,new_analytic_fits=0))
        vec.close();records=[]
        for steps in task['checkpoints']:
            residual=PPO.load(out/f'checkpoint-{steps}.zip',device='cpu')
            for source in task['sources']:
                plain,base=parent_factory(cfg,stack,student,actor,source)
                def make(*a,**kw):return attach_contact_residual(base(*a,**kw),norm,task['residual_scale'],residual)
                r=run_transient(out,f'contact-residual-{steps}',source['protocol'],source['seed'],make,stack.adapter,False)
                env=plain.instances[-1];folder=ROOT/r['folder'];save_traction_probe(plain,folder,source['protocol']);save_calls(env,folder)
                dump(folder/'selector.json',env.recovery_selection_log);np.savez_compressed(folder/'initial-state.npz',**env.validation_initial)
                first=next((c['step'] for c in env.contact_calls if c['changed']),None);assert raw_equal(r,source,first)
                for k,v in pressure_metrics(rows(folder),source['protocol']).items():assert r[k]==v,k
                r.update(checkpoint=steps,source=source,kind='pressure',profile=source['profile'],controller='145-plus-contact-residual',first_residual_change=first,
                         residual_frames=sum(c['actor_applied'] for c in env.contact_calls),costs=costs(folder),old_pass_lost=bool(source['passed'] and not r['passed']),added_termination=bool(not source['terminated'] and r['terminated']))
                dump(folder/'result.json',r);records.append(r);dump(out/'evaluation-progress.json',records)
                print('EVAL155',steps,r['case'],r['passed'],r['peak_tilt'],r['recovery_seconds'],r['terminated'],flush=True)
        qualified=[n for n in task['checkpoints'] if all(r['passed'] for r in records if r['checkpoint']==n)]
        reports=[dict(checkpoint=n,passes=sum(r['passed'] for r in records if r['checkpoint']==n),
                      old_pass_losses=[r['case'] for r in records if r['checkpoint']==n and r['old_pass_lost']],added_terminations=[r['case'] for r in records if r['checkpoint']==n and r['added_termination']]) for n in task['checkpoints']]
        for n in code:assert sha(ROOT/n)==hashes[n],n
        assert sha(ROOT/'docs/data/go2-current/selection.json')==hashes['docs/data/go2-current/selection.json']
        summary=dict(config=cfg,records=records,checkpoints=reports,qualified_checkpoint=qualified[0] if qualified else None,
            new_full_evaluation_attempts=len(records),partial_prefix_physical_runs=4,partial_prefix_control_steps=sum(s['first_learner_step']+1 for s in task['sources']),
            PPO_interactions=model.num_timesteps,exposure=exposure,new_normal_runs=0,new_independent_conditions=0,formal_selection_changed=False,visual_acceptance='pending',
            validation_level='lean development; no full independent network audit or archive rebuild',elapsed_seconds=time.perf_counter()-start)
        dump(out/'summary.json',summary);public=ROOT/cfg['public'];public.mkdir(exist_ok=False);dump(public/'summary.json',summary)
        print('COMPLETE155',json.dumps(reports),summary['qualified_checkpoint'],'seconds',summary['elapsed_seconds'],flush=True)
    except BaseException:(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf8');raise
    finally:stack.close()


if __name__=='__main__':main()
