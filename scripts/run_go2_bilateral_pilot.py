"""111 frozen preflight, then a finite actual-interaction PPO experiment."""
import copy,csv,json,pickle,random,shutil,time,traceback
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from stable_baselines3.common.logger import configure
from archive_project import ROOT
from diagnose_go2_update import read,sha,dump
from go2_bilateral_pilot import Stack,InteractionEnv,capture,restore
from go2_neighbor_entry_causal import snapshot
from go2_transition_env_v3 import Go2TransitionEnvV3
from screen_go2_mass_course import exact,norm_state
from go2_quality_guard import equal_state
from go2_update_capture import capture_update
from go2_action_transfer import seed_audit
CONFIG='configs/go2-bilateral-pilot.json'


class AppliedOnlyBuffer(RolloutBuffer):
    def reset(self):
        super().reset();self.actor_applied=np.zeros((self.buffer_size,self.n_envs),dtype=bool)
    def get(self,batch_size=None):
        assert self.full
        if not self.generator_ready:
            for name in ['observations','actions','values','log_probs','advantages','returns']:
                self.__dict__[name]=self.swap_and_flatten(self.__dict__[name])
            self.actor_applied=self.swap_and_flatten(self.actor_applied[...,None]).ravel();self.generator_ready=True
        ids=np.flatnonzero(self.actor_applied);assert len(ids)>0
        ids=np.random.permutation(ids);batch_size=batch_size or len(ids)
        for i in range(0,len(ids),batch_size):yield self._get_samples(ids[i:i+batch_size])


class PilotPPO(PPO):
    def train(self):
        capture_update(self,self.pilot_out/'rollouts',self.num_timesteps//(self.n_steps*self.n_envs))
        PPO.train(self)


class PilotCallback(BaseCallback):
    def __init__(self,out,cfg):super().__init__();self.out=out;self.spec=cfg['bilateral_pilot'];self.counts={};self.last=time.perf_counter();self.saved=set()
    def _on_rollout_start(self):
        if self.num_timesteps in self.spec['checkpoints'] and self.num_timesteps not in self.saved:
            save_checkpoint(self.model,self.out,self.num_timesteps);self.saved.add(self.num_timesteps)
    def _on_step(self):
        b=self.model.rollout_buffer;b.actor_applied[b.pos]=[i['actor_applied'] for i in self.locals['infos']]
        for i in self.locals['infos']:
            c=self.counts.setdefault(i['record'],dict(transitions=0,effective_actor_transitions=0));c['transitions']+=1;c['effective_actor_transitions']+=int(i['actor_applied'])
        if time.perf_counter()-self.last>25:
            dump(self.out/'exposure-progress.json',self.counts);print('PILOT_TRAIN '+json.dumps(dict(steps=self.num_timesteps,exposure=self.counts)),flush=True);self.last=time.perf_counter()
        return True


def save_checkpoint(model,out,steps):
    model.save(out/f'checkpoint-{steps}');model.get_vec_normalize_env().save(out/f'checkpoint-{steps}-vecnormalize.pkl')
    torch.save(dict(policy=copy.deepcopy(model.policy.state_dict()),Adam=copy.deepcopy(model.policy.optimizer.state_dict()),
        Python=random.getstate(),NumPy=np.random.get_state(),Torch=torch.get_rng_state().clone(),timesteps=model.num_timesteps),out/f'checkpoint-{steps}-optimizer-RNG.pt')
    print('CHECKPOINT '+str(steps),flush=True)


def preflight(cfg,out):
    stack=Stack(cfg);records=cfg['bilateral_pilot']['sources'];caches=[];checks=[]
    try:
        for record in records:
            env=stack.make(record);env.reset(seed=record['seed'],options={'command':record['protocol']['schedule'][0]['command']})
            entry=319 if record['direction']=='positive' else 320;old=pickle.loads((ROOT/record['folder']/'snapshots.pkl').read_bytes())['queries'][entry]
            with np.load(ROOT/record['folder']/'states.npz') as z:source={k:z[k].copy() for k in z.files}
            maximum=record['steps'];actual={k:[] for k in source};control=None
            for i in range(maximum):
                if i==entry:
                    # drive prepares force just before native control; frozen entry
                    # already stores that force. No integration/forward here.
                    env.data.xfrc_applied[:]=0
                    # Force prep uses the same body frame as drive.
                    from validate_go2_context_states import local_force
                    f=local_force(record['protocol']['events'],i);R=env.data.xmat[env.base].reshape(3,3)
                    env.data.xfrc_applied[env.base]=np.r_[R@f[:3],R@f[3:]]
                    control=snapshot(env,env.modify_joint_action);assert exact(control,old),(record['case'],'full entry')
                    caches.append(capture(env))
                obs,a,(_,_,term,trunc,info)=stack.drive(env,record)
                values=dict(observations=obs,actions=a,qpos=env.data.qpos,qvel=env.data.qvel,targets=env.data.ctrl,
                    world_wrench=env.data.xfrc_applied[env.base],feet_world=env.data.geom_xpos[env.meter.feet],heading=env.data.xmat[env.base].reshape(3,3)[:,0])
                for k in actual:actual[k].append(values[k].copy())
                assert bool(term)==bool(record['terminated'] and i+1==maximum)
            for k in source:np.testing.assert_array_equal(np.asarray(actual[k]),source[k],err_msg=record['case']+k)
            # Different fresh model/data instance, exact cached suffix. This is
            # an additional physical run, not a claim of independent evidence.
            fresh=stack.make(record);restore(fresh,caches[-1]);suffix={k:[] for k in source}
            for i in range(entry,maximum):
                obs,a,(_,_,term,trunc,info)=stack.drive(fresh,record)
                values=dict(observations=obs,actions=a,qpos=fresh.data.qpos,qvel=fresh.data.qvel,targets=fresh.data.ctrl,
                    world_wrench=fresh.data.xfrc_applied[fresh.base],feet_world=fresh.data.geom_xpos[fresh.meter.feet],heading=fresh.data.xmat[fresh.base].reshape(3,3)[:,0])
                for k in suffix:suffix[k].append(values[k].copy())
            for k in source:np.testing.assert_array_equal(np.asarray(suffix[k]),source[k][entry:],err_msg='restored '+record['case']+k)
            folder=out/'preflight'/record['case'];folder.mkdir(parents=True);np.savez_compressed(folder/'full.npz',**{k:np.asarray(v) for k,v in actual.items()});np.savez_compressed(folder/'suffix.npz',**{k:np.asarray(v) for k,v in suffix.items()});(folder/'entry.pkl').write_bytes(pickle.dumps(control,protocol=5))
            checks.append(dict(case=record['case'],entry=entry,full_raw_exact=True,suffix_raw_exact=True,full_entry_exact=True,steps=maximum,suffix_steps=maximum-entry))
            env.close();fresh.close();print('PREFLIGHT '+record['case']+' EXACT',flush=True)
        dump(out/'preflight.json',dict(passed=True,new_physical_runs=8,full_replays=4,suffix_replays=4,checks=checks,observed_entry_reset_only=True))
    finally:stack.close()
    return caches


def main():
    cfg=read(ROOT/CONFIG);spec=cfg['bilateral_pilot'];out=ROOT/cfg['run'];out.mkdir(exist_ok=False);(out/'source').mkdir()
    for p in (ROOT/'scripts').glob('*.py'):shutil.copy2(p,out/'source'/p.name)
    dump(out/'acceptance.json',cfg);shutil.copy2(ROOT/cfg['predeclared_plan'],out/'predeclared-plan.md')
    names={CONFIG,cfg['predeclared_plan'],*spec['inputs']};inputs={n:sha(ROOT/n) for n in names};dump(out/'input-sha256.json',inputs)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    start=time.perf_counter();caches=preflight(cfg,out)
    envs=DummyVecEnv([lambda w=w:InteractionEnv(cfg,spec['sources'],caches,w,out/'episodes'/f'worker-{w}') for w in range(spec['workers'])])
    norm=VecNormalize.load(ROOT/spec['parent_norm'],envs);norm.training=False;norm.norm_reward=False;before=norm_state(norm)
    parent=PPO.load(ROOT/spec['parent_model'],device='cpu')
    model=PilotPPO('MlpPolicy',norm,learning_rate=spec['learning_rate'],n_steps=spec['n_steps'],batch_size=spec['batch_size'],n_epochs=spec['n_epochs'],gamma=.99,gae_lambda=.95,clip_range=.15,ent_coef=0.,target_kl=.02,seed=spec['seed'],policy_kwargs=copy.deepcopy(parent.policy_kwargs),rollout_buffer_class=AppliedOnlyBuffer,device='cpu')
    state=model.policy.state_dict();actor={k:v.clone() for k,v in parent.policy.state_dict().items() if k=='log_std' or k.startswith(('mlp_extractor.policy_net.','action_net.'))};state.update(actor);model.policy.load_state_dict(state)
    assert all(torch.equal(model.policy.state_dict()[k],v) for k,v in actor.items());assert not model.policy.optimizer.state_dict()['state']
    assert any(not torch.equal(v,parent.policy.state_dict()[k]) for k,v in state.items() if k.startswith(('mlp_extractor.value_net.','value_net.')))
    for key,p in model.policy.named_parameters():assert p.requires_grad,key
    seed_receipt=seed_audit(model,spec['seed']);model.pilot_out=out;model.set_logger(configure(str(out/'logs'),['csv']))
    model.save(out/'initial-model');norm.save(out/'initial-vecnormalize.pkl');dump(out/'initialization-audit.json',dict(actor_and_logstd_exact=True,fresh_critic=True,fresh_Adam=True,all_parameters_trainable=True,seed=seed_receipt,ignored_action_samples_excluded=True))
    callback=PilotCallback(out,cfg);train_start=time.perf_counter()
    try:
        model.learn(total_timesteps=spec['training_steps'],callback=callback);save_checkpoint(model,out,spec['training_steps'])
        assert exact(before,norm_state(norm));dump(out/'exposure.json',callback.counts)
    finally:norm.close()
    for n,h in inputs.items():assert sha(ROOT/n)==h,n
    report=dict(config=cfg,preflight_physical_runs=8,training_samples=model.num_timesteps,rollout_updates=model.num_timesteps//(spec['n_steps']*spec['workers']),training_seconds=time.perf_counter()-train_start,exposure=callback.counts,normalization_unchanged=True,formal_selection_changed=False,research_candidate_changed=False,status='trained_awaiting_fixed_screen',elapsed_seconds=time.perf_counter()-start)
    dump(out/'training-summary.json',report);dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()});print('PILOT_COMPLETE '+json.dumps({k:v for k,v in report.items() if k not in ['config','exposure']}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        cfg=read(ROOT/CONFIG);out=ROOT/cfg['run']
        if out.exists():(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
