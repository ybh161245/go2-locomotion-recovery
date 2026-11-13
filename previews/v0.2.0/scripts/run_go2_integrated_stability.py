"""253 actual call-site interaction via queue, frozen input normalization."""
import math,threading,queue,traceback,shutil,pickle,time,random
from pathlib import Path
import numpy as np
import torch,gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from go2_157 import ROOT,RUNTIME,read,sha,write_new,load_stack
from go2_speed_hold import candidate_inputs
from go2_variable_speed_heading_hold import attach as attach212
from run_go2_full_heading_feedback import attach as attach197
from go2_dual_speed import attach as attach248
from go2_dual_heading_handoff import attach as attach249
from go2_action_transfer import norm_load
from go2_sequence import validate
from run_go2_dual_speed_handoff import run,metrics,PARENT
RUN=ROOT/'artifacts/runs/20261111-integrated-stability'
PUBLIC=ROOT/'docs/data/go2-integrated-stability'
PLAN=ROOT/'docs/experiments/2026-11-11-integrated-stability-course.md'
CONFIG=ROOT/'configs/go2-integrated-stability.json'
OLD=ROOT/'artifacts/runs/20261108-dual-heading-handoff'
class StopWorker(Exception):pass

class Course(gym.Env):
    observation_space=spaces.Box(-np.inf,np.inf,(50,),np.float32)
    action_space=spaces.Box(-1,1,(12,),np.float32)
    def __init__(self,stack,factory,reference,mechanism,norm):
        self.stack,self.factory,self.reference,self.mechanism,self.norm=stack,factory,reference,mechanism,norm
        self.episode=0;self.warmup_steps=0;self.training_steps=0;self.falls=0;self.thread=None
        self.events=queue.Queue();self.actions=queue.Queue();self.previous=None;self.trace=[]
    def make(self,actor):
        _,f=self.factory(self.reference)
        return attach249(attach248(attach212(attach197(f('external',episode_seconds=math.nextafter(22.,math.inf)),self.stack),self.mechanism),actor,self.norm))
    def predict(self,obs,deterministic=True):
        self.calls+=1;self.current_obs=np.asarray(obs).copy()
        if self.episode==1 and self.i==300:
            with np.load(OLD/'mixed/states.npz') as z:expected=self.norm.normalize_obs(z['policy_input'][300].copy())
            np.testing.assert_array_equal(obs,expected)
            write_new(PUBLIC/'preflight.json',dict(warmup_exact_steps=300,first_actual_input_exact=True,new_optimizer_updates=0,warmup_physical_steps=300))
        reward,info=(0.,{}) if self.previous is None else self.previous
        self.events.put(('obs',self.current_obs.copy(),reward,False,False,info))
        action=self.actions.get()
        if action is None:raise StopWorker()
        self.current_action=np.asarray(action).copy()
        return self.current_action,None
    def worker(self):
        env=None;self.trace=[];self.previous=None;self.calls=0
        try:
            env=self.make(self);env.reset(seed=648101,options={'command':[0.,0.,0.]})
            with np.load(OLD/'mixed/states.npz') as z:old={k:z[k] for k in ('qpos','qvel','targets')}
            for self.i in range(600):
                i=self.i;env.command[:]=[0. if i<100 else .4 if i<300 else .8,0.,0.];env.data.xfrc_applied[:]=0
                outer=env.observe().copy();adapter=self.stack.adapter[0].predict(self.stack.adapter[1].normalize_obs(outer),deterministic=True)[0]
                before=self.calls
                _,base,term,trunc,info=env.step(adapter)
                c=env.dual_calls[-1];v,w=env.velocity()
                penalty=min((max(float(np.linalg.norm(v[:2]-env.filtered_command[:2]))-.08,0)/.08)**2,4)+min((max(abs(float(w[2]))-.15,0)/.15)**2,4) if i>=300 else 0.
                np.testing.assert_array_equal(np.clip(env.action_targets(c['actual']),*env.model.actuator_ctrlrange.T),env.data.ctrl)
                if i<300:
                    self.warmup_steps+=1
                    for k,a in [('qpos',env.data.qpos),('qvel',env.data.qvel),('targets',env.data.ctrl)]:np.testing.assert_array_equal(a,old[k][i])
                    assert not term
                else:
                    assert self.calls==before+1,'Missing high actor call: stop course'
                    np.testing.assert_array_equal(c['high'],np.clip(self.current_action,-1,1))
                    self.training_steps+=1
                self.trace.append(dict(step=i,qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),targets=env.data.ctrl.copy(),
                    observation=self.current_obs.copy() if i>=300 else np.zeros(50,np.float32),sample_action=self.current_action.copy() if i>=300 else np.zeros(12,np.float32),
                    actual=c['actual'].copy(),high=c['high'].copy(),low=c['low'].copy(),weight=c['weight'],velocity=v.copy(),angular=w.copy(),
                    base_reward=base,penalty=penalty,reward=base-penalty,terminated=term,learner=i>=300))
                self.previous=(float(base-penalty),dict(physical_terminated=bool(term),step=i))
                if term or i==599:
                    self.falls+=int(term)
                    self.events.put(('obs',self.current_obs.copy(),float(base-penalty),True,False,dict(physical_terminated=bool(term),finite_horizon=i==599)))
                    break
        except StopWorker:pass
        except Exception:self.events.put(('error',traceback.format_exc()))
        finally:
            if self.trace:np.savez_compressed(RUN/f'train-episode-{self.episode:04}.npz',**{k:np.asarray([r[k] for r in self.trace]) for k in self.trace[0]})
            if env is not None:env.close()
    def receive(self):
        event=self.events.get(timeout=180)
        if event[0]=='error':raise RuntimeError(event[1])
        return event[1:]
    def reset(self,seed=None,options=None):
        if self.thread:self.thread.join(timeout=30);assert not self.thread.is_alive()
        self.episode+=1
        if self.episode>1000:raise RuntimeError('Warmup budget exceeded')
        self.thread=threading.Thread(target=self.worker,daemon=True);self.thread.start()
        obs,_,_,_,info=self.receive();return obs,info
    def step(self,action):self.actions.put(action.copy());return self.receive()
    def close(self):
        if self.thread and self.thread.is_alive():self.actions.put(None);self.thread.join(timeout=30)

class Archive(BaseCallback):
    def __init__(self,start):super().__init__();self.start=start;self.roll=0;self.began=time.monotonic()
    def save(self,n):
        self.model.save(RUN/f'checkpoint-{n}');self.model.get_vec_normalize_env().save(RUN/f'checkpoint-{n}-rewardnorm.pkl')
    def _on_step(self):
        if time.monotonic()-self.began>1500:raise TimeoutError('253 wall-clock cap')
        return True
    def _on_rollout_start(self):
        n=self.model.num_timesteps-self.start
        if n==32768:self.save(n)
        self.roll+=1
        with (RUN/f'rng-{self.roll:03}.pkl').open('xb') as f:pickle.dump(dict(numpy=np.random.get_state(),python=random.getstate(),torch=torch.get_rng_state()),f)
        if self.roll%8==1:print('253 training',n,flush=True)
    def _on_rollout_end(self):
        b=self.model.rollout_buffer
        np.savez_compressed(RUN/f'buffer-{self.roll:03}.npz',**{k:getattr(b,k).copy() for k in ('observations','actions','rewards','episode_starts','values','log_probs','advantages','returns')})

def evaluate(course,actor,label):
    mixed=[(0,100,[0.,0.,0.],'stand'),(100,300,[.4,0.,0.],'low'),(300,600,[.8,0.,0.],'high'),(600,800,[.4,0.,0.],'low'),(800,1100,[0.,0.,0.],'stop')]
    task=read(ROOT/'configs/go2-variable-speed-new-conditions.json')['task'];rows=[]
    with np.load(OLD/'mixed/states.npz') as d,np.load(OLD/'mixed/telemetry.npz') as t:
        jumps=np.linalg.norm(np.diff(d['targets'],axis=0),axis=1)
        baseline=dict(power=float(t['power'][100:800].mean()),jump_max=float(jumps.max()),jump_p95=float(np.percentile(jumps,95)))
    with np.load(OLD/'mixed/physics.npz') as p:limits=np.max(abs(p['actuator_forcerange']),axis=1)
    with np.load(OLD/'mixed/telemetry.npz') as t:baseline['torque']=float(np.max(abs(t['actuator_force'])/limits))
    for name,schedule,seed in [('mixed',mixed,648101),('low-replay',validate(task['sequence']),513101)]:
        env=course.make(actor);folder=RUN/f'{label}-{name}'
        try:
            d,t=run(env,course.stack,schedule,seed,folder)
            if name=='mixed':
                r=metrics(d,t);jump=np.linalg.norm(np.diff(d['targets'],axis=0),axis=1)
                costs=dict(power=float(t['power'][100:min(800,len(t['power']))].mean()),jump_max=float(jump.max()),jump_p95=float(np.percentile(jump,95)),torque=float(np.max(abs(t['actuator_force'])/limits)))
                cost_ok=all(costs[k]<=baseline[k]*1.1 for k in ('power','jump_max','jump_p95')) and costs['torque']<=min(1,baseline['torque']+.05)
                r.update(costs=costs,baseline_costs=baseline,cost_budget_passed=bool(cost_ok));r['passed']=bool(r['passed'] and cost_ok)
            else:
                with np.load(OLD/'low-replay/states.npz') as old:
                    for k in old.files:np.testing.assert_array_equal(d[k],old[k])
                r=dict(passed=True,all_shared_exact=True)
            r.update(name=name,checkpoint=label,steps=len(d['qpos']),folder=folder.relative_to(ROOT).as_posix());write_new(folder/'result.json',r);rows.append(r);print(r,flush=True)
        finally:env.close()
    return rows

def main():
    torch.set_num_threads(1)
    inputs=dict(read(ROOT/'configs/go2-dual-heading-handoff.json')['inputs'])
    for p in (Path(__file__),PLAN,OLD/'mixed/states.npz',ROOT/'scripts/go2_dual_heading_handoff.py',ROOT/'scripts/go2_dual_speed.py'):inputs[p.relative_to(ROOT).as_posix()]=sha(p)
    for p,h in inputs.items():assert sha(ROOT/p)==h,p
    write_new(CONFIG,dict(experiment=253,inputs=inputs,training_steps=65536,warmup_step_cap=300000,checkpoints=[32768,65536],seed=653101,eval_budget=4,learning_rate=1e-5))
    RUN.mkdir(exist_ok=False);PUBLIC.mkdir(exist_ok=False)
    for p in (Path(__file__),PLAN,CONFIG):shutil.copy2(p,RUN/p.name)
    _,mechanism=candidate_inputs();_,stack,factory,_=load_stack(read(RUNTIME));norm=norm_load(PARENT/'checkpoint-65536-norm.pkl')
    course=Course(stack,factory,read(ROOT/read(RUNTIME)['cases']['normal']['reference_record']),mechanism,norm)
    vec=VecNormalize(DummyVecEnv([lambda:course]),norm_obs=False,norm_reward=True)
    vec.ret_rms=pickle.loads(pickle.dumps(norm.ret_rms))
    model=PPO.load(PARENT/'checkpoint-65536.zip',env=vec,device='cpu',n_steps=512)
    model.set_random_seed(653101);model.learning_rate=1e-5;model.lr_schedule=lambda _:1e-5
    for g in model.policy.optimizer.param_groups:g['lr']=1e-5
    model.save(RUN/'initial-model');shutil.copy2(PARENT/'checkpoint-65536-norm.pkl',RUN/'frozen-observation-norm.pkl')
    start=model.num_timesteps;callback=Archive(start);began=time.monotonic()
    try:
        model.learn(total_timesteps=65536,reset_num_timesteps=False,callback=callback)
        callback.save(65536)
    finally:
        course.close()
        write_new(PUBLIC/'training-counts.json',dict(training_steps=course.training_steps,warmup_steps=course.warmup_steps,episodes=course.episode,physical_falls=course.falls,elapsed_s=time.monotonic()-began,optimizer_updates=int(model._n_updates)))
    assert course.training_steps==65536
    rows=[]
    try:
        for n in (32768,65536):rows+=evaluate(course,PPO.load(RUN/f'checkpoint-{n}.zip',device='cpu'),str(n))
    finally:stack.close()
    write_new(PUBLIC/'summary.json',dict(experiment=253,records=rows,new_evaluations=4,G3_evaluations_used=12,training_steps=65536,qualified_checkpoints=[n for n in ('32768','65536') if all(r['passed'] for r in rows if r['checkpoint']==n)]))
    for p,h in inputs.items():assert sha(ROOT/p)==h,p
if __name__=='__main__':main()
