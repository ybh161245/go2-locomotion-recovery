"""Physical normal-motion update selection on frozen internal cases."""
import copy,json,math,random
from contextlib import contextmanager
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from go2_transition_env_v3 import Go2TransitionEnvV3
from go2_turn_residual_env import configured
from go2_motion_metrics import MotionMeter,LEGS,sample_row,summarize
from evaluate_go2_robust import ROOT,save_run
from go2_guarded_retention import snapshot,restore

@contextmanager
def preserve_rng(model=None):
    py=random.getstate();npstate=np.random.get_state();ts=torch.get_rng_state();mode=model.policy.training if model is not None else None
    try:yield
    finally:
        random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(ts)
        if model is not None:model.policy.set_training_mode(mode)
        assert random.getstate()==py and torch.equal(torch.get_rng_state(),ts)
        current=np.random.get_state();assert current[0]==npstate[0] and np.array_equal(current[1],npstate[1]) and current[2:]==npstate[2:]

def equal_state(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_state(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_state(v,w) for v,w in zip(a,b))
    return a==b

def stride(ids,rows,states):
    lengths=[]
    for leg in range(4):
        armed=False;last_contact=True;streak=0;last=None
        for i in ids:
            contact=rows[i][LEGS[leg]+'_contact'];streak=streak+1 if contact==last_contact else 1
            if not contact and streak>=2:armed=True
            if contact and streak==2 and armed:
                if last is not None and .2<(i-last)*.02<1.5:
                    direction=states['heading'][i]+states['heading'][last];direction[2]=0;direction/=np.linalg.norm(direction)
                    lengths.append(float(np.dot(states['feet_world'][i,leg]-states['feet_world'][last,leg],direction)))
                last=i;armed=False
            last_contact=contact
    return float(np.median(lengths)) if lengths else None

class QualityEvaluator:
    def __init__(self,model,cfg):
        self.model=model;self.cfg=cfg
        with preserve_rng(model):
            folder=ROOT/'artifacts/runs'/cfg['parent'];stem=cfg['parent_checkpoint']
            self.joint_norm=VecNormalize.load(folder/(stem+'-vecnormalize.pkl'),DummyVecEnv([lambda:Go2TransitionEnvV3('external')]))
            self.joint_norm.training=False;self.joint_norm.norm_reward=False
            work=ROOT/'artifacts/runs'/cfg['working_stack'];self.cls=configured(json.loads((work/'config.json').read_text()))
            self.adapter=PPO.load(work/'final-model.zip',device='cpu');self.adapter_norm=VecNormalize.load(work/'final-vecnormalize.pkl',DummyVecEnv([lambda:self.make_env()]))
            self.adapter_norm.training=False;self.adapter_norm.norm_reward=False

    def make_env(self,seconds=12):
        env=self.cls('external',episode_seconds=seconds);env.model=copy.copy(env.model);env.data=mujoco.MjData(env.model);env.meter=MotionMeter(env.model);env.teacher=self.model;env.teacher_norm=self.joint_norm
        return env

    def run_case(self,out,label,case):
        schedule=[];start=0
        for s in case['segments']:
            end=start+s['seconds']*50;schedule.append(dict(s,start=start,end=end));start=end
        env=self.make_env(start*.02);env.reset(seed=case['seed'],options={'command':schedule[0]['command']})
        rows=[];states={k:[] for k in ['qpos','qvel','targets','world_wrench','feet_world','heading']};term=False
        try:
            for i in range(start):
                s=next(s for s in schedule if s['start']<=i<s['end']);cmd=s['command'];env.command[:]=cmd;obs=env.observe()
                action,_=self.adapter.predict(self.adapter_norm.normalize_obs(obs.copy()),deterministic=True);_,_,term,trunc,info=env.step(action)
                for k,v in [('qpos',env.data.qpos),('qvel',env.data.qvel),('targets',env.data.ctrl),('world_wrench',np.zeros(6)),('feet_world',env.data.geom_xpos[env.meter.feet]),('heading',env.data.xmat[env.base].reshape(3,3)[:,0])]:states[k].append(v.copy())
                row=sample_row(env,env.meter,info,env.data.time,s['phase']);row.update(vy=info['vy'],terminated=term,command_vx=cmd[0],command_yaw=cmd[2]);rows.append(row)
                if term or trunc:break
        finally:env.close()
        states={k:np.array(v) for k,v in states.items()};flat={'complete':float(len(rows)==start and not term)};segments={}
        q=states['qpos'][:,3:7];w,x,y,z=q.T;angles=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
        for s in schedule:
            phase=s['phase'];ids=[i for i in range(s['start']+50,min(s['end'],len(rows)))];g=[rows[i] for i in ids]
            if not g:continue
            m=summarize(g);cmd=s['command'];moving=np.linalg.norm(cmd)>.025
            m['velocity_rmse']=float(np.sqrt(np.mean([(r['vx']-cmd[0])**2+r['vy']**2 for r in g])));m['yaw_rmse']=float(np.sqrt(np.mean([(r['yaw_rate']-cmd[2])**2 for r in g])))
            m['peak_tilt']=max(max(abs(r['roll_deg']),abs(r['pitch_deg'])) for r in rows[s['start']:min(s['end'],len(rows))])
            for key in ['velocity_rmse','yaw_rmse','peak_tilt','height_std','joint_speed_rms_mean','roll_deg_std','pitch_deg_std']:flat[phase+'/'+key]=m[key]
            for leg in LEGS:
                for key in ['clearance_p95_m','loaded_foot_speed_mean_m_s','contact_fraction','longest_unloaded_seconds']:flat[f'{phase}/{leg}/{key}']=m[leg][key]
            if moving and cmd[2]==0:m['stride']=stride(ids,rows,states);flat[phase+'/stride']=m['stride']
            if cmd[2] and len(rows)>=s['end']:
                value=float(np.degrees(angles[s['end']-1]-angles[s['start']-1]));target=float(np.degrees(cmd[2]*(s['end']-s['start'])*.02));m.update(angle=value,angle_target=target);flat[phase+'/angle_error']=abs(value-target)
            segments[phase]=m
        result=dict(steps=len(rows),terminated=bool(term),metrics=flat,segments=segments,schedule=schedule)
        return save_run(out,label,case['id'],case['seed'],result,states,rows)

    def evaluate(self,out,label,limits=None):
        with preserve_rng(self.model):
            cases={c['id']:self.run_case(out,label,c) for c in self.cfg['quality_guard']['cases']}
        report=dict(results=cases,rng_and_policy_mode_restored=True)
        if limits is not None:
            report['checks']={key:check_limits(value['metrics'],limits[key]) for key,value in cases.items()};report['passed']=all(c['passed'] for group in report['checks'].values() for c in group)
        return report

    def close(self):self.adapter_norm.close();self.joint_norm.close()

def build_limits(report,cfg):
    q=cfg['quality_guard'];b=cfg['normal_budget'];limits={}
    for case in q['cases']:
        old=report['results'][case['id']]['metrics'];rules=[]
        def add(key,direction,value):rules.append(dict(metric=key,direction=direction,limit=float(value)))
        add('complete','min',1)
        for s in case['segments']:
            p=s['phase'];moving=np.linalg.norm(s['command'])>.025
            add(p+'/peak_tilt','max',8)
            for metric,cap in [('velocity_rmse',.1),('yaw_rmse',.15)]:
                v=old[p+'/'+metric];add(p+'/'+metric,'max',min(cap,v+max(.1*v,.005)))
            if moving:
                v=old[p+'/height_std'];add(p+'/height_std','max',v+max(.15*v,.001))
                if s['command'][2]==0:add(p+'/stride','min',old[p+'/stride']*q['stride_preserved_ratio'])
            else:
                for metric,cap,inc in [('joint_speed_rms_mean',.3,.05),('roll_deg_std',.5,.1),('pitch_deg_std',.5,.1)]:add(p+'/'+metric,'max',min(cap,old[p+'/'+metric]+inc))
            for leg in LEGS:
                key=f'{p}/{leg}/loaded_foot_speed_mean_m_s';v=old[key];add(key,'max',v+max(.1*v,.01))
                if moving:
                    add(f'{p}/{leg}/clearance_p95_m','min',max(q['foot_min_m'],old[f'{p}/{leg}/clearance_p95_m']*q['foot_preserved_ratio']))
                    add(f'{p}/{leg}/contact_fraction','min',q['moving_contact_fraction'][0]);add(f'{p}/{leg}/contact_fraction','max',q['moving_contact_fraction'][1])
                    add(f'{p}/{leg}/longest_unloaded_seconds','max',max(q['max_unloaded_floor_s'],old[f'{p}/{leg}/longest_unloaded_seconds']+q['unloaded_increase_s']))
            if s['command'][2]:add(p+'/angle_error','max',max(q['turn_abs_floor_deg'],old[p+'/angle_error']+q['turn_error_increase_deg']))
        limits[case['id']]=rules
    return limits

def check_limits(metrics,limits):
    result=[]
    for r in limits:
        v=metrics.get(r['metric']);passed=v is not None and math.isfinite(v) and (v>=r['limit'] if r['direction']=='min' else v<=r['limit'])
        result.append(dict(r,value=v,passed=bool(passed)))
    return result

class PhysicalQualityPPO(PPO):
    def _excluded_save_params(self):return super()._excluded_save_params()+['quality_evaluator']

    def train(self):
        before=snapshot(self);super().train();index=len(self.quality_update_log)+1;out=self.quality_out
        report=self.quality_evaluator.evaluate(out/'quality-evaluations',f'update-{index:02d}',self.quality_limits);accepted=report['passed']
        candidates=out/'update-candidates';candidates.mkdir(exist_ok=True);path=candidates/f'update-{index:02d}.zip';assert not path.exists();self.save(path)
        if not accepted:
            restore(self,before)
            assert all(torch.equal(v,self.policy.state_dict()[k]) for k,v in before[0].items())
            assert equal_state(before[1],self.policy.optimizer.state_dict())
        self.quality_update_log.append(dict(attempt=index,added_steps=self.num_timesteps-self.quality_training_start,accepted=accepted,assessment=report,candidate=path.relative_to(out).as_posix()))
        (out/'quality-updates.json').write_text(json.dumps(self.quality_update_log,indent=2));self.logger.record('train/quality_update_accepted',int(accepted))
        failed=sum(not c['passed'] for group in report['checks'].values() for c in group)
        print(f'QUALITY_UPDATE {index} accepted={accepted} failed_internal_checks={failed}',flush=True)

def verify_identity_target_hook(current,old):
    """Prove049's default extension is algebraically the old mapping; no whitelist bypass."""
    import ast,copy
    before=ast.parse(old);after=ast.parse(current)
    old_class=next(n for n in before.body if isinstance(n,ast.ClassDef) and n.name=='Go2VelocityEnv')
    new_class=next(n for n in after.body if isinstance(n,ast.ClassDef) and n.name=='Go2VelocityEnv')
    old_step=next(n for n in old_class.body if isinstance(n,ast.FunctionDef) and n.name=='step')
    new_step=next(n for n in new_class.body if isinstance(n,ast.FunctionDef) and n.name=='step')
    assignment=lambda node:next(n for n in node.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='desired' for t in n.targets))
    original=assignment(old_step).value
    hook=next(n for n in new_class.body if isinstance(n,ast.FunctionDef) and n.name=='action_targets')
    assert ast.dump(hook.args,include_attributes=False)==ast.dump(ast.parse('def f(self, action): pass').body[0].args,include_attributes=False)
    statements=[n for n in hook.body if not(isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
    assert len(statements)==1 and isinstance(statements[0],ast.Return)
    assert ast.dump(statements[0].value,include_attributes=False)==ast.dump(original,include_attributes=False)
    assert ast.dump(assignment(new_step).value,include_attributes=False)==ast.dump(ast.parse('self.action_targets(action)',mode='eval').body,include_attributes=False)
    assignment(new_step).value=copy.deepcopy(original);new_class.body.remove(hook)
    assert ast.dump(after,include_attributes=False)==ast.dump(before,include_attributes=False)
    return True

def setup(model,env,cfg,out):
    import ast,hashlib
    from zipfile import ZipFile
    reference=ROOT/cfg['quality_guard']['reference_folder']
    for name,h in json.loads((reference/'manifest.json').read_text()).items():assert hashlib.sha256((reference/name).read_bytes()).hexdigest()==h
    original=json.loads((reference/'acceptance.json').read_text())
    for key in ['parent','parent_checkpoint','working_stack','normal_budget']:assert cfg[key]==original[key],key
    guard=lambda x:{k:v for k,v in x.items() if k not in ['reference_folder','description']}
    assert guard(cfg['quality_guard'])==guard(original['quality_guard'])
    for name,h in json.loads((reference/'model-inputs.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    fingerprints={}
    for name in ['go2_env.py','go2_model.py','go2_comfort_env.py','go2_transition_env_v2.py','go2_transition_env_v3.py','go2_turn_residual_env.py','go2_motion_metrics.py','go2_sequence_env.py','go2_guarded_retention.py']:
        current=(ROOT/'scripts'/name).read_bytes();old=(reference/'source'/name).read_bytes()
        if current!=old:
            assert name=='go2_env.py' and cfg.get('quality_source_compatibility')=='identity_target_hook_v1',name
            assert verify_identity_target_hook(current.decode('utf-8'),old.decode('utf-8'))
            fingerprints['compatibility:'+name]=dict(proof='Full module AST equals old after inlining exactly identical default target expression.',old_sha256=hashlib.sha256(old).hexdigest(),new_sha256=hashlib.sha256(current).hexdigest())
        fingerprints[name]=hashlib.sha256(current).hexdigest()
    protected=['preserve_rng','equal_state','stride','QualityEvaluator','build_limits','check_limits','PhysicalQualityPPO']
    def pieces(path):
        source=path.read_text(encoding='utf-8');return {n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in protected}
    current=pieces(ROOT/'scripts/go2_quality_guard.py');old=pieces(reference/'source/go2_quality_guard.py');assert current==old
    fingerprints.update({f'quality_component:{k}':hashlib.sha256(v.encode()).hexdigest() for k,v in current.items()})
    # The reference receipt originally omitted adapter config; compare its sealed047 entry.
    if cfg.get('rl_repair') or cfg.get('quality_source_compatibility'):
        name=f'artifacts/runs/{cfg["working_stack"]}/config.json';archive=ROOT/'artifacts/source-snapshots/20261025-physical-quality-guard/stage-47-training-and-screen.zip'
        with ZipFile(archive) as z:old_hash=json.loads(z.read('ARCHIVE-MANIFEST.json'))[name]
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==old_hash;fingerprints[name]=old_hash
    model.quality_limits=json.loads((reference/'limits.json').read_text());model.quality_out=out;model.quality_training_start=model.num_timesteps;model.quality_update_log=[];model.quality_evaluator=QualityEvaluator(model,cfg)
    (out/'quality-reference.json').write_text(json.dumps(dict(folder=cfg['quality_guard']['reference_folder'],manifest_sha256=hashlib.sha256((reference/'manifest.json').read_bytes()).hexdigest(),parent_adapter_budget_rules_equal_reference=True,protected_fingerprints=fingerprints),indent=2))
