"""253 audit actual queued PPO observations/actions and reward components."""
import json
import numpy as np
from run_go2_integrated_stability import ROOT,RUN,PUBLIC,CONFIG,read,sha,write_new
def main():
    obs=[];actions=[];reward_error=0.;steps=warm=falls=0
    for p in sorted(RUN.glob('train-episode-*.npz')):
        with np.load(p) as z:
            m=z['learner'];warm+=int((~m).sum());steps+=int(m.sum());falls+=int(z['terminated'].sum())
            obs.append(z['observation'][m]);actions.append(z['sample_action'][m])
            # Original filtered command ramps .4 -> .8 by .016 per control step.
            idx=z['step'][m];cmd=np.minimum(.8,.4+(idx-299)*.016)
            v=z['velocity'][m];w=z['angular'][m,2]
            pe=np.sqrt((v[:,0]-cmd)**2+v[:,1]**2)
            penalty=np.minimum((np.maximum(pe-.08,0)/.08)**2,4)+np.minimum((np.maximum(abs(w)-.15,0)/.15)**2,4)
            reward_error=max(reward_error,float(abs(penalty-z['penalty'][m]).max(initial=0)))
            np.testing.assert_allclose(z['reward'],z['base_reward']-z['penalty'],rtol=0,atol=1e-12)
            expected=(1-z['weight'][m,None])*z['low'][m].astype(float)+z['weight'][m,None]*z['high'][m].astype(float)
            np.testing.assert_array_equal(expected,z['actual'][m])
            np.testing.assert_array_equal(z['high'][m],np.clip(z['sample_action'][m],-1,1))
    obs=np.concatenate(obs);actions=np.concatenate(actions);offset=0
    for p in sorted(RUN.glob('buffer-*.npz')):
        with np.load(p) as z:
            n=len(z['observations']);np.testing.assert_array_equal(z['observations'][:,0],obs[offset:offset+n])
            np.testing.assert_array_equal(np.clip(z['actions'][:,0],-1,1),actions[offset:offset+n]);offset+=n
    counts=read(PUBLIC/'training-counts.json')
    assert steps==offset==65536==counts['training_steps'];assert warm==counts['warmup_steps'];assert falls==counts['physical_falls']
    assert reward_error<1e-10
    for p,h in read(CONFIG)['inputs'].items():assert sha(ROOT/p)==h,p
    write_new(PUBLIC/'targeted-audit.json',dict(actual_input_steps_exact=offset,clipped_sample_actual_high_exact=True,native_blend_exact=True,
        reward_penalty_max_error=reward_error,warmup_steps=warm,physical_falls=falls,all_inputs_unchanged=True,
        scope='No full optimizer replay or independent training seed; finite-horizon endings distinct from physical terminations.'))
    print('253 targeted audit passed',steps,warm,falls,flush=True)
if __name__=='__main__':main()
