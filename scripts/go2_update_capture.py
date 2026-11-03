"""Opt-in complete pre-update evidence capture; preserve the original normal guard."""
import copy,hashlib,json,pickle,random
from pathlib import Path
import numpy as np
import torch
from go2_quality_guard import PhysicalQualityPPO,equal_state


def capture_update(model,folder,attempt):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'update-{attempt:02d}-before.pkl';assert not path.exists()
    b=model.rollout_buffer;assert b.full and not b.generator_ready
    py=random.getstate();nr=np.random.get_state();tr=torch.get_rng_state().clone();mode=model.policy.training
    env=model.get_vec_normalize_env()
    record=dict(attempt=attempt,buffer=b,
        context={k:copy.deepcopy(getattr(model,k)) for k in ['num_timesteps','_last_obs','_last_episode_starts','_last_original_obs','_episode_num','_current_progress_remaining','_n_updates']},
        rng=(py,nr,tr),policy=copy.deepcopy(model.policy.state_dict()),Adam=copy.deepcopy(model.policy.optimizer.state_dict()),
        normalization={key:copy.deepcopy(getattr(env,key)) for key in ['obs_rms','ret_rms','returns']} if env is not None else None,
        policy_training_mode=mode)
    with path.open('wb') as f:pickle.dump(record,f,protocol=pickle.HIGHEST_PROTOCOL)
    assert random.getstate()==py and torch.equal(torch.get_rng_state(),tr)
    current=np.random.get_state();assert current[0]==nr[0] and np.array_equal(current[1],nr[1]) and current[2:]==nr[2:]
    assert model.policy.training==mode
    assert equal_state(model.policy.state_dict(),record['policy']) and equal_state(model.policy.optimizer.state_dict(),record['Adam'])
    receipt=dict(attempt=attempt,file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        transitions=b.buffer_size*b.n_envs,full_unflattened_buffer=True,policy_Adam_RNG_and_mode_unchanged=True)
    path.with_suffix('.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt


class CapturedQualityPPO(PhysicalQualityPPO):
    def train(self):
        capture_update(self,self.quality_out/'rollouts',len(self.quality_update_log)+1)
        super().train()
