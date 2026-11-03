"""Fixed reference-bank guard around PPO updates; empirical action/KL bounds."""
import copy,json,math
from pathlib import Path
import numpy as np
import torch
from go2_retention_ppo import RetentionPPO

def snapshot(model):
    return copy.deepcopy(model.policy.state_dict()),copy.deepcopy(model.policy.optimizer.state_dict())

def restore(model,state):
    model.policy.load_state_dict(state[0]);model.policy.optimizer.load_state_dict(state[1]);model.policy.optimizer.zero_grad()

def metrics(model):
    with torch.no_grad():
        current=model.policy.get_distribution(model.retention_obs).distribution.mean
        delta=.35*(current.clamp(-1,1)-model.retention_targets.clamp(-1,1))
        std=model.policy.log_std.exp();refstd=model.retention_reference_log_std.exp()
        kl=(torch.log(std/refstd)+(refstd.square()+(model.retention_targets-current).square())/(2*std.square())-.5).sum(dim=1)
        groups={}
        for label in sorted(set(model.retention_group_labels)):
            ids=np.flatnonzero(model.retention_group_labels==label)
            groups[str(label)]=dict(steps=len(ids),rms_rad=float(delta[ids].square().mean().sqrt()),mean_kl=float(kl[ids].mean()))
        return dict(bank_steps=len(current),clipped_target_rms_rad=float(delta.square().mean().sqrt()),mean_kl=float(kl.mean()),groups=groups)

def acceptable(report,cfg):
    values=[report['clipped_target_rms_rad'],report['mean_kl']]+[v[k] for v in report['groups'].values() for k in ['rms_rad','mean_kl']]
    return all(math.isfinite(v) for v in values) and report['clipped_target_rms_rad']<=cfg['rms_budget_rad'] and report['mean_kl']<=cfg['kl_budget'] and all(v['rms_rad']<=cfg['group_rms_budget_rad'] and v['mean_kl']<=cfg['group_kl_budget'] for v in report['groups'].values())

class GuardedRetentionPPO(RetentionPPO):
    def _excluded_save_params(self):
        return super()._excluded_save_params()+['retention_reference_log_std','retention_group_labels','retention_guard_dir']

    def train(self):
        before=snapshot(self)
        super().train()
        candidate=metrics(self);accepted=acceptable(candidate,self.retention_cfg)
        index=len(self.retention_update_log)+1;out=Path(self.retention_guard_dir);file=None
        if not accepted:
            folder=out/'rejected-updates';folder.mkdir(exist_ok=True);path=folder/f'update-{index:02d}-candidate.zip';assert not path.exists()
            self.save(path);file=path.relative_to(out).as_posix();restore(self,before)
        after=metrics(self);assert acceptable(after,self.retention_cfg)
        self.retention_update_log.append(dict(attempt=index,added_steps=self.num_timesteps-self.retention_training_start,accepted=accepted,candidate=candidate,after=after,rejected_candidate=file))
        (out/'retention-updates.json').write_text(json.dumps(self.retention_update_log,indent=2))
        self.logger.record('train/retention_update_accepted',int(accepted));self.logger.record('train/retention_bank_rms_rad',after['clipped_target_rms_rad']);self.logger.record('train/retention_bank_kl',after['mean_kl'])
        print(f'RETENTION_UPDATE {index} accepted={accepted} candidate_rms={candidate["clipped_target_rms_rad"]:.6f} candidate_kl={candidate["mean_kl"]:.6f}',flush=True)
