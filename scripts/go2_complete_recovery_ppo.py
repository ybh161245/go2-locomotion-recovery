"""112 actor learns only applied actions; critic learns every resulting transition."""
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from run_go2_bilateral_pilot import PilotPPO


class AllValueMaskedActorBuffer(RolloutBuffer):
    def reset(self):
        super().reset();self.actor_applied=np.zeros((self.buffer_size,self.n_envs),dtype=bool)

    def get(self,batch_size=None):
        assert self.full
        if not self.generator_ready:
            for name in ['observations','actions','values','log_probs','advantages','returns']:
                self.__dict__[name]=self.swap_and_flatten(self.__dict__[name])
            self.actor_applied=self.swap_and_flatten(self.actor_applied[...,None]).ravel();self.generator_ready=True
        ids=np.random.permutation(self.buffer_size*self.n_envs);batch_size=batch_size or len(ids)
        for i in range(0,len(ids),batch_size):
            chosen=ids[i:i+batch_size];self.batch_actor_mask=self.to_torch(self.actor_applied[chosen]);yield self._get_samples(chosen)


class CompletePPO(PilotPPO):
    """Same PPO clipped surrogate; separate actor mask from full-state value loss.

    After handback the selected action cannot affect physics. Those samples
    contribute value/GAE, and cannot create policy or entropy gradients.
    """
    def train(self):
        from go2_update_capture import capture_update
        capture_update(self,self.pilot_out/'rollouts',self.num_timesteps//(self.n_steps*self.n_envs))
        self.train_without_capture()

    def train_without_capture(self):
        self.policy.set_training_mode(True);self._update_learning_rate(self.policy.optimizer)
        clip=self.clip_range(self._current_progress_remaining);losses=[];kls=[];active_counts=[];value_counts=[];keep=True
        for epoch in range(self.n_epochs):
            for data in self.rollout_buffer.get(self.batch_size):
                mask=self.rollout_buffer.batch_actor_mask;values,logp,entropy=self.policy.evaluate_actions(data.observations,data.actions);values=values.flatten()
                advantages=data.advantages
                if self.normalize_advantage and int(mask.sum())>1:
                    chosen=advantages[mask];advantages=(advantages-chosen.mean())/(chosen.std()+1e-8)
                ratio=torch.exp(logp-data.old_log_prob)
                if bool(mask.any()):
                    actor=-torch.min(advantages*ratio,advantages*torch.clamp(ratio,1-clip,1+clip))[mask].mean()
                    ent_loss=-(entropy[mask].mean() if entropy is not None else -logp[mask].mean())
                    with torch.no_grad():
                        delta=logp[mask]-data.old_log_prob[mask];kl=float(((torch.exp(delta)-1)-delta).mean())
                else:actor=logp.sum()*0;ent_loss=logp.sum()*0;kl=0.
                value=F.mse_loss(data.returns,values);loss=actor+self.ent_coef*ent_loss+self.vf_coef*value
                if self.target_kl is not None and kl>1.5*self.target_kl:keep=False;break
                self.policy.optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.policy.parameters(),self.max_grad_norm);self.policy.optimizer.step()
                losses.append(float(loss.detach()));kls.append(kl);active_counts.append(int(mask.sum()));value_counts.append(len(mask))
            self._n_updates+=1
            if not keep:break
        assert losses and all(np.isfinite(losses))
        self.logger.record('train/loss',np.mean(losses));self.logger.record('train/approx_kl',np.mean(kls));self.logger.record('train/actor_presentations',sum(active_counts));self.logger.record('train/value_presentations',sum(value_counts));self.logger.record('train/n_updates',self._n_updates)
