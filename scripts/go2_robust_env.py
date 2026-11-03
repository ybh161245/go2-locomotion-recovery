"""First full-joint robustness curriculum; nominal dynamics and reward unchanged."""
import json
from pathlib import Path
import numpy as np
from go2_transition_env_v3 import Go2TransitionEnvV3

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT/'configs/go2-robust-finetune.json').read_text(encoding='utf-8'))


class Go2RobustEnv(Go2TransitionEnvV3):
    def __init__(self, command_mode='mixed', worker=0, archive=None, config=None, **kwargs):
        self.config = CONFIG if config is None else config
        self.worker = worker
        self.archive = Path(archive) if archive else None
        if self.archive: self.archive.mkdir(parents=True, exist_ok=False)
        self.total_steps = 0
        self.episode = -1
        self.events = []
        self.trace = None
        self.episode_meta = {}
        super().__init__(command_mode, **kwargs)

    def flush_trace(self, reason):
        if self.archive and self.trace and self.trace['action']:
            stem = self.archive/f'episode-{self.episode:05d}'
            np.savez_compressed(str(stem)+'.npz', **{k: np.asarray(v) for k,v in self.trace.items()})
            meta = dict(self.episode_meta, end_reason=reason, steps=len(self.trace['action']))
            Path(str(stem)+'.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        self.trace = None

    def reset(self, *, seed=None, options=None):
        self.flush_trace('reset')
        obs, info = super().reset(seed=seed, options=options)
        self.data.xfrc_applied[:] = 0
        self.episode += 1
        level = max(i for i,c in enumerate(self.config['curriculum']) if self.total_steps*self.config['workers'] >= c['start_steps'])
        spec = self.config['curriculum'][level]
        self.events = []
        if self.np_random.random() >= self.config['normal_episode_probability']:
            count = int(self.np_random.integers(spec['pulses'][0], spec['pulses'][1]+1))
            start = int(self.np_random.integers(100, 401))
            for j in range(count):
                angle = float(self.np_random.uniform(0, 2*np.pi))
                force = float(self.np_random.uniform(*spec['force_N']))
                self.events.append(dict(start_step=start+50*j, duration_steps=int(self.np_random.integers(spec['duration_steps'][0], spec['duration_steps'][1]+1)), local_force=[force*np.cos(angle), force*np.sin(angle), 0.]))
        if options and 'events' in options: self.events = options['events']
        self.episode_meta = dict(worker=self.worker, episode=self.episode, reset_seed=seed, level=level,
                                 global_steps_at_reset=self.total_steps*self.config['workers'], events=self.events)
        self.trace = {k: [] for k in ['observation','action','qpos','qvel','target','world_wrench','command','reward','terminated','truncated']} if self.archive else None
        return obs, info

    def step(self, action):
        local = np.zeros(3)
        for event in self.events:
            if event['start_step'] <= self.step_count < event['start_step']+event['duration_steps']:
                local += np.asarray(event['local_force'])
        wrench = np.r_[self.data.xmat[self.base].reshape(3,3)@local, np.zeros(3)]
        self.data.xfrc_applied[:] = 0
        self.data.xfrc_applied[self.base] = wrench
        before = self.observe().copy(); command = self.command.copy()
        obs, reward, term, trunc, info = super().step(action)
        reward = self.adjust_reward(reward, info)
        self.total_steps += 1
        if self.trace is not None:
            values = [before, np.asarray(action).copy(), self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy(), wrench, command, reward, term, trunc]
            for key,value in zip(self.trace,values): self.trace[key].append(value)
        info['push_force_N'] = float(np.linalg.norm(local))
        info['curriculum_level'] = self.episode_meta['level']
        if term or trunc: self.flush_trace('terminated' if term else 'time_limit')
        return obs, reward, term, trunc, info

    def adjust_reward(self, reward, info):
        return reward

    def close(self):
        self.flush_trace('training_end_partial')
        super().close()
