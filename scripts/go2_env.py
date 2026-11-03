"""Go2 commanded-velocity task with native torque-limited joint position servos."""
from functools import lru_cache

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from go2_model import load_model, joint_map, set_home

ENV_CONFIG = dict(physics_dt=0.002, control_substeps=10, control_dt=0.02,
                  kp=40.0, kd=1.0, action_scale=0.35, episode_seconds=12.0,
                  nominal_angles=[0.0,0.8,-1.6]*4, clock_hz=2.0,
                  command_interval_seconds=4.0, velocity_error_scale=0.16,
                  yaw_error_scale=0.25, min_height=0.16, max_height=0.50,
                  min_projected_up=0.65)


@lru_cache(maxsize=1)
def servo_model():
    model = load_model()
    mapping = joint_map(model)
    if [r['qpos_adr'] for r in mapping] != list(range(7,19)) or [r['actuator_id'] for r in mapping] != list(range(12)):
        raise RuntimeError('Unexpected Go2 joint ordering')
    # Position-servo torque is kp*(q_target-q)-kd*qdot. Native MuJoCo computes
    # it at every physics substep; force limits preserve upstream motor limits.
    original_torque_limits = model.actuator_ctrlrange.copy()
    model.actuator_gainprm[:,0] = ENV_CONFIG['kp']
    model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
    model.actuator_biasprm[:,0] = 0
    model.actuator_biasprm[:,1] = -ENV_CONFIG['kp']
    model.actuator_biasprm[:,2] = -ENV_CONFIG['kd']
    model.actuator_forcelimited[:] = 1
    model.actuator_forcerange[:] = original_torque_limits
    model.actuator_ctrlrange[:] = np.array([r['range'] for r in mapping])
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 640
    return model


class Go2VelocityEnv(gym.Env):
    metadata = {'render_modes':['rgb_array'], 'render_fps':25}

    def __init__(self, command_mode='mixed', render_mode=None, episode_seconds=None):
        super().__init__()
        if command_mode not in ('forward','mixed','turn','external'):
            raise ValueError(command_mode)
        self.command_mode, self.render_mode = command_mode, render_mode
        self.episode_seconds = ENV_CONFIG['episode_seconds'] if episode_seconds is None else float(episode_seconds)
        self.model = servo_model()
        self.data = mujoco.MjData(self.model)
        self.base = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'base')
        self.floor = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
        self.feet = set(mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ('FL','FR','RL','RR'))
        self.nominal = np.array(ENV_CONFIG['nominal_angles'])
        self.action_space = spaces.Box(-1.0,1.0,(12,),dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf,np.inf,(50,),dtype=np.float32)
        self.command = np.zeros(3)
        self.last_action = np.zeros(12)
        self._velocity = np.zeros(6)
        self.renderer = None
        self.step_count = 0

    def velocity(self):
        # MuJoCo BODY's local spatial frame is its inertial frame, which need
        # not align with the robot's named body axes. Read world-oriented
        # spatial velocity, then explicitly rotate into the base body frame.
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,
                                self.base,self._velocity,0)
        rotation=self.data.xmat[self.base].reshape(3,3)
        return rotation.T@self._velocity[3:],rotation.T@self._velocity[:3]

    def gravity(self):
        return self.data.xmat[self.base].reshape(3,3).T @ np.array([0.,0.,-1.])

    def observe(self):
        lin,ang = self.velocity()
        phase = 2*np.pi*ENV_CONFIG['clock_hz']*self.step_count*ENV_CONFIG['control_dt']
        return np.concatenate((lin*2,ang*.25,self.gravity(),
                               self.command*np.array([2,2,.25]),
                               self.data.qpos[7:]-self.nominal,self.data.qvel[6:]*.05,
                               self.last_action,[np.sin(phase),np.cos(phase)])).astype(np.float32)

    def _sample_command(self):
        if self.command_mode == 'forward':
            self.command[:] = [.4,0,0]
        elif self.command_mode == 'turn':
            self.command[:] = [.2,0,self.np_random.choice([-.5,.5])]
        elif self.command_mode == 'mixed':
            kind = self.np_random.uniform()
            if kind < .15:
                self.command[:] = 0
            elif kind < .7:
                self.command[:] = [self.np_random.uniform(.15,.6),0,0]
            else:
                self.command[:] = [self.np_random.uniform(0,.4),0,self.np_random.uniform(-.6,.6)]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        set_home(self.model,self.data)
        self.data.qpos[7:] = self.nominal + self.np_random.uniform(-.015,.015,12)
        self.data.qpos[2] = .32
        self.data.ctrl[:] = self.nominal
        self.data.qvel[:] = 0
        self.last_action[:] = 0
        self.step_count = 0
        if options and 'command' in options:
            self.command[:] = options['command']
        else:
            self._sample_command()
        mujoco.mj_forward(self.model,self.data)
        return self.observe(),{}

    def step(self, action):
        action_limit=getattr(self,'joint_action_limit',1.)
        action = np.clip(np.asarray(action,dtype=np.float64),-action_limit,action_limit)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError('Expected finite 12-dimensional action')
        desired = self.action_targets(action)
        desired = np.clip(desired,self.model.actuator_ctrlrange[:,0],self.model.actuator_ctrlrange[:,1])
        self.data.ctrl[:] = desired
        mujoco.mj_step(self.model,self.data,nstep=ENV_CONFIG['control_substeps'])
        self.step_count += 1
        lin,ang = self.velocity()
        grav = self.gravity()
        error_lin = float(np.square(lin[:2]-self.command[:2]).sum())
        error_yaw = float((ang[2]-self.command[2])**2)
        bad_contact = False
        for contact in self.data.contact[:self.data.ncon]:
            a,b = int(contact.geom1),int(contact.geom2)
            if self.floor in (a,b) and (b if a == self.floor else a) not in self.feet:
                bad_contact = True
                break
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        terminated = bool(not finite or not ENV_CONFIG['min_height'] < self.data.qpos[2] < ENV_CONFIG['max_height']
                          or -grav[2] < ENV_CONFIG['min_projected_up'] or bad_contact)
        tracking = 2*np.exp(-error_lin/ENV_CONFIG['velocity_error_scale']) + np.exp(-error_yaw/ENV_CONFIG['yaw_error_scale'])
        costs = (.5*lin[2]**2 + .05*np.square(ang[:2]).sum() + .5*np.square(grav[:2]).sum()
                 + .5*((self.data.qpos[2]-.29)/.1)**2
                 + .0002*np.square(self.data.actuator_force).sum()
                 + .01*np.square(action-self.last_action).sum()
                 + .0001*np.square(self.data.qvel[6:]).sum())
        reward = float(tracking + .25 - costs - (5.0 if terminated else 0))
        info = dict(vx=float(lin[0]),vy=float(lin[1]),yaw_rate=float(ang[2]),
                    command_vx=float(self.command[0]),command_yaw=float(self.command[2]),
                    velocity_error=float(np.sqrt(error_lin)),yaw_error=float(np.sqrt(error_yaw)),
                    height=float(self.data.qpos[2]),tracking_reward=float(tracking),costs=float(costs))
        self.last_action[:] = action
        truncated = self.step_count >= int(self.episode_seconds/ENV_CONFIG['control_dt'])
        interval = int(ENV_CONFIG['command_interval_seconds']/ENV_CONFIG['control_dt'])
        if self.command_mode != 'external' and self.step_count % interval == 0:
            self._sample_command()
        obs = self.observe()
        if not np.isfinite(obs).all() or not np.isfinite(reward):
            raise RuntimeError('Non-finite Go2 transition')
        return obs,reward,terminated,bool(truncated),info

    def action_targets(self, action):
        """Default joint mapping; bounded diagnostic overrides keep policy actions intact."""
        return self.nominal + ENV_CONFIG['action_scale']*action

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model,height=640,width=960)
        camera = mujoco.MjvCamera()
        camera.type=mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid=self.base
        camera.distance,camera.azimuth,camera.elevation=1.35,135,-22
        option=mujoco.MjvOption()
        option.geomgroup[3]=0
        self.renderer.update_scene(self.data,camera=camera,scene_option=option)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer=None
