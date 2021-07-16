import os
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from gym import utils
from gym.envs.mujoco import mujoco_env

import random
import test


class Model():
    def __init__(self, task='walk', latency=0, xyz_noise_std=0.0, rpy_noise_std=0.0, min_obs_stack=1):
        # PID params
        self.Kp = 0.8
        self.Ki = 0
        self.Kd = 1

        self.task = task

        self.n_delay_steps = latency # 1 step = 50 ms
        self.n_past_obs = self.n_delay_steps + min_obs_stack

        self.xyz_noise_std = xyz_noise_std
        self.rpy_noise_std = rpy_noise_std

        self.pre_point_first_xyz = [0, 0, 0]
        self.pre_point_second_xyz = [0, 0, 0]
        self.pre_point_third_xyz = [0, 0, 0]

        self.int_err, self.past_err = 0, 0
        self.delayed_meas = [(np.random.random(3), np.random.random(3))]
        self.past_obses = deque([np.zeros(29)]*self.n_past_obs, maxlen=self.n_past_obs)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        # mujoco_env.MujocoEnv.__init__(self, f'{dir_path}/mujoco.xml', 5)
        # utils.EzPickle.__init__(self)

    def step(self, setpoints):
        # self.prev_body_xyz, self.prev_body_rpy = self.delayed_meas[0]

        # # compute torques
        # joint_positions = self.sim.data.qpos.flat[-8:]
        # joint_velocities = self.sim.data.qvel.flat[-8:]

        # #
        # # motor control using a PID controller
        # #

        # # limit motor maximum speed (this matches the real servo motors)
        # timestep = self.dt
        # vel_limit = 0.1  # rotational units/s
        # #motor_setpoints = np.clip(2 * setpoints, joint_positions - timestep*vel_limit, joint_positions + timestep*vel_limit)

        # # joint positions are scaled somehow roughly between -1.8...1.8
        # # to meet these limits, multiply setpoints by two.
        # err = 2 * setpoints - joint_positions
        # self.int_err += err
        # d_err = err - self.past_err
        # self.past_err = err

        # torques = np.minimum(
        #     1,
        #     np.maximum(-1, self.Kp * err + self.Ki * self.int_err + self.Kd * d_err),
        # )

        # # clip available torque if the joint is moving too fast
        # lowered_torque = 0.0
        # torques = np.clip(torques,
        #     np.minimum(-lowered_torque, (-vel_limit-np.minimum(0, joint_velocities)) / vel_limit),
        #     np.maximum(lowered_torque, (vel_limit-np.maximum(0, joint_velocities)) / vel_limit))

        # self.do_simulation(torques, self.frame_skip)
        # ob = self._get_obs()

        # if self.task == 'walk':
        #     reward = ob[0]
        # elif self.task == 'sleep':
        #     reward = -np.square(ob[3])
        # elif self.task == 'turn':
        #     goal = np.array([0, 0, 0])
        #     body_rpy = np.arctan2(ob[7:10], ob[10:13])
        #     reward = -np.square(goal[0]-body_rpy[0])
        # else:
        #     raise Exception('Unknown task')

        # state = self.state_vector()
        # notdone = np.isfinite(state).all()
        # done = not notdone
        
        ob = self._get_obs()
        # TODO : calculate reward
        reward = abs(ob[0] - self.pre_point_first_xyz\
                +ob[1] - self.pre_point_second_xyz\
                +ob[2] - self.pre_point_third_xyz)

        self.pre_point_first_xyz = self.point_first_xyz
        self.pre_point_second_xyz = self.point_second_xyz
        self.pre_point_third_xyz = self.point_third_xyz
        self.pre_foot_first_xy = self.foot_first_xy
        self.pre_foot_second_xy = self.foot_second_xy
        self.pre_foot_third_xy = self.foot_third_xy
        self.pre_foot_forth_xy = self.foot_forth_xy

        return ob, reward, {}

    def _get_obs(self):
        # body_xyz = np.copy(self.sim.data.qpos.flat[:3])
        # body_quat = np.copy(self.sim.data.qpos.flat[3:7])
        # body_rpy = R.from_quat(body_quat).as_euler('xyz')

        # # add noise
        # body_xyz += np.random.randn(3) * self.xyz_noise_std
        # body_rpy += np.random.randn(3) * self.rpy_noise_std

        # self.delayed_meas.append((body_xyz, body_rpy))

        # body_xyz, body_rpy = self.delayed_meas[0]

        # joint_positions = self.sim.data.qpos.flat[-8:]
        # joint_positions_vel = self.sim.data.qvel.flat[-8:]

        obs = test.receive(self)

        self.point_first_xyz = obs[0:3]
        self.point_second_xyz = obs[3:6]
        self.point_third_xyz = obs[6:9]
        self.foot_first_xy = obs[9:11]
        self.foot_second_xy = obs[11:13]
        self.foot_third_xy = obs[13:15]
        self.foot_forth_xy = obs[15:17]

        # body_xyz_vel = body_xyz - self.prev_body_xyz
        # body_rpy_vel = body_rpy - self.prev_body_rpy

        obs = np.concatenate([
            self.point_first_xyz,
            self.point_second_xyz,
            self.point_third_xyz,
            self.foot_first_xy,
            self.foot_second_xy,
            self.foot_third_xy,
            self.foot_forth_xy,
        ])

        # obs = np.concatenate([
        #     body_xyz_vel,
        #     body_xyz[-1:],
        #     body_rpy_vel,
        #     np.sin(body_rpy),
        #     np.cos(body_rpy),
        #     joint_positions,
        #     joint_positions_vel,
        # ])

        self.past_obses.append(obs)

        return np.concatenate(self.past_obses)

    def reset_model(self):
        # self.int_err = 0
        # self.past_err = 0

        # qpos = self.init_qpos
        # qpos[-8:] = qpos[-8:] + self.np_random.uniform(size=8, low=-.1, high=.1)
        # qvel = self.init_qvel #+ self.np_random.randn(self.model.nv) * .1
        # self.set_state(qpos, qvel)

        # self.prev_body_xyz = np.copy(self.sim.data.qpos.flat[:3])
        # self.prev_body_rpy = R.from_quat(np.copy(self.sim.data.qpos.flat[3:7])).as_euler('xyz')

        # body_xyz = np.copy(self.sim.data.qpos.flat[:3])
        # body_quat = np.copy(self.sim.data.qpos.flat[3:7])
        # body_rpy = R.from_quat(body_quat).as_euler('xyz')

        # self.delayed_meas = deque([(body_xyz, body_rpy)]*(self.n_delay_steps+1), maxlen=(self.n_delay_steps+1))
        # self.past_obses = deque([np.zeros(29)]*self.n_past_obs, maxlen=self.n_past_obs)

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 2

    def action_space_sample(self):
        # print("hello")
        return np.random.randint(2, size=4)

