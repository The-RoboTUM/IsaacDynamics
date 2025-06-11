# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

# from isaaclab_assets.robots.cartpole import CARTPOLE_CFG
from isaaclab_assets.custom.pendulum_grav_simple import PENDULUM_GRAV_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg

# from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform


@configclass
class PendulumSimpleDirectEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    action_scale = 100.0  # [N]
    action_space = 1
    observation_space = 2
    state_space = 0

    # sim
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    stiffness_enable = False
    setpoint = math.pi / 2
    robot_cfg: ArticulationCfg = PENDULUM_GRAV_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # cart_dof_name = "slider_to_cart"
    pendulum_dof_name = "base_to_pendulum"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # reset
    # max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range = [
        -math.pi,
        math.pi,
    ]  # the range in which the pole angle is sampled from on reset [rad]

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pend_pos = -1.0
    # rew_scale_cart_vel = -0.01
    rew_scale_pend_vel = -0.005


class PendulumSimpleDirectEnv(DirectRLEnv):
    cfg: PendulumSimpleDirectEnvCfg

    def __init__(self, cfg: PendulumSimpleDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pendulum_dof_idx, _ = self.pendulum.find_joints(self.cfg.pendulum_dof_name)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.pendulum.data.joint_pos
        self.joint_vel = self.pendulum.data.joint_vel

        # spring
        self.stiffness = 50
        self.spring_enable = cfg.stiffness_enable
        self.setpoint = cfg.setpoint

    def _setup_scene(self):
        self.pendulum = Articulation(self.cfg.robot_cfg)
        # add ground plane
        # spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["pendulum"] = self.pendulum
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        angle = self.joint_pos[:, self._pendulum_dof_idx[0]].unsqueeze(dim=1)
        error = self.setpoint - angle
        # print(error)
        # vel = self.joint_vel[:, self._pendulum_dof_idx[0]].unsqueeze(dim=1)

        actions = self.actions + error * self.stiffness * int(self.spring_enable)

        self.pendulum.set_joint_effort_target(actions, joint_ids=self._pendulum_dof_idx)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, self._pendulum_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pendulum_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pend_pos,
            self.cfg.rew_scale_pend_vel,
            # self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pendulum_dof_idx[0]],
            self.joint_vel[:, self._pendulum_dof_idx[0]],
            # self.joint_pos[:, self._cart_dof_idx[0]],
            # self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.pendulum.data.joint_pos
        self.joint_vel = self.pendulum.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # out_of_bounds = torch.any(
        #     torch.abs(self.joint_pos[:, self._pendulum_dof_idx]) > self.cfg.max_cart_pos,
        #     dim=1,
        # )
        # out_of_bounds = out_of_bounds | torch.any(
        #     torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1
        # )
        return time_out, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.pendulum._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.pendulum.data.default_joint_pos[env_ids]
        joint_pos[:, self._pendulum_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0],
            self.cfg.initial_pole_angle_range[1],
            joint_pos[:, self._pendulum_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.pendulum.data.default_joint_vel[env_ids]

        default_root_state = self.pendulum.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.pendulum.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.pendulum.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.pendulum.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pend_pos: float,
    # rew_scale_cart_vel: float,
    rew_scale_pend_vel: float,
    pend_pos: torch.Tensor,
    pend_vel: torch.Tensor,
    # cart_pos: torch.Tensor,
    # cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pend_pos = rew_scale_pend_pos * torch.sum(torch.square(math.pi / 2 - pend_pos).unsqueeze(dim=1), dim=-1)
    # rew_cart_vel = rew_scale_cart_vel * torch.sum(
    #     torch.abs(cart_vel).unsqueeze(dim=1), dim=-1
    # )
    rew_pend_vel = rew_scale_pend_vel * torch.sum(torch.abs(pend_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pend_pos + rew_pend_vel
    return total_reward
