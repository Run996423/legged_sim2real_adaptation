from isaacgym import gymtorch, gymapi
import torch
import numpy as np

from legged_gym.envs import LeggedRobot
from legged_gym.envs.go1_adaptation.wheeled_go1_utils import quat_to_rpy, quat_from_rpy, quat_to_rot_mat, BMxBV
from legged_gym.envs.go1_adaptation.wheeled_go1_kin import position_level_forward_kinematics, compute_q40


class Go1AdaptationOracle(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.history_length = cfg.env.history_length
        self.property_dim = cfg.env.property_dim
        self.state_dim = cfg.env.state_dim
        self.num_supervised_latent = cfg.env.num_supervised_latent
        self.property = torch.zeros(cfg.env.num_envs, self.property_dim, device=sim_device, dtype=torch.float32)
        self.nominal_property = torch.empty_like(self.property)

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.joint_idx_gym2mjc = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        self.joint_idx_mjc2gym = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        self.hip_pos_base_frame = torch.tensor([[+0.1881, -0.04675, 0.0],
                                                [+0.1881, +0.04675, 0.0],
                                                [-0.1881, -0.04675, 0.0],
                                                [-0.1881, +0.04675, 0.0]], device=self.device)
        self.default_contact_pos_cfcb = torch.tensor([[+0.20, -0.16, -0.26],
                                                      [+0.20, +0.16, -0.26],
                                                      [-0.20, -0.16, -0.26],
                                                      [-0.20, +0.16, -0.26]], device=self.device)

        self.state_history = torch.zeros(self.num_envs, self.history_length + 1, self.state_dim,
                                         device=self.device, dtype=torch.float32)
        self.action_history = torch.zeros(self.num_envs, self.history_length, self.num_actions,
                                          device=self.device, dtype=torch.float32)
        self.obs_history = torch.zeros(self.num_envs, self.history_length, self.num_obs,
                                       device=self.device, dtype=torch.float32)

        self.disturbance_force = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device, dtype=torch.float32)
        self.supervised_latent_ground_truth = torch.zeros(self.num_envs, self.num_supervised_latent, device=self.device, dtype=torch.float32)

    def _process_rigid_body_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_base_mass:
            self.nominal_property[env_id, 0] = props[0].mass
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
            props[0].invMass = 1 / props[0].mass
            self.property[env_id, 0] = props[0].mass
        if self.cfg.domain_rand.randomize_base_inertia:
            inertia = props[0].inertia
            self.nominal_property[env_id, 1] = inertia.x.x
            self.nominal_property[env_id, 2] = inertia.y.y
            self.nominal_property[env_id, 3] = inertia.z.z
            rng = self.cfg.domain_rand.added_inertia_range
            inertia.x.x += np.random.uniform(rng[0], rng[1])
            inertia.y.y += np.random.uniform(rng[0], rng[1])
            inertia.z.z += np.random.uniform(rng[0], rng[1])
            inertia_numpy = np.array([
                [inertia.x.x, inertia.x.y, inertia.x.z],
                [inertia.y.x, inertia.y.y, inertia.y.z],
                [inertia.z.x, inertia.z.y, inertia.z.z]
            ])
            inv_inertia_numpy = np.linalg.inv(inertia_numpy)
            props[0].invInertia.x.x = inv_inertia_numpy[0, 0]
            props[0].invInertia.x.y = inv_inertia_numpy[0, 1]
            props[0].invInertia.x.z = inv_inertia_numpy[0, 2]
            props[0].invInertia.y.x = inv_inertia_numpy[1, 0]
            props[0].invInertia.y.y = inv_inertia_numpy[1, 1]
            props[0].invInertia.y.z = inv_inertia_numpy[1, 2]
            props[0].invInertia.z.x = inv_inertia_numpy[2, 0]
            props[0].invInertia.z.y = inv_inertia_numpy[2, 1]
            props[0].invInertia.z.z = inv_inertia_numpy[2, 2]
            self.property[env_id, 1] = inertia.x.x
            self.property[env_id, 2] = inertia.y.y
            self.property[env_id, 3] = inertia.z.z
        return props

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.state_history[env_ids] = 0.
        self.action_history[env_ids] = 0.
        self.obs_history[env_ids] = 0.

    def _get_noise_scale_vec(self, cfg):
        noise_vec_for_single_obs = torch.zeros(55, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec_for_single_obs[0:3] = noise_scales.lin_vel
        noise_vec_for_single_obs[3:6] = noise_scales.ang_vel
        noise_vec_for_single_obs[6:9] = noise_scales.gravity
        noise_vec_for_single_obs[9:21] = noise_scales.dof_pos
        noise_vec_for_single_obs[21:33] = noise_scales.dof_vel
        noise_vec_for_single_obs[33:45] = 0.  # previous actions
        noise_vec_for_single_obs[45:48] = 0.  # commands
        noise_vec_for_single_obs[48:51] = noise_scales.disturbance  # disturbance
        noise_vec_for_single_obs[51:55] = 0  # contact_state
        noise_vec_for_single_obs = noise_vec_for_single_obs * noise_level
        return noise_vec_for_single_obs

    def compute_observations(self):
        single_obs = torch.cat((self.base_lin_vel,
                                self.base_ang_vel,
                                self.projected_gravity,
                                self.dof_pos[:, self.joint_idx_gym2mjc],
                                self.dof_vel[:, self.joint_idx_gym2mjc],
                                self.actions,
                                self.commands[:, 0:3],
                                self.disturbance_force[:, 0, :],
                                torch.greater(self.contact_forces[:, self.feet_indices, 2], 1.).float()[:, [1, 0, 3, 2]]
                                ), dim=-1)
        if self.add_noise:
            single_obs = single_obs + (2 * torch.rand_like(self.noise_scale_vec) - 1) * self.noise_scale_vec
        self.obs_history = torch.cat(
            (single_obs[:, None, 0:51], self.obs_history[:, :-1, :]), dim=1
        )
        self.obs_buf = torch.cat((self.obs_history.flatten(1, 2), single_obs[:, 0:51]), dim=1)
        self.state_history = torch.cat(
            (single_obs[:, None, 3:33], self.state_history[:, :-1, :]), dim=1)
        self.action_history = torch.cat((self.actions[:, None, :], self.action_history[:, :-1, :]), dim=1)
        self.privileged_obs_buf = single_obs[:, 0:51].clone()
        self.supervised_latent_ground_truth = torch.cat((single_obs[:, 0:3], single_obs[:, -7:]), dim=1)

    def _compute_torques(self, actions):
        # 在这里才能在每个physics_dt都apply force，如果写在_post_physics_step_callback里面，则每decimation才apply一次
        if self.cfg.domain_rand.disturbance:
            env_ids = (self.episode_length_buf % int(self.cfg.domain_rand.disturbance_interval_s / self.dt) == 0).nonzero(as_tuple=False).flatten()
            # reset disturbance forces
            magnitude_range = self.cfg.domain_rand.disturbance_force_range
            magnitude = torch.rand(self.num_envs, 1, device=self.device) * (
                    magnitude_range[1] - magnitude_range[0]) + magnitude_range[0]
            cos_theta = torch.rand(self.num_envs, 1, device=self.device) * 2 - 1
            theta = torch.acos(cos_theta)
            phi = torch.rand(self.num_envs, 1, device=self.device) * 2 * torch.pi
            if self.cfg.domain_rand.disturbance == 'along_to_cmd':
                phi = torch.where(
                    torch.linalg.norm(self.commands[:, 0:2], dim=1, keepdim=True) > 0.1,
                    torch.atan2(self.commands[:, 1:2], self.commands[:, 0:1]),
                    (torch.rand_like(phi) * 2 - 1) * torch.pi * 2
                )
                theta = (torch.rand_like(theta) * 2 - 1) * torch.pi / 12 + torch.pi / 2
            elif self.cfg.domain_rand.disturbance == 'opposite_to_cmd':
                phi = torch.where(
                    torch.linalg.norm(self.commands[:, 0:2], dim=1, keepdim=True) > 0.1,
                    torch.atan2(self.commands[:, 1:2], self.commands[:, 0:1]) + torch.pi,
                    (torch.rand_like(phi) * 2 - 1) * torch.pi * 2
                )
                theta = (torch.rand_like(theta) * 2 - 1) * torch.pi / 12 + torch.pi / 2
            elif self.cfg.domain_rand.disturbance == 'along/opposite_to_cmd':
                phi = torch.where(
                    torch.linalg.norm(self.commands[:, 0:2], dim=1, keepdim=True) > 0.1,
                    torch.atan2(self.commands[:, 1:2], self.commands[:, 0:1])
                    + torch.randint(0, 2, phi.shape, device=phi.device).to(phi.dtype) * torch.pi,  # 抵抗或顺着命令
                    (torch.rand_like(phi) * 2 - 1) * torch.pi * 2
                )
                theta = (torch.rand_like(theta) * 2 - 1) * torch.pi / 12 + torch.pi / 2
            elif self.cfg.domain_rand.disturbance == 'up/down':
                phi = (torch.rand_like(phi) * 2 - 1) * torch.pi * 2
                theta = torch.randint(0, 2, theta.shape, device=theta.device).to(theta.dtype) * torch.pi + (
                        torch.rand_like(theta) * 2 - 1) * torch.pi / 6
            elif self.cfg.domain_rand.disturbance == 'lateral_to_cmd':
                phi = torch.where(
                    torch.linalg.norm(self.commands[:, 0:2], dim=1, keepdim=True) > 0.1,
                    torch.atan2(self.commands[:, 1:2], self.commands[:, 0:1])
                    + torch.randint(0, 2, phi.shape, device=phi.device).to(phi.dtype) * torch.pi / 2,
                    (torch.rand_like(phi) * 2 - 1) * torch.pi * 2
                )
                theta = (torch.rand_like(theta) * 2 - 1) * torch.pi / 12 + torch.pi / 2
            elif self.cfg.domain_rand.disturbance == '+x':
                phi = torch.zeros_like(phi)
                theta = torch.zeros_like(theta) + torch.pi / 2
            elif self.cfg.domain_rand.disturbance == '+y':
                phi = torch.zeros_like(phi) + torch.pi / 2
                theta = torch.zeros_like(theta) + torch.pi / 2
            elif self.cfg.domain_rand.disturbance == '+z':
                phi = torch.zeros_like(phi)
                theta = torch.zeros_like(theta)
            elif self.cfg.domain_rand.disturbance == 'x':
                phi = torch.zeros_like(phi) + torch.randint(0, 2, phi.shape, device=phi.device).to(phi.dtype) * torch.pi
                theta = torch.zeros_like(theta) + torch.pi / 2
            elif self.cfg.domain_rand.disturbance == 'y':
                phi = torch.zeros_like(phi) + torch.pi / 2 + torch.randint(0, 2, phi.shape, device=phi.device).to(phi.dtype) * torch.pi
                theta = torch.zeros_like(theta) + torch.pi / 2
            elif self.cfg.domain_rand.disturbance == 'z':
                phi = torch.zeros_like(phi)
                theta = torch.zeros_like(theta) + torch.randint(0, 2, phi.shape, device=phi.device).to(phi.dtype) * torch.pi
            elif self.cfg.domain_rand.disturbance == 'x':
                phi = torch.zeros_like(phi) + torch.randint(0, 2, phi.shape, device=phi.device).to(phi.dtype) * torch.pi
                theta = torch.zeros_like(theta) + torch.pi * 0.5
            elif self.cfg.domain_rand.disturbance == 'y':
                phi = torch.zeros_like(phi) + torch.pi / 2 + torch.randint(0, 2, phi.shape, device=phi.device).to(phi.dtype) * torch.pi
                theta = torch.zeros_like(theta) + torch.pi * 0.5
            sin_theta = torch.sin(theta)
            cos_theta = torch.cos(theta)
            x = magnitude * sin_theta * torch.cos(phi)
            y = magnitude * sin_theta * torch.sin(phi)
            z = magnitude * cos_theta
            base_disturbance = torch.stack([x, y, z], dim=-1)
            self.disturbance_force[env_ids, 0:1, :] = base_disturbance[env_ids]
            # self.disturbance_force *= torch.greater(torch.linalg.norm(self.disturbance_force, dim=-1, keepdim=True), 5).float()
            self.gym.apply_rigid_body_force_at_pos_tensors(
                sim=self.sim,
                forceTensor=gymtorch.unwrap_tensor(self.disturbance_force),
                posTensor=None,
                space=gymapi.LOCAL_SPACE)

        leg_joint_pos_target = self.default_dof_pos[:, self.joint_idx_gym2mjc] + actions * 0.2

        leg_joint_pos = self.dof_pos[:, self.joint_idx_gym2mjc].clone()
        leg_joint_vel = self.dof_vel[:, self.joint_idx_gym2mjc].clone()

        kp = self.p_gains[self.joint_idx_gym2mjc].clone()
        kd = self.d_gains[self.joint_idx_gym2mjc].clone()

        leg_torque = kp * (leg_joint_pos_target - leg_joint_pos) + kd * (0. - leg_joint_vel)
        torques = leg_torque[:, self.joint_idx_mjc2gym].clone()
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reward_alive(self):
        return torch.ones_like(self.base_lin_vel[:, 0])

    def _reward_ugly_contact_pos(self):
        joint_pos = self.dof_pos[:, self.joint_idx_gym2mjc]
        leg_joint_pos = joint_pos
        base_quat = torch.empty_like(self.base_quat)
        base_quat[:, 0] = self.base_quat[:, 3]
        base_quat[:, 1:4] = self.base_quat[:, 0:3]

        base_rpy = quat_to_rpy(base_quat)
        base_rot_mat = quat_to_rot_mat(base_quat)
        base_rot_mat_only_yaw = quat_to_rot_mat(quat_from_rpy(
            base_rpy * torch.tensor([0., 0., 1.], device=self.device)))

        N = self.num_envs
        zeros = lambda *shape: torch.zeros(shape, device=self.device)
        q = torch.cat((leg_joint_pos.reshape(-1, 4, 3), zeros(N, 4, 1)), dim=2)
        q40 = zeros(N, 4)
        l1 = torch.tensor([-1.0, +1.0, -1.0, +1.0], device=self.device) * 0.08
        l2 = torch.tensor([+1.0, +1.0, +1.0, +1.0], device=self.device) * 0.213
        l3 = torch.tensor([+1.0, +1.0, +1.0, +1.0], device=self.device) * 0.213
        d1 = torch.tensor([+1.0, +1.0, +1.0, +1.0], device=self.device) * 0.0
        d2 = torch.tensor([+1.0, +1.0, +1.0, +1.0], device=self.device) * 0.0
        d3 = torch.tensor([-1.0, +1.0, -1.0, +1.0], device=self.device) * 0.0
        r = torch.tensor([+1.0, +1.0, +1.0, +1.0], device=self.device) * 0.0
        p0, p1, p2, p3, p4, p5, R0, R1, R2, R3, R4 = position_level_forward_kinematics(
            q, q40, l1, l2, l3, d1, d2, d3, r)
        q40 = compute_q40(base_rot_mat[:, None, :, :].repeat(1, 4, 1, 1) @ R4)
        p0, p1, p2, p3, p4, p5, R0, R1, R2, R3, R4 = position_level_forward_kinematics(
            q, q40, l1, l2, l3, d1, d2, d3, r)

        contact_pos_cfcb = (
            BMxBV(base_rot_mat_only_yaw.swapaxes(-2, -1)[:, None, :, :].repeat(1, 4, 1, 1),
                  BMxBV(base_rot_mat[:, None, :, :].repeat(1, 4, 1, 1),
                        p5 + self.hip_pos_base_frame)))

        return torch.sum(torch.sum(torch.square(
            contact_pos_cfcb[:, :, 0:2] - self.default_contact_pos_cfcb[:, 0:2]), dim=2), dim=1)

    def _reward_foot_slip(self):
        feet_vel = torch.linalg.norm(self.feet_vel, dim=2)
        contact_state = self.contact_forces[:, self.feet_indices, 2] > 1.
        return torch.sum(feet_vel * contact_state.float(), dim=1)
