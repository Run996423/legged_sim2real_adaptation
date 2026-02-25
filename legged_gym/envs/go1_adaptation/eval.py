from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.gamepad import Gamepad

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import time
import mujoco
import mujoco_viewer

command_time_list = [
    (0.0, 0.0, 0.0, 4.0),
    (1.0, 0.0, 0.0, 4.0),
    (0.0, 1.0, 0.0, 4.0),
    (0.0, 0.0, 1.5, 4.0),
    (1.0, 0.0, 1.5, 4.0),
]


def get_command_at_time(time_point):
    accumulated_time = 0
    for elem in command_time_list:
        x, y, z, duration = elem
        next_time = accumulated_time + duration
        if time_point <= next_time:
            return x, y, z
        accumulated_time = next_time
    last_command = command_time_list[-1]
    return last_command[0], last_command[1], last_command[2]


class InferenceModel(torch.nn.Module):
    def __init__(self, obs_history_normalizer, current_obs_normalizer, encoder, actor):
        super().__init__()
        self.obs_history_normalizer = obs_history_normalizer
        self.current_obs_normalizer = current_obs_normalizer
        self.encoder = encoder
        self.actor = actor

    def forward(self, obs_history, current_obs):
        obs_history = self.obs_history_normalizer(obs_history)
        current_obs = self.current_obs_normalizer(current_obs)
        latent = self.encoder(obs_history.flatten(1, 2))
        mean = self.actor(torch.cat((current_obs, latent), dim=-1))
        return mean


class InferenceModelWithOverrideForce(torch.nn.Module):
    def __init__(self, obs_history_normalizer, current_obs_normalizer, encoder, actor, supervised_latent_ground_truth_normalizer):
        super().__init__()
        self.obs_history_normalizer = obs_history_normalizer
        self.current_obs_normalizer = current_obs_normalizer
        self.supervised_latent_ground_truth_normalizer = supervised_latent_ground_truth_normalizer
        self.encoder = encoder
        self.actor = actor

    def forward(self, obs_history, current_obs, override_force):
        obs_history = self.obs_history_normalizer(obs_history)
        current_obs = self.current_obs_normalizer(current_obs)
        latent = self.encoder(obs_history.flatten(1, 2))
        pred_quantity = self.supervised_latent_ground_truth_normalizer.inverse(latent[:, 0:6])
        override_quantity = pred_quantity.clone()
        override_quantity[:, 3:6] = override_force
        override_quantity = self.supervised_latent_ground_truth_normalizer(override_quantity)
        latent[:, 0:6] = override_quantity
        mean = self.actor(torch.cat((current_obs, latent), dim=-1))
        return mean


class GetVelForce(torch.nn.Module):
    def __init__(self, obs_history_normalizer, supervised_latent_ground_truth_normalizer, encoder):
        super().__init__()
        self.obs_history_normalizer = obs_history_normalizer
        self.supervised_latent_ground_truth_normalizer = supervised_latent_ground_truth_normalizer
        self.encoder = encoder

    def forward(self, obs_history):
        obs_history = self.obs_history_normalizer(obs_history)
        latent = self.encoder(obs_history.flatten(1, 2))
        pred_quantity = self.supervised_latent_ground_truth_normalizer.inverse(latent[:, 0:6])
        pred_vel_force = pred_quantity[:, 0:6]
        return pred_vel_force


def play(args):
    # 实时曲线画图
    mjcf_path = 'resources/robots/go1w/xml/go1w_stair.xml'
    mjc_model = mujoco.MjModel.from_xml_path(filename=mjcf_path, assets=None)
    mjc_data = mujoco.MjData(mjc_model)
    mjc_viewer = mujoco_viewer.MujocoViewer(mjc_model, mjc_data)
    mjc_viewer.add_line_to_fig(line_name="Fx", fig_idx=2)
    mjc_viewer.add_line_to_fig(line_name="gt", fig_idx=2)
    mjc_viewer.add_line_to_fig(line_name="Fy", fig_idx=1)
    mjc_viewer.add_line_to_fig(line_name="gt", fig_idx=1)
    mjc_viewer.add_line_to_fig(line_name="Fz", fig_idx=0)
    mjc_viewer.add_line_to_fig(line_name="gt", fig_idx=0)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # override env_cfg for testing
    env_cfg.env.num_envs = 10
    env_cfg.env.episode_length_s = 10000

    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False

    env_cfg.commands.resampling_time = 10000
    env_cfg.commands.heading_command = False

    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.friction_range = [0.5, 1.25]
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.added_mass_range = [-0., 0.]
    env_cfg.domain_rand.randomize_base_inertia = True
    env_cfg.domain_rand.added_inertia_range = [0., 0.]
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = True
    env_cfg.domain_rand.disturbance_interval_s = 3
    env_cfg.domain_rand.disturbance_force_range = (0., 50.)

    env_cfg.noise.add_noise = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()
    obs_dict, extra = env.get_observations()

    # load policy
    train_cfg.resume = True
    on_policy_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = on_policy_runner.get_inference_policy(env.device)

    # export policy to jit
    # inference_model = InferenceModel(
    #     on_policy_runner.obs_normalizer_dict['obs_history'],
    #     on_policy_runner.obs_normalizer_dict['current_obs'],
    #     on_policy_runner.alg.policy.encoder,
    #     on_policy_runner.alg.policy.actor
    # )
    # inference_model.eval()
    # example_obs_history = torch.rand_like(obs_dict['obs_history'][0:1, ...])
    # example_current_obs = torch.rand_like(obs_dict['current_obs'][0:1, ...])
    # print(f'input shape: {example_obs_history.shape}, {example_current_obs.shape}')
    # example_output = inference_model(example_obs_history, example_current_obs)
    # print(f'output shape: {example_output.shape}')
    # jit_model = torch.jit.trace(inference_model, example_inputs=(example_obs_history, example_current_obs))
    # jit_model.save('inference_model_jit_new.pt')
    # checkpoint = {
    #     'state_dict': inference_model.state_dict()
    # }
    # torch.save(checkpoint, 'inference_model_state_dict.pth', _use_new_zipfile_serialization=False)
    #
    # inference_model_with_override = InferenceModelWithOverrideForce(
    #     on_policy_runner.obs_normalizer_dict['obs_history'],
    #     on_policy_runner.obs_normalizer_dict['current_obs'],
    #     on_policy_runner.alg.policy.encoder,
    #     on_policy_runner.alg.policy.actor,
    #     on_policy_runner.obs_normalizer_dict['supervised_latent_ground_truth']
    # )
    # inference_model_with_override.eval()
    # example_obs_history = torch.rand_like(obs_dict['obs_history'][0:1, ...])
    # example_current_obs = torch.rand_like(obs_dict['current_obs'][0:1, ...])
    # example_override_force = torch.rand_like(obs_dict['current_obs'][0:1, 0:3])
    # print(f'input shape: {example_obs_history.shape}, {example_current_obs.shape}, {example_override_force.shape}')
    # example_output = inference_model_with_override(example_obs_history, example_current_obs, example_override_force)
    # print(f'output shape: {example_output.shape}')
    # jit_model = torch.jit.trace(inference_model_with_override, example_inputs=(example_obs_history, example_current_obs, example_override_force))
    # jit_model.save('inference_model_with_override_force_jit_new.pt')
    # checkpoint = {
    #     'state_dict': inference_model_with_override.state_dict()
    # }
    # torch.save(checkpoint, 'inference_model_with_override_force_state_dict.pth', _use_new_zipfile_serialization=False)
    #
    # get_vel_force_model = GetVelForce(
    #     obs_history_normalizer=on_policy_runner.obs_normalizer_dict['obs_history'],
    #     supervised_latent_ground_truth_normalizer=on_policy_runner.obs_normalizer_dict['supervised_latent_ground_truth'],
    #     encoder=on_policy_runner.alg.policy.encoder
    # )
    # get_vel_force_model.eval()
    # example_obs_history = torch.rand_like(obs_dict['obs_history'][0:1, ...])
    # print(f'input shape: {example_obs_history.shape}')
    # example_output = get_vel_force_model(example_obs_history)
    # print(f'output shape: {example_output.shape}')
    # jit_model = torch.jit.trace(get_vel_force_model, example_inputs=example_obs_history)
    # jit_model.save('get_vel_force_model_jit_new.pt')
    # checkpoint = {
    #     'state_dict': get_vel_force_model.state_dict()
    # }
    # torch.save(checkpoint, 'get_vel_force_model_state_dict.pth', _use_new_zipfile_serialization=False)

    # for log
    record_duration = 20
    command_history = []
    actual_history = []
    pred_history = []
    time_history = []
    is_recording = True

    gamepad = Gamepad()
    use_sync = True

    pred_disturbance_smooth = 0
    smooth_factor = 1.0

    with tqdm(desc=f'simulating', total=None, unit='s@sim') as pbar:
        for i in range(10000 * int(env.max_episode_length)):
            step_start_time = time.time()

            with torch.no_grad():
                # 监视预测精度
                obs_history = obs_dict["obs_history"]
                supervised_latent_ground_truth = obs_dict["supervised_latent_ground_truth"]
                latent = on_policy_runner.alg.policy.encoder(
                    on_policy_runner.obs_normalizer_dict['obs_history'](obs_history).flatten(1, 2))
                pred_quantity = on_policy_runner.obs_normalizer_dict['supervised_latent_ground_truth'].inverse(latent[:, 0:6])
                pred_lin_vel = pred_quantity[:, 0:3]
                pred_disturbance = pred_quantity[:, 3:6]

                # 仿真step
                actions = policy(obs_dict)
                obs_dict, rews, dones, infos = env.step(actions.detach())

            elapsed = env.episode_length_buf[env.tracked_env_id].item() * env.dt

            # 使用手柄获取command
            left_stick_x, left_stick_y, right_stick_x, right_stick_y = gamepad.read()
            cmds = - 1.0 * left_stick_y, - 1.0 * left_stick_x, - 1.5 * right_stick_y
            # 使用预定义的command
            # cmds = get_command_at_time(elapsed)

            cmd1, cmd2, cmd3 = cmds
            env.commands[:, 0] = cmd1
            env.commands[:, 1] = cmd2
            env.commands[:, 2] = cmd3
            env.commands[:, 0] *= (torch.norm(env.commands[:, 0:1], dim=1) > 0.2)
            env.commands[:, 1] *= (torch.norm(env.commands[:, 1:2], dim=1) > 0.2)
            env.commands[:, 2] *= (torch.norm(env.commands[:, 2:3], dim=1) > 0.2)

            # env.disturbance_force[:, 0, 0] = cmd1 * 30
            # env.disturbance_force[:, 0, 1] = cmd2 * 30
            # env.disturbance_force[:, 0, 2] = cmd3 * 30

            pred_disturbance_smooth = smooth_factor * pred_disturbance + (1 - smooth_factor) * pred_disturbance_smooth
            # env.commands[:, 0] = pred_disturbance_smooth[:, 0] / 40
            # env.commands[:, 1] = pred_disturbance_smooth[:, 1] / 40
            # env.commands[:, 2] = 0

            # 画图
            # command = env.commands[env.tracked_env_id, :].detach().cpu().numpy()
            # actual = torch.cat((env.base_lin_vel[env.tracked_env_id, 0:2],
            #                     env.base_ang_vel[env.tracked_env_id, 2:3],
            #                     env.base_lin_vel[env.tracked_env_id, 2:3],
            #                     env.disturbance_force[env.tracked_env_id, 0, :],
            #                     torch.greater(env.contact_forces[env.tracked_env_id, env.feet_indices, 2], 1.).float()[[1, 0, 3, 2]])).detach().cpu().numpy()
            # pred = torch.cat((pred_lin_vel[env.tracked_env_id, 0:2],
            #                   pred_lin_vel[env.tracked_env_id, 0:1] * 0,
            #                   pred_lin_vel[env.tracked_env_id, 2:3],
            #                   pred_disturbance_smooth[env.tracked_env_id, 0:3],
            #                   pred_contact_state[env.tracked_env_id])).detach().cpu().numpy()
            mjc_viewer.render()
            mjc_viewer.add_data_to_line(line_name="Fx", line_data=pred_disturbance_smooth[env.tracked_env_id, 0].item(), fig_idx=2)
            mjc_viewer.add_data_to_line(line_name="gt", line_data=supervised_latent_ground_truth[env.tracked_env_id, 3].item(), fig_idx=2)
            mjc_viewer.add_data_to_line(line_name="Fy", line_data=pred_disturbance_smooth[env.tracked_env_id, 1].item(), fig_idx=1)
            mjc_viewer.add_data_to_line(line_name="gt", line_data=supervised_latent_ground_truth[env.tracked_env_id, 4].item(), fig_idx=1)
            mjc_viewer.add_data_to_line(line_name="Fz", line_data=pred_disturbance_smooth[env.tracked_env_id, 2].item(), fig_idx=0)
            mjc_viewer.add_data_to_line(line_name="gt", line_data=supervised_latent_ground_truth[env.tracked_env_id, 5].item(), fig_idx=0)

            # if is_recording and elapsed <= record_duration:
            #     time_history.append(elapsed)
            #     command_history.append(command.copy())
            #     actual_history.append(actual.copy())
            #     pred_history.append(pred.copy())
            # if is_recording and elapsed > record_duration:
            #     is_recording = False
            #     time_array = np.array(time_history)
            #     command_array = np.array(command_history)
            #     actual_array = np.array(actual_history)
            #     pred_array = np.array(pred_history)
            #     fig, axes = plt.subplots(11, 1, figsize=(10, 30))
            #     labels = ['vx', 'vy', 'wz', 'vz', 'fx', 'fy', 'fz', 'FR', 'FL', 'RR', 'RL']
            #     for j in range(11):
            #         if j in (0, 1, 2):
            #             axes[j].plot(time_array, command_array[:, j], 'b-', label='Command', linewidth=1)
            #         axes[j].plot(time_array, actual_array[:, j], 'r-', label=f'Actual', linewidth=1)
            #         if j != 2:
            #             axes[j].plot(time_array, pred_array[:, j], 'g--', label=f'Pred', linewidth=1)
            #         axes[j].set_ylabel(f'{labels[j]}')
            #         axes[j].legend()
            #         axes[j].grid(True, alpha=0.3)
            #     axes[0].set_ylim(-1.7, 1.7)
            #     axes[1].set_ylim(-1.2, 1.2)
            #     axes[2].set_ylim(-1.7, 1.7)
            #     axes[3].set_ylim(-1.0, 1.0)
            #     axes[4].set_ylim(-50, 50)
            #     axes[5].set_ylim(-50, 50)
            #     axes[6].set_ylim(-50, 50)
            #     axes[7].set_ylim(-0.2, 1.2)
            #     axes[8].set_ylim(-0.2, 1.2)
            #     axes[9].set_ylim(-0.2, 1.2)
            #     axes[10].set_ylim(-0.2, 1.2)
            #     axes[0].set_title('Command vs Actual vs Pred Values')
            #     axes[-1].set_xlabel('Time (s)')
            #     plt.tight_layout()
            #     fig_name = f'command_vs_actual_vs_pred.png'
            #     plt.savefig(fig_name, dpi=150, bbox_inches='tight')
            #     plt.close()
            #     print(f"图像已保存为 {fig_name}")

            # 超速时间补偿
            pbar.update(env_cfg.sim.dt * env_cfg.control.decimation)
            step_end_time = time.time()
            surplus_time = env_cfg.sim.dt * env_cfg.control.decimation - (step_end_time - step_start_time)
            if surplus_time > 0 and use_sync:
                time.sleep(surplus_time)


if __name__ == '__main__':
    play(get_args())
