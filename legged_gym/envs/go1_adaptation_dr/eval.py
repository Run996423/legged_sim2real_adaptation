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
    def __init__(self, obs_normalizer, actor):
        super().__init__()
        self.obs_normalizer = obs_normalizer
        self.actor = actor

    def forward(self, obs):
        obs = self.obs_normalizer(obs)
        mean = self.actor(obs)
        return mean


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
    obs, extra = env.get_observations()

    # load policy
    train_cfg.resume = True
    on_policy_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = on_policy_runner.get_inference_policy(env.device)

    # export policy to jit
    # inference_model = InferenceModel(
    #     on_policy_runner.obs_normalizer,
    #     on_policy_runner.alg.policy.actor
    # )
    # inference_model.eval()
    # example_obs = torch.rand_like(obs[0:1, ...])
    # print(f'input shape: {example_obs.shape}')
    # example_output = inference_model(example_obs)
    # print(f'output shape: {example_output.shape}')
    # jit_model = torch.jit.trace(inference_model, example_inputs=example_obs)
    # jit_model.save('inference_model_jit_new.pt')
    # checkpoint = {
    #     'state_dict': inference_model.state_dict()
    # }
    # torch.save(checkpoint, 'inference_model_state_dict.pth', _use_new_zipfile_serialization=False)

    # for log
    record_duration = 20
    command_history = []
    actual_history = []
    time_history = []
    is_recording = True

    gamepad = Gamepad()
    use_sync = True

    with tqdm(desc=f'simulating', total=None, unit='s@sim') as pbar:
        for i in range(10000 * int(env.max_episode_length)):
            step_start_time = time.time()

            with torch.no_grad():
                actions = policy(obs)
                # actions = inference_model(obs)
                obs, rews, dones, infos = env.step(actions.detach())

            elapsed = env.episode_length_buf[env.tracked_env_id].item() * env.dt

            # 使用手柄获取command
            # left_stick_x, left_stick_y, right_stick_x, right_stick_y = gamepad.read()
            # cmds = - 1.0 * left_stick_y, - 1.0 * left_stick_x, - 1.5 * right_stick_y
            # 使用预定义的command
            cmds = get_command_at_time(elapsed)

            cmd1, cmd2, cmd3 = cmds
            env.commands[:, 0] = cmd1
            env.commands[:, 1] = cmd2
            env.commands[:, 2] = cmd3
            env.commands[:, 0] *= (torch.norm(env.commands[:, 0:1], dim=1) > 0.2)
            env.commands[:, 1] *= (torch.norm(env.commands[:, 1:2], dim=1) > 0.2)
            env.commands[:, 2] *= (torch.norm(env.commands[:, 2:3], dim=1) > 0.2)

            # env.disturbance_force[:, 0, 0] = cmd1 * 50
            # env.disturbance_force[:, 0, 1] = cmd2 * 50
            # env.disturbance_force[:, 0, 2] = cmd3 * 50

            # pred_disturbance_smooth = smooth_factor * pred_disturbance + (1 - smooth_factor) * pred_disturbance_smooth
            # env.commands[:, 0] = pred_disturbance_smooth[:, 0] / 40
            # env.commands[:, 1] = pred_disturbance_smooth[:, 1] / 40
            # env.commands[:, 2] = 0

            # 画图
            command = env.commands[env.tracked_env_id, :].detach().cpu().numpy()
            actual = torch.cat((env.base_lin_vel[env.tracked_env_id, 0:2],
                                env.base_ang_vel[env.tracked_env_id, 2:3],
                                env.base_lin_vel[env.tracked_env_id, 2:3],
                                env.disturbance_force[env.tracked_env_id, 0, :])).detach().cpu().numpy()
            mjc_viewer.render()
            # mjc_viewer.add_data_to_line(line_name="Fx", line_data=pred_disturbance_smooth[env.tracked_env_id, 0].item(), fig_idx=2)
            mjc_viewer.add_data_to_line(line_name="gt", line_data=env.disturbance_force[env.tracked_env_id, 0, :][0].item(), fig_idx=2)
            # mjc_viewer.add_data_to_line(line_name="Fy", line_data=pred_disturbance_smooth[env.tracked_env_id, 1].item(), fig_idx=1)
            mjc_viewer.add_data_to_line(line_name="gt", line_data=env.disturbance_force[env.tracked_env_id, 0, :][1].item(), fig_idx=1)
            # mjc_viewer.add_data_to_line(line_name="Fz", line_data=pred_disturbance_smooth[env.tracked_env_id, 2].item(), fig_idx=0)
            mjc_viewer.add_data_to_line(line_name="gt", line_data=env.disturbance_force[env.tracked_env_id, 0, :][2].item(), fig_idx=0)

            if is_recording and elapsed <= record_duration:
                time_history.append(elapsed)
                command_history.append(command.copy())
                actual_history.append(actual.copy())
            if is_recording and elapsed > record_duration:
                is_recording = False
                time_array = np.array(time_history)
                command_array = np.array(command_history)
                actual_array = np.array(actual_history)
                fig, axes = plt.subplots(7, 1, figsize=(10, 19))
                labels = ['vx', 'vy', 'wz', 'vz', 'fx', 'fy', 'fz']
                for j in range(7):
                    if j in (0, 1, 2):
                        axes[j].plot(time_array, command_array[:, j], 'b-', label='Command', linewidth=1)
                    axes[j].plot(time_array, actual_array[:, j], 'r-', label=f'Actual', linewidth=1)
                    axes[j].set_ylabel(f'{labels[j]}')
                    axes[j].legend()
                    axes[j].grid(True, alpha=0.3)
                axes[0].set_ylim(-1.7, 1.7)
                axes[1].set_ylim(-1.2, 1.2)
                axes[2].set_ylim(-1.7, 1.7)
                axes[3].set_ylim(-1.0, 1.0)
                axes[4].set_ylim(-50, 50)
                axes[5].set_ylim(-50, 50)
                axes[6].set_ylim(-50, 50)
                axes[0].set_title('Command vs Actual Values')
                axes[-1].set_xlabel('Time (s)')
                plt.tight_layout()
                fig_name = f'command_vs_actual.png'
                plt.savefig(fig_name, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"图像已保存为 {fig_name}")

            # 超速时间补偿
            pbar.update(env_cfg.sim.dt * env_cfg.control.decimation)
            step_end_time = time.time()
            surplus_time = env_cfg.sim.dt * env_cfg.control.decimation - (step_end_time - step_start_time)
            if surplus_time > 0 and use_sync:
                time.sleep(surplus_time)


if __name__ == '__main__':
    play(get_args())
