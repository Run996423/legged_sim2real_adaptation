from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Go1AdaptationImVaeCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actions = 12
        state_dim = 30
        num_observations = state_dim + num_actions + 3
        history_length = 10
        property_dim = 4
        num_supervised_latent = 3 + 3 + 4  # lin_vel, disturbance, contact_state
        num_privileged_obs = num_observations + 3 + 3 + 4  # lin_vel, disturbance, contact_state

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.5, 1.5]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1.5, 1.5]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.27]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FR_hip_joint": 0.,
            "FL_hip_joint": 0.,
            "RR_hip_joint": 0.,
            "RL_hip_joint": 0.,

            "FR_thigh_joint": 1.,
            "FL_thigh_joint": 1.,
            "RR_thigh_joint": 1.,
            "RL_thigh_joint": 1.,

            "FR_calf_joint": -2.,
            "FL_calf_joint": -2.,
            "RR_calf_joint": -2.,
            "RL_calf_joint": -2.,
        }

    class control:
        # PD Drive parameters:
        stiffness = {'hip': 30., 'thigh': 30., 'calf': 30.}  # [N*m/rad]
        damping = {'hip': 1., 'thigh': 1., 'calf': 1.}  # [N*m*s/rad]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1_with_foot.urdf"
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["base", "hip", "thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        collapse_fixed_joints = True
        replace_cylinder_with_capsule = False

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-0., 0.]
        recompute_inertia = False
        randomize_base_inertia = True
        added_inertia_range = [0., 0.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        disturbance = True
        disturbance_interval_s = 11
        disturbance_force_range = (0., 70.)

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.26
        only_positive_rewards = True

        class scales(LeggedRobotCfg.rewards.scales):
            action_rate = -0.02
            ang_vel_xy = -0.05
            collision = -1.0
            dof_acc = -2.5e-7
            feet_air_time = 0.5
            lin_vel_z = -2.0
            orientation = -5.0
            torques = -2.5e-4
            tracking_ang_vel = 0.5
            tracking_lin_vel = 1.0
            alive = 0.5
            base_height = -50.0
            dof_pos_limits = -10.0
            ugly_contact_pos = -10.0
            foot_slip = -0.2
            stand_still = -1.0

        soft_dof_pos_limit = 0.9

    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.3
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            disturbance = 2.

    class sim(LeggedRobotCfg.sim):
        dt = 0.002


class Go1AdaptationImVaeCfgPPO:
    seed = 1
    runner_class_name = 'OnPolicyRunnerForPPOVAE'

    class policy:
        class_name = 'ActorCriticVAE'
        beta = 1.0
        encoder_hidden_dims = [512, 256, 128]
        num_latent = 20
        init_noise_std = 1.0
        noise_std_type = 'scalar'
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class algorithm:
        class_name = 'PPOVAE'
        # training params
        value_loss_coef = 1.0
        vae_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        normalize_advantage_per_mini_batch = False

    empirical_normalization = True
    num_steps_per_env = 24  # per iteration
    max_iterations = 2000  # number of policy updates

    # logging
    save_interval = 50  # check for potential saves every this many iterations
    experiment_name = 'go1_adaptation_series'
    run_name = 'imvae'

    # load and resume
    resume = False
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = None  # updated from load_run and chkpt
