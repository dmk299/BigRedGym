from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg


class Go2BoxStepCfg(Go2Cfg):
    class env(Go2Cfg.env):
        # One box = 4 moves x 3 beats. At the slowest sampled tempo (60 bpm)
        # a box takes 12 s, so 25 s always fits two full boxes.
        episode_length_s = 25

    class init_state(Go2Cfg.init_state):
        # Feet directly under the hips at 0.30 m leg height, as in CPG-RL.
        # CPG foot paths are offsets from this stance. Unlike go2trot's
        # stance (feet 9 cm behind the hips), body weight puts almost no
        # moment on the thigh joints, so the soft PD gains below hold it
        # far better. Values are the IK solution for (0, +/-0.0955, -0.30).
        default_joint_angles = {
            "hip_joint": 0.0,
            "thigh_joint": 0.7895,
            "calf_joint": -1.5789,
        }
        reset_mode = "reset_to_basic"
        pos = [0.0, 0.0, 0.36]  # x,y,z [m]; small drop onto the nominal stance

    class control(Go2Cfg.control):
        # Same as go2trot. Higher gains go unstable at a 50 Hz physics rate;
        # raising them needs desired_sim_frequency >= 200.
        stiffness = {"hip": 20.0, "thigh": 20.0, "calf": 20.0}
        damping = {"hip": 0.5, "thigh": 0.5, "calf": 0.5}
        ctrl_frequency = 50
        desired_sim_frequency = 50

    class cpg:
        # Per-leg amplitude-controlled phase oscillators (CPG-RL, Eqs. 3-8),
        # uncoupled. Policy actions in [-1, 1] map linearly onto these ranges.
        amplitude_range = [1.0, 2.0]  # mu; stride = max_step_length*(r - 1)
        frequency_range = [0.0, 1.5]  # omega [Hz]; one step per bar ~0.47 Hz
        direction_rate_range = [-0.75, 0.75]  # psi [Hz]; turns the step direction
        swing_fraction_range = [0.15, 0.5]  # option B: share of cycle in swing
        convergence_gain = 50.0  # a; amplitude settles in roughly 0.1 s
        integration_substeps = 10  # keeps Euler stable at the 50 Hz control rate
        max_step_length = 0.15  # d_step [m]: half of the swing excursion at r=2
        ground_clearance = 0.06  # g_c [m]
        ground_penetration = 0.01  # g_p [m]

    class dance:
        # Sampled per episode. Slower than a slow waltz (84-90 bpm) so each
        # beat (0.86-1.0 s) leaves room for one pair to land before the other
        # lifts. Speed up later by resuming from a checkpoint with a faster
        # range; observation and action sizes do not change.
        tempo_bpm_range = [60.0, 70.0]
        beats_per_move = 3
        step_forward = 0.20  # [m] front-back size of the box
        step_side = 0.15  # [m] left-right size of the box
        randomize_start_move = True

    class commands(Go2Cfg.commands):
        resampling_time = 1000.0  # unused; velocity commands stay zero

    class push_robots(Go2Cfg.push_robots):
        # Random pushes train disturbance rejection, the opposite of the
        # partner-following stretch goal.
        toggle = False

    class domain_randomization(Go2Cfg.domain_randomization):
        class startup(Go2Cfg.domain_randomization.startup):
            link_mass_scale_range = [0.9, 1.2]

        class episode(Go2Cfg.domain_randomization.episode):
            scale_ranges = {
                "p_gains": [0.9, 1.1],
                "d_gains": [0.8, 1.2],
            }

    class reward_settings(Go2Cfg.reward_settings):
        # min_base_height saturates above this; the nominal stance puts the
        # base at ~0.32 m.
        base_height_target = 0.30
        # Width of the min_base_height score in meters. go2trot's version
        # still paid ~75% at 0.22 m, so the robot crouched; at 0.05 m a
        # crouch to 0.22 m pays ~8%.
        base_height_sigma = 0.05
        # Elliptical placement score exp(-(dx/sx)^2 - (dy/sy)^2) in the box
        # frame: looser fore-aft to leave foot spacing flexible. Start wide.
        placement_sigma = [0.15, 0.10]  # [m] fore-aft, lateral
        heading_sigma = 0.25  # [rad]
        # A foot counts as planted above this share of total body weight.
        # One foot in four-foot stance carries roughly 0.25.
        contact_load_threshold = 0.05
        touchdown_min_air_time = 0.1  # [s]; filters chatter and scuffs
        max_touchdowns_per_move = 1  # per leg

    class scaling(Go2Cfg.scaling):
        base_height = 0.3
        dof_pos = 4 * [1.0472, 2.53075, 0.94247]
        dof_pos_obs = dof_pos
        # Option D: residual radians per policy unit, same as go2trot's
        # action scale. With these soft gains the robot sags ~10 cm under
        # its own weight unless the policy adds posture corrections, so the
        # residual needs this much authority (see box-step notes).
        dof_pos_residual = 4 * [0.2, 0.3, 0.3]


class Go2BoxStepRunnerCfg(Go2RunnerCfg):
    class actor(Go2RunnerCfg.actor):
        hidden_dims = [256, 256, 128]
        activation = "elu"
        obs = [
            "base_ang_vel",
            "projected_gravity",
            "heading_obs",
            "dof_pos_obs",
            "dof_vel",
            "cpg_obs",
            "cpg_actions",
            "dof_pos_residual",
            "dance_phase_obs",
            "dance_move_obs",
            "dance_tempo",
        ]
        normalize_obs = False
        smooth_exploration = False
        actions = ["cpg_actions", "dof_pos_residual"]
        add_noise = False
        disable_actions = False

        class noise(Go2RunnerCfg.actor.noise):
            pass

    class critic(Go2RunnerCfg.critic):
        hidden_dims = [128, 64]
        activation = "elu"
        # Privileged: the critic also sees where the robot and feet are
        # relative to the box targets, which the real robot cannot measure.
        obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "heading_obs",
            "dof_pos_obs",
            "dof_vel",
            "cpg_obs",
            "cpg_actions",
            "dof_pos_residual",
            "dance_phase_obs",
            "dance_move_obs",
            "dance_tempo",
            "base_box_obs",
            "foot_target_error",
        ]
        normalize_obs = False

        class reward:
            class weights:
                foot_placement = 4.0
                heading = 1.0
                # Event penalty: rewards are scaled by dt (0.02 s), so this
                # costs 0.2 per extra step. Kept light; raise once balance
                # is reliable.
                extra_steps = 10.0
                foot_slip = 1.0
                # Only the scheduled pair may leave the ground on each beat.
                stance_timing = 2.0
                orientation = 1.0
                min_base_height = 2.0
                torques = 5.0e-6
                action_rate = 0.02
                action_rate2 = 0.01

            class termination_weight:
                termination = 0.01

    class algorithm(Go2RunnerCfg.algorithm):
        gamma = 0.99
        lam = 0.95
        batch_size = 2**15
        max_gradient_steps = 24
        clip_param = 0.2
        learning_rate = 1.0e-3
        max_grad_norm = 1.0
        rollout_size = 2**16
        use_clipped_value_loss = True
        entropy_coef = 0.001
        schedule = "adaptive"
        desired_kl = 0.01
        lr_range = [2e-5, 1e-2]
        lr_ratio = 1.5

    class runner(Go2RunnerCfg.runner):
        experiment_name = "go2boxstep"
        max_iterations = 1500
