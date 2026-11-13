"""Go2 waltz box step with a learned central pattern generator (CPG).

Step pattern (robot always faces forward, pace pairing):
    left pair = FL + RL, right pair = FR + RR
    corners, in order: back-left (start), front-left, front-right, back-right
    each move is one bar of 3/4: beat 1 lead pair steps, beat 2 trail pair
    steps, beat 3 hold. The pair on the outside of the box leads.

Controller (adapted from Bellegarda & Ijspeert, CPG-RL, RA-L 2022):
    A. A dance clock (bar phase, move index, tempo) is observed but does not
       drive the legs; timing comes from the rewards.
    B. Each leg's oscillator spends a policy-chosen fraction of its cycle in
       swing instead of the paper's fixed half cycle.
    D. Small residual joint offsets are added on top of the CPG/IK targets.
    The oscillators are uncoupled, as in the paper.
"""

import torch

from gym.envs.go2.go2 import Go2
from gym.envs.go2.go2_kinematics import Go2LegKinematics
from gym.utils.sampling import masked_update, torch_rand_float

# Leg indices in canonical order FL, FR, RL, RR.
LEFT_LEGS = (0, 2)
RIGHT_LEGS = (1, 3)
# Lead pair for each move: 0 = left pair, 1 = right pair.
MOVE_LEAD_PAIR = (0, 1, 1, 0)
TWO_PI = 2.0 * torch.pi


class Go2BoxStep(Go2):
    def __init__(self, cfg, device, headless, backend):
        super().__init__(cfg, device, headless, backend)

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------
    def _init_buffers(self):
        super()._init_buffers()
        n, dev = self.num_envs, self.device
        cpg = self.cfg.cpg
        self.kinematics = Go2LegKinematics(device=dev)

        self._default_actuated = self.default_dof_pos.index_select(
            1, self.actuated_dof_indices
        )
        self._actuated_limits = self.dof_pos_limits.index_select(
            0, self.actuated_dof_indices
        )
        # Nominal foot positions (hip frames) at the default joint angles.
        self._nominal_foot_hip = self.kinematics.forward(
            self._default_actuated.view(1, 4, 3)
        )[0]
        # Nominal foot positions relative to the base origin, xy only.
        self._nominal_foot_body_xy = (
            self._nominal_foot_hip + self.kinematics.hip_positions
        )[:, :2]

        # ---- policy actions (set by the runner through set_states) ----
        # cpg_actions per leg: amplitude, frequency, direction rate, swing
        # fraction; each in [-1, 1] after clipping, laid out as 4 blocks of 4.
        self.cpg_actions = torch.zeros(n, 16, device=dev)
        self.dof_pos_residual = torch.zeros(n, self.num_actuators, device=dev)

        # ---- oscillator states ----
        self.cpg_r = torch.full((n, 4), cpg.amplitude_range[0], device=dev)
        self.cpg_r_dot = torch.zeros(n, 4, device=dev)
        self.cpg_theta = torch.zeros(n, 4, device=dev)
        self.cpg_phi = torch.zeros(n, 4, device=dev)
        self.cpg_swing_fraction = torch.full(
            (n, 4), sum(cpg.swing_fraction_range) / 2, device=dev
        )
        self.cpg_obs = torch.zeros(n, 4 * 7, device=dev)
        self.foot_target_body = torch.zeros(n, 4, 3, device=dev)

        # ---- dance clock (a command; a partner signal can replace it) ----
        self.bar_phase = torch.zeros(n, 1, device=dev)
        self.move_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self.bars_per_second = torch.ones(n, 1, device=dev)
        self.dance_phase_obs = torch.zeros(n, 2, device=dev)
        self.dance_move_obs = torch.zeros(n, 4, device=dev)
        self.dance_tempo = torch.zeros(n, 1, device=dev)

        # ---- box frame (movable origin for the partner-lead stretch goal) ----
        self.box_origin = torch.zeros(n, 2, device=dev)
        self.box_yaw = torch.zeros(n, device=dev)
        sx, sy = self.cfg.dance.step_forward, self.cfg.dance.step_side
        # Corner positions of the base in the box frame: BL, FL, FR, BR.
        self._corners = torch.tensor(
            [[0.0, 0.0], [sx, 0.0], [sx, -sy], [0.0, -sy]], device=dev
        )
        self.foot_target_error = torch.zeros(n, 8, device=dev)
        self.base_box_obs = torch.zeros(n, 2, device=dev)
        self.heading_obs = torch.zeros(n, 2, device=dev)

        # ---- contact and touchdown bookkeeping ----
        self.foot_load = torch.zeros(n, 4, device=dev)
        self.foot_in_contact = torch.zeros(n, 4, dtype=torch.bool, device=dev)
        self.foot_air_time = torch.zeros(n, 4, device=dev)
        self.touchdowns_this_move = torch.zeros(n, 4, device=dev)
        self.extra_touchdowns = torch.zeros(n, 4, device=dev)

        self._update_dance_observations()

    # ------------------------------------------------------------------
    # Commands: velocity commands are unused; the dance clock replaces them.
    # ------------------------------------------------------------------
    def _resample_commands(self, command_mask):
        self.commands.masked_fill_(command_mask.unsqueeze(1), 0.0)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        if not hasattr(self, "cpg_r"):
            return
        n, dev = self.num_envs, self.device
        cpg, dance = self.cfg.cpg, self.cfg.dance

        # Oscillators start at rest: unit amplitude (no step), facing forward,
        # every leg somewhere in stance. Phases are spread so legs do not
        # start in lockstep.
        mask2 = reset_mask.unsqueeze(1)
        self.cpg_r.masked_fill_(mask2, cpg.amplitude_range[0])
        self.cpg_r_dot.masked_fill_(mask2, 0.0)
        swing0 = sum(cpg.swing_fraction_range) / 2
        self.cpg_swing_fraction.masked_fill_(mask2, swing0)
        theta0 = torch_rand_float(TWO_PI * swing0, TWO_PI, (n, 4), device=dev)
        masked_update(self.cpg_theta, theta0, reset_mask)
        self.cpg_phi.masked_fill_(mask2, 0.0)
        self.cpg_actions.masked_fill_(mask2, 0.0)
        self.dof_pos_residual.masked_fill_(mask2, 0.0)

        # Dance clock: random tempo and (optionally) random starting move.
        bpm = torch_rand_float(
            dance.tempo_bpm_range[0], dance.tempo_bpm_range[1], (n, 1), device=dev
        )
        masked_update(
            self.bars_per_second, bpm / 60.0 / dance.beats_per_move, reset_mask
        )
        self.bar_phase.masked_fill_(mask2, 0.0)
        if dance.randomize_start_move:
            start = torch.randint(0, 4, (n,), device=dev)
        else:
            start = torch.zeros(n, dtype=torch.long, device=dev)
        masked_update(self.move_idx, start, reset_mask)

        # Box frame: place the origin so the robot stands on the starting
        # corner of its first move, facing the box's forward direction.
        yaw = self._base_yaw()
        corner = self._corners[self.move_idx]
        origin = self.root_states[:, :2] - self._rotate(corner, yaw)
        masked_update(self.box_yaw, yaw, reset_mask)
        masked_update(self.box_origin, origin, reset_mask)

        self.foot_in_contact.masked_fill_(mask2, True)
        self.foot_air_time.masked_fill_(mask2, 0.0)
        self.touchdowns_this_move.masked_fill_(mask2, 0.0)
        self.extra_touchdowns.masked_fill_(mask2, 0.0)

        self._compute_cpg_targets()
        masked_update(self.dof_pos_target, self._cpg_dof_pos_target(), reset_mask)
        masked_update(self.dof_pos_history, self.dof_pos_target.tile(1, 3), reset_mask)
        self._update_dance_observations()
        self._update_cpg_observations()
        self._update_box_observations()

    # ------------------------------------------------------------------
    # Control step
    # ------------------------------------------------------------------
    def _pre_decimation_step(self):
        super()._pre_decimation_step()
        self._integrate_cpg(self.dt)
        self._compute_cpg_targets()
        self.dof_pos_target = self._cpg_dof_pos_target()

    def _post_decimation_step(self):
        super()._post_decimation_step()
        self._update_contacts()
        self._advance_dance_clock()
        self._update_dance_observations()
        self._update_cpg_observations()
        self._update_box_observations()

    # ------------------------------------------------------------------
    # CPG
    # ------------------------------------------------------------------
    def _cpg_setpoints(self):
        """Map clipped policy outputs in [-1, 1] to oscillator setpoints."""
        cpg = self.cfg.cpg
        a = torch.clamp(self.cpg_actions, -1.0, 1.0).view(self.num_envs, 4, 4)

        def scale(x, bounds):
            lo, hi = bounds
            return lo + 0.5 * (x + 1.0) * (hi - lo)

        mu = scale(a[..., 0], cpg.amplitude_range)
        omega = TWO_PI * scale(a[..., 1], cpg.frequency_range)
        psi = TWO_PI * scale(a[..., 2], cpg.direction_rate_range)
        swing = scale(a[..., 3], cpg.swing_fraction_range)
        return mu, omega, psi, swing

    def _integrate_cpg(self, dt):
        """Euler-integrate the amplitude-controlled phase oscillators."""
        cpg = self.cfg.cpg
        mu, omega, psi, swing = self._cpg_setpoints()
        a = cpg.convergence_gain
        h = dt / cpg.integration_substeps
        for _ in range(cpg.integration_substeps):
            r_ddot = a * (0.25 * a * (mu - self.cpg_r) - self.cpg_r_dot)
            self.cpg_r = self.cpg_r + h * self.cpg_r_dot
            self.cpg_r_dot = self.cpg_r_dot + h * r_ddot
        theta = self.cpg_theta + dt * omega
        # Latch a new swing fraction only when a cycle wraps (start of swing),
        # so changing it never makes the foot target jump mid-cycle.
        wrapped = theta >= TWO_PI
        self.cpg_swing_fraction = torch.where(wrapped, swing, self.cpg_swing_fraction)
        self.cpg_theta = torch.remainder(theta, TWO_PI)
        phi = self.cpg_phi + dt * psi
        self.cpg_phi = torch.atan2(torch.sin(phi), torch.cos(phi))

    def _cpg_warped_phase(self):
        """Phase in [0, 2pi) with swing occupying [0, pi) (option B)."""
        swing_end = TWO_PI * self.cpg_swing_fraction
        in_swing = self.cpg_theta < swing_end
        swing_beta = torch.pi * self.cpg_theta / swing_end
        stance_beta = torch.pi + torch.pi * (self.cpg_theta - swing_end) / (
            TWO_PI - swing_end
        )
        return torch.where(in_swing, swing_beta, stance_beta), in_swing

    def _compute_cpg_targets(self):
        """Oscillator states -> foot positions in the hip frames (Eqs. 6-8)."""
        cpg = self.cfg.cpg
        beta, in_swing = self._cpg_warped_phase()
        stride = -cpg.max_step_length * (self.cpg_r - 1.0) * torch.cos(beta)
        dx = stride * torch.cos(self.cpg_phi)
        dy = stride * torch.sin(self.cpg_phi)
        lift = torch.where(
            in_swing, cpg.ground_clearance, cpg.ground_penetration
        ) * torch.sin(beta)
        offset = torch.stack((dx, dy, lift), dim=-1)
        self.foot_target_body = self._nominal_foot_hip + offset

    def _cpg_dof_pos_target(self):
        q = self.kinematics.inverse(self.foot_target_body).view(self.num_envs, -1)
        q = q + self.dof_pos_residual
        q = torch.clamp(q, self._actuated_limits[:, 0], self._actuated_limits[:, 1])
        return q - self._default_actuated

    # ------------------------------------------------------------------
    # Dance clock and box frame
    # ------------------------------------------------------------------
    def _advance_dance_clock(self):
        self.bar_phase += self.dt * self.bars_per_second
        new_bar = (self.bar_phase >= 1.0).squeeze(1)
        self.bar_phase = torch.remainder(self.bar_phase, 1.0)
        self.move_idx = torch.where(
            new_bar, torch.remainder(self.move_idx + 1, 4), self.move_idx
        )
        self.touchdowns_this_move.masked_fill_(new_bar.unsqueeze(1), 0.0)

    def _beat(self):
        beats = self.cfg.dance.beats_per_move
        return torch.clamp((self.bar_phase.squeeze(1) * beats).long(), max=beats - 1)

    def _update_dance_observations(self):
        angle = TWO_PI * self.bar_phase
        self.dance_phase_obs = torch.cat((torch.sin(angle), torch.cos(angle)), dim=1)
        self.dance_move_obs = torch.nn.functional.one_hot(self.move_idx, 4).float()
        self.dance_tempo = self.bars_per_second.clone()

    def _update_cpg_observations(self):
        self.cpg_obs = torch.cat(
            (
                self.cpg_r - 1.0,
                self.cpg_r_dot / self.cfg.cpg.convergence_gain,
                torch.cos(self.cpg_theta),
                torch.sin(self.cpg_theta),
                torch.cos(self.cpg_phi),
                torch.sin(self.cpg_phi),
                self.cpg_swing_fraction,
            ),
            dim=1,
        )

    def _base_yaw(self):
        x, y, z, w = self.root_states[:, 3:7].unbind(dim=1)
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _rotate(xy, yaw):
        c, s = torch.cos(yaw), torch.sin(yaw)
        if xy.dim() == 3:
            c, s = c.unsqueeze(1), s.unsqueeze(1)
        x, y = xy[..., 0], xy[..., 1]
        return torch.stack((c * x - s * y, s * x + c * y), dim=-1)

    def _heading_error(self):
        err = self._base_yaw() - self.box_yaw
        return torch.atan2(torch.sin(err), torch.cos(err))

    def _feet_in_box_frame(self):
        feet_world = self._rigid_body_pos[:, self.feet_indices, :2]
        rel = feet_world - self.box_origin.unsqueeze(1)
        return self._rotate(rel, -self.box_yaw)

    def _foot_targets_box(self):
        """Per-foot placement targets in the box frame and their active masks.

        Beat gating (beat index 0, 1, 2 within the current move):
            lead pair:  swing (inactive), new corner, new corner
            trail pair: old corner, swing (inactive), new corner
        """
        beat = self._beat()
        lead_is_right = torch.tensor(MOVE_LEAD_PAIR, device=self.device)[
            self.move_idx
        ].bool()
        is_right_leg = torch.tensor([False, True, False, True], device=self.device)
        is_lead = is_right_leg.unsqueeze(0) == lead_is_right.unsqueeze(1)

        old_corner = self._corners[self.move_idx]
        new_corner = self._corners[torch.remainder(self.move_idx + 1, 4)]
        beat2 = beat.unsqueeze(1)
        use_old = (~is_lead) & (beat2 == 0)
        corner = torch.where(
            use_old.unsqueeze(-1), old_corner.unsqueeze(1), new_corner.unsqueeze(1)
        )
        active = torch.where(is_lead, beat2 >= 1, beat2 != 1)
        targets = corner + self._nominal_foot_body_xy.unsqueeze(0)
        return targets, active

    def _update_box_observations(self):
        targets, _ = self._foot_targets_box()
        self.foot_target_error = (self._feet_in_box_frame() - targets).reshape(
            self.num_envs, 8
        )
        base_rel = self.root_states[:, :2] - self.box_origin
        base_box = self._rotate(base_rel, -self.box_yaw)
        new_corner = self._corners[torch.remainder(self.move_idx + 1, 4)]
        self.base_box_obs = base_box - new_corner
        err = self._heading_error().unsqueeze(1)
        self.heading_obs = torch.cat((torch.sin(err), torch.cos(err)), dim=1)

    # ------------------------------------------------------------------
    # Contacts
    # ------------------------------------------------------------------
    def _foot_load_fraction(self):
        """Upward foot force as a fraction of the robot's total weight."""
        weight = 9.81 * self._backend.link_mass.sum(dim=1, keepdim=True)
        return torch.clamp(
            self.contact_forces[:, self.feet_indices, 2] / weight, 0.0, 1.0
        )

    def _update_contacts(self):
        rs = self.cfg.reward_settings
        self.foot_load = self._foot_load_fraction()
        on = self.foot_load > rs.contact_load_threshold
        off = self.foot_load < 0.5 * rs.contact_load_threshold
        was_in_contact = self.foot_in_contact
        in_contact = torch.where(on, True, torch.where(off, False, was_in_contact))

        touchdown = in_contact & ~was_in_contact
        real_step = touchdown & (self.foot_air_time >= rs.touchdown_min_air_time)
        self.touchdowns_this_move += real_step.float()
        allowed = rs.max_touchdowns_per_move
        self.extra_touchdowns = (
            real_step & (self.touchdowns_this_move > allowed)
        ).float()

        self.foot_air_time = torch.where(
            in_contact,
            torch.zeros_like(self.foot_air_time),
            self.foot_air_time + self.dt,
        )
        self.foot_in_contact = in_contact

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _reward_foot_placement(self):
        """Elliptical closeness of each planted foot to its beat's target."""
        rs = self.cfg.reward_settings
        targets, active = self._foot_targets_box()
        err = self._feet_in_box_frame() - targets
        sx, sy = rs.placement_sigma
        score = torch.exp(-(err[..., 0] / sx).square() - (err[..., 1] / sy).square())
        planted = self.foot_load > rs.contact_load_threshold
        return torch.mean(score * (active & planted).float(), dim=1)

    def _reward_heading(self):
        """Keep facing the box's forward direction."""
        err = self._heading_error()
        return torch.exp(-err.square() / self.cfg.reward_settings.heading_sigma**2)

    def _reward_extra_steps(self):
        """Penalize touchdowns beyond the allowed count per leg per move."""
        return -torch.sum(self.extra_touchdowns, dim=1)

    def _reward_foot_slip(self):
        """Penalize horizontal foot speed while the foot is loaded."""
        vel = self._rigid_body_lin_vel[:, self.feet_indices, :2]
        slip = vel.square().sum(dim=-1) * self.foot_in_contact.float()
        return -torch.mean(slip, dim=1)
