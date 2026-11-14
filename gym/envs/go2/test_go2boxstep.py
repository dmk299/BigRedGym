from types import SimpleNamespace

import pytest
import torch

from gym.envs.go2.go2boxstep import Go2BoxStep
from gym.envs.go2.go2boxstep_config import Go2BoxStepCfg

SX, SY = 0.20, 0.15
CORNERS = torch.tensor([[0.0, 0.0], [SX, 0.0], [SX, -SY], [0.0, -SY]])


def _task(num_envs=1):
    task = Go2BoxStep.__new__(Go2BoxStep)
    task.cfg = Go2BoxStepCfg()
    task.num_envs = num_envs
    task.device = "cpu"
    task.dt = 0.02
    task._corners = CORNERS.clone()
    task._nominal_foot_body_xy = torch.zeros(4, 2)
    task.move_idx = torch.zeros(num_envs, dtype=torch.long)
    task.bar_phase = torch.zeros(num_envs, 1)
    return task


# Lead pair per move (True = right pair leads): forward, right, back, left.
LEAD_IS_RIGHT = [False, True, True, False]
RIGHT_LEG = torch.tensor([False, True, False, True])


@pytest.mark.parametrize("move", range(4))
@pytest.mark.parametrize("beat", range(3))
def test_beat_gating_matches_box_step_table(move, beat):
    task = _task()
    task.move_idx[:] = move
    task.bar_phase[:] = (beat + 0.5) / 3
    targets, active = task._foot_targets_box()

    is_lead = RIGHT_LEG == LEAD_IS_RIGHT[move]
    old, new = CORNERS[move], CORNERS[(move + 1) % 4]
    for leg in range(4):
        if is_lead[leg]:
            assert active[0, leg].item() == (beat >= 1)
            torch.testing.assert_close(targets[0, leg], new)
        else:
            assert active[0, leg].item() == (beat != 1)
            expected = old if beat == 0 else new
            torch.testing.assert_close(targets[0, leg], expected)


def test_outside_pair_leads_every_move():
    # Forward from back-left and left back to back-left: left pair leads.
    # Right to front-right and back to back-right: right pair leads.
    task = _task(num_envs=4)
    task.move_idx = torch.arange(4)
    task.bar_phase[:] = 0.1  # beat 1: only the trail pair is planted
    _, active = task._foot_targets_box()
    swinging_right = ~active[:, 1] & ~active[:, 3]
    swinging_left = ~active[:, 0] & ~active[:, 2]
    assert swinging_left.tolist() == [True, False, False, True]
    assert swinging_right.tolist() == [False, True, True, False]


def test_extra_steps_count_only_real_touchdowns_beyond_one_per_move():
    task = _task()
    task.feet_indices = torch.arange(4)
    task._backend = SimpleNamespace(link_mass=torch.tensor([[10.0]]))
    task.foot_in_contact = torch.ones(1, 4, dtype=torch.bool)
    task.foot_air_time = torch.zeros(1, 4)
    task.touchdowns_this_move = torch.zeros(1, 4)
    task.extra_touchdowns = torch.zeros(1, 4)
    weight = 98.1

    def step(fl_load_fraction):
        task.contact_forces = torch.zeros(1, 4, 3)
        task.contact_forces[0, :, 2] = 0.25 * weight
        task.contact_forces[0, 0, 2] = fl_load_fraction * weight
        task._update_contacts()
        return task.extra_touchdowns[0, 0].item()

    # A real step: 0.2 s airborne, then land. First step of the move is free.
    for _ in range(10):
        assert step(0.0) == 0.0
    assert step(0.25) == 0.0
    # A brief scuff (one 0.02 s lift) is not counted as a step.
    step(0.0)
    assert step(0.25) == 0.0
    # A second real step in the same move is penalized.
    for _ in range(10):
        step(0.0)
    assert step(0.25) == 1.0
    assert task._reward_extra_steps().item() == -1.0


def test_swing_fraction_warps_phase_continuously():
    task = _task()
    task.cpg_swing_fraction = torch.full((1, 4), 0.2)
    theta = torch.linspace(0, 2 * torch.pi - 1e-4, 2001)
    beta = []
    for t in theta:
        task.cpg_theta = torch.full((1, 4), t.item())
        b, in_swing = task._cpg_warped_phase()
        beta.append(b[0, 0].item())
    beta = torch.tensor(beta)
    # Monotonic, spans [0, 2pi), swing covers the first 20% of the cycle.
    assert (beta.diff() > 0).all()
    assert beta[0].item() == pytest.approx(0.0)
    assert beta[-1].item() == pytest.approx(2 * torch.pi, abs=1e-3)
    idx = int(0.2 * (len(theta) - 1))
    assert beta[idx].item() == pytest.approx(torch.pi, abs=1e-2)


def test_dance_clock_advances_moves_and_resets_step_counts():
    task = _task()
    task.bars_per_second = torch.full((1, 1), 1.0)
    task.touchdowns_this_move = torch.ones(1, 4)
    task.bar_phase[:] = 0.99
    task._advance_dance_clock()
    assert task.move_idx.item() == 1
    assert task.bar_phase.item() == pytest.approx(0.01, abs=1e-6)
    assert task.touchdowns_this_move.sum().item() == 0.0
    task.move_idx[:] = 3
    task.bar_phase[:] = 0.995
    task._advance_dance_clock()
    assert task.move_idx.item() == 0


def test_stance_timing_allows_only_the_scheduled_pair_to_lift():
    task = _task()
    task.move_idx[:] = 0  # forward move: left pair (FL, RL) leads
    planted = 0.25
    # Beat 1, lead pair airborne, trail planted: correct, no penalty.
    task.bar_phase[:] = 0.5 / 3
    task.foot_load = torch.tensor([[0.0, planted, 0.0, planted]])
    assert task._reward_stance_timing().item() == 0.0
    # Beat 1, both pairs airborne (two steps at once): trail pair penalized.
    task.foot_load = torch.zeros(1, 4)
    assert task._reward_stance_timing().item() == pytest.approx(-0.5)
    # Beat 2, trail pair airborne, lead planted: correct.
    task.bar_phase[:] = 1.5 / 3
    task.foot_load = torch.tensor([[planted, 0.0, planted, 0.0]])
    assert task._reward_stance_timing().item() == 0.0
    # Hold beat: any lifted foot is penalized.
    task.bar_phase[:] = 2.5 / 3
    task.foot_load = torch.tensor([[planted, planted, planted, 0.0]])
    assert task._reward_stance_timing().item() == pytest.approx(-0.25)


def test_min_base_height_penalizes_a_crouch():
    task = _task(num_envs=3)
    task.base_height = torch.tensor([[0.32], [0.28], [0.22]])
    r = task._reward_min_base_height()
    assert r[0].item() == 1.0
    assert r[1].item() == pytest.approx(0.85, abs=0.01)
    assert r[2].item() < 0.1


def test_tempo_and_episode_length_fit_two_boxes():
    cfg = Go2BoxStepCfg()
    slowest_box_s = 4 * cfg.dance.beats_per_move * 60.0 / cfg.dance.tempo_bpm_range[0]
    assert cfg.env.episode_length_s >= 2 * slowest_box_s
