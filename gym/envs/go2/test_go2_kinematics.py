import math

import torch

from gym.envs.go2.go2_kinematics import Go2LegKinematics


def _random_joint_angles(n, generator):
    # Inside the Go2 limits, away from the singular straight-knee pose.
    lo = torch.tensor([-0.6, -0.5, -2.6])
    hi = torch.tensor([0.6, 2.0, -0.9])
    return lo + (hi - lo) * torch.rand(n, 4, 3, generator=generator)


def test_inverse_recovers_joint_angles():
    kin = Go2LegKinematics()
    q = _random_joint_angles(512, torch.Generator().manual_seed(0))
    # The solver picks the foot-below-hip branch, the only one used when
    # standing; poses with the foot raised above the thigh joint are excluded.
    z_plane = -0.213 * torch.cos(q[..., 1]) - 0.213 * torch.cos(q[..., 1] + q[..., 2])
    below = z_plane < -0.05
    q_ik = kin.inverse(kin.forward(q))
    assert below.float().mean() > 0.8
    torch.testing.assert_close(q_ik[below], q[below], atol=1e-4, rtol=0)


def test_forward_matches_hand_computed_default_stance():
    kin = Go2LegKinematics()
    q = torch.tensor([0.0, 0.96, -1.36]).expand(1, 4, 3)
    p = kin.forward(q)[0]
    x = -0.213 * math.sin(0.96) - 0.213 * math.sin(0.96 - 1.36)
    z = -0.213 * math.cos(0.96) - 0.213 * math.cos(0.96 - 1.36)
    expected = torch.tensor(
        [
            [x, 0.0955, z],
            [x, -0.0955, z],
            [x, 0.0955, z],
            [x, -0.0955, z],
        ]
    )
    torch.testing.assert_close(p, expected, atol=1e-6, rtol=0)


def test_unreachable_targets_stay_finite():
    kin = Go2LegKinematics()
    p = torch.tensor([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0]]).expand(4, 2, 3)
    p = p.permute(1, 0, 2).contiguous()
    assert torch.isfinite(kin.inverse(p)).all()
