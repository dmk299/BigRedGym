"""Batched closed-form leg kinematics for the Unitree Go2.

Geometry is taken from resources/robots/go2/urdf/go2.urdf. Each leg has a hip
abduction joint about the body x axis followed by thigh and calf joints about
the y axis. Positions are expressed in a frame parallel to the base frame with
its origin at the leg's hip joint. Legs use the canonical order FL, FR, RL, RR
and joints the order hip, thigh, calf, matching ``GO2_DOF_NAMES``.
"""

import torch

GO2_HIP_POSITIONS = (
    (0.1934, 0.0465, 0.0),  # FL
    (0.1934, -0.0465, 0.0),  # FR
    (-0.1934, 0.0465, 0.0),  # RL
    (-0.1934, -0.0465, 0.0),  # RR
)
# Lateral offset from hip joint to thigh joint; positive for left legs.
GO2_HIP_TO_THIGH_Y = (0.0955, -0.0955, 0.0955, -0.0955)
GO2_THIGH_LENGTH = 0.213
GO2_CALF_LENGTH = 0.213


class Go2LegKinematics:
    """Forward and inverse kinematics for all four legs, batched over envs."""

    def __init__(self, device="cpu"):
        self.device = device
        self.hip_positions = torch.tensor(GO2_HIP_POSITIONS, device=device)
        self.hip_to_thigh_y = torch.tensor(GO2_HIP_TO_THIGH_Y, device=device)
        self.l1 = GO2_THIGH_LENGTH
        self.l2 = GO2_CALF_LENGTH

    def forward(self, q):
        """Joint angles (..., 4, 3) -> foot positions in hip frames (..., 4, 3)."""
        q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2]
        # Sagittal plane of the thigh/calf chain (before hip abduction).
        x = -self.l1 * torch.sin(q2) - self.l2 * torch.sin(q2 + q3)
        z = -self.l1 * torch.cos(q2) - self.l2 * torch.cos(q2 + q3)
        d = self.hip_to_thigh_y.expand_as(q1)
        c1, s1 = torch.cos(q1), torch.sin(q1)
        y_rot = d * c1 - z * s1
        z_rot = d * s1 + z * c1
        return torch.stack((x, y_rot, z_rot), dim=-1)

    def forward_body(self, q):
        """Joint angles (..., 4, 3) -> foot positions in the base frame."""
        return self.forward(q) + self.hip_positions

    def inverse(self, p):
        """Foot positions in hip frames (..., 4, 3) -> joint angles (..., 4, 3).

        Unreachable targets are projected onto the workspace boundary, so the
        result is always finite. The knee always bends backward (calf < 0),
        matching the Go2 joint limits.
        """
        px, py, pz = p[..., 0], p[..., 1], p[..., 2]
        d = self.hip_to_thigh_y.expand_as(px)

        # Hip abduction: the leg plane sits at lateral distance |d| from the
        # hip axis, so the in-plane leg height is the remaining yz distance.
        r_yz_sq = py.square() + pz.square()
        z_plane = -torch.sqrt(torch.clamp(r_yz_sq - d.square(), min=1e-6))
        q1 = torch.atan2(pz, py) - torch.atan2(z_plane, d)
        q1 = torch.atan2(torch.sin(q1), torch.cos(q1))

        # Planar two-link solution for thigh and calf.
        cos_q3 = (px.square() + z_plane.square() - self.l1**2 - self.l2**2) / (
            2.0 * self.l1 * self.l2
        )
        q3 = -torch.acos(torch.clamp(cos_q3, -1.0 + 1e-6, 1.0 - 1e-6))
        a = self.l1 + self.l2 * torch.cos(q3)
        b = self.l2 * torch.sin(q3)
        q2 = torch.atan2(-px, -z_plane) - torch.atan2(b, a)
        return torch.stack((q1, q2, q3), dim=-1)

    def inverse_body(self, p_body):
        """Foot positions in the base frame (..., 4, 3) -> joint angles."""
        return self.inverse(p_body - self.hip_positions)
