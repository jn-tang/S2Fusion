"""Simply wraps call to pytorch3d's rotation utilities
"""

import torch
from pytorch3d.transforms.rotation_conversions import (
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    matrix_to_axis_angle,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    euler_angles_to_matrix,
    rotation_6d_to_matrix,
    random_rotations
    )

# Four basic representation for rotations:
# 1. axis-angle
# 2. quaternion
# 3. rotation matrix (and corresponding 6D representation)
# 4. euler_angle

def quaternion_to_euler_angles(quaternions: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as quaternion to

    Args:
        quaternions (torch.Tensor): _description_
        convention (str): _description_

    Returns:
        torch.Tensor: _description_
    """
    mat = quaternion_to_matrix(quaternions)
    return matrix_to_euler_angles(mat, convention)


def axis_angle_to_euler_angles(axis_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """_summary_

    Args:
        axis_angles (torch.Tensor): _description_
        convention (str): _description_

    Returns:
        torch.Tensor: _description_
    """
    quats = axis_angle_to_quaternion(axis_angles)
    return quaternion_to_euler_angles(quats, convention)


def euler_angles_to_quaternion(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """_summary_

    Args:
        euler_angles (torch.Tensor): _description_
        convention (str): _description_

    Returns:
        torch.Tensor: _description_
    """
    mat = euler_angles_to_matrix(euler_angles, convention)
    return matrix_to_quaternion(mat)


def euler_angles_to_axis_angles(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """_summary_

    Args:
        euler_angles (torch.Tensor): _description_
        convention (str): _description_

    Returns:
        torch.Tensor: _description_
    """
    mat = euler_angles_to_matrix(euler_angles, convention)
    return matrix_to_axis_angle(mat)


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py.

    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.
    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def _rotation_axis(euler_angles: torch.Tensor, axis: str) -> torch.Tensor:
    """
    Return the rotation axis for one of the rotations about an axis
    of which Euler angles describes, for each axis given.

    Args:
        euler_angles (torch.Tensor): _description_
        axis (str): _description_

    Returns:
        torch.Tensor: _description_
    """
    _axis_convert = {'X': 0, 'Y': 1, 'Z': 2}
    rot_axis = torch.zeros_like(euler_angles)
    rot_axis[..., _axis_convert[axis]] = 1
    return rot_axis


def euler_angle_rotation_axes(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """_summary_

    Args:
        euler_angles (torch.Tensor): _description_
        convention (str): _description_

    Returns:
        torch.Tensor: _description_
    """
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
        ]

    axes = [
        _rotation_axis(euler_angles, c)
        for c in convention
        ]

    a = (
        torch.matmul(matrices[2], torch.matmul(matrices[1], axes[0].unsqueeze(dim=-1))),
        torch.matmul(matrices[2], axes[1].unsqueeze(dim=-1)),
        axes[2].unsqueeze(dim=-1)
        )
    return torch.cat(a, dim=-1)

def axis_angle_to_rotation_6d(axis_angles: torch.Tensor):
    mats = axis_angle_to_matrix(axis_angles)
    return matrix_to_rotation_6d(mats)


def rotation_6d_to_axis_angle(rot6d: torch.Tensor):
    return matrix_to_axis_angle(rotation_6d_to_matrix(rot6d))
