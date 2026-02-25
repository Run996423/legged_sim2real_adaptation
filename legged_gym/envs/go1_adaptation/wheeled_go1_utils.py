"""约定欧拉角对应的旋转矩阵为R=Rz@Ry@Rx"""
import yaml
import torch


def read_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
        return data


def save_yaml(data, file_path):
    with open(file_path, 'x') as file:
        yaml.dump(data, file)


def quat_mul(a, b):
    """
    四元数乘法
    :param a: torch.Tensor, (N, 4), wxyz
    :param b: torch.Tensor, (N, 4), wxyz
    :return: torch.Tensor, (N, 4), wxyz
    """
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    quat = torch.stack([w, x, y, z], dim=1)
    return quat


def quat_inv(q):
    """
    四元数取逆
    :param q: torch.Tensor, (N, 4), wxyz
    :return: torch.Tensor, (N, 4), wxyz
    """
    q_inv = q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device)
    return q_inv


def quat_rotate(q, v):
    """
    按四元数对向量做旋转
    :param q: torch.Tensor, (N, 4), wxyz
    :param v: torch.Tensor, (N, 3)
    :return: torch.Tensor, (N, 3)
    """
    q_w = q[:, 0]
    q_vec = q[:, 1:4]
    a = v * (2.0 * q_w ** 2 - 1.0)[:, None]
    b = torch.cross(q_vec, v, dim=1) * q_w[:, None] * 2.0
    c = q_vec * torch.matmul(q_vec[:, None, :], v[:, :, None])[..., 0] * 2.0
    return a + b + c


def quat_rotate_inverse(q, v):
    """
    按四元数对向量做反旋转
    :param q: torch.Tensor, (N, 4), wxyz
    :param v: torch.Tensor, (N, 3)
    :return: torch.Tensor, (N, 3)
    """
    q_w = q[:, 0]
    q_vec = q[:, 1:4]
    a = v * (2.0 * q_w ** 2 - 1.0)[:, None]
    b = torch.cross(q_vec, v, dim=1) * q_w[:, None] * 2.0
    c = q_vec * torch.matmul(q_vec[:, None, :], v[:, :, None])[..., 0] * 2.0
    return a - b + c


def quat_to_axis_angle(q):
    """
    单位四元数->轴角
    :param q: torch.Tensor, (N, 4), wxyz
    :return: torch.Tensor & torch.Tensor, (N, 3) & (N, 1)
    """
    angle = 2 * torch.acos(torch.clamp(q[:, 0], min=-0.99999, max=0.99999))[:, None]
    norm = torch.clamp(torch.norm(q[:, 1:4], dim=1), min=1e-5, max=1)[:, None]
    axis = q[:, 1:4] / norm
    return axis, angle


@torch.jit.script
def quat_from_rot_mat(R):
    """
    从旋转矩阵到四元数. 使用Shepperd方法确保数值稳定
    :param R: torch.Tensor, (N, 3, 3)
    :return: torch.Tensor, (N, 4), wxyz
    """
    # 计算矩阵的迹
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # 初始化四元数
    batch_size = R.shape[0]
    q = torch.zeros(batch_size, 4, device=R.device, dtype=R.dtype)

    # 情况1: trace > 0
    mask1 = trace > 0.0
    s1 = torch.sqrt(trace[mask1] + 1.0) * 2.0  # s = 4 * w
    q[mask1, 0] = 0.25 * s1  # w
    q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s1  # x
    q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s1  # y
    q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s1  # z

    # 情况2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s2 = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2.0  # s = 4 * x
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2  # w
    q[mask2, 1] = 0.25 * s2  # x
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2  # y
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2  # z

    # 情况3: R[1,1] > R[2,2]
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    s3 = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2.0  # s = 4 * y
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3  # w
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3  # x
    q[mask3, 2] = 0.25 * s3  # y
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3  # z

    # 情况4: 其他情况
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s4 = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2.0  # s = 4 * z
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4  # w
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4  # x
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4  # y
    q[mask4, 3] = 0.25 * s4  # z

    # 确保w为正（标准化四元数）
    q = torch.where(q[:, 0:1] < 0.0, -q, q)

    # 归一化以确保单位四元数
    norm = torch.norm(q, dim=1, keepdim=True)
    norm = torch.clamp(norm, min=1e-6)
    q = q / norm

    return q


@torch.jit.script
def quat_to_rot_mat(q):
    """
    从四元数到旋转矩阵
    :param q: torch.Tensor, (N, 4), wxyz
    :return: torch.Tensor, (N, 3, 3)
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    Nq = w * w + x * x + y * y + z * z
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ = y * Y, y * Z
    zZ = z * Z
    rotation_matrix = torch.stack([torch.stack([1.0 - (yY + zZ), xY - wZ, xZ + wY], dim=-1),
                                   torch.stack([xY + wZ, 1.0 - (xX + zZ), yZ - wX], dim=-1),
                                   torch.stack([xZ - wY, yZ + wX, 1.0 - (xX + yY)], dim=-1)], dim=-2)
    return rotation_matrix


def angle_regularization(x):
    """
    将弧度角化成-pi~pi之间
    :param x: torch.Tensor, (...)
    :return: torch.Tensor, (...)
    """
    return (x + torch.pi) % (2 * torch.pi) - torch.pi


@torch.jit.script
def quat_to_rpy(q):
    """
    从四元数到欧拉角
    :param q: torch.Tensor, (N, 4), wxyz
    :return: torch.Tensor, (N, 3), rpy, -pi~pi
    """
    qw, qx, qy, qz = 0, 1, 2, 3

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = (q[:, qw] * q[:, qw]
                 - q[:, qx] * q[:, qx]
                 - q[:, qy] * q[:, qy]
                 + q[:, qz] * q[:, qz])
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * torch.pi / 2.0, torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = (q[:, qw] * q[:, qw]
                 + q[:, qx] * q[:, qx]
                 - q[:, qy] * q[:, qy]
                 - q[:, qz] * q[:, qz])
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    rpy = torch.stack([roll, pitch, yaw], dim=1)
    rpy = angle_regularization(rpy)

    return rpy


@torch.jit.script
def quat_from_rpy(rpy):
    """
    从欧拉角到四元数
    :param rpy: torch.Tensor, (N, 3)
    :return: torch.Tensor, (N, 4), wxyz
    """
    roll, pitch, yaw = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    q = torch.stack([qw, qx, qy, qz], dim=1)

    return q


def auxiliary_angle_method(a, b, c):
    """
    用辅助角公式求解x: a cos(x) + b sin(x) = c, 返回-pi~pi的两个解(可能相等)
    :param a: torch.Tensor, (...)
    :param b: torch.Tensor, (...)
    :param c: torch.Tensor, (...)
    :return: torch.Tensor, (..., 2)
    """
    alpha = torch.atan2(b, a)
    t = torch.clamp(c / torch.sqrt(a ** 2 + b ** 2), min=-0.99999, max=0.99999)
    x = torch.stack([alpha - torch.acos(t), alpha + torch.acos(t)], dim=-1)
    x = angle_regularization(x)
    return x


def convert_to_skew_symmetric(x):
    """
    把x转换为对应的叉乘矩阵X, 从而: x叉乘y = X@y
    :param x: torch.Tensor, (..., 3)
    :return: torch.Tensor, (..., 3, 3)
    """
    zero = torch.zeros_like(x[..., 0])
    skew = torch.stack([torch.stack([zero, -x[..., 2], x[..., 1]], dim=-1),
                        torch.stack([x[..., 2], zero, -x[..., 0]], dim=-1),
                        torch.stack([-x[..., 1], x[..., 0], zero], dim=-1)], dim=-2)
    return skew


@torch.jit.script
def BMxBV(A, B):
    """
    (..., M, N) x (..., N) -> (..., M)
    :param A: torch.Tensor, (..., M, N)
    :param B: torch.Tensor, (..., N)
    :return: torch.Tensor, (..., M)
    """
    return torch.einsum('...ij,...j->...i', A, B)


def compute_C_rpy(rpy):
    """
    w = C_rpy @ d_rpy
    :param rpy: torch.Tensor, (..., 3)
    :return: (..., 3, 3)
    """
    roll, pitch, yaw = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    zero = torch.zeros_like(roll)
    C_rpy = torch.stack([torch.stack([torch.cos(pitch) * torch.cos(yaw), -torch.sin(yaw), zero], dim=-1),
                         torch.stack([torch.cos(pitch) * torch.sin(yaw), torch.cos(yaw), zero], dim=-1),
                         torch.stack([-torch.sin(pitch), zero, zero + 1.], dim=-1)], dim=-2)
    return C_rpy


def compute_C_rpy_inv(rpy):
    """
    d_rpy = C_rpy_inv @ w
    :param rpy: torch.Tensor, (..., 3)
    :return: (..., 3, 3)
    """
    roll, pitch, yaw = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    zero = torch.zeros_like(roll)
    C_rpy_inv = torch.stack(
        [torch.stack([torch.cos(yaw) / torch.cos(pitch), torch.sin(yaw) / torch.cos(pitch), zero], dim=-1),
         torch.stack([-torch.sin(yaw), torch.cos(yaw), zero], dim=-1),
         torch.stack([torch.cos(yaw) * torch.tan(pitch), torch.sin(yaw) * torch.tan(pitch), zero + 1.], dim=-1)],
        dim=-2)
    return C_rpy_inv


def torch_rand_float(shape=None, a_min=0.0, a_max=1.0, device='cuda', dtype=torch.float32, like=None):
    if like is None:
        return torch.rand(size=shape, device=device, dtype=dtype) * (a_max - a_min) + a_min
    else:
        return torch.rand(size=like.shape, device=like.device, dtype=like.dtype) * (a_max - a_min) + a_min
