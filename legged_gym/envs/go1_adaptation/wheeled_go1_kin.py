import time

import torch

from legged_gym.envs.go1_adaptation.wheeled_go1_utils import BMxBV, auxiliary_angle_method, convert_to_skew_symmetric, \
    quat_from_rpy, quat_to_rot_mat


class WheeledGo1Kin(object):
    """
    kinematics tree for wheeled go1:
    连接定义: hip - joint1 - link1(l1) - joint2 - link2(l2) - joint3 - link3(l3) - offset(d1~d3) - joint4 - link4(r)
        frame0: o=hip, 姿态同base frame
        frame1: o=link1近端, +x=joint1, +y=link1
        frame2: o=link2近端, +y=joint2, -z=link2
        frame3: o=link3近端, +y=joint3, -z=link3
        frame4: o=link4近端, +y=joint4, -z=link4
        frame5: o=link4远端, 姿态同frame4
    l1: hip link length
    l2: thigh link length
    l3: calf link length
    d1: toe-wheel offset along calf (向着躯干下方)
    d2: toe-wheel lateral offset (向着躯干后方)
    d3: toe-wheel offset inward (向着躯干内侧)
    r: wheel radius
    """

    def __init__(self, device):
        self.l1 = torch.tensor([-1.0, +1.0, -1.0, +1.0], device=device) * 0.08
        self.l2 = torch.tensor([+1.0, +1.0, +1.0, +1.0], device=device) * 0.213
        self.l3 = torch.tensor([+1.0, +1.0, +1.0, +1.0], device=device) * 0.213
        self.d1 = torch.tensor([+1.0, +1.0, +1.0, +1.0], device=device) * 0.02
        self.d2 = torch.tensor([+1.0, +1.0, +1.0, +1.0], device=device) * 0.015
        self.d3 = torch.tensor([+1.0, -1.0, +1.0, -1.0], device=device) * 0.025
        self.r = torch.tensor([+1.0, +1.0, +1.0, +1.0], device=device) * 0.075

        self.p0 = None
        self.p1 = None
        self.p2 = None
        self.p3 = None
        self.p4 = None
        self.p5 = None

        self.R0 = None
        self.R1 = None
        self.R2 = None
        self.R3 = None
        self.R4 = None

        self.J1 = None
        self.J2 = None
        self.J3 = None
        self.J4 = None
        self.J5 = None

        self.dJ1 = None
        self.dJ2 = None
        self.dJ3 = None
        self.dJ4 = None
        self.dJ5 = None

        self.axes = ()

    def position_level_forward_kinematics(self, q, q40):
        """
        位置级正运动学
        :param q: tensor, (N, 4, 4)
        :param q40: tensor, (N, 4)
        :return:
        """
        N = q.shape[0]
        device = q.device

        q1, q2, q3, q4 = q[:, :, 0], q[:, :, 1], q[:, :, 2], q[:, :, 3]
        zero = torch.zeros_like(q1)
        one = torch.ones_like(q1)

        p0 = torch.zeros((N, 4, 3), device=device)
        R0 = torch.eye(3, device=device).repeat(N, 4, 1, 1)

        p1 = p0
        R1 = torch.matmul(R0, torch.stack([one, zero, zero,
                                           zero, torch.cos(q1), -torch.sin(q1),
                                           zero, torch.sin(q1), torch.cos(q1)], dim=2).reshape(N, 4, 3, 3))

        p2 = p1 + BMxBV(R1, torch.stack([zero, self.l1 * one, zero], dim=2))
        R2 = torch.matmul(R1, torch.stack([torch.cos(q2), zero, torch.sin(q2),
                                           zero, one, zero,
                                           -torch.sin(q2), zero, torch.cos(q2)], dim=2).reshape(N, 4, 3, 3))

        p3 = p2 + BMxBV(R2, torch.stack([zero, zero, -self.l2 * one], dim=2))
        R3 = torch.matmul(R2, torch.stack([torch.cos(q3), zero, torch.sin(q3),
                                           zero, one, zero,
                                           -torch.sin(q3), zero, torch.cos(q3)], dim=2).reshape(N, 4, 3, 3))

        p4 = p3 + BMxBV(R3, torch.stack([-self.d2 * one,
                                         self.d3 * one,
                                         -(self.l3 + self.d1) * one], dim=2))
        R4 = torch.matmul(R3, torch.stack([torch.cos(q4 + q40), zero, torch.sin(q4 + q40),
                                           zero, one, zero,
                                           -torch.sin(q4 + q40), zero, torch.cos(q4 + q40)], dim=2).reshape(N, 4, 3, 3))

        p5 = p4 + BMxBV(R4, torch.stack([zero, zero, -self.r * one], dim=2))

        self.p0, self.p1, self.p2, self.p3, self.p4, self.p5 = p0, p1, p2, p3, p4, p5
        self.R0, self.R1, self.R2, self.R3, self.R4 = R0, R1, R2, R3, R4

    def compute_joint_axis(self):
        """
        计算各joint的轴在hip frame(frame0)中的坐标
        :return:
        """
        N = self.p0.shape[0]
        device = self.p0.device

        zero = torch.zeros((N, 4), device=device)
        one = torch.ones((N, 4), device=device)

        axis_1 = BMxBV(self.R1, torch.stack([one, zero, zero], dim=2))
        axis_2 = BMxBV(self.R2, torch.stack([zero, one, zero], dim=2))
        axis_3 = BMxBV(self.R3, torch.stack([zero, one, zero], dim=2))
        axis_4 = BMxBV(self.R4, torch.stack([zero, one, zero], dim=2))
        axis_0 = torch.zeros_like(axis_1)

        self.axes = (axis_0, axis_1, axis_2, axis_3, axis_4)

    def velocity_level_forward_kinematics(self):
        """
        速度级正运动学
        :return:
        """
        N = self.p0.shape[0]
        device = self.p0.device

        ps = [self.p0, self.p1, self.p2, self.p3, self.p4, self.p5]
        Js = []

        self.compute_joint_axis()
        axes = self.axes

        for i in range(1, 5):
            J = torch.zeros((N, 4, 6, 4), device=device)
            for j in range(1, i + 1):
                J[:, :, 0:3, j - 1] = torch.cross(axes[j], ps[i] - ps[j], dim=-1)
                J[:, :, 3:6, j - 1] = axes[j]
            Js.append(J)
        J = torch.zeros((N, 4, 6, 4), device=device)
        for j in range(1, 5):
            J[:, :, 0:3, j - 1] = torch.cross(axes[j], ps[5] - ps[j], dim=-1)
            J[:, :, 3:6, j - 1] = axes[j]
        Js.append(J)

        self.J1, self.J2, self.J3, self.J4, self.J5 = Js

    def acceleration_level_forward_kinematics(self, dq):
        """
        加速度级正运动学
        :param dq: tensor, (N, 4, 4)
        :return:
        """
        N = self.p0.shape[0]
        device = self.p0.device

        ps = [self.p0, self.p1, self.p2, self.p3, self.p4, self.p5]
        Js = [torch.zeros_like(self.J1), self.J1, self.J2, self.J3, self.J4, self.J5]
        dJs = []

        self.compute_joint_axis()
        axes = self.axes

        v = [torch.zeros((N, 4, 3), device=device)] + [BMxBV(Js[i][:, :, 0:3, :], dq) for i in range(1, 6)]
        w = [torch.zeros((N, 4, 3), device=device)] + [BMxBV(Js[i][:, :, 3:6, :], dq) for i in range(1, 6)]
        d_axis = [torch.zeros((N, 4, 3), device=device)] + [torch.cross(w[i], axes[i], dim=-1) for i in range(1, 5)]

        for i in range(1, 5):
            dJ = torch.zeros((N, 4, 6, 4), device=device)
            for j in range(1, i + 1):
                dJ[:, :, 0:3, j - 1] = (torch.cross(d_axis[j], ps[i] - ps[j], dim=-1)
                                        + torch.cross(axes[j], v[i] - v[j], dim=-1))
                dJ[:, :, 3:6, j - 1] = d_axis[j]
            dJs.append(dJ)
        dJ = torch.zeros((N, 4, 6, 4), device=device)
        for j in range(1, 5):
            dJ[:, :, 0:3, j - 1] = (torch.cross(d_axis[j], ps[5] - ps[j], dim=-1)
                                    + torch.cross(axes[j], v[5] - v[j], dim=-1))
            dJ[:, :, 3:6, j - 1] = d_axis[j]
        dJs.append(dJ)

        self.dJ1, self.dJ2, self.dJ3, self.dJ4, self.dJ5 = dJs

    def position_level_inverse_kinematics_from_p4(self, p4):
        """
        逆运动学, 从p4推q1, q2, q3
        :param p4: tensor, (N, 4, 3)
        :return: tensor, (N, 4, 3)
        """
        x, y, z = p4[:, :, 0], p4[:, :, 1], p4[:, :, 2]
        l1 = self.l1 + self.d3  # 等效hip link长度
        k = ((self.d1 + self.l3) ** 2 + self.d2 ** 2) ** 0.5  # 等效小腿长度
        alpha = torch.arctan(self.d2 / (self.d1 + self.l3))  # 等效膝关节偏置角
        cos_q3a = (x ** 2 + y ** 2 + z ** 2 - l1 ** 2 - self.l2 ** 2 - k ** 2) / (2 * self.l2 * k)
        q3a = - torch.arccos(torch.clip(cos_q3a, min=-1, max=1))
        q3 = q3a - alpha
        l = torch.sqrt(torch.clip(self.l2 ** 2 + k ** 2 + 2 * self.l2 * k * torch.cos(q3a), min=1e-6, max=None))
        u = torch.arcsin(torch.clip(-x / l, min=-1, max=1))
        q2 = u + torch.arccos(torch.clip((self.l2 ** 2 + l ** 2 - k ** 2) / (2.0 * self.l2 * l), min=-1.0, max=1.0))
        q1 = torch.arctan2(l * torch.cos(u) * y + l1 * z, l1 * y - l * torch.cos(u) * z)
        q = torch.stack([q1, q2, q3], dim=2)
        return q

    @staticmethod
    def compute_q40(R):
        """
        计算触地点(轮子上竖直高度最低的点)对应的轮子关节角
        :param R: tensor, (N, 4, 3, 3), frame4相对world frame的旋转
        :return: tensor, (N, 4)
        """
        r31, r33 = R[:, :, 2, 0], R[:, :, 2, 2]
        phi = torch.arctan2(r33, r31)
        q40 = torch.pi / 2 - phi
        return q40

    def compute_contact_pos_minus_p4(self, contact_pos, base_rot_mat):
        """
        计算轮心和轮子触地点构成的向量, 也即contact_pos - p4
        :param contact_pos: tensor, (N, 4, 3), 触地点在hip frame中的坐标
        :param base_rot_mat: tensor, (N, 3, 3), 也即hip frame相对world frame的旋转
        :return: tensor, (N, 4, 3)
        """
        N = contact_pos.shape[0]
        device = contact_pos.device

        l1 = self.l1 + self.d3  # 等效hip link长度
        q1 = auxiliary_angle_method(contact_pos[:, :, 1], contact_pos[:, :, 2], l1)
        idx = torch.logical_and(torch.greater_equal(q1[:, :, 0], -0.863), torch.less_equal(q1[:, :, 0], 0.863))
        q1 = torch.where(idx, q1[:, :, 0], q1[:, :, 1])
        link1_hip_frame = torch.stack([torch.zeros_like(q1), torch.sin(q1), -torch.cos(q1)], dim=2)
        x_axis_hip_frame = torch.tensor([1.0, 0.0, 0.0], device=device)
        # n表示"joint 1的轴, 轮子触地点构成的平面"的法向量, 该平面也是"joint 1的轴, 轮心构成的平面"
        n_hip_frame = torch.cross(x_axis_hip_frame.repeat(N, 4, 1), link1_hip_frame, dim=-1)
        n_world_frame = BMxBV(base_rot_mat[:, None, :, :].repeat(1, 4, 1, 1), n_hip_frame)
        # b表示轮心和轮子触地点构成的向量, 也即p4 - p3
        b_world_frame = torch.stack([n_world_frame[:, :, 0] * n_world_frame[:, :, 2],
                                     n_world_frame[:, :, 1] * n_world_frame[:, :, 2],
                                     - torch.sum(torch.square(n_world_frame[:, :, 0:2]), dim=2)], dim=2)
        norm_of_b = torch.norm(b_world_frame, dim=2, keepdim=True)
        b_world_frame = b_world_frame / norm_of_b * self.r[:, None]
        b_hip_frame = BMxBV(torch.swapaxes(base_rot_mat, 1, 2)[:, None, :, :].repeat(1, 4, 1, 1), b_world_frame)
        return b_hip_frame

    def compute_whole_J_dJ(self, base_rot_mat, base_ang_vel, dq, hip_pos_base_frame):
        """
        计算触地点和轮心的whole Jacobian和JacobianTimeVariation
        :param base_rot_mat: tensor, (N, 3, 3)
        :param base_ang_vel: tensor, (N, 3)
        :param dq: tensor, (N, 4, 4)
        :param hip_pos_base_frame: tensor, (4, 3)
        :return: J_contact, J_foot, dJ_contact, dJ_foot, all tensor of (N, 12, 22)
        """
        N = base_rot_mat.shape[0]
        device = base_rot_mat.device

        eye3 = torch.eye(3, device=device)
        ps = [self.p0, self.p1, self.p2, self.p3, self.p4, self.p5]
        Js = [torch.zeros_like(self.J1), self.J1, self.J2, self.J3, self.J4, self.J5]
        dJs = [torch.zeros_like(self.dJ1), self.dJ1, self.dJ2, self.dJ3, self.dJ4, self.dJ5]

        def compute_whole_J(frame_id):
            J = torch.zeros((N, 12, 22), device=device)
            J[:, 0:3, 6:10] = base_rot_mat @ Js[frame_id][:, 0, 0:3, :]
            J[:, 3:6, 10:14] = base_rot_mat @ Js[frame_id][:, 1, 0:3, :]
            J[:, 6:9, 14:18] = base_rot_mat @ Js[frame_id][:, 2, 0:3, :]
            J[:, 9:12, 18:22] = base_rot_mat @ Js[frame_id][:, 3, 0:3, :]
            pos = BMxBV(base_rot_mat[:, None, :, :].repeat(1, 4, 1, 1), ps[frame_id] + hip_pos_base_frame)
            skew_mat = convert_to_skew_symmetric(pos)
            J[:, 0:3, 0:3] = eye3
            J[:, 0:3, 3:6] = - skew_mat[:, 0, :, :]
            J[:, 3:6, 0:3] = eye3
            J[:, 3:6, 3:6] = - skew_mat[:, 1, :, :]
            J[:, 6:9, 0:3] = eye3
            J[:, 6:9, 3:6] = - skew_mat[:, 2, :, :]
            J[:, 9:12, 0:3] = eye3
            J[:, 9:12, 3:6] = - skew_mat[:, 3, :, :]
            return J

        J_foot, J_contact = compute_whole_J(4), compute_whole_J(5)

        def compute_whole_dJ(frame_id):
            dJ = torch.zeros((N, 12, 22), device=device)
            dJ[:, 0:3, 6:10] = base_rot_mat @ dJs[frame_id][:, 0, 0:3, :]
            dJ[:, 3:6, 10:14] = base_rot_mat @ dJs[frame_id][:, 1, 0:3, :]
            dJ[:, 6:9, 14:18] = base_rot_mat @ dJs[frame_id][:, 2, 0:3, :]
            dJ[:, 9:12, 18:22] = base_rot_mat @ dJs[frame_id][:, 3, 0:3, :]
            pos = BMxBV(base_rot_mat[:, None, :, :].repeat(1, 4, 1, 1), ps[frame_id] + hip_pos_base_frame)
            v_r = BMxBV(base_rot_mat[:, None, :, :].repeat(1, 4, 1, 1), BMxBV(Js[frame_id][:, :, 0:3, :], dq))
            factor = torch.cross(base_ang_vel[:, None, :].repeat(1, 4, 1), pos, dim=-1) + 2. * v_r
            skew_mat = convert_to_skew_symmetric(factor)
            dJ[:, 0:3, 0:3] = 0.0
            dJ[:, 0:3, 3:6] = - skew_mat[:, 0, :, :]
            dJ[:, 3:6, 0:3] = 0.0
            dJ[:, 3:6, 3:6] = - skew_mat[:, 1, :, :]
            dJ[:, 6:9, 0:3] = 0.0
            dJ[:, 6:9, 3:6] = - skew_mat[:, 2, :, :]
            dJ[:, 9:12, 0:3] = 0.0
            dJ[:, 9:12, 3:6] = - skew_mat[:, 3, :, :]
            return dJ

        dJ_foot, dJ_contact = compute_whole_dJ(4), compute_whole_dJ(5)

        return J_contact, J_foot, dJ_contact, dJ_foot


@torch.jit.script
def position_level_forward_kinematics(q, q40, l1, l2, l3, d1, d2, d3, r):
    """位置级正运动学"""
    N = q.shape[0]
    device = q.device

    q1, q2, q3, q4 = q[:, :, 0], q[:, :, 1], q[:, :, 2], q[:, :, 3]
    zero = torch.zeros_like(q1)
    one = torch.ones_like(q1)

    p0 = torch.zeros((N, 4, 3), device=device)
    R0 = torch.eye(3, device=device).repeat(N, 4, 1, 1)

    p1 = p0
    R1 = torch.matmul(R0, torch.stack([one, zero, zero,
                                       zero, torch.cos(q1), -torch.sin(q1),
                                       zero, torch.sin(q1), torch.cos(q1)], dim=2).reshape(N, 4, 3, 3))

    p2 = p1 + BMxBV(R1, torch.stack([zero, l1 * one, zero], dim=2))
    R2 = torch.matmul(R1, torch.stack([torch.cos(q2), zero, torch.sin(q2),
                                       zero, one, zero,
                                       -torch.sin(q2), zero, torch.cos(q2)], dim=2).reshape(N, 4, 3, 3))

    p3 = p2 + BMxBV(R2, torch.stack([zero, zero, - l2 * one], dim=2))
    R3 = torch.matmul(R2, torch.stack([torch.cos(q3), zero, torch.sin(q3),
                                       zero, one, zero,
                                       -torch.sin(q3), zero, torch.cos(q3)], dim=2).reshape(N, 4, 3, 3))

    p4 = p3 + BMxBV(R3, torch.stack([- d2 * one,
                                     d3 * one,
                                     - (l3 + d1) * one], dim=2))
    R4 = torch.matmul(R3, torch.stack([torch.cos(q4 + q40), zero, torch.sin(q4 + q40),
                                       zero, one, zero,
                                       -torch.sin(q4 + q40), zero, torch.cos(q4 + q40)], dim=2).reshape(N, 4, 3, 3))

    p5 = p4 + BMxBV(R4, torch.stack([zero, zero, - r * one], dim=2))

    return p0, p1, p2, p3, p4, p5, R0, R1, R2, R3, R4


@torch.jit.script
def compute_joint_axis(R1, R2, R3, R4):
    """计算各joint的轴在hip frame(frame0)中的坐标"""
    N = R1.shape[0]
    device = R1.device

    zero = torch.zeros((N, 4), device=device)
    one = torch.ones((N, 4), device=device)

    axis_1 = BMxBV(R1, torch.stack([one, zero, zero], dim=2))
    axis_2 = BMxBV(R2, torch.stack([zero, one, zero], dim=2))
    axis_3 = BMxBV(R3, torch.stack([zero, one, zero], dim=2))
    axis_4 = BMxBV(R4, torch.stack([zero, one, zero], dim=2))
    axis_0 = torch.zeros_like(axis_1)

    return axis_0, axis_1, axis_2, axis_3, axis_4


@torch.jit.script
def velocity_level_forward_kinematics(p0, p1, p2, p3, p4, p5, R1, R2, R3, R4):
    """速度级正运动学"""
    N = p0.shape[0]
    device = p0.device

    ps = [p0, p1, p2, p3, p4, p5]
    axes = compute_joint_axis(R1, R2, R3, R4)
    Js = []

    for i in range(1, 5):
        J = torch.zeros((N, 4, 6, 4), device=device)
        for j in range(1, i + 1):
            J[:, :, 0:3, j - 1] = torch.cross(axes[j], ps[i] - ps[j], dim=-1)
            J[:, :, 3:6, j - 1] = axes[j]
        Js.append(J)
    J = torch.zeros((N, 4, 6, 4), device=device)
    for j in range(1, 5):
        J[:, :, 0:3, j - 1] = torch.cross(axes[j], ps[5] - ps[j], dim=-1)
        J[:, :, 3:6, j - 1] = axes[j]
    Js.append(J)

    J1, J2, J3, J4, J5 = Js
    return J1, J2, J3, J4, J5


@torch.jit.script
def acceleration_level_forward_kinematics(p0, p1, p2, p3, p4, p5, R1, R2, R3, R4, J1, J2, J3, J4, J5, dq):
    """加速度级正运动学"""
    N = p0.shape[0]
    device = p0.device

    ps = [p0, p1, p2, p3, p4, p5]
    Js = [torch.zeros_like(J1), J1, J2, J3, J4, J5]
    dJs = []

    axes = compute_joint_axis(R1, R2, R3, R4)

    v = [torch.zeros((N, 4, 3), device=device)] + [BMxBV(Js[i][:, :, 0:3, :], dq) for i in range(1, 6)]
    w = [torch.zeros((N, 4, 3), device=device)] + [BMxBV(Js[i][:, :, 3:6, :], dq) for i in range(1, 6)]
    d_axis = [torch.zeros((N, 4, 3), device=device)] + [torch.cross(w[i], axes[i], dim=-1) for i in range(1, 5)]

    for i in range(1, 5):
        dJ = torch.zeros((N, 4, 6, 4), device=device)
        for j in range(1, i + 1):
            dJ[:, :, 0:3, j - 1] = (torch.cross(d_axis[j], ps[i] - ps[j], dim=-1)
                                    + torch.cross(axes[j], v[i] - v[j], dim=-1))
            dJ[:, :, 3:6, j - 1] = d_axis[j]
        dJs.append(dJ)
    dJ = torch.zeros((N, 4, 6, 4), device=device)
    for j in range(1, 5):
        dJ[:, :, 0:3, j - 1] = (torch.cross(d_axis[j], ps[5] - ps[j], dim=-1)
                                + torch.cross(axes[j], v[5] - v[j], dim=-1))
        dJ[:, :, 3:6, j - 1] = d_axis[j]
    dJs.append(dJ)

    dJ1, dJ2, dJ3, dJ4, dJ5 = dJs

    return dJ1, dJ2, dJ3, dJ4, dJ5


@torch.jit.script
def position_level_inverse_kinematics_from_p4(p4, l1, l2, l3, d1, d2, d3):
    """逆运动学, 从p4推q1, q2, q3"""
    x, y, z = p4[:, :, 0], p4[:, :, 1], p4[:, :, 2]
    l1 = l1 + d3  # 等效hip link长度
    k = ((d1 + l3) ** 2 + d2 ** 2) ** 0.5  # 等效小腿长度
    alpha = torch.arctan(d2 / (d1 + l3))  # 等效膝关节偏置角
    cos_q3a = (x ** 2 + y ** 2 + z ** 2 - l1 ** 2 - l2 ** 2 - k ** 2) / (2 * l2 * k)
    q3a = - torch.arccos(torch.clip(cos_q3a, min=-1, max=1))
    q3 = q3a - alpha
    l = torch.sqrt(torch.clip(l2 ** 2 + k ** 2 + 2 * l2 * k * torch.cos(q3a), min=1e-6, max=None))
    u = torch.arcsin(torch.clip(-x / l, min=-1, max=1))
    q2 = u + torch.arccos(torch.clip((l2 ** 2 + l ** 2 - k ** 2) / (2.0 * l2 * l), min=-1.0, max=1.0))
    q1 = torch.arctan2(l * torch.cos(u) * y + l1 * z, l1 * y - l * torch.cos(u) * z)
    q = torch.stack([q1, q2, q3], dim=2)
    return q


def compute_q40(R, ground_normal=None):
    """计算触地点(轮子上竖直高度最低的点)对应的轮子关节角"""
    if ground_normal is None:
        ground_normal = torch.tensor((0., 0., 1.), device=R.device).expand(R.shape[:-2] + (3,))
    n_local = torch.einsum('nkij,nkj->nki', R.transpose(-2, -1), ground_normal)
    n_x, n_z = n_local[:, :, 0], n_local[:, :, 2]
    phi = torch.arctan2(n_z, n_x)
    q40 = torch.pi / 2 - phi
    return q40


@torch.jit.script
def compute_contact_pos_minus_p4(contact_pos, base_rot_mat, l1, d3, r):
    """计算轮心和轮子触地点构成的向量, 也即contact_pos - p4"""
    N = contact_pos.shape[0]
    device = contact_pos.device

    l1 = l1 + d3  # 等效hip link长度
    q1 = auxiliary_angle_method(contact_pos[:, :, 1], contact_pos[:, :, 2], l1)
    idx = torch.logical_and(torch.greater_equal(q1[:, :, 0], -0.863), torch.less_equal(q1[:, :, 0], 0.863))
    q1 = torch.where(idx, q1[:, :, 0], q1[:, :, 1])
    link1_hip_frame = torch.stack([torch.zeros_like(q1), torch.sin(q1), -torch.cos(q1)], dim=2)
    x_axis_hip_frame = torch.tensor([1.0, 0.0, 0.0], device=device)
    # n表示"joint 1的轴, 轮子触地点构成的平面"的法向量, 该平面也是"joint 1的轴, 轮心构成的平面"
    n_hip_frame = torch.cross(x_axis_hip_frame.repeat(N, 4, 1), link1_hip_frame, dim=-1)
    n_world_frame = BMxBV(base_rot_mat[:, None, :, :].repeat(1, 4, 1, 1), n_hip_frame)
    # b表示轮心和轮子触地点构成的向量, 也即p4 - p3
    b_world_frame = torch.stack([n_world_frame[:, :, 0] * n_world_frame[:, :, 2],
                                 n_world_frame[:, :, 1] * n_world_frame[:, :, 2],
                                 - torch.sum(torch.square(n_world_frame[:, :, 0:2]), dim=2)], dim=2)
    norm_of_b = torch.norm(b_world_frame, dim=2, keepdim=True)
    b_world_frame = b_world_frame / norm_of_b * r[:, None]
    b_hip_frame = BMxBV(torch.swapaxes(base_rot_mat, 1, 2)[:, None, :, :].repeat(1, 4, 1, 1), b_world_frame)
    return b_hip_frame


@torch.jit.script
def compute_whole_J_dJ(base_rot_mat, base_ang_vel, dq, hip_pos_base_frame,
                       p0, p1, p2, p3, p4, p5, J1, J2, J3, J4, J5, dJ1, dJ2, dJ3, dJ4, dJ5):
    """计算触地点和轮心的whole Jacobian和JacobianTimeVariation"""
    N = base_rot_mat.shape[0]
    device = base_rot_mat.device

    eye3 = torch.eye(3, device=device)
    ps = [p0, p1, p2, p3, p4, p5]
    Js = [torch.zeros_like(J1), J1, J2, J3, J4, J5]
    dJs = [torch.zeros_like(dJ1), dJ1, dJ2, dJ3, dJ4, dJ5]

    J_foot = torch.zeros((N, 12, 22), device=device)
    J_foot[:, 0:3, 6:10] = base_rot_mat @ Js[4][:, 0, 0:3, :]
    J_foot[:, 3:6, 10:14] = base_rot_mat @ Js[4][:, 1, 0:3, :]
    J_foot[:, 6:9, 14:18] = base_rot_mat @ Js[4][:, 2, 0:3, :]
    J_foot[:, 9:12, 18:22] = base_rot_mat @ Js[4][:, 3, 0:3, :]
    pos = BMxBV(base_rot_mat[:, None, :, :].repeat(1, 4, 1, 1), ps[4] + hip_pos_base_frame)
    skew_mat = convert_to_skew_symmetric(pos)
    J_foot[:, 0:3, 0:3] = eye3
    J_foot[:, 0:3, 3:6] = - skew_mat[:, 0, :, :]
    J_foot[:, 3:6, 0:3] = eye3
    J_foot[:, 3:6, 3:6] = - skew_mat[:, 1, :, :]
    J_foot[:, 6:9, 0:3] = eye3
    J_foot[:, 6:9, 3:6] = - skew_mat[:, 2, :, :]
    J_foot[:, 9:12, 0:3] = eye3
    J_foot[:, 9:12, 3:6] = - skew_mat[:, 3, :, :]

    J_contact = torch.zeros((N, 12, 22), device=device)
    J_contact[:, 0:3, 6:10] = base_rot_mat @ Js[5][:, 0, 0:3, :]
    J_contact[:, 3:6, 10:14] = base_rot_mat @ Js[5][:, 1, 0:3, :]
    J_contact[:, 6:9, 14:18] = base_rot_mat @ Js[5][:, 2, 0:3, :]
    J_contact[:, 9:12, 18:22] = base_rot_mat @ Js[5][:, 3, 0:3, :]
    pos = BMxBV(base_rot_mat[:, None, :, :].repeat(1, 4, 1, 1), ps[5] + hip_pos_base_frame)
    skew_mat = convert_to_skew_symmetric(pos)
    J_contact[:, 0:3, 0:3] = eye3
    J_contact[:, 0:3, 3:6] = - skew_mat[:, 0, :, :]
    J_contact[:, 3:6, 0:3] = eye3
    J_contact[:, 3:6, 3:6] = - skew_mat[:, 1, :, :]
    J_contact[:, 6:9, 0:3] = eye3
    J_contact[:, 6:9, 3:6] = - skew_mat[:, 2, :, :]
    J_contact[:, 9:12, 0:3] = eye3
    J_contact[:, 9:12, 3:6] = - skew_mat[:, 3, :, :]

    dJ_foot = torch.zeros((N, 12, 22), device=device)
    dJ_foot[:, 0:3, 6:10] = base_rot_mat @ dJs[4][:, 0, 0:3, :]
    dJ_foot[:, 3:6, 10:14] = base_rot_mat @ dJs[4][:, 1, 0:3, :]
    dJ_foot[:, 6:9, 14:18] = base_rot_mat @ dJs[4][:, 2, 0:3, :]
    dJ_foot[:, 9:12, 18:22] = base_rot_mat @ dJs[4][:, 3, 0:3, :]
    pos = BMxBV(base_rot_mat[:, None, :, :].repeat(1, 4, 1, 1), ps[4] + hip_pos_base_frame)
    v_r = BMxBV(base_rot_mat[:, None, :, :].repeat(1, 4, 1, 1), BMxBV(Js[4][:, :, 0:3, :], dq))
    factor = torch.cross(base_ang_vel[:, None, :].repeat(1, 4, 1), pos, dim=-1) + 2. * v_r
    skew_mat = convert_to_skew_symmetric(factor)
    dJ_foot[:, 0:3, 0:3] = 0.0
    dJ_foot[:, 0:3, 3:6] = - skew_mat[:, 0, :, :]
    dJ_foot[:, 3:6, 0:3] = 0.0
    dJ_foot[:, 3:6, 3:6] = - skew_mat[:, 1, :, :]
    dJ_foot[:, 6:9, 0:3] = 0.0
    dJ_foot[:, 6:9, 3:6] = - skew_mat[:, 2, :, :]
    dJ_foot[:, 9:12, 0:3] = 0.0
    dJ_foot[:, 9:12, 3:6] = - skew_mat[:, 3, :, :]

    dJ_contact = torch.zeros((N, 12, 22), device=device)
    dJ_contact[:, 0:3, 6:10] = base_rot_mat @ dJs[5][:, 0, 0:3, :]
    dJ_contact[:, 3:6, 10:14] = base_rot_mat @ dJs[5][:, 1, 0:3, :]
    dJ_contact[:, 6:9, 14:18] = base_rot_mat @ dJs[5][:, 2, 0:3, :]
    dJ_contact[:, 9:12, 18:22] = base_rot_mat @ dJs[5][:, 3, 0:3, :]
    pos = BMxBV(base_rot_mat[:, None, :, :].repeat(1, 4, 1, 1), ps[5] + hip_pos_base_frame)
    v_r = BMxBV(base_rot_mat[:, None, :, :].repeat(1, 4, 1, 1), BMxBV(Js[5][:, :, 0:3, :], dq))
    factor = torch.cross(base_ang_vel[:, None, :].repeat(1, 4, 1), pos, dim=-1) + 2. * v_r
    skew_mat = convert_to_skew_symmetric(factor)
    dJ_contact[:, 0:3, 0:3] = 0.0
    dJ_contact[:, 0:3, 3:6] = - skew_mat[:, 0, :, :]
    dJ_contact[:, 3:6, 0:3] = 0.0
    dJ_contact[:, 3:6, 3:6] = - skew_mat[:, 1, :, :]
    dJ_contact[:, 6:9, 0:3] = 0.0
    dJ_contact[:, 6:9, 3:6] = - skew_mat[:, 2, :, :]
    dJ_contact[:, 9:12, 0:3] = 0.0
    dJ_contact[:, 9:12, 3:6] = - skew_mat[:, 3, :, :]

    return J_contact, J_foot, dJ_contact, dJ_foot


def main():
    N = 1000
    runs = 100
    device = 'cuda'

    base_quat = quat_from_rpy((torch.rand((N, 3), device=device) - 0.5) * 0.2)  # (N, 4)
    base_rot_mat = quat_to_rot_mat(base_quat)  # (N, 3, 3)
    base_ang_vel = (torch.rand((N, 3), device=device) - 0.5) * 0.3  # (N, 3)
    q = (torch.tensor([[[-0.07125754, 1.29578570, -2.35611180, 0.16725932],
                        [-0.09827993, 0.95574780, -2.09038700, 0.17049010],
                        [-0.07868110, 1.40632810, -2.14806440, -0.17608170],
                        [-0.18465565, 1.39047780, -2.12072370, -0.16750483]]], device=device)
         + (torch.rand((N, 4, 4), device=device) - 0.5) * 0.05)  # (N, 4, 4)
    dq = (torch.tensor([[[0.24453469, 0.44326484, 0.09895979, 0.82018815],
                         [0.61054411, 0.65396317, 0.15327909, 0.62887442],
                         [0.55817806, 0.09751281, 0.69155223, 0.63302070],
                         [0.43529932, 0.83517665, 0.40806022, 0.53288333]]], device=device)
          + (torch.rand((N, 4, 4), device=device) - 0.5) * 0.05)  # (N, 4, 4)
    hip_pos_base_frame = torch.tensor([[+0.1881, -0.04675, 0.0],
                                       [+0.1881, +0.04675, 0.0],
                                       [-0.1881, -0.04675, 0.0],
                                       [-0.1881, +0.04675, 0.0]], device=device)

    """面向对象、class风格的Kinematics"""
    times = []
    for i in range(runs + 1):
        torch.cuda.synchronize()
        start = time.time()

        # 计算轮子触地点
        kin = WheeledGo1Kin(device=device)
        kin.position_level_forward_kinematics(q=q, q40=torch.zeros_like(q[:, :, 3]))
        q40 = kin.compute_q40(base_rot_mat[:, None, :, :].repeat(1, 4, 1, 1) @ kin.R4)
        kin.position_level_forward_kinematics(q=q, q40=q40)
        contact_pos_hip_frame = kin.p5

        # 计算J和dJ
        kin.velocity_level_forward_kinematics()
        kin.acceleration_level_forward_kinematics(dq)

        # 已知触地点, 求腿关节电机位置q1~q3
        p4 = contact_pos_hip_frame - kin.compute_contact_pos_minus_p4(contact_pos_hip_frame, base_rot_mat)
        q_from_IK = kin.position_level_inverse_kinematics_from_p4(p4)

        # 计算完整的jacobian
        J_contact, J_foot, dJ_contact, dJ_foot = kin.compute_whole_J_dJ(
            base_rot_mat, base_ang_vel, dq, hip_pos_base_frame)

        torch.cuda.synchronize()
        times.append(time.time() - start)
    print(sum(times[1:]) / runs)

    """纯函数式风格的Kinematics"""
    times = []
    for i in range(runs + 2):
        torch.cuda.synchronize()
        start = time.time()

        # 计算轮子触地点
        kin = WheeledGo1Kin(device=device)
        kin.p0, kin.p1, kin.p2, kin.p3, kin.p4, kin.p5, kin.R0, kin.R1, kin.R2, kin.R3, kin.R4 = (
            position_level_forward_kinematics(
                q=q, q40=torch.zeros_like(q[:, :, 3]),
                l1=kin.l1, l2=kin.l2, l3=kin.l3, d1=kin.d1, d2=kin.d2, d3=kin.d3, r=kin.r))
        q40 = compute_q40(base_rot_mat[:, None, :, :].repeat(1, 4, 1, 1) @ kin.R4)
        kin.p0, kin.p1, kin.p2, kin.p3, kin.p4, kin.p5, kin.R0, kin.R1, kin.R2, kin.R3, kin.R4 = (
            position_level_forward_kinematics(
                q=q, q40=q40, l1=kin.l1, l2=kin.l2, l3=kin.l3, d1=kin.d1, d2=kin.d2, d3=kin.d3, r=kin.r))
        contact_pos_hip_frame = kin.p5

        # 计算J和dJ
        kin.J1, kin.J2, kin.J3, kin.J4, kin.J5 = velocity_level_forward_kinematics(
            p0=kin.p0, p1=kin.p1, p2=kin.p2, p3=kin.p3, p4=kin.p4, p5=kin.p5, R1=kin.R1, R2=kin.R2, R3=kin.R3,
            R4=kin.R4)
        kin.dJ1, kin.dJ2, kin.dJ3, kin.dJ4, kin.dJ5 = acceleration_level_forward_kinematics(
            p0=kin.p0, p1=kin.p1, p2=kin.p2, p3=kin.p3, p4=kin.p4, p5=kin.p5, R1=kin.R1, R2=kin.R2, R3=kin.R3,
            R4=kin.R4,
            J1=kin.J1, J2=kin.J2, J3=kin.J3, J4=kin.J4, J5=kin.J5, dq=dq)

        # 已知触地点, 求腿关节电机位置q1~q3
        p4 = contact_pos_hip_frame - compute_contact_pos_minus_p4(
            contact_pos=contact_pos_hip_frame, base_rot_mat=base_rot_mat, l1=kin.l1, d3=kin.d3, r=kin.r)
        q_from_IK = position_level_inverse_kinematics_from_p4(
            p4=p4, l1=kin.l1, l2=kin.l2, l3=kin.l3, d1=kin.d1, d2=kin.d2, d3=kin.d3)

        # 计算完整的jacobian
        J_contact, J_foot, dJ_contact, dJ_foot = compute_whole_J_dJ(
            base_rot_mat=base_rot_mat, base_ang_vel=base_ang_vel, dq=dq, hip_pos_base_frame=hip_pos_base_frame,
            p0=kin.p0, p1=kin.p1, p2=kin.p2, p3=kin.p3, p4=kin.p4, p5=kin.p5,
            J1=kin.J1, J2=kin.J2, J3=kin.J3, J4=kin.J4, J5=kin.J5,
            dJ1=kin.dJ1, dJ2=kin.dJ2, dJ3=kin.dJ3, dJ4=kin.dJ4, dJ5=kin.dJ5)

        torch.cuda.synchronize()
        times.append(time.time() - start)
    print(sum(times[2:]) / runs)


if __name__ == '__main__':
    main()
