#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023-08-28 10:16 p.m.
# @Author  : shangfeng
# @Organization: University of Calgary
# @File    : wireframe_util.py
# @IDE     : PyCharm
import torch
import numpy as np
from scipy.spatial.distance import cdist


def roty_batch_tensor(t):
    input_shape = t.shape
    output = torch.zeros(
        tuple(list(input_shape) + [3, 3]), dtype=torch.float32, device=t.device
    )
    c = torch.cos(t)
    s = torch.sin(t)
    output[..., 0, 0] = c
    output[..., 0, 2] = s
    output[..., 1, 1] = 1
    output[..., 2, 0] = -s
    output[..., 2, 2] = c
    return output


def parametrization_to_wireframe_tensor(center, size, direction_cls_class):
    assert isinstance(center, torch.Tensor)
    assert isinstance(size, torch.Tensor)
    assert isinstance(direction_cls_class, torch.Tensor)

    reshape_final = False
    if direction_cls_class.ndim == 2:
        assert size.ndim == 3
        assert center.ndim == 3
        bsize = size.shape[0]
        nprop = size.shape[1]
        size = size.reshape(-1, size.shape[-1])
        direction_cls_class = direction_cls_class.reshape(-1)
        center = center.reshape(-1, 3)
        reshape_final = True

    input_shape = direction_cls_class.shape
    l = torch.unsqueeze(size[..., 0], -1)
    w = torch.unsqueeze(size[..., 1], -1)
    h = torch.unsqueeze(size[..., 2], -1)
    lines_3d = torch.zeros(
        tuple(list(input_shape) + [2, 3]), device=size.device, dtype=torch.float32
    )
    x = torch.cat((l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2), -1)
    y = torch.cat((w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2), -1)
    z = torch.cat((h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2), -1)
    lines_3d[..., :, 0] = torch.stack((x[torch.arange(x.size(0)), direction_cls_class],
                                       x[torch.arange(x.size(0)), (direction_cls_class + 2) % 4 + 4]), dim=1)
    lines_3d[..., :, 1] = torch.stack((y[torch.arange(y.size(0)), direction_cls_class],
                                       y[torch.arange(y.size(0)), (direction_cls_class + 2) % 4 + 4]), dim=1)
    lines_3d[..., :, 2] = torch.stack((z[torch.arange(z.size(0)), direction_cls_class],
                                       z[torch.arange(z.size(0)), (direction_cls_class + 2) % 4 + 4]), dim=1)

    lines_3d = lines_3d + torch.unsqueeze(center, -2)
    if reshape_final:
        lines_3d = lines_3d.reshape(bsize, nprop, 2, 3)
    return lines_3d


def parametrization_to_wireframe(size, center, direction_cls_class):
    """box_size: [x1,x2,...,xn,3]
        angle: [x1,x2,...,xn]
        center: [x1,x2,...,xn,3]
    Return:
        [x1,x3,...,xn,8,3]
    """
    input_shape = direction_cls_class.shape
    l = np.expand_dims(size[..., 0], -1)  # [x1,...,xn,1]
    w = np.expand_dims(size[..., 1], -1)
    h = np.expand_dims(size[..., 2], -1)
    lines_3d = np.zeros(tuple(list(input_shape) + [2, 3]))
    x = np.concatenate((l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2), -1)
    y = np.concatenate((w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2), -1)
    z = np.concatenate((h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2), -1)
    lines_3d[:, :, 0] = np.stack((x[np.arange(x.shape[0]), direction_cls_class],
                                  x[np.arange(x.shape[0]), (direction_cls_class + 2) % 4 + 4]), axis=1)
    lines_3d[:, :, 1] = np.stack((y[np.arange(y.shape[0]), direction_cls_class],
                                  y[np.arange(y.shape[0]), (direction_cls_class + 2) % 4 + 4]), axis=1)
    lines_3d[:, :, 2] = np.stack((z[np.arange(z.shape[0]), direction_cls_class],
                                  z[np.arange(z.shape[0]), (direction_cls_class + 2) % 4 + 4]), axis=1)
    lines_3d += np.expand_dims(center, -2)
    return lines_3d


def hausdorff_distance_line_tensor(p_line: torch.Tensor, t_line: torch.Tensor, sample_points=20):
    r"""
    :param p_line: torch Tensor (B, N, 2, 3), here N is number queries
    :param t_line: torch Tensor (B, M, 6), here M is the number of gt lines
    :param sample_points: int, sample sample_points from start points to end points and include them
    :return: B x N x M matrix
    """
    assert isinstance(p_line, torch.Tensor)
    assert isinstance(t_line, torch.Tensor)

    bsize = p_line.size(0)
    N, M = p_line.size(1), t_line.size(1)

    # Sample Points
    if t_line.dim() == 3:
        t_line = t_line.view(bsize, M, 2, 3)
    all_lines = torch.cat((p_line, t_line), dim=1)
    weights = torch.linspace(0, 1, sample_points, device=all_lines.device).view(1, 1, sample_points, 1)
    # B, N+M, sample_points, 3
    all_points = all_lines[:, :, 0, :].unsqueeze(2) + weights * (
            all_lines[:, :, 1, :].unsqueeze(2) - all_lines[:, :, 0, :].unsqueeze(2))

    # Calculate Hausdorff distance
    distance_matrix = torch.cdist(all_points[:, :N, :, :].view(bsize, -1, 3),
                                  all_points[:, N:N + M, :, :].view(bsize, -1, 3), p=2)  # p=2 means Euclidean Distance
    distance_matrix = distance_matrix.view(bsize, N, sample_points, M, sample_points).transpose(2, 3)
    h_pt_value = distance_matrix.min(-1)[0].max(-1, keepdim=True)[0]  # h(Prediction, Target)
    h_tp_value = distance_matrix.min(-2)[0].max(-1, keepdim=True)[0]  # h(Target, Prediction)
    hausdorff_matrix = torch.cat((h_pt_value, h_tp_value), dim=-1)
    hausdorff_matrix = hausdorff_matrix.max(-1)[0]

    return hausdorff_matrix


def hausdorff_distance_line(p_line, t_line, sample_points=20):
    r"""
    :param p_line: (N, 2, 3), here N is number queries
    :param t_line: (M, 2, 3), here M is the number of gt lines
    :param sample_points: int, sample sample_points from start points to end points and include them
    :return: N x M matrix
    """
    N, M = p_line.shape[0], t_line.shape[0]
    if N == 0:
        return np.array([])

    # Sample Points
    all_lines = np.concatenate((p_line, t_line), axis=0)
    weights = np.linspace(0, 1, sample_points).reshape(1, sample_points, 1)
    # N+M, sample_points, 3
    all_points = all_lines[:, 0, :][:, np.newaxis, :] + weights * (
            all_lines[:, 1, :][:, np.newaxis, :] - all_lines[:, 0, :][:, np.newaxis, :])

    # Calculate Hausdorff distance
    distance_matrix = cdist(all_points[:N, :, :].reshape(-1, 3), all_points[N:N + M, :, :].reshape(-1, 3),
                            'euclidean')  # p=2 means Euclidean Distance
    distance_matrix = distance_matrix.reshape(N, sample_points, M, sample_points)
    distance_matrix = np.transpose(distance_matrix, axes=(0, 2, 1, 3))
    h_pt_value = distance_matrix.min(-1).max(-1, keepdims=True)  # h(Prediction, Target)
    h_tp_value = distance_matrix.min(-2).max(-1, keepdims=True)  # h(Target, Prediction)
    hausdorff_matrix = np.concatenate((h_pt_value, h_tp_value), axis=-1)
    hausdorff_matrix = hausdorff_matrix.max(-1)

    return hausdorff_matrix


def line_length_similarity_tensor(p_size, t_size):
    r"""
    :param p_size: torch Tensor (B, N, 3), here N is number queries
    :param t_size: torch Tensor (B, M, 3), here M is the number of gt lines
    :return: B x N x M matrix
    """
    assert isinstance(p_size, torch.Tensor)
    assert isinstance(t_size, torch.Tensor)

    p_length = torch.sqrt(torch.sum(p_size ** 2, dim=-1)).unsqueeze(-1)
    t_length = torch.sqrt(torch.sum(t_size ** 2, dim=-1)).unsqueeze(1)

    length_similarity = torch.min(p_length, t_length) / torch.max(p_length, t_length)

    return 1 - length_similarity


def line_cosine_similarity(p_line, t_line):
    r"""
    Calculate the cosine similarity between lines
    :param p_line: torch Tensor (B, N, 2, 3), here N is number queries
    :param t_line: torch Tensor (B, M, 6), here M is the number of gt lines
    :return: B x N x M matrix
    """
    assert isinstance(p_line, torch.Tensor)
    assert isinstance(t_line, torch.Tensor)

    bsize, M, _ = t_line.size()

    p_vector = p_line[..., 1, :] - p_line[..., 0, :]
    t_vector = t_line[..., 0:3] - t_line[..., 3:6]
    dot_product = torch.einsum('bni,bmi->bnm', p_vector, t_vector)
    p_norm = torch.norm(p_vector, dim=-1)
    t_norm = torch.norm(t_vector, dim=-1)

    epsilon = torch.tensor(1e-6, dtype=p_norm.dtype, device=p_norm.device)
    p_norm = torch.where(p_norm < epsilon, epsilon, p_norm)
    t_norm = torch.where(t_norm < epsilon, epsilon, t_norm)

    similarity = abs(dot_product / (p_norm.unsqueeze(2) * t_norm.unsqueeze(1)))

    return 1 - similarity


def nms_3d_lines_semantic(predicted_line_corners, predicted_confident_prob, predicted_line_class, nms_threshold):
    hausdorff_mat = hausdorff_distance_line_tensor(predicted_line_corners.unsqueeze(0),
                                                   predicted_line_corners.unsqueeze(0))
    hausdorff_mat = hausdorff_mat.squeeze(0).cpu().detach().numpy()
    predicted_confident_prob = predicted_confident_prob.cpu().detach().numpy()
    predicted_line_class = predicted_line_class.cpu().detach().numpy()
    I = np.argsort(predicted_confident_prob)

    mask_ind = []
    while I.size != 0:
        last = I.size
        i = I[-1]
        cls1 = predicted_line_class[i]
        cls2 = predicted_line_class[I[: last - 1]]
        mask_ind.append(i)

        h_mat = hausdorff_mat[i, I[: last - 1]]
        h_mat[~(cls1 == cls2)] = 1

        I = np.delete(I, np.concatenate(([last - 1], np.where(h_mat <= nms_threshold)[0])))

    return mask_ind
