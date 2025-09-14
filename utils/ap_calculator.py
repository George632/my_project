#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023-09-05 2:46 p.m.
# @Author  : shangfeng
# @Organization: University of Calgary
# @File    : ap_calculator.py
# @IDE     : PyCharm

import numpy as np
from utils.wireframe_util import nms_3d_lines_semantic
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from collections import OrderedDict
from utils.wireframe_util import hausdorff_distance_line
from datasets.building3d import save_wireframe_model


class APCalculator(object):
    r"""
    Calculating Average Precision
    """

    def __init__(self, distance_thresh=0.1, nms_thresh=0.1, confidence_thresh=0.7, eps=0.05, min_samples=2):
        r"""
        :param distance_thresh: the distance thresh
        :param confidence_thresh: the edges confident thresh
        """
        self.distance_thresh = distance_thresh
        self.nms_thresh = nms_thresh
        self.confidence_thresh = confidence_thresh
        self.eps = eps
        self.min_samples = min_samples
        self.gt_map_cls = []
        self.pred_map_cls = []
        self.max_distance = np.empty((0, 1))
        self.centroid = np.empty((0, 3))
        self.scan_idx = np.empty((0, 1))
        self.ap_dict = OrderedDict()
        self.reset()
        # self.ap_dict = {'tp_corners': 0, 'num_pred_corners': 0, 'num_label_corners': 0, 'distance': 0, 'tp_edges': 0,
        #                 'num_pred_edges': 0, 'num_label_edges': 0, 'average_corner_offset': 0, 'corners_precision': 0,
        #                 'corners_recall': 0, 'corners_f1': 0, 'edges_precision': 0, 'edges_recall': 0, 'edges_f1': 0}

    def make_gt_list(self, gt_lines_corners, gt_line_numbers):
        batch_gt_map_cls = []
        bsize = gt_lines_corners.shape[0]
        for i in range(bsize):
            batch_gt_map_cls.append(
                gt_lines_corners[i, :gt_line_numbers[i]].reshape(-1, 2, 3)
            )
        return batch_gt_map_cls

    def parse_predictions(self, predicted_line_corners, predicted_confident_prob, predicted_line_class):
        r"""
        :param predicted_line_corners: B, N, 2, 3
        :param predicted_confident_prob: B, N
        :param predicted_line_class: B, N

        bsize: batch size
        pred_mask: B, N
        :return:
        """
        predicted_confident_prob = predicted_confident_prob.squeeze(-1)
        bsize, N = predicted_confident_prob.size()
        pred_mask = np.zeros((bsize, N))
        for i in range(bsize):
            mask_ind = nms_3d_lines_semantic(predicted_line_corners[i], predicted_confident_prob[i],
                                             predicted_line_class[i], self.nms_thresh)
            assert len(pred_mask) > 0
            pred_mask[i, mask_ind] = 1

        predicted_line_corners = predicted_line_corners.cpu().detach().numpy()
        predicted_confident_prob = predicted_confident_prob.cpu().detach().numpy()

        batch_pred_map_cls = []
        for i in range(bsize):
            batch_pred_map_cls.append(
                np.array([
                    predicted_line_corners[i, j]
                    for j in range(N)
                    if pred_mask[i, j] == 1
                       and predicted_confident_prob[i, j] >= self.confidence_thresh
                ])
            )
        return batch_pred_map_cls

    def step_meter(self, outputs, targets):
        if "outputs" in outputs:
            outputs = outputs["outputs"]
        batch_pred_map_cls, batch_gt_map_cls = self.step(
            predicted_line_corners=outputs["lines"],
            predicted_confident_prob=outputs["confident_prob"],
            predicted_line_class=outputs["direction_cls_class"],
            gt_line_corners=targets["wf_edges_points"],
            gt_line_numbers=targets["wf_line_number"],
            gt_vertices=targets['wf_vertices'],
            gt_lines=targets['wf_edges']
        )

        self.accumulate(batch_pred_map_cls, batch_gt_map_cls, targets['max_distance'], targets['centroid'],
                        targets['scan_idx'])

    def step(
            self,
            predicted_line_corners,
            predicted_confident_prob,
            predicted_line_class,
            gt_line_corners,
            gt_line_numbers,
            gt_vertices,
            gt_lines,
    ):
        r"""
        :param predicted_line_corners: B, N, 2, 3
        :param predicted_confident_prob: B, N
        :param gt_line_corners: B, ngt, 6
        :param gt_line_numbers: B
        :param gt_vertices: B, n_vertices, 3
        :param gt_lines: B, ngt, 2
        :return:
        """
        gt_lines_corners = gt_line_corners.cpu().detach().numpy()
        gt_line_numbers = gt_line_numbers.cpu().detach().numpy()
        batch_gt_map_cls = self.make_gt_list(
            gt_lines_corners, gt_line_numbers
        )  # get line corners without any padding items

        batch_pred_map_cls = self.parse_predictions(
            predicted_line_corners,
            predicted_confident_prob,
            predicted_line_class
        )  # parse lines with line probability

        return batch_pred_map_cls, batch_gt_map_cls

    def accumulate(self, batch_pred_map_cls, batch_gt_map_cls, max_distance, centroid, scan_idx):
        bsize = len(batch_pred_map_cls)
        assert bsize == len(batch_gt_map_cls)
        self.max_distance = np.concatenate((self.max_distance, max_distance.cpu().detach().numpy()[:, np.newaxis]),
                                           axis=0)
        self.centroid = np.concatenate((self.centroid, centroid.cpu().detach().numpy()), axis=0)
        self.scan_idx = np.concatenate((self.scan_idx, scan_idx.cpu().detach().numpy()[:, np.newaxis]), axis=0).astype(
            np.int64)
        self.gt_map_cls.extend(batch_gt_map_cls)
        self.pred_map_cls.extend(batch_pred_map_cls)

    def computer_corners_metrics(self, p_corners, l_corners, max_distance, centroid, return_indices=False):
        # ----------------------- Corners Eval based on Hungarian Mather algorithms ---------------------------
        distance_matrix = cdist(p_corners, l_corners)
        predict_indices, label_indices = linear_sum_assignment(distance_matrix)
        mask = distance_matrix[predict_indices, label_indices] <= self.distance_thresh
        tp_corners_predict_indices, tp_corners_label_indices = predict_indices[mask], label_indices[mask]
        tp_corners = len(tp_corners_predict_indices)
        tp_fp_corners = len(p_corners)
        tp_fn_corners = len(l_corners)

        # ----------------------- Corners Average Corners Offset-----------------------------------------------
        distances = np.linalg.norm((p_corners[tp_corners_predict_indices] * max_distance + centroid) - (
                l_corners[tp_corners_label_indices] * max_distance + centroid), axis=1)
        distances = np.sum(distances)

        if not return_indices:
            return tp_corners, tp_fp_corners, tp_fn_corners, distances
        else:
            return tp_corners, tp_fp_corners, tp_fn_corners, distances, tp_corners_predict_indices, tp_corners_label_indices

    def computer_edges(self, edges, vertices):
        index = []
        for edge in edges:
            indices = []
            for point in edge:
                matching_indices = np.where((vertices == point).all(axis=1))[0]
                if len(matching_indices) > 0:
                    indices.append(matching_indices[0])
                else:
                    indices.append(-1)

            index.append(indices)

        return np.sort(np.array(index), axis=-1)

    def compute_metrics(self, return_ap_dict=False, save_wireframe=False):
        r"""
            :param save_wireframe:  save wireframe models
            :param return_ap_dict:
            :param batch: batch_size, predicted_corners, predicted_edges, predicted_score, wf_vertices, wf_edges, centroid,
            max_distance
            : test case
                batch_size = np.array([1])
                p_corners: n_pred_corners * 3, type = np.array
                l_corners: n_gt_corners * 3, type = np.array
                predicted_edges: n_pro * 2 * 3, type = np.array
                label_edges: n_gt * 2 * 3, type = np.array
                centroid = np.array([1, 2, 3])
                max_distance = np.array([[2]])
                predicted_score = np.array([[0.8, 0.8, 0.3, 1]]
            :return: AP Dict
        """
        if not self.pred_map_cls:
            return None
        batch_size = len(self.pred_map_cls)
        assert batch_size == len(self.gt_map_cls)
        for b in range(batch_size):
            label_edges = self.gt_map_cls[b]
            all_vertices = label_edges.reshape(-1, 3)
            l_corners = np.unique(all_vertices, axis=0)
            l_edges_index = self.computer_edges(label_edges, l_corners)

            precision_dict = OrderedDict()
            # ----------------------- Original Edge Precision and Recall---------------------------
            original_pred_edges = self.pred_map_cls[b]
            original_hausdorff_distance = hausdorff_distance_line(original_pred_edges, label_edges)
            if original_hausdorff_distance.size != 0:
                original_predict_indices, original_label_indices = linear_sum_assignment(original_hausdorff_distance)
                mask = original_hausdorff_distance[original_predict_indices, original_label_indices] <= 0.1
                original_tp_edges = sum(mask)
                original_tp_fp_edges = len(original_pred_edges)
                original_tp_fn_edges = len(l_edges_index)
            else:
                original_tp_edges, original_tp_fp_edges, original_tp_fn_edges = 0, 0, len(l_edges_index)
            precision_dict.update({
                'original_tp_edges': original_tp_edges,
                'original_tp_fp_edges': original_tp_fp_edges,
                'original_tp_fn_edges': original_tp_fn_edges,
            })

            for eps in [self.eps, self.eps + 0.05]:
                # ----------------------- DBSCAN to parse corners-----------------------------
                p_corners, pred_edges = dbscan_to_parse_corners(self.pred_map_cls[b], eps, self.min_samples)

                if p_corners.size != 0:
                    all_tp_corners, all_tp_fp_corners, all_tp_fn_corners, all_distances = \
                        self.computer_corners_metrics(p_corners, l_corners, self.max_distance[b], self.centroid[b])

                    if pred_edges.size != 0:
                        pred_edges_corners = pred_edges.reshape(-1, 3)
                        pred_edges_corners = np.unique(pred_edges_corners, axis=0)
                        p_edges_index = self.computer_edges(pred_edges, pred_edges_corners)
                        p_edges_index = np.unique(p_edges_index, axis=0)
                        if save_wireframe and eps == (self.eps+0.05):
                            save_wireframe_model(pred_edges_corners, p_edges_index, './results/' +str(self.scan_idx[b][0])+'.obj')
                            # print(self.scan_idx[b])
                        tp_corners, tp_fp_corners, tp_fn_corners, distances, predict_indices, label_indices \
                            = self.computer_corners_metrics(pred_edges_corners, l_corners, self.max_distance[b],
                                                            self.centroid[b], return_indices=True)

                        # ------------------------------- Edges Eval ------------------------------
                        if predict_indices.size != 0 and label_indices.size != 0:
                            corners_map = {key: value for key, value in zip(predict_indices, label_indices)}
                            for i, _ in enumerate(p_edges_index):
                                for j in range(2):
                                    p_edges_index[i, j] = corners_map[p_edges_index[i, j]] if p_edges_index[
                                                                                                  i, j] in corners_map else -1
                                p_edges_index[i] = sorted(p_edges_index[i])

                            tp_edges = np.sum([np.any(np.all(e == l_edges_index, axis=1)) for e in p_edges_index])
                            tp_fp_edges = len(p_edges_index)
                            tp_fn_edges = len(l_edges_index)
                        else:
                            tp_edges, tp_fp_edges, tp_fn_edges = 0, 0, len(l_edges_index)

                    else:
                        tp_corners, tp_fp_corners, tp_fn_corners, distances = 0, 0, len(l_corners), 0
                        tp_edges, tp_fp_edges, tp_fn_edges = 0, 0, len(l_edges_index)

                else:
                    all_tp_corners, all_tp_fp_corners, all_tp_fn_corners, all_distances = 0, 0, len(l_corners), 0
                    tp_corners, tp_fp_corners, tp_fn_corners, distances = 0, 0, len(l_corners), 0
                    tp_edges, tp_fp_edges, tp_fn_edges = 0, 0, len(l_edges_index)

                precision_dict.update({'%s_all_tp_corners' % eps: all_tp_corners,
                                       '%s_all_tp_fp_corners' % eps: all_tp_fp_corners,
                                       '%s_all_tp_fn_corners' % eps: all_tp_fn_corners,
                                       '%s_all_distances' % eps: all_distances,
                                       '%s_tp_edges' % eps: tp_edges,
                                       '%s_tp_fp_edges' % eps: tp_fp_edges,
                                       '%s_tp_fn_edges' % eps: tp_fn_edges,
                                       '%s_tp_corners' % eps: tp_corners,
                                       '%s_tp_fp_corners' % eps: tp_fp_corners,
                                       '%s_tp_fn_corners' % eps: tp_fn_corners,
                                       '%s_distances' % eps: distances})

            for key, value in precision_dict.items():
                if key in self.ap_dict:
                    self.ap_dict[key] += value
                else:
                    self.ap_dict[key] = value

        # ------------------------------------------ All Edges ---------------------------------------------
        self.ap_dict['original_edges_precision'] = self.ap_dict['original_tp_edges'] / \
                                                   (self.ap_dict['original_tp_fp_edges'] + 1e-6)
        self.ap_dict['original_edges_recall'] = self.ap_dict['original_tp_edges'] / \
                                                (self.ap_dict['original_tp_fn_edges'] + 1e-6)
        self.ap_dict['original_edges_f1'] = 2 * self.ap_dict['original_edges_precision'] * \
                                            self.ap_dict['original_edges_recall'] / (
                                                    self.ap_dict['original_edges_precision'] +
                                                    self.ap_dict['original_edges_recall'] + 1e-6)

        for eps in [self.eps, self.eps + 0.05]:
            # ------------------------------------------ All Corners ---------------------------------------------
            self.ap_dict['%s_all_average_corner_offset' % eps] = self.ap_dict['%s_all_distances' % eps] / \
                                                                 (self.ap_dict['%s_all_tp_corners' % eps] + 1e-6)
            self.ap_dict['%s_all_corners_precision' % eps] = self.ap_dict['%s_all_tp_corners' % eps] / \
                                                             (self.ap_dict['%s_all_tp_fp_corners' % eps] + 1e-6)
            self.ap_dict['%s_all_corners_recall' % eps] = self.ap_dict['%s_all_tp_corners' % eps] / \
                                                          (self.ap_dict['%s_all_tp_fn_corners' % eps] + 1e-6)
            self.ap_dict['%s_all_corners_f1' % eps] = 2 * self.ap_dict['%s_all_corners_precision' % eps] * \
                                                      self.ap_dict['%s_all_corners_recall' % eps] / (
                                                              self.ap_dict['%s_all_corners_precision' % eps] +
                                                              self.ap_dict['%s_all_corners_recall' % eps] + 1e-6)

            # ------------------------------------------ TP Corners -------------------------------------------
            self.ap_dict['%s_average_corner_offset' % eps] = self.ap_dict['%s_distances' % eps] / \
                                                             (self.ap_dict['%s_tp_corners' % eps] + 1e-6)
            self.ap_dict['%s_corners_precision' % eps] = self.ap_dict['%s_tp_corners' % eps] / \
                                                         (self.ap_dict['%s_tp_fp_corners' % eps] + 1e-6)
            self.ap_dict['%s_corners_recall' % eps] = self.ap_dict['%s_tp_corners' % eps] / \
                                                      (self.ap_dict['%s_tp_fn_corners' % eps] + 1e-6)
            self.ap_dict['%s_corners_f1' % eps] = 2 * self.ap_dict['%s_corners_precision' % eps] * \
                                                  self.ap_dict['%s_corners_recall' % eps] / (
                                                          self.ap_dict['%s_corners_precision' % eps] +
                                                          self.ap_dict['%s_corners_recall' % eps] + 1e-6)

            # ------------------------------------------ Edge Corners -------------------------------------------
            self.ap_dict['%s_edges_precision' % eps] = self.ap_dict['%s_tp_edges' % eps] / \
                                                       (self.ap_dict['%s_tp_fp_edges' % eps] + 1e-6)
            self.ap_dict['%s_edges_recall' % eps] = self.ap_dict['%s_tp_edges' % eps] / \
                                                    (self.ap_dict['%s_tp_fn_edges' % eps] + 1e-6)
            self.ap_dict['%s_edges_f1' % eps] = 2 * self.ap_dict['%s_edges_precision' % eps] * \
                                                self.ap_dict['%s_edges_recall' % eps] / (
                                                        self.ap_dict['%s_edges_precision' % eps] +
                                                        self.ap_dict['%s_edges_recall' % eps] + 1e-6)
        if return_ap_dict:
            return self.ap_dict

    def metrics_to_str(self, best_ap_dict=None):
        if not self.pred_map_cls:
            return None
        if best_ap_dict is not None:
            self.ap_dict = best_ap_dict
        ap_str = "Accuracy and Precision \n"
        ap_str += '-------------------------- All Edges ------------------------------\n'
        ap_str += 'original_edges_precision:       ' + str(self.ap_dict['original_edges_precision']) + '\n'
        ap_str += 'original_edges_recall:          ' + str(self.ap_dict['original_edges_recall']) + '\n'
        ap_str += 'original_edges_f1:              ' + str(self.ap_dict['original_edges_f1']) + '\n'
        ap_str += '\n'

        for eps in [self.eps, self.eps + 0.05]:
            ap_str += '-------------------------- ' + str(eps) + ' ------------------------------\n'
            ap_str += '%s_all_average_corner_offset:   ' % eps + str(self.ap_dict['%s_all_average_corner_offset' % eps]) + '\n'
            ap_str += '%s_all_corners_precision:       ' % eps + str(self.ap_dict['%s_all_corners_precision' % eps]) + '\n'
            ap_str += '%s_all_corners_recall:          ' % eps + str(self.ap_dict['%s_all_corners_recall' % eps]) + '\n'
            ap_str += '%s_all_corners_f1:              ' % eps + str(self.ap_dict['%s_all_corners_f1' % eps]) + '\n'

            ap_str += '\n'
            ap_str += '%s_average_corner_offset:   ' % eps + str(self.ap_dict['%s_average_corner_offset' % eps]) + '\n'
            ap_str += '%s_corners_precision:       ' % eps + str(self.ap_dict['%s_corners_precision' % eps]) + '\n'
            ap_str += '%s_corners_recall:          ' % eps + str(self.ap_dict['%s_corners_recall' % eps]) + '\n'
            ap_str += '%s_corners_f1:              ' % eps + str(self.ap_dict['%s_corners_f1' % eps]) + '\n'

            ap_str += '\n'
            ap_str += '%s_edges_precision:       ' % eps + str(self.ap_dict['%s_edges_precision' % eps]) + '\n'
            ap_str += '%s_edges_recall:          ' % eps + str(self.ap_dict['%s_edges_recall' % eps]) + '\n'
            ap_str += '%s_edges_f1:              ' % eps + str(self.ap_dict['%s_edges_f1' % eps]) + '\n'
            ap_str += '\n'

        return ap_str

    def metrics_to_dict(self):
        metrics_dict = {}
        if not self.pred_map_cls:
            return metrics_dict
        metrics_dict['original_edges_precision'] = self.ap_dict['original_edges_precision'] * 100
        metrics_dict['original_edges_recall'] = self.ap_dict['original_edges_recall'] * 100
        metrics_dict['original_edges_f1'] = self.ap_dict['original_edges_f1'] * 100
        for eps in [self.eps, self.eps + 0.05]:
            metrics_dict['%s_all_average_corner_offset' % eps] = self.ap_dict[
                                                                     '%s_all_average_corner_offset' % eps] * 100
            metrics_dict['%s_all_corners_precision' % eps] = self.ap_dict['%s_all_corners_precision' % eps] * 100
            metrics_dict['%s_all_corners_recall' % eps] = self.ap_dict['%s_all_corners_recall' % eps] * 100
            metrics_dict['%s_all_corners_f1' % eps] = self.ap_dict['%s_all_corners_f1' % eps] * 100

            metrics_dict['%s_average_corner_offset' % eps] = self.ap_dict['%s_average_corner_offset' % eps] * 100
            metrics_dict['%s_corners_precision' % eps] = self.ap_dict['%s_corners_precision' % eps] * 100
            metrics_dict['%s_corners_recall' % eps] = self.ap_dict['%s_corners_recall' % eps] * 100
            metrics_dict['%s_corners_f1' % eps] = self.ap_dict['%s_corners_f1' % eps] * 100

            metrics_dict['%s_edges_precision' % eps] = self.ap_dict['%s_edges_precision' % eps] * 100
            metrics_dict['%s_edges_recall' % eps] = self.ap_dict['%s_edges_recall' % eps] * 100
            metrics_dict['%s_edges_f1' % eps] = self.ap_dict['%s_edges_f1' % eps] * 100

        return metrics_dict

    def reset(self):
        self.pred_map_cls = []
        self.gt_map_cls = []
        self.max_distance = np.empty((0, 1))
        self.centroid = np.empty((0, 3))
        self.scan_idx = np.empty((0, 1))
        self.ap_dict = OrderedDict()


def dbscan_to_parse_corners(pred_edges, eps, min_samples):
    npoints = pred_edges.shape[0]
    if npoints == 0:
        return np.array([]), np.array([])
    pred_vertices = pred_edges.reshape(-1, 3)
    pred_vertices_copy = np.copy(pred_vertices)

    # get cluster center
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(pred_vertices)
    unique_labels = set(labels) - {-1}
    cluster_centers = []

    for label in unique_labels:
        cluster_mask = (labels == label)
        cluster_vertices = pred_vertices[cluster_mask]
        cluster_center = np.mean(cluster_vertices, axis=0)
        cluster_centers.append(cluster_center)
        pred_vertices_copy[cluster_mask] = cluster_center

    # generate new edges
    line_label = labels.reshape(npoints, 2)
    line_mask = np.any(line_label == -1, axis=1)
    pred_vertices_copy = pred_vertices_copy.reshape(npoints, 2, -1)
    edges_vertices = pred_vertices_copy[~line_mask]
    # edges_vertices = edges_vertices.reshape(-1, 2, 3)

    cluster_centers = np.array(cluster_centers)
    return cluster_centers, edges_vertices
