# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.dist import all_reduce_average
from scipy.optimize import linear_sum_assignment
from utils.wireframe_util import hausdorff_distance_line_tensor, line_cosine_similarity, line_length_similarity_tensor


class Matcher(nn.Module):
    def __init__(self, cost_hausdorff_distance, cost_cosine, cost_length, cost_confidence, cost_center):
        r"""
        :param cost_hausdorff_distance:
        :param cost_cosine:
        :param cost_confidence:
        :param cost_center:
        """
        super().__init__()
        self.cost_hausdorff_distance = cost_hausdorff_distance  # 2
        self.cost_cosine = cost_cosine  # 1
        self.cost_length = cost_length # 1
        self.cost_confidence = cost_confidence  # 0
        self.cost_center = cost_center  # 0

    @torch.no_grad()
    def forward(self, outputs, targets):

        batchsize = outputs['confident_scores'].shape[0]
        nqueries = outputs['confident_scores'].shape[1]
        ngt = targets['wf_line_class'].shape[1]
        nactual_gt = targets['wf_line_number']

        # line hausdorff_distance_line cost
        hausdorff_mat = hausdorff_distance_line_tensor(outputs['lines'], targets['wf_edges_points'])
        outputs['hausdorff'] = hausdorff_mat

        # line cosine similarity
        cosine_mat = line_cosine_similarity(outputs['lines'], targets['wf_edges_points'])

        # line length similarity
        length_mat = line_length_similarity_tensor(outputs['size'], targets['wf_sizes'])

        # confident cost: batch x nqueries x 1
        confident_mat = -torch.sigmoid(outputs["confident_scores"])

        # center cost: batch x nqueries x ngt
        center_mat = outputs["center_dist"].detach()

        final_cost = (
                self.cost_hausdorff_distance * hausdorff_mat
                + self.cost_cosine * cosine_mat
                + self.cost_confidence * confident_mat
                + self.cost_center * center_mat
                + self.cost_length * length_mat
        )

        final_cost = final_cost.detach().cpu().numpy()
        assignments = []

        # auxiliary variables useful for batched loss computation
        batch_size, nprop = final_cost.shape[0], final_cost.shape[1]
        per_prop_gt_inds = torch.zeros(
            [batch_size, nprop], dtype=torch.int64, device=hausdorff_mat.device
        )
        proposal_matched_mask = torch.zeros(
            [batch_size, nprop], dtype=torch.float32, device=hausdorff_mat.device
        )
        confident_matched_mask = torch.zeros(
            [batch_size, nprop], dtype=torch.float32, device=hausdorff_mat.device
        )
        for b in range(batchsize):
            assign = []
            if nactual_gt[b] > 0:
                assign = linear_sum_assignment(final_cost[b, :, : nactual_gt[b]])
                assign = [
                    torch.from_numpy(x).long().to(device=hausdorff_mat.device)
                    for x in assign
                ]
                per_prop_gt_inds[b, assign[0]] = assign[1]
                proposal_matched_mask[b, assign[0]] = 1

                # generate confident match mask
                confidence_prob = np.clip(1 - np.min(final_cost[b, :, : nactual_gt[b]], axis=-1), a_min=0, a_max=1)
                confident_matched_mask[b] = torch.from_numpy(confidence_prob).to(hausdorff_mat.device)
            assignments.append(assign)

        return {
            "assignments": assignments,
            "per_prop_gt_inds": per_prop_gt_inds,
            "proposal_matched_mask": proposal_matched_mask,
            "confident_matched_mask": confident_matched_mask
        }, outputs


class SetCriterion(nn.Module):
    def __init__(self, cfg, matcher, loss_weight_dict):
        super().__init__()
        self.cfg = cfg
        self.matcher = matcher
        self.loss_weight_dict = loss_weight_dict

        # semcls_percls_weights = torch.ones(cfg.Model.RegLine.num_semcls + 1)
        # semcls_percls_weights[-1] = loss_weight_dict["loss_no_object_weight"]
        # del loss_weight_dict["loss_no_object_weight"]
        # self.register_buffer("semcls_percls_weights", semcls_percls_weights)

        self.loss_functions = {
            "loss_sem_cls": self.loss_confidence,
            "loss_center": self.loss_center,
            "loss_size": self.loss_size,
            "loss_directional_cls": self.loss_directional_cls,
            "loss_hausdorff_distance": self.loss_hausdorff_distance,
            # this isn't used during training and is logged for debugging.
            # thus, this loss does not have a loss_weight associated with it.
            "loss_cardinality": self.loss_cardinality,
        }

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, assignments):
        # Count the number of predictions that are objects
        # Cardinality is the error between predicted #objects and ground truth objects

        pred_logits = outputs["confident_scores"].squeeze(-1)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        pred_objects = (pred_logits.sigmoid() >= 0.5).sum(1)
        card_err = F.l1_loss(pred_objects.float(), targets["wf_line_number"])
        return {"loss_cardinality": card_err}

    def loss_confidence(self, outputs, targets, assignments):

        pred_logits = outputs['confident_scores'].squeeze(-1)
        gt_confidence = assignments["confident_matched_mask"]
        # weights = torch.where(gt_confidence == 1, torch.tensor(1.0, device=pred_logits.device),
        #                       torch.tensor(0.2, device=pred_logits.device))
        loss = F.binary_cross_entropy_with_logits(pred_logits, gt_confidence, reduction='mean')
        # loss = F.binary_cross_entropy()

        return {"loss_object_confidence": loss}

    def loss_directional_cls(self, outputs, targets, assignments):
        direction_logits = outputs["direction_cls"]

        if targets["num_lines_replica"] > 0:
            gt_direction_label = targets["wf_line_class"].squeeze(-1)

            gt_direction_label = torch.gather(
                gt_direction_label, 1, assignments["per_prop_gt_inds"]
            )
            direction_cls_loss = F.cross_entropy(
                direction_logits.transpose(2, 1), gt_direction_label, reduction="none"
            )
            direction_cls_loss = (
                    direction_cls_loss * assignments["proposal_matched_mask"]
            ).sum()

            direction_cls_loss /= targets["num_lines"]
        else:
            direction_cls_loss = torch.zeros(1, device=direction_logits.device).squeeze()
        return {"loss_directional_cls": direction_cls_loss}

    def loss_center(self, outputs, targets, assignments):
        center_dist = outputs["center_dist"]
        if targets["num_lines_replica"] > 0:
            center_loss = torch.gather(
                center_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
            ).squeeze(-1)
            # zero-out non-matched proposals
            center_loss = center_loss * assignments["proposal_matched_mask"]
            center_loss = center_loss.sum()

            if targets["num_lines"] > 0:
                center_loss /= targets["num_lines"]
        else:
            center_loss = torch.zeros(1, device=center_dist.device).squeeze()

        return {"loss_center": center_loss}

    def loss_hausdorff_distance(self, outputs, targets, assignments):
        hausdorff_dist = outputs["hausdorff"]

        hausdorff_loss = torch.gather(
            hausdorff_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
        ).squeeze(-1)
        # zero-out non-matched proposals
        hausdorff_loss = hausdorff_loss * assignments["proposal_matched_mask"]
        hausdorff_loss = hausdorff_loss.sum()

        if targets["num_lines"] > 0:
            hausdorff_loss /= targets["num_lines"]

        return {"loss_hausdorff_distance": hausdorff_loss}

    def loss_size(self, outputs, targets, assignments):
        gt_line_sizes = targets["wf_sizes"]
        pred_line_sizes = outputs["size"]

        if targets["num_lines_replica"] > 0:

            # construct gt_lines_sizes as [batch x nprop x 3] matrix by using proposal to gt matching
            gt_line_sizes = torch.stack(
                [
                    torch.gather(
                        gt_line_sizes[:, :, x], 1, assignments["per_prop_gt_inds"]
                    )
                    for x in range(gt_line_sizes.shape[-1])
                ],
                dim=-1,
            )
            size_loss = F.l1_loss(pred_line_sizes, gt_line_sizes, reduction="none").sum(
                dim=-1
            )

            # zero-out non-matched proposals
            size_loss *= assignments["proposal_matched_mask"]
            size_loss = size_loss.sum()

            size_loss /= targets["num_lines"]
        else:
            size_loss = torch.zeros(1, device=pred_line_sizes.device).squeeze()
        return {"loss_size": size_loss}

    def single_output_forward(self, outputs, targets):
        center_dist = torch.cdist(
            outputs["center_normalized"], targets["wf_centers"], p=1
        )
        outputs["center_dist"] = center_dist
        assignments, outputs = self.matcher(outputs, targets)

        losses = {}

        for k in self.loss_functions:
            loss_wt_key = k + "_weight"
            if (
                    loss_wt_key in self.loss_weight_dict
                    and self.loss_weight_dict[loss_wt_key] > 0
            ) or loss_wt_key not in self.loss_weight_dict:
                # only compute losses with loss_wt > 0
                # certain losses like cardinality are only logged and have no loss weight
                curr_loss = self.loss_functions[k](outputs, targets, assignments)
                losses.update(curr_loss)

        final_loss = 0
        for k in self.loss_weight_dict:
            if self.loss_weight_dict[k] > 0:
                losses[k.replace("_weight", "")] *= self.loss_weight_dict[k]
                final_loss += losses[k.replace("_weight", "")]
        return final_loss, losses

    def forward(self, outputs, targets):
        nactual_gt = targets['wf_line_number']
        num_boxes = torch.clamp(all_reduce_average(nactual_gt.sum()), min=1).item()
        targets["num_lines"] = num_boxes
        targets[
            "num_lines_replica"
        ] = nactual_gt.sum().item()  # number of lines on this worker for dist training

        loss, loss_dict = self.single_output_forward(outputs["outputs"], targets)

        if "aux_outputs" in outputs:
            for k in range(len(outputs["aux_outputs"])):
                interm_loss, interm_loss_dict = self.single_output_forward(
                    outputs["aux_outputs"][k], targets
                )

                loss += interm_loss
                for interm_key in interm_loss_dict:
                    loss_dict[f"{interm_key}_{k}"] = interm_loss_dict[interm_key]
        return loss, loss_dict


def build_criterion(cfg):
    matcher = Matcher(
        cost_hausdorff_distance=cfg.Loss.Matcher.matcher_hausdorff_distance_cost,
        cost_cosine=cfg.Loss.Matcher.matcher_consine_cost,
        cost_length=cfg.Loss.Matcher.matcher_length_cost,
        cost_center=cfg.Loss.Matcher.matcher_center_cost,
        cost_confidence=cfg.Loss.Matcher.matcher_confident_cost,
    )

    loss_weight_dict = {
        "loss_object_confidence_weight": cfg.Loss.Weights.loss_object_confidence_weight,
        "loss_directional_cls_weight": cfg.Loss.Weights.loss_directional_cls_weight,
        "loss_center_weight": cfg.Loss.Weights.loss_center_weight,
        "loss_size_weight": cfg.Loss.Weights.loss_size_weight,
        "loss_hausdorff_distance_weight": cfg.Loss.Weights.loss_hausdorff_distance_weight,
    }

    criterion = SetCriterion(cfg, matcher, loss_weight_dict)
    return criterion
