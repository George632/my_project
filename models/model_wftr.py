# Copyright (c) Facebook, Inc. and its affiliates.
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from third_party.pointnet2.pointnet2_utils import furthest_point_sample

from models.helpers import GenericMLP
from models.position_embedding import PositionEmbeddingCoordsSine
from models.transformer import (MaskedTransformerEncoder, TransformerDecoder,
                                TransformerDecoderLayer, TransformerEncoder,
                                TransformerEncoderLayer)
from utils.wireframe_util import parametrization_to_wireframe_tensor


class WireframeProcessor(object):
    """
    Class to convert 3DETR MLP head outputs into bounding boxes
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def compute_predicted_center(self, center_offset, query_xyz, centroid, max_distance):
        center_normalized = query_xyz + center_offset
        max_distance = max_distance.reshape(max_distance.shape[0], 1, 1)
        centroid = centroid.unsqueeze(1)
        center_unnormalized = center_normalized * max_distance + centroid
        return center_normalized, center_unnormalized

    def parametrization_to_wireframe(
            self, center_normalized, size, direction_cls_class
    ):
        return parametrization_to_wireframe_tensor(
            center_normalized, size, direction_cls_class
        )


class ModelWFTR(nn.Module):
    """
    Main WFTR model. Consists of the following learnable sub-models
    - pre_encoder: takes raw point cloud, subsamples it and projects into "D" dimensions
                Input is a Nx3 matrix of N point coordinates
                Output is a N'xD matrix of N' point features
    - encoder: series of self-attention blocks to extract point features
                Input is a N'xD matrix of N' point features
                Output is a N''xD matrix of N'' point features.
                N'' = N' for regular encoder; N'' = N'//2 for masked encoder
    - query computation: samples a set of B coordinates from the N'' points
                and outputs a BxD matrix of query features.
    - decoder: series of self-attention and cross-attention blocks to produce BxD box features
                Takes N''xD features from the encoder and BxD query features.
    - mlp_heads: Predicts bounding box parameters and classes from the BxD box features
    """

    def __init__(
            self,
            cfg,
            pre_encoder,
            encoder,
            decoder,
            encoder_dim=256,
            decoder_dim=256,
            position_embedding="fourier",
            mlp_dropout=0.3,
            num_queries=256,
    ):
        super().__init__()
        self.cfg = cfg
        self.pre_encoder = pre_encoder
        self.encoder = encoder
        # hasattr: whether the encoder model includes the masking_radius submodule
        if hasattr(self.encoder, "masking_radius"):
            hidden_dims = [encoder_dim]
        else:
            hidden_dims = [encoder_dim, encoder_dim]
        # non-linear projections
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=hidden_dims,
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )
        # positional embedding
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim, pos_type=position_embedding, normalize=False
        )

        # seed points non-parametric query embedding
        self.query_projection = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )
        self.decoder = decoder
        # mlp bounding box heads
        self.build_mlp_heads(self.cfg, decoder_dim, mlp_dropout)

        self.num_queries = num_queries
        self.wireframe_processor = WireframeProcessor(self.cfg)

    def build_mlp_heads(self, cfg, decoder_dim, mlp_dropout):
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )

        # Semantic class of the box
        # add 1 for background/not-an-object class
        confidence_scores = mlp_func(output_dim=1)

        # geometry of the box
        center_offset = mlp_func(output_dim=3)
        size_lwh = mlp_func(output_dim=3)
        direction_cls = mlp_func(output_dim=cfg.Model.RegLine.direction_cls)

        mlp_heads = [
            ("confidence_scores", confidence_scores),
            ("center_offset", center_offset),
            ("size_lwh", size_lwh),
            ("direction_cls", direction_cls),
        ]
        self.mlp_heads = nn.ModuleDict(mlp_heads)

    def get_query_embeddings(self, encoder_xyz, point_cloud_dims):
        query_inds = furthest_point_sample(encoder_xyz, self.num_queries)
        query_inds = query_inds.long()
        query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds) for x in range(3)]
        query_xyz = torch.stack(query_xyz)
        query_xyz = query_xyz.permute(1, 2, 0)

        # Gater op above can be replaced by the three lines below from the pointnet2 codebase
        # xyz_flipped = encoder_xyz.transpose(1, 2).contiguous()
        # query_xyz = gather_operation(xyz_flipped, query_inds.int())
        # query_xyz = query_xyz.transpose(1, 2)
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        # pos_embed: (B, 256, num_queries)
        # query_embed: (B, 256, num_queries)
        return query_xyz, query_embed

    def _break_up_pc(self, pc):
        # pc may contain color/normals.

        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def run_encoder(self, point_clouds):
        xyz, features = self._break_up_pc(point_clouds)
        pre_enc_xyz, pre_enc_features = self.pre_encoder(xyz, features)
        # xyz: batch x npoints x 3
        # features: batch x channel x npoints

        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        pre_enc_features = pre_enc_features.permute(2, 0, 1)  # (N, B, C)

        # xyz points are in batch x npoint x channel order
        enc_xyz, enc_features, _ = self.encoder(
            pre_enc_features, xyz=pre_enc_xyz
        )
        return enc_xyz, enc_features

    def get_wireframe_prediction(self, query_xyz, wireframe_features, inputs):
        """
        Parameters:
            query_xyz: batch x nqueries x 3 tensor of query XYZ coords
            point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            wireframe_features: num_layers x num_queries x batch x channel
        """
        point_cloud_dims = None
        # wireframe_features change to (num_layers x batch) x channel x num_queries
        wireframe_features = wireframe_features.permute(0, 2, 3, 1)
        num_layers, batch, channel, num_queries = (
            wireframe_features.shape[0],
            wireframe_features.shape[1],
            wireframe_features.shape[2],
            wireframe_features.shape[3],
        )
        wireframe_features = wireframe_features.reshape(num_layers * batch, channel, num_queries)

        # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
        confident_scores = self.mlp_heads["confidence_scores"](wireframe_features).transpose(1, 2)
        center_offset = (
                self.mlp_heads["center_offset"](wireframe_features).sigmoid().transpose(1, 2) - 0.5
        )
        size_lwh = (
                self.mlp_heads["size_lwh"](wireframe_features).sigmoid().transpose(1, 2) * 2
        )
        direction_cls = self.mlp_heads["direction_cls"](wireframe_features).transpose(1, 2)

        # reshape outputs to num_layers x batch x nqueries x noutput
        confident_scores = confident_scores.reshape(num_layers, batch, num_queries, -1)
        center_offset = center_offset.reshape(num_layers, batch, num_queries, -1)
        size_lwh = size_lwh.reshape(num_layers, batch, num_queries, -1)
        direction_cls = direction_cls.reshape(num_layers, batch, num_queries, -1)

        outputs = []
        for l in range(num_layers):
            # wireframe processor converts outputs so we can get a wireframe
            (
                center_normalized,
                center_unnormalized,
            ) = self.wireframe_processor.compute_predicted_center(
                center_offset[l], query_xyz, inputs['centroid'], inputs['max_distance']
            )
            direction_cls_class, size = direction_cls[l].argmax(dim=-1).detach(), size_lwh[l]
            lines = self.wireframe_processor.parametrization_to_wireframe(
                center_normalized, size, direction_cls_class
            )
            confident_prob = torch.sigmoid(confident_scores[l])

            wireframe_prediction = {
                "confident_scores": confident_scores[l],
                "confident_prob": confident_prob,
                "center_normalized": center_normalized.contiguous(),
                "center_unnormalized": center_unnormalized,
                "direction_cls": direction_cls[l],
                "direction_cls_class": direction_cls_class,
                "size": size_lwh[l],
                "lines": lines,
            }
            outputs.append(wireframe_prediction)

        # intermediate decoder layer outputs are only used during training
        aux_outputs = outputs[:-1]
        outputs = outputs[-1]

        return {
            "outputs": outputs,  # output from last layer of decoder
            "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
        }

    def forward(self, inputs, encoder_only=False):
        point_clouds = inputs["point_clouds"]

        # pre_encoder and encoder
        enc_xyz, enc_features = self.run_encoder(point_clouds)
        # GenericMLP
        enc_features = self.encoder_to_decoder_projection(
            enc_features.permute(1, 2, 0)  # (N, B, C) -> (B, C, N)
        ).permute(2, 0, 1)  # (N, B, C)
        # encoder features: npoints x batch x channel
        # encoder xyz: npoints x batch x 3

        if encoder_only:
            # return: batch x npoints x channels
            return enc_xyz, enc_features.transpose(0, 1)

        query_xyz, query_embed = self.get_query_embeddings(enc_xyz, None)
        # query_embed: batch x channel x n_queries (B, 256, n_queries)
        enc_pos = self.pos_embedding(enc_xyz)
        # enc_pos: batch x channel x n_queries

        # decoder expects: npoints x batch x channel
        enc_pos = enc_pos.permute(2, 0, 1)
        query_embed = query_embed.permute(2, 0, 1)
        tgt = torch.zeros_like(query_embed)
        # tgt: (B, C, N_queries)
        wireframe_features = self.decoder(
            tgt, enc_features, query_pos=query_embed, pos=enc_pos
        )[0]
        # Actually, self.decoder return outputs and attention matrix, but here
        # attention matrix is []. So [0] to get the outputs.

        wireframe_predictions = self.get_wireframe_prediction(
            query_xyz, wireframe_features, inputs
        )
        return wireframe_predictions


def build_preencoder(cfg):
    r"""
        return: new_feature, new_xyz, new_ind,  (B, N, C), (B, N, 3), (B, N)
        new_feature: (B, N, C)
        new_xyz: (B, N, 3)
        new_ind: (B, N)
    """
    # [3, 64, 128, 256]
    mlp_dims = [4 * int(cfg.Dataset.use_color) + int(cfg.Dataset.use_intensity), 64, 128, cfg.Model.Encoder.enc_dim]
    preencoder = PointnetSAModuleVotes(
        radius=0.2,
        nsample=64,
        npoint=cfg.Model.PreEncoder.preenc_npoints,
        mlp=mlp_dims,
        normalize_xyz=True,
    )
    return preencoder


def build_encoder(cfg):
    if cfg.Model.Encoder.enc_type == "vanilla":
        encoder_layer = TransformerEncoderLayer(
            d_model=cfg.Model.Encoder.enc_dim,
            nhead=cfg.Model.Encoder.enc_nhead,
            dim_feedforward=cfg.Model.Encoder.enc_ffn_dim,
            dropout=cfg.Model.Encoder.enc_dropout,
            activation=cfg.Model.Encoder.enc_activation,
        )
        encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=cfg.Model.Encoder.enc_nlayers
        )
    elif cfg.Model.Encoder.enc_type in ["masked"]:
        encoder_layer = TransformerEncoderLayer(
            d_model=cfg.Model.Encoder.enc_dim,
            nhead=cfg.Model.Encoder.enc_nhead,
            dim_feedforward=cfg.Model.Encoder.enc_ffn_dim,
            dropout=cfg.Model.Encoder.enc_dropout,
            activation=cfg.Model.Encoder.enc_activation,
        )
        interim_downsampling = PointnetSAModuleVotes(
            radius=0.4,
            nsample=32,
            npoint=cfg.Model.Encoder.preenc_npoints // 2,
            mlp=[cfg.Model.Encoder.enc_dim, 256, 256, cfg.Model.Encoder.enc_dim],
            normalize_xyz=True,
        )

        masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
        encoder = MaskedTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=3,
            interim_downsampling=interim_downsampling,
            masking_radius=masking_radius,
        )
    else:
        raise ValueError(f"Unknown encoder type {cfg.Model.Encoder.enc_type}")
    return encoder


def build_decoder(cfg):
    decoder_layer = TransformerDecoderLayer(
        d_model=cfg.Model.Decoder.dec_dim,
        nhead=cfg.Model.Decoder.dec_nhead,
        dim_feedforward=cfg.Model.Decoder.dec_ffn_dim,
        dropout=cfg.Model.Decoder.dec_dropout,
    )
    decoder = TransformerDecoder(
        decoder_layer, num_layers=cfg.Model.Decoder.dec_nlayers, return_intermediate=True
    )
    return decoder


def build_wftr(cfg):
    # process encoder
    pre_encoder = build_preencoder(cfg)
    # encoder
    encoder = build_encoder(cfg)
    # decoder
    decoder = build_decoder(cfg)
    # 3DETR model
    model = ModelWFTR(
        cfg,
        pre_encoder,
        encoder,
        decoder,
        encoder_dim=cfg.Model.Encoder.enc_dim,
        decoder_dim=cfg.Model.Decoder.dec_dim,
        mlp_dropout=cfg.Model.RegLine.mlp_dropout,
        num_queries=cfg.Model.RegLine.nqueries,
    )
    # Box Processor
    # output_processor = BoxProcessor(dataset_config)
    output_processor = torch.tensor(0)
    return model, output_processor
