import torch
from torch import nn
import math
import numpy as np
from util.math import cal_slot_distance_batch

class PositionEmbedding(nn.Module):
    def __init__(self, d_model, device, dropout=0.1, max_len=500):
        super(PositionEmbedding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (- math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class TimeEmbedding(nn.Module):
    r"""
    Ref: https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs/blob/master/module.py
    The input of ts should be like [E, 1] with all time interval as values.
    """
    def __init__(self, args, embedding_dim):
        super(TimeEmbedding, self).__init__()
        self.time_dim = embedding_dim
        self.expand_dim = self.time_dim
        self.factor = args.phase_factor
        self.use_linear_trans = args.use_linear_trans

        self.basis_freq = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float())
        self.phase = nn.Parameter(torch.zeros(self.time_dim).float())
        if self.use_linear_trans:
            self.dense = nn.Linear(self.time_dim, self.expand_dim, bias=False)
            nn.init.xavier_normal_(self.dense.weight)

    def forward(self, ts):
        if ts.dim() == 1:
            dim = 1
            edge_len = ts.size().numel()
        else:
            edge_len, dim = ts.size()
        ts = ts.view(edge_len, dim)
        map_ts = ts * self.basis_freq.view(1, -1)
        map_ts += self.phase.view(1, -1)
        harmonic = torch.cos(map_ts)
        if self.use_linear_trans:
            harmonic = harmonic.type(self.dense.weight.dtype)
            harmonic = self.dense(harmonic)
        return harmonic

#HST-LSTM model
class DistanceEmbeddingHSTLSTM(nn.Module):
    r"""
    First determine the position of diffrent slot bins, and do linear interpolation within different slots
    with the embedding of the slots as a trainable parameters.
    """
    def __init__(self, args, embedding_dim, spatial_slots):
        super(DistanceEmbeddingHSTLSTM, self).__init__()
        self.dist_dim = embedding_dim
        self.spatial_slots = spatial_slots
        self.embed_q = nn.Embedding(len(spatial_slots), self.dist_dim)
        self.device = args.gpu

    def place_parameters(self, ld, hd, l, h):
        if self.device == 'cpu':
            ld = torch.from_numpy(np.array(ld)).type(torch.FloatTensor)
            hd = torch.from_numpy(np.array(hd)).type(torch.FloatTensor)
            l = torch.from_numpy(np.array(l)).type(torch.LongTensor)
            h = torch.from_numpy(np.array(h)).type(torch.LongTensor)
        else:     
            ld = torch.from_numpy(np.array(ld, dtype=np.float16)).type(torch.FloatTensor).to(self.device)
            hd = torch.from_numpy(np.array(hd, dtype=np.float16)).type(torch.FloatTensor).to(self.device)
            l = torch.from_numpy(np.array(l, dtype=np.float16)).type(torch.LongTensor).to(self.device)
            h = torch.from_numpy(np.array(h, dtype=np.float16)).type(torch.LongTensor).to(self.device)
        return ld, hd, l, h

    def cal_inter(self, ld, hd, l, h, embed):
        """
        Calculate a linear interpolation.
        :param ld: Distances to lower bound, shape (batch_size, step)
        :param hd: Distances to higher bound, shape (batch_size, step)
        :param l: Lower bound indexes, shape (batch_size, step)
        :param h: Higher bound indexes, shape (batch_size, step)
        """
        # Fetch the embed of higher and lower bound.
        # Each result shape (batch_size, step, input_size)
        l_embed = embed(l)
        h_embed = embed(h)
        return torch.stack([hd], -1) * l_embed + torch.stack([ld], -1) * h_embed

    def forward(self, dist):
        self.spatial_slots = sorted(self.spatial_slots)
        d_ld, d_hd, d_l, d_h = self.place_parameters(*cal_slot_distance_batch(dist, self.spatial_slots))
        batch_q = self.cal_inter(d_ld, d_hd, d_l, d_h, self.embed_q)
        return batch_q

#STAN model
class DistanceEmbedding_STAN(nn.Module):
    r"""
    Interpolating between min and max distance value, only need to initial minimum distance embedding and maximum
    distance embedding.
    """

    def __init__(self, args, embedding_dim, spatial_slots):
        super(DistanceEmbedding_STAN, self).__init__()
        self.dist_dim = embedding_dim
        self.min_d, self.max_d_ch2tj, self.max_d_tj2tj = spatial_slots
        self.embed_min = nn.Embedding(1, self.dist_dim)
        self.embed_max = nn.Embedding(1, self.dist_dim)
        self.embed_max_traj = nn.Embedding(1, self.dist_dim)
        self.quantile = args.quantile

    def forward(self, dist, dist_type):
        if dist_type == 'ch2tj':
            emb_low, emb_high = self.embed_min.weight, self.embed_max.weight
            max_d = self.max_d_ch2tj
        else:
            emb_low, emb_high = self.embed_min.weight, self.embed_max_traj.weight
            max_d = self.max_d_tj2tj

        # if you want to clip in case of outlier maxmimum exist, please uncomment the line below
        # max_d = torch.quantile(dist, self.quantile)
        dist = dist.clip(0, max_d)
        vsl = (dist - self.min_d).unsqueeze(-1).expand(-1, self.dist_dim)
        vsu = (max_d - dist).unsqueeze(-1).expand(-1, self.dist_dim)

        space_interval = (emb_low * vsu + emb_high * vsl) / (max_d - self.min_d)
        return space_interval

#no complex model
class DistanceEmbedding_S(nn.Module):
    r"""
    Only need to initial just on embedding, and directly do scalar*vector multiply.
    """
    def __init__(self, args, embedding_dim, spatial_slots):
        super(DistanceEmbedding_S, self).__init__()
        self.args = args
        self.dist_dim = embedding_dim
        self.min_d, self.max_d, self.max_d_traj = spatial_slots
        self.embed_unit = nn.Embedding(1, self.dist_dim)

    def forward(self, dist):
        dist = dist.unsqueeze(-1).expand(-1, self.dist_dim)
        return dist * self.embed_unit.weight

class CheckinEmbedding(nn.Module):
    def __init__(
        self,
        embed_size,
        fusion_type,
        dataset_args
    ):
        super(CheckinEmbedding, self).__init__()
        self.embed_size = embed_size
        self.fusion_type = fusion_type
        self.user_embedding = nn.Embedding(
            dataset_args.num_user + 1,
            self.embed_size,
            padding_idx=dataset_args.padding_user_id
        )
        self.poi_embedding = nn.Embedding(
            dataset_args.num_poi + 1,
            self.embed_size,
            padding_idx=dataset_args.padding_poi_id

        )
        self.category_embedding = nn.Embedding(
            dataset_args.num_category + 1,
            self.embed_size,
            padding_idx=dataset_args.padding_poi_category
        )
        self.dayofweek_embedding = nn.Embedding(8, self.embed_size, padding_idx=dataset_args.padding_weekday_id)
        self.hourofday_embedding = nn.Embedding(25, self.embed_size, padding_idx=dataset_args.padding_hour_id)
        if self.fusion_type == 'concat':
            self.output_embed_size = 5 * self.embed_size
        elif self.fusion_type == 'add':
            self.output_embed_size = embed_size
        else:
            raise ValueError(f"Get wrong fusion type {self.fusion_type}")

    def forward(self, data):
        embedding_list = [
            self.user_embedding(data[..., 0].long()),
            self.poi_embedding(data[..., 1].long()),
            self.category_embedding(data[..., 2].long()),
            self.dayofweek_embedding(data[..., 6].long()),
            self.hourofday_embedding(data[..., 7].long())
        ]
        if self.fusion_type == 'concat':
            self.output_embed_size = len(embedding_list) * self.embed_size
            return torch.cat(embedding_list, -1)
        elif self.fusion_type == 'add':
            return torch.squeeze(sum(embedding_list))
        else:
            raise ValueError(f"Get wrong fusion type {self.fusion_type}")

class EdgeEmbedding(torch.nn.Module):
    def __init__(self, embed_size, fusion_type, num_edge_type):
        super(EdgeEmbedding, self).__init__()
        self.embed_size = embed_size
        self.fusion_type = fusion_type
        self.edge_type_embedding = nn.Embedding(num_edge_type, self.embed_size)
        self.output_embed_size = self.embed_size

    def forward(self, data):
        embedding_list = [self.edge_type_embedding(data.long())]

        if self.fusion_type == 'concat':
            self.output_embed_size = len(embedding_list) * self.embed_size
            return torch.cat(embedding_list, -1)
        elif self.fusion_type == 'add':
            return sum(embedding_list)
        else:
            raise ValueError(f"Get wrong fusion type {self.fusion_type}")
    