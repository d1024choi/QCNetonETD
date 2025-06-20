
from typing import Dict

import torch
import torch.nn as nn
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from fvcore.nn import FlopCountAnalysis

from layers.attention_layer import AttentionLayer
from layers.fourier_embedding import FourierEmbedding
from utils import angle_between_2d_vectors
from utils import merge_edges
from utils import weight_init
from utils import wrap_angle

from torch_cluster import radius
class QCNetMapEncoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 pl2pl_radius: float,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 **kwargs) -> None:
        super(QCNetMapEncoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.pl2pl_radius = pl2pl_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_map_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        if dataset == 'ETRI_Dataset':
            if input_dim == 2:
                input_dim_x_pt = 1    # point dimension
                input_dim_r_pt2pl = 3    # point_to_polygon

                input_dim_r_pl2pl = 3  # polygon_to_polygon
            elif input_dim == 3:
                input_dim_x_pt = 2
                input_dim_r_pt2pl = 4
                input_dim_r_pl2pl = 4
            else:
                raise ValueError('{} is not a valid dimension'.format(input_dim))
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))


        self.x_pt_emb = FourierEmbedding(input_dim=input_dim_x_pt, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.x_pl_emb = FourierEmbedding(input_dim=2, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pt2pl_emb = FourierEmbedding(input_dim=input_dim_r_pt2pl, hidden_dim=hidden_dim,
                                            num_freq_bands=num_freq_bands)
        self.r_pl2pl_emb = FourierEmbedding(input_dim=input_dim_r_pl2pl, hidden_dim=hidden_dim,
                                            num_freq_bands=num_freq_bands)
        self.pt2pl_layers = nn.ModuleList([AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(self.num_layers)])
        self.pl2pl_layers = nn.ModuleList([AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(self.num_layers)])
        self.apply(weight_init)

    # def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
    #     pos_pt = data['map_point']['position'][:, :self.input_dim].contiguous()   # N x 2
    #     orient_pt = data['map_point']['orientation'].contiguous()                 # N
    #     pos_pl = data['map_polygon']['position'][:, :self.input_dim].contiguous() # M x 2
    #     orient_pl = data['map_polygon']['orientation'].contiguous()               # M
    #     orient_vector_pl = torch.stack([orient_pl.cos(), orient_pl.sin()], dim=-1)# M x 2
    #
    #     if self.dataset == 'ETRI_Dataset':
    #         if self.input_dim == 2:
    #             x_pt = data['map_point']['magnitude'].unsqueeze(-1)               # N x 1
    #             x_pl = None
    #         else:
    #             raise ValueError('{} is not a valid dimension'.format(self.input_dim))
    #     else:
    #         raise ValueError('{} is not a valid dataset'.format(self.dataset))
    #
    #     map_polygon_batch = data['map_polygon']['batch'] if isinstance(data, Batch) else None # M (int64)
    #     map_point_batch = data['map_point']['batch'] if isinstance(data, Batch) else None     # N (int64)

    def forward(self, pos_pt: torch.Tensor,
                orient_pt: torch.Tensor,
                pos_pl: torch.Tensor,
                orient_pl: torch.Tensor,
                orient_vector_pl: torch.Tensor,
                x_pt: torch.Tensor,
                x_pl: torch.Tensor,
                map_polygon_batch: torch.Tensor,
                map_point_batch: torch.Tensor,
                edge_index_pt2pl,
                edge_index_pl2pl) -> Dict[str, torch.Tensor]:

        x_pt = self.x_pt_emb(continuous_inputs=x_pt, categorical_embs= None)
        x_pl = self.x_pl_emb(continuous_inputs=pos_pl, categorical_embs= None)


        rel_pos_pt2pl = pos_pt[edge_index_pt2pl[0]] - pos_pl[edge_index_pt2pl[1]]
        rel_orient_pt2pl = wrap_angle(orient_pt[edge_index_pt2pl[0]] - orient_pl[edge_index_pt2pl[1]])
        if self.input_dim == 2:
            r_pt2pl = torch.stack(
                [torch.norm(rel_pos_pt2pl[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pt2pl[1]],
                                          nbr_vector=rel_pos_pt2pl[:, :2]),
                 rel_orient_pt2pl], dim=-1)
        elif self.input_dim == 3:
            r_pt2pl = torch.stack(
                [torch.norm(rel_pos_pt2pl[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pt2pl[1]],
                                          nbr_vector=rel_pos_pt2pl[:, :2]),
                 rel_pos_pt2pl[:, -1],
                 rel_orient_pt2pl], dim=-1)
        else:
            raise ValueError('{} is not a valid dimension'.format(self.input_dim))
        r_pt2pl = self.r_pt2pl_emb(continuous_inputs=r_pt2pl, categorical_embs=None)
        # batch_pl = data['map_polygon']['batch'] if isinstance(data, Batch) else None
        rel_pos_pl2pl = pos_pl[edge_index_pl2pl[0]] - pos_pl[edge_index_pl2pl[1]]
        rel_orient_pl2pl = wrap_angle(orient_pl[edge_index_pl2pl[0]] - orient_pl[edge_index_pl2pl[1]])
        if self.input_dim == 2:
            r_pl2pl = torch.stack(
                [torch.norm(rel_pos_pl2pl[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pl2pl[1]],
                                          nbr_vector=rel_pos_pl2pl[:, :2]),
                 rel_orient_pl2pl], dim=-1)
        elif self.input_dim == 3:
            r_pl2pl = torch.stack(
                [torch.norm(rel_pos_pl2pl[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pl2pl[1]],
                                          nbr_vector=rel_pos_pl2pl[:, :2]),
                 rel_pos_pl2pl[:, -1],
                 rel_orient_pl2pl], dim=-1)
        else:
            raise ValueError('{} is not a valid dimension'.format(self.input_dim))
        r_pl2pl = self.r_pl2pl_emb(continuous_inputs=r_pl2pl, categorical_embs=None)

        for i in range(self.num_layers):
            x_pl = self.pt2pl_layers[i]((x_pt, x_pl), r_pt2pl, edge_index_pt2pl)
            x_pl = self.pl2pl_layers[i](x_pl, r_pl2pl, edge_index_pl2pl)
        x_pl = x_pl.repeat_interleave(repeats=self.num_historical_steps,
                                      dim=0).reshape(-1, self.num_historical_steps, self.hidden_dim)

        return {'x_pt': x_pt, 'x_pl': x_pl}